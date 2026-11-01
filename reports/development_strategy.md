# ResNet18 development strategy

Recorded: 2026-11-01. Status: experimental plan, not implemented results.
Baseline: pretrained ResNet18, 416/450 correct (92.44%), selected checkpoint 24.
See baseline_report.md and experiment_log.csv for the recorded comparison.

## Objective and project organization

Improve validation performance and generalization to unfamiliar generators and
photo sources, while preserving the full experiment history. Keep this repository,
the shared model factory and evaluation code. Do not duplicate the entire project
or overwrite the baseline with a new method. Add method-specific modules only
when implementing a genuinely new training objective.

Use stages B0 (baseline), S1 (supervised optimization), D1 (additional labeled
data), P1 (pseudo-labeling), P2 (consistency learning), F1 (final evaluation).
For every run record stage, unique run ID, parent checkpoint if any, Git revision,
effective config, seed, data manifest hash, software/device information, best and
completed epochs, metrics, checkpoint, and training history. Append results instead
of replacing historical log entries. Keep dataset versions as separate manifests.

Existing code: pretrained_experiments.py already creates timestamped run folders,
effective settings and validation summaries. Complete environment/revision capture,
semi-supervised training and dataset acquisition are still future implementation.
Only results supplied by the user have been archived locally: the actual baseline
weights and histories still need to be preserved on LiU. Put reports and code in
Git; store large model/data artifacts separately with hashes and locations.

## Experimental sequence

| Stage | Question | Controlled comparison | Advancement criterion |
|---|---|---|---|
| B0 | What is our starting point? | Preserve all three existing results | Recover remote configurations and histories |
| S1a | Does LR scheduling help? | Constant / plateau / cosine; same initial LR, data, seed and stopping rule | Lowest validation loss, with accuracy and class recall reported |
| S1b | Is initial LR suitable? | 3e-5 / 1e-4 / 3e-4 with chosen schedule fixed | Repeat preferred settings over seeds 42, 43, 44 |
| S1c | Does a different fine-tuning strategy help? | Train all layers versus classifier-only warm-up, then unfreeze | Improvement across seeds, not just a single best run |
| D1 | Does more labeled data help? | Nested subsets of 2,100 / 5,000 / 10,000 training images if eligible data permits | Better validation and source-held-out development performance |
| P1 | Can extra images help without their labels? | Labeled-only control versus labeled + pseudo-labeled, from same parent weights | Better validation and acceptable pseudo-label error in diagnostic audit |
| P2 | Does consistency improve P1? | P1 versus a FixMatch-inspired objective | Stable gain across seeds and controlled augmentations |
| F1 | Does the selected method generalize? | Frozen configuration on an untouched external set | Report all outcomes, including failures |

The numbers above are planned budgets, not available verified image counts.
Run S1a first rather than the full grid. Keep architecture and image size fixed
initially. LR and augmentation experiments should precede larger architectures.
Longer training is a control: compare SSL against supervised continuation from the
same parent checkpoint for a matched optimizer-step budget. For new supervised
comparisons start each candidate from the same ImageNet initialization protocol.

## Data boundaries

Keep the current test set for historical reporting; repeated observation has made
it unsuitable as the sole untouched final test. Construct a separate external
development set for source-shift tuning and a locked external final test before
new data enters training. Reserve whole generator families and photo sources
where possible, not just random files. Document any ImageNet overlap uncertainty
because the backbone was pretrained on ImageNet.

No validation, development-test or final-test image may enter the unlabeled pool.
Remove exact pixel duplicates and inspect near-duplicates/related subjects across
old and new data, including all evaluation partitions. Group related images in one
partition. Do not claim a newly held-out generator is unseen if it occurred in a
parent checkpoint's training data; start that experiment from ImageNet instead.

Match class distributions in resolution, encoding and content where practical.
Do not solve class imbalance by selecting data solely from the current model's
predictions. Record class/source/generator counts and reasons for exclusion.
Preserve raw images; create transformations consistently across both classes.
For Photo/AI scope, exclude non-AI artwork and ambiguous mixed edits by a documented
rule applied independently of model correctness. Verify source labels: 'real' can
mean non-generated imagery rather than strictly camera photographs.

## Candidate data sources (checked 2026-11-01)

1. [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage):
   the current dataset's source; the card lists 28,000 training rows with labels
   and generator IDs. First candidate for a modest labeled-data expansion and
   controlled hidden-label experiments. Exclude our existing images and their
   near-duplicates before sampling. This is not an independent external source.
   The card declares CC-BY-NC-SA-4.0; record provenance and applicable terms.
2. [GenImage](https://github.com/GenImage-Dataset/GenImage): larger parent benchmark.
   Useful if the small source cannot supply the desired sample, but shares lineage
   with the existing data. Do not present it as an independent dataset without
   source/generator-level separation and overlap checks.
3. [Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics):
   candidate for more varied generators and external evaluation. Metadata includes
   generator and real-source fields. Audit photographic scope and source overlap
   before selecting a subset; different dataset names do not imply independence.
   The authors' [download page](https://jespark.net/projects/2024/community_forensics/download_dataset.html)
   lists 278 GB even for Train-small. Begin with metadata and selected shards,
   not a full download. Actual subset transfer size and suitable example counts
   must be measured. The card declares CC-BY-4.0, while the authors also specify
   research use and per-model license conditions; inspect applicable metadata.

Acquisition workflow: inspect schema and shards, pin source revision, sample a
small pilot of both classes, verify labels/decoding/duplicates, then download a
bounded subset on LiU. Save source ID, revision, label, generator, image dimensions,
hash and split in a versioned manifest. No new images have been downloaded yet.

## Semi-supervised experiment design

Pseudo-labeling combines real labels with model-generated labels, so it is
semi-supervised rather than purely unsupervised. First use a labeled source but
hide labels for a fixed subset of the TRAINING pool. Keep hidden labels in a
separate diagnostic file inaccessible to the training loader. This provides an
audit of pseudo-label accuracy without requiring unknown-origin web imagery.

Compare three conditions on identical evaluation data: labeled subset only;
labeled subset plus the hidden-label pool treated as unlabeled; and all those
training images with their true labels. The last is a full-supervision reference,
not a guaranteed numerical upper bound. Never substitute pseudo-labels for known
labels merely to increase production training data: hiding labels is an experiment.

For P1, use the selected supervised model as a frozen teacher. Accept high-confidence
predictions (e.g. initial threshold 0.95, selected using validation diagnostics),
and train a student from the same parent weights on labeled and accepted images.
Keep true-labeled samples in every batch, limit pseudo-labeled contribution, log
acceptance rate and class/source balance, and retain rejected images separately.
Confidence is not correctness or a calibrated probability. Begin with one round;
do not repeatedly reinforce labels without checking results.

Objective: L = L_supervised + lambda_u * L_pseudo. L_pseudo is cross-entropy against
the teacher's detached hard labels for accepted images. Specify normalization
explicitly (e.g. divide by the whole unlabeled batch size, including rejected
examples contributing zero); define zero accepted examples as zero pseudo loss.
Tune lambda_u conservatively on validation, and compare against matched-budget
supervised continuation to separate extra training from the benefit of new data.

P2 can add weak/strong-view consistency following
[FixMatch](https://research.google/pubs/fixmatch-simplifying-semi-supervised-learning-with-consistency-and-confidence/).
For image forensics, verify that transformations preserve useful evidence:
strong blurring, downsampling or compression may erase the traces being learned.
This is an adaptation to test, not a claim that standard FixMatch is optimal here.
[Confirmation bias research](https://arxiv.org/abs/1908.02983) explains why incorrect
pseudo-labels can be reinforced. Measure that failure mode explicitly.

Self-supervised representation learning without Photo/AI labels is a separate
possible later study; prioritize supervised expansion and controlled SSL first.

## Reporting and selection

Select on validation loss with validation accuracy, macro F1 and both class recalls
reported. Record runtime, data volume and parameter count too. Report mean and
spread across seeds rather than only the highest score. Keep threshold selection
on validation; final evaluation uses a fixed threshold and fixed preprocessing.
Report per-source/generator performance and controlled JPEG/resize robustness.
The final set must never be used to choose pseudo-label thresholds or schedules.
An unsuccessful SSL experiment is still a useful result: report it and keep the
stronger supervised checkpoint rather than forcing adoption of the new method.
