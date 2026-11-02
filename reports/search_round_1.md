# Pretrained ResNet18: first hyperparameter search

Recorded: 2026-11-02.
Study ID: 20261101_185748_129895 (LiU).
Evidence: nine rows from validation_ranking.csv supplied by the project author.
All nine trials have COMPLETE status. Raw histories, checkpoints and the remote
study configuration have not been inspected for this analysis.

## Purpose and procedure

Investigate combinations of initial learning rate, weight decay, batch size,
trainable layers and learning-rate schedule for ImageNet-pretrained ResNet18.
The implemented search maximizes validation macro F1 and selects each trial's
checkpoint by that metric. Test data is not used by this search.
The project validation set contains 450 images, balanced across Photo and AI.
Confirm the remote manifest hash and seed from the study artifacts for the final
report. The CSV alone does not verify these or the exact stopping settings.

## Observed results

| Trial | Trainable layers | Initial LR | Weight decay | Batch | Schedule | Selected epoch | Val accuracy | Macro F1 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 6 | Last block + head | 0.000192650 | 0.00000105679 | 64 | Cosine | 21 | 90.22% | 0.902204836 |
| 7 | Last block + head | 0.000141389 | 0.00000115446 | 64 | Cosine | 11 | 90.22% | 0.902204836 |
| 5 | Last block + head | 0.000697828 | 0.00000225928 | 64 | Cosine | 11 | 90.22% | 0.902173913 |
| 0 | Last block + head | 0.000056115 | 0.00635122101 | 16 | Plateau | 20 | 89.56% | 0.895530276 |
| 8 | All layers | 0.000011401 | 0.00004255007 | 32 | Cosine | 19 | 88.22% | 0.882221641 |
| 3 | All layers | 0.000790262 | 0.00728665374 | 16 | Cosine | 36 | 84.44% | 0.844432153 |
| 1 | Head only | 0.000870602 | 0.00213683291 | 16 | Plateau | 24 | 82.00% | 0.819799778 |
| 2 | Head only | 0.000038396 | 0.00002920434 | 32 | Constant | 89 | 80.89% | 0.808582621 |
| 4 | Head only | 0.000032928 | 0.00044677528 | 64 | Constant | 100 | 79.33% | 0.793037965 |

Selected epoch denotes the saved checkpoint, not total completed epochs.
The three leading trials each classified 406 of 450 validation examples correctly.
Their nearly identical macro F1 scores do not establish a meaningful winner;
they may nevertheless make errors on different images.

## Interpretation

Last-block fine-tuning produced 89.56-90.22% accuracy in four tested combinations,
compared with 84.44-88.22% for all-layer fine-tuning (two combinations) and
79.33-82.00% for head-only training (three combinations). This suggests prioritizing
last-block fine-tuning, but it is not an isolated causal comparison of layer freezing:
the other hyperparameters changed simultaneously.

The three leading configurations share batch size 64, cosine scheduling and low
weight decay, with initial LR spanning about 1.4e-4 to 7.0e-4. This identifies a
promising region rather than proving that any individual setting is optimal.
Trials 6 and 7 were selected adaptively by the search; these are not independent
seed replications or evidence of statistical significance.

## Decision log

| Decision | Reason | Limitation |
|---|---|---|
| Deprioritize head-only training in the next round | Three tested configurations were consistently weaker | Not proof that every head-only configuration is inferior |
| Use last-block fine-tuning as the main approach | All four tested configurations were competitive | Other hyperparameters confound the comparison |
| Use trial 6 as a reference | Among the tied leading trials; moderate initial LR | Not a demonstrated unique winner |
| Retain all-layer training as a control | Only two configurations were tested | Its best settings may be different |
| Retest batch 32 and alternative schedules | Their individual effects are unresolved | Constant LR was tested only with a frozen backbone |
| Do not infer a shorter epoch budget from checkpoint epochs | Cosine timing changes with the epoch limit | Actual histories and stopping points remain to be reviewed |

## Proposed follow-up (not run)

Implementation update, 2026-11-02: search_pretrained.py now implements this fixed
15-run plan as its default. It exports variant_summary.csv and can continue missing
combinations. The actual training runs remain to be performed on LiU.

Repeat the following five configurations with the same three seeds (42, 43, 44):
trial 6 reference; reference with batch size 32; reference with plateau scheduling;
reference with constant LR; reference with all layers trainable. Change only the
named setting in each variant, retaining the reference's exact LR and weight decay.
This is 15 planned runs. Use a common maximum epoch budget and stopping rules,
recording runtime and optimizer-update counts because changing batch size changes
the number of updates per epoch. Report mean and spread, and per-class performance.

These local comparisons support interpretation but cannot exhaust all interactions.
Preserve this search and create a separately named experiment for follow-up.
Select using validation, then use a separate untouched dataset for final evaluation.
The earlier 92.44% result was test accuracy and must not be directly compared with
these validation percentages as evidence of improvement or deterioration.
