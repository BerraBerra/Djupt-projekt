# Project journal — Photo or AI?

Use this page to follow the decisions in order. For each future experiment, append a dated entry under **Experiment history** with the question, the result, the decision it led to, and a link to the evidence. Update **Where to resume** when the next question changes.

## Experiment history

### Starting point — define the task and fix the split

An early prototype included an animated-image class. We narrowed the question to whether an image was a photograph or AI-generated and trained directly on RGB pixels. We fixed a balanced sample of 3,000 labeled Tiny-GenImage images: 2,100 for training, 450 for validation and 450 for testing. This let us compare models on the same images. [Dataset preparation and attribution](../data/README.md) · [Tiny-GenImage source](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)

### 1 November 2026 — choose a model family

On the original test split, the small CNN scored 81.56% accuracy, randomly initialized ResNet18 scored 69.33%, and ImageNet-pretrained ResNet18 scored 92.44%. We chose pretrained ResNet18 for further study. The comparison covers the training settings used for each model, including their starting learning rates. [Baseline report](baseline_report.md) · [Recorded test results](experiment_log.csv)

### 2 November 2026 — explore fine-tuning settings

Across nine completed validation trials, three configurations reached 90.22% accuracy. All three trained the last residual block and classifier with batch size 64 and cosine decay. Several settings varied together. We took trial 6 as a reference and designed a follow-up that changed one setting at a time. [Nine trials and decision log](search_round_1.md)

### 2 November 2026 — repeat the comparison across seeds

We ran five configurations with seeds 42, 43 and 44. Training all layers achieved the highest mean validation accuracy (91.11%), followed by the batch-32 variant (90.67%). We selected trial 4, seed 42, epoch 19 as the working checkpoint. It scored 92.00% on validation. We kept these weights fixed for testing on new sources. [All 15 runs, parameter values and checkpoint selection](search_round_2.md)

### 3 November 2026 — prepare an external evaluation

We assembled a separate, balanced set of 2,000 labeled images: 1,000 photos from COCO and FFHQ and 200 AI images from each of five generators. We screened for duplicates against the original dataset and within the new sample. This gave us a way to measure performance when the image sources change. [Sampling protocol and limitations](external_dataset.md) · [Community Forensics-Eval source](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval)

### 15 November 2026 — measure performance on the new sources

The selected checkpoint correctly classified 1,247 of 2,000 external images (62.35%). It labeled 149 of 200 DFGAN images as photos and 137 of 183 FFHQ photos as AI. Those errors in opposite directions motivated a closer look at image content and processing. The errors alone leave their cause open. [Results by source](external_dataset.md#recall-by-generator-and-photo-source) · [Findings and paths to the LiU artifacts](project_status_2026-11-15.md)

## Where to resume

1. **Inspect the errors.** Sample correct and incorrect predictions within DFGAN and within FFHQ. Compare subject matter, dimensions and encoding. Write down one explanation that could be tested on fresh images.
2. **Test whether more training data helps.** If the audit suggests missing coverage, collect fresh labeled photos and AI images from relevant sources. Compare training on the original data with training on the expanded set, using the same ResNet18 settings and seeds. Evaluate both versions on the same held-out sources. Keep the current external images out of training and reserve fresh sources for final evaluation.
3. **Keep later questions separate.** Pseudo-labeling and posters with known background origin are possible future experiments. [Detailed continuation plan](project_status_2026-11-15.md#where-to-resume) · [Research strategy](development_strategy.md)
