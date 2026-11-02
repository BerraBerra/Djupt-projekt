# Photo versus AI: baseline model comparison

Follow-up recorded on 2026-11-02: [first hyperparameter search](search_round_1.md).
It documents nine completed validation trials and the rationale for the next
experiments. The plan below is retained as historical context; the subsequent
Optuna search selects by validation macro F1 rather than validation loss.

Recorded: 2026-11-01. Source: evaluation output supplied by the project author
from a LiU computer. This is the date the results were recorded, not a verified
training date. The remote checkpoints and training histories have not been inspected.

## Research question

How do architecture and ImageNet pretraining affect classification of photographs
and AI-generated images with a limited labeled training set?

## Experimental setup

The project uses supervised binary classification directly from RGB images.
The documented dataset contains 3,000 images: 2,100 training, 450 validation and
450 test images, balanced between the two classes in each split. The supplied
evaluation confirms 450 test images, including 225 examples per class.

The implemented comparison uses a fixed manifest, selects checkpoints by lowest
validation loss, and evaluates these saved checkpoints on the test split.
The small CNN has 101,858 trainable parameters. Both ResNet18 variants have
11,177,538 parameters after replacing the original classifier with two outputs.
The pretrained model starts with ImageNet backbone weights and a randomly
initialized classifier. The current implementation trains all layers.

The project code uses AdamW and cross-entropy with optional label smoothing.
However, the exact remote learning rates, scheduler, seed, epoch limit,
normalization, batch size, software versions and Git revision are not present in
the supplied evaluation output. They must be recovered from the saved checkpoint
configuration and training environment before claiming an exactly reproducible run.
In particular, checkpoint epoch is not the number of epochs actually completed.

## Results

| Model | Checkpoint epoch | Correct / 450 | Errors | Test accuracy | Macro F1 (printed) |
|---|---:|---:|---:|---:|---:|
| Small CNN | 46 | 367 | 83 | 81.56% | 0.816 |
| ResNet18, random initialization | 11 | 312 | 138 | 69.33% | 0.693 |
| ResNet18, pretrained | 24 | 416 | 34 | 92.44% | 0.924 |

The exact counts come from the confusion matrices; percentages and macro F1
are rounded as printed. These are the authoritative results for this comparison,
superseding earlier approximate conversational figures (83%, 70%, 93%). An earlier
CNN run reported as 83% is a separate experiment, not this run.

Rows are true classes and columns predicted classes, both ordered Photo, AI:

```text
Small CNN                  Random ResNet18            Pretrained ResNet18
             Photo  AI                  Photo  AI                  Photo  AI
Photo          184  41     Photo          157  68     Photo          211  14
AI              42 183     AI              70 155     AI              20 205
```

For pretrained ResNet18, 14 photographs were incorrectly labeled AI and 20
AI-generated images were incorrectly labeled photographs. Photo recall was
211/225 = 93.78%; AI recall was 205/225 = 91.11%.

## Interpretation

Pretrained ResNet18 outperformed the small CNN by 49 correct predictions
(10.89 percentage points) and random ResNet18 by 104 (23.11 percentage points).
Its error count was about 59% lower than the small CNN's (34 versus 83).

The results support using pretraining under the tested conditions. They do not
isolate pretraining as the sole cause: the intended settings use a different
starting learning rate for the pretrained model. The small CNN outperforming
random ResNet18 also shows that more parameters alone did not improve this run.
Whether random ResNet18 overfit or was insufficiently optimized requires its
training and validation curves; test accuracy alone cannot answer that question.

## Limitations

- One reported run per model; variation across random seeds is unknown.
- The full remote training configuration is unverified.
- Shared source characteristics can support classification without generalizing
  to new generators, cameras or processing methods.
- Accuracy on this balanced test set does not establish deployment performance
  or calibrated probabilities.
- Test results have already influenced the choice to focus on pretrained ResNet18.
  Further tuning should use validation, followed by a fresh held-out evaluation
  for a stronger final generalization claim.

## Next experiments

1. Preserve all baseline checkpoints, histories and evaluation files on LiU.
2. Compare constant, plateau and cosine LR schedules for pretrained ResNet18,
   keeping the initial LR, epoch budget, early stopping, data and seed fixed.
3. Select the schedule using validation loss, reporting validation accuracy too.
4. With the selected schedule fixed, investigate initial LR (for example 3e-5,
   1e-4 and 3e-4), then repeat the preferred configuration across several seeds.
5. Evaluate robustness on a new source/generator-held-out dataset. Keep its labels
   out of training and parameter selection.

These are planned experiments, not results or guaranteed improvements.
`pretrained_experiments.py` implements step 2 with a new timestamped output
directory for each invocation. It starts each experiment from ImageNet weights,
not from the baseline checkpoint, and never evaluates the test split.
