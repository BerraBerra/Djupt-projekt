# Controlled ResNet18 follow-up: three seeds

Recorded: 2026-11-02. Study: 20261102_084414_406073 (LiU).
Evidence: 15 COMPLETE rows supplied from validation_ranking.csv. Remote histories
and checkpoint files have not been inspected. This is validation, not test performance.

## Design

Five configurations were evaluated with seeds 42, 43 and 44. All use pretrained
ResNet18, initial LR 0.0001926496250871329 and weight decay
1.0567900894501647e-06. Reference: last block and classifier trainable, batch 64,
cosine schedule. Each variant changes only the indicated setting. Checkpoints
are selected by validation macro F1. The documented validation set has 450 images.
Seed replication measures training variation on this split, not uncertainty over
new datasets. The CSV does not establish completed epoch counts or training times.

## Results by seed

Each cell lists trial ID / selected epoch / macro F1 / accuracy.

| Variant | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| reference | 0 / 21 / 0.902204836 / 90.22% | 5 / 29 / 0.897769700 / 89.78% | 10 / 34 / 0.906664823 / 90.67% |
| batch32 | 1 / 25 / 0.904364630 / 90.44% | 6 / 15 / 0.904443973 / 90.44% | 11 / 17 / 0.911095306 / 91.11% |
| plateau | 2 / 8 / 0.895555040 / 89.56% | 7 / 23 / 0.897775759 / 89.78% | 12 / 10 / 0.897775759 / 89.78% |
| constant | 3 / 21 / 0.904440197 / 90.44% | 8 / 39 / 0.908852430 / 90.89% | 13 / 10 / 0.897775759 / 89.78% |
| all_layers | 4 / 19 / 0.920000000 / 92.00% | 9 / 10 / 0.906664823 / 90.67% | 14 / 22 / 0.906659292 / 90.67% |

## Aggregated comparison

Sample standard deviation is across three seeds (n-1 denominator), not a confidence
interval. Rankings use mean macro F1; percentages below refer to mean accuracy.

| Variant | Mean macro F1 | Sample SD F1 | Mean accuracy |
|---|---:|---:|---:|
| all_layers | 0.911108 | 0.007701 | 91.11% |
| batch32 | 0.906635 | 0.003863 | 90.67% |
| constant | 0.903689 | 0.005576 | 90.37% |
| reference | 0.902213 | 0.004448 | 90.22% |
| plateau | 0.897036 | 0.001282 | 89.70% |

## Interpretation and decision

All-layer fine-tuning is the leading configuration by average macro F1 and accuracy,
but it is not a proven global optimum. It beats the reference for seeds 42 and 43
and is effectively tied at seed 44 (macro F1 is very slightly lower). Its mean
accuracy advantage over batch32 is only 0.44 percentage points, equivalent to two
correct predictions per run on average; batch32 wins their seed-44 comparison.
Three seeds and one small validation split do not establish statistical superiority.

This revises the first round's impression: all-layer fine-tuning can be competitive
with suitable settings. The controlled follow-up, rather than the architecture
alone or one isolated trial, supports the decision.

Proceed with all_layers as the provisional main configuration; preserve batch32
as a close alternative. Do not spend the next budget on more head-only or plateau
experiments under these data conditions. This is prioritization, not universal
rejection of these methods. Do not infer 20 epochs suffice from selected epochs;
the cosine schedule and training budget interact.

Trial 4 (seed 42, checkpoint epoch 19) is the best individual validation checkpoint.
If choosing a single artifact by validation score, record that selection explicitly.
Its 92% validation accuracy is not the same measurement as the earlier baseline's
92.44% test accuracy. Do not choose between seeds by looking at their test scores.

## Next steps

1. Preserve all 15 checkpoints, settings, histories and this report.
2. Freeze the selected configuration. Use trial_0004/best_model.pt as the working
   artifact if following the best-validation-checkpoint rule; retain its provenance.
3. Evaluate that fixed artifact once on the historical test set for continuity,
   saving results in its own folder. Label this test as previously observed rather
   than untouched final evidence. Alternatively, report all three all_layers seed
   checkpoints as a predeclared group without selecting the best test seed.
4. Prioritize a new, provenance-checked external evaluation dataset before further
   tuning. The present gains are small enough that source generalization is a more
   informative next question than another broad hyperparameter search.
5. For a later poster task, retain this model and fine-tune a separately versioned
   model using appropriate labeled composites and disjoint backgrounds/templates.

No checkpoints were copied from LiU and no test evaluation was performed locally
as part of writing this report.
