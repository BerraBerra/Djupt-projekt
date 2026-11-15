# Photo versus AI classification: findings and continuation plan

**Status on 15 November 2026.** The selected ResNet18 checkpoint correctly classified 1,247 of 2,000 images (62.35%) in the external evaluation. Its accuracy on the original validation split was 92.00%. The 29.65 percentage point gap makes performance on new image sources the main open question for this project.

## How we reached this result

1. We trained a small CNN, a randomly initialized ResNet18 and an ImageNet-pretrained ResNet18 on 2,100 labeled images. Pretrained ResNet18 led the initial comparison with 92.44% accuracy on the original 450-image **test** split. That score belongs to the original baseline checkpoint. [Baseline report](baseline_report.md)
2. Nine validation trials identified fine-tuning the last block as a promising reference. We then compared five variants across the same three seeds. Fine-tuning all layers had the highest mean validation accuracy (91.11%). Trial 4 used seed 42, batch size 64, cosine decay and an initial learning rate of approximately 0.000193. We selected its epoch-19 checkpoint by validation macro F1 before evaluating it externally. [First search](search_round_1.md) · [Three-seed comparison](search_round_2.md)
3. The fixed checkpoint evaluated 1,000 photos and 1,000 AI images from a custom Community Forensics subset. It correctly identified 695 photos and 552 AI images. It flagged 305 photos as AI and classified 448 AI images as photos. External macro F1 was 0.62 as printed. [Dataset protocol and full results](external_dataset.md)

## What today's evaluation established

AI recall ranged from 25.5% on DFGAN to 80.5% on FLUX-dev, with 200 examples per generator. DFGAN and Firefly Image 2 accounted for 266 of the 448 missed AI images. Errors in the other direction also depended on source. The model flagged 137 of 183 FFHQ photos as AI, compared with 168 of 817 COCO photos. The larger COCO group therefore had more influence on overall photo recall.

These findings identify where the detector struggled on this sample. Subject matter, resolution and encoding can all vary across sources. Their individual effects remain open. The project has a working supervised classifier, a reproducible model comparison and a documented measurement of generalization to a second dataset.

## Where to resume

**First session back:** Open `test_predictions.csv` from the LiU evaluation folder below and join it to `data/external_community/manifest.csv` by `path`. With a recorded random seed, sample 20 correct and 20 incorrect predictions from each of DFGAN and FFHQ. Compare subject matter, file format, original dimensions and prediction score within each source. Record one testable explanation for the observed errors and the evidence that would contradict it.

**Next supervised experiment:** Use that audit to choose fresh labeled training images from relevant sources. Keep the existing checkpoint as the comparison model. Train the new version with the same configuration and seeds, varying the training data first. Track photo false positives and AI recall per generator alongside macro F1. Once these external results guide development, reserve new, untouched sources for the final evaluation.

**Later research options:** To measure the benefit of unlabeled images, compare supervised continuation with pseudo-labeling from the same starting weights and training budget. Keep posters as a separate experiment with known background origin and disjoint backgrounds and templates across splits. Both questions can reuse the selected ResNet18. [Longer-term strategy](development_strategy.md)

## LiU artifacts

Paths are relative to the LiU project directory. Results in this report are transcribed from the supplied LiU output.

- Selected weights: `outputs/search_pretrained/20261102_084414_406073/trial_0004/best_model.pt`
- Training configuration and learning curves: `settings.json` and `history.csv` beside those weights.
- External evaluation: `outputs/external_evaluation/20261115_142005_812160/` contains `evaluation.txt`, `source_results.csv`, `test_predictions.csv` and `evaluation_provenance.json`.
