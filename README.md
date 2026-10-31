# Image Origin Classifier

A small PyTorch CNN that classifies RGB images as `photo` or `ai_generated`. The model learns directly from pixels, not precomputed features.

## Quick start

Use Python 3.11 or newer. From the project folder:

```bash
python -m pip install -r requirements.txt
python training.py
python evaluate.py
python predict.py "path/to/image.jpg"
```

`predict.py` also accepts a folder. CUDA is used automatically during training when
available; prediction works on CPU. For a GPU server, install the PyTorch build
compatible with that server's driver/environment before running a large experiment.

Edit **settings.py** to change epochs, learning rate, batch size, image size or paths.
The defaults are 224x224, 50 epochs, batch size 32 and learning rate 0.001.
The terminal shows the current epoch, train/validation phase, percentage, batches,
estimated remaining time, loss and accuracy. No HTML report or JSON config is required.

## Project structure

```text
training.py                Start training
settings.py                All experiment settings and default paths
evaluate.py                Test the saved model
predict.py                 Classify your own images
image_classifier/
    model.py               CNN architecture and checkpoint loading
    losses.py              Cross-entropy objective
    data.py                Read images, transforms and fixed splits
    training.py            Optimization, validation and checkpoint selection
    evaluation.py          Test metrics and predictions
    inference.py           Prediction for new images
    __init__.py            Python package marker
data/
    images/
        photo/
        ai_generated/
    manifest.csv           Image paths, labels, splits and source metadata
    README.md              Dataset sources, attribution and limitations
outputs/                   Created by training/evaluation/prediction
tests/test_model.py        Small automated correctness checks
requirements.txt           Runtime dependencies
```

The root training.py is the short entry point. image_classifier/training.py
contains the actual training loop. The model architecture is defined in only one
place: image_classifier/model.py.

## Data

The current local sample has 3,000 images: 1,500 per class.
Each class has 1,050 training images, 225 validation images and 225 test images.
Images and manifest.csv must remain together. The manifest fixes the split;
training only shuffles the order of training examples, not their split membership.

- `photo`: photographs, according to source labels.
- `ai_generated`: generated content, regardless of visual style.

This is binary classification: photographs versus AI-generated images.
Non-AI illustrations, cartoons and mixed/edited content are outside the validated scope.
See data/README.md for sources, revisions, usage restrictions and sampling limitations.

Images are resized to a square without preserving their aspect ratio. This can
introduce distortions; the transform is in image_classifier/data.py. Keep the same
transform for validation, test and prediction when comparing an experiment.
Only training uses random horizontal flips. Pixels are normalized to [-1, 1].

## Architecture and learning objective

```text
[B, 3, 224, 224]
  -> Conv(3 -> 32, 3x3, padding=1), BatchNorm, ReLU, MaxPool
[B, 32, 112, 112]
  -> Conv(32 -> 64, 3x3, padding=1), BatchNorm, ReLU, MaxPool
[B, 64, 56, 56]
  -> Conv(64 -> 128, 3x3, padding=1), BatchNorm, ReLU, MaxPool
[B, 128, 28, 28]
  -> Global average pooling, Flatten
[B, 128]
  -> Linear(128 -> 64), ReLU, Dropout(0.25)
[B, 64]
  -> Linear(64 -> 2)
[B, 2] logits
```

There are 101,858 trainable parameters. All layers are trained from random
initialization; no pretrained weights or frozen layers are used.

Cross-entropy receives logits directly. With label smoothing epsilon=0.05:

```text
p_k = softmax(logits)_k
q_k = (1 - epsilon) * indicator(k == target) + epsilon / 2
loss = -sum_k q_k * log(p_k)
```

AdamW updates weights after each training batch. Weight decay is 0.0001.
Validation does not update weights. The checkpoint with the lowest validation loss
is saved; early stopping triggers after 10 epochs without improvement by default.
The test split is used separately by evaluate.py, never for checkpoint selection.

## Outputs: only useful artifacts

| File | Created by | Purpose |
|---|---|---|
| best_model.pt | training.py | Model weights, settings, class order, epoch and manifest hash |
| history.csv | training.py | Training/validation loss and accuracy per epoch |
| evaluation.txt | evaluate.py | Accuracy, per-class precision/recall/F1 and confusion matrix |
| test_predictions.csv | evaluate.py | Each test image's label, prediction and scores |
| predictions.csv | predict.py | Predictions for your own images |

Outputs are created only when you run the corresponding command. Reusing the
same output directory can overwrite previous results; change OUTPUT in settings.py
for a new experiment. There are no bundled old models: run training first.
Checkpoints store settings internally, so no separate config JSON is needed.
A checkpoint is for inference, not exact training resumption: optimizer/RNG state
is not stored. Changing architecture can make an old checkpoint incompatible.

## Interpretation and research limitations

Softmax values sum to one, but are not calibrated probabilities or proof of origin.
The default review flag below 0.7 confidence is a heuristic, not a validated detector.
Source/style differences can be easier to learn than genuine AI traces.
Both classes come from the Tiny-GenImage sample. Exact duplicates were
removed, but near-duplicates and shared subjects may remain. A stronger study
needs source/generator-held-out testing and more diverse, matched image content.

Pseudo-labeling is not implemented. A future extension would be semi-supervised
learning and should not add validation/test images to training.

## Sharing on GitHub/GitLab and running on another computer

Commit source code, tests, requirements and documentation. Images, manifest.csv,
outputs and virtual environments are excluded by .gitignore; data/README.md is kept
for attribution. No dataset downloader is included in this cleaned project.
Transfer the existing data/images and data/manifest.csv separately, subject to the
source licenses. A fresh clone therefore needs the dataset before training.

On LiU: clone the code, install dependencies, transfer data, check CUDA availability,
then run python training.py in the appropriate compute session. No particular
scheduler or host configuration is assumed. Copy outputs/best_model.pt back to
use the trained model locally. The old pilot datasets and checkpoints were removed.

## Tests

```bash
python -m unittest discover -s tests -v
```
