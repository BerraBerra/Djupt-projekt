# Image Origin Classifier

A PyTorch project comparing a small CNN, randomly initialized ResNet18 and pretrained ResNet18 for `photo` versus `ai_generated`. Models learn directly from pixels.

## Quick start

Use Python 3.11 or newer. From the project folder:

### First-time setup on Linux / LiU

Create an isolated Python environment once on each computer, then activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first command creates the `.venv/` directory. Activation makes `python` and
`python -m pip` use that environment in the current terminal. Packages are installed
inside the environment, separately from the system Python. The prompt usually
starts with `(.venv)` when activated.

### Each new terminal or login

Go to the project folder and activate the existing environment before running:

```bash
source .venv/bin/activate
python training.py
python evaluate.py
python predict.py "path/to/image.jpg"
```

You do not need to recreate the environment or reinstall packages on every login.
Run `deactivate` to leave it. Activation only affects the current terminal.
Check which interpreter is active with:

```bash
python -c "import sys; print(sys.executable)"
```

The path should point inside the project's `.venv/` directory. Activation is optional
if you explicitly use `.venv/bin/python training.py` instead.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python training.py
```

If activation is unavailable, run `.\.venv\Scripts\python.exe training.py` directly
and install packages with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
Create a separate environment on each computer; do not copy a Windows environment
to Linux. `.venv/` is excluded from Git and does not contain your dataset.

`predict.py` also accepts a folder. CUDA is used automatically during training when
available; prediction works on CPU. For a GPU server, install the PyTorch build
compatible with that server's driver/environment before running a large experiment.

Edit **settings.py** to change epochs, learning rate, batch size, image size or paths.
The defaults are 224x224, 50 epochs, batch size 32 and learning rate 0.001.
The terminal shows the current epoch, train/validation phase, percentage, batches,
estimated remaining time, loss and accuracy. No HTML report or JSON config is required.

## Model and learning rate settings

Select `MODEL` in settings.py: `cnn`, `resnet18_random`, or `resnet18_pretrained`.
`TRAINING` contains shared settings. `MODEL_SETTINGS` contains per-model overrides:
both random models start at LR 0.001, while pretrained ResNet18 starts at 0.0001.
These are starting points, not validated optimal settings. All layers are trainable;
there is no freezing stage. ResNet18 pretrained weights download on the first
training run (internet required); saved checkpoints load offline.

Set `TRAINING['learning_rate_type']` to one of:

| Value | Behavior |
|---|---|
| `constant` | Keep the starting LR; default for an initial baseline |
| `linear` | Decrease linearly to `min_learning_rate` at the last planned epoch |
| `cosinedecay` | Decrease on a cosine curve to the same minimum |
| `plateau` | Reduce LR by `lr_factor` after more than `lr_patience` epochs without sufficient validation-loss improvement |

Each model has its own optimizer and scheduler. Plateau reductions can occur at
different epochs despite shared settings. `patience` controls early stopping,
separately from `lr_patience`; allow time for training after LR reductions.
Early stopping can end linear/cosine runs before they reach their minimum LR.
The LR actually used in each epoch appears in the terminal and history.csv.

```bash
python training.py        # Train the selected MODEL
python compare_models.py  # Train all three sequentially; summarize validation
python evaluate_compare.py  # Evaluate all three saved models on the test split
```

Results live in `outputs/<model_name>/`. comparison.csv summarizes the best
validation-loss epoch for each model; it does not automatically evaluate test data.
`evaluate_compare.py` requires all three checkpoints and uses the same test split
for each model. It prints a table sorted by test accuracy and saves
`outputs/test_comparison.csv` with accuracy, macro F1, per-class recall, test count
and checkpoint epoch. Per-model evaluation.txt and test_predictions.csv are also
updated. It never trains or replaces weights. Use validation for subsequent tuning;
repeated model selection on test scores compromises an independent final test.
After selecting settings, use `python evaluate.py` and `python predict.py image.jpg`
for the selected MODEL, or pass `--checkpoint` explicitly. Re-running a model
overwrites its outputs: copy them elsewhere before another experiment.

All three models use the same manifest, resize, augmentation and ImageNet
normalization. The two ResNets differ in initialization and default starting LR;
set their learning rates equal to isolate initialization more strictly. A single
seed is not evidence of a stable improvement; repeat with several seeds later.
The new normalization also changes the small CNN's training conditions relative
to the original experiment. Old checkpoints retain their original normalization.
To use an old checkpoint at `outputs/best_model.pt`, pass that path explicitly.
Starting training always starts a new run; it does not resume a checkpoint.

## Project structure

## Automated pretrained ResNet18 search

Install requirements, then run `python search_pretrained.py`. This runs only
ImageNet-pretrained ResNet18. Edit search_settings.py for the search budget and
ranges. The defaults are 24 sequential trials over initial LR, weight decay,
batch size, fine-tuning mode and LR schedule. All other training settings come
from settings.py. Each trial starts from ImageNet weights, not the previous trial.
Frozen blocks retain frozen BatchNorm running statistics as well as frozen weights.

Trials select checkpoints and early-stop on validation macro F1 (ties keep the
earlier checkpoint). Plateau scheduling still monitors validation loss. TPE chooses
combinations; median pruning may end weak trials after a 15-epoch warm-up and six
startup trials. This is a search budget, not a guarantee of a global optimum.

Each invocation creates outputs/search_pretrained/<timestamp>/ with study.db,
validation_ranking.csv, effective settings, environment information and separate
trial folders containing checkpoints and histories. JSON files here are generated
reproducibility records, not additional user-edited configuration files. Pruned
and failed trials are recorded but excluded from the completed-trial ranking.
Test data is never evaluated during search.

The ranking prints automatically after every trial. To display it later without
training or evaluating the test split:

```bash
python search_pretrained.py --study-dir outputs/search_pretrained/YOUR_TIMESTAMP --report-only
```

To add another SEARCH['trials'] trials to that study, omit --report-only.
Changed code, data manifest or effective settings require a new study. Resuming
adds trials; it does not resume an interrupted trial's optimizer or reproduce the
exact uninterrupted sampler sequence. Run only one process per study. CUDA OOM
trials are marked failed and skipped; unexpected errors stop for investigation.

The first stage uses one seed. Its highest-scoring trial is a provisional winner;
confirm finalists with multiple seeds before a final test. Automatic finalist
seed replication is not part of this search script yet. Existing compare_models.py
and evaluate_compare.py operate on the old baseline folders, not these trials.

The recorded baseline results and limitations are in `reports/baseline_report.md`;
`reports/experiment_log.csv` stores the exact counts and printed metrics provided
from the LiU evaluation. These files are versioned, unlike generated outputs.

To investigate pretrained ResNet18 further, run:

```bash
python pretrained_experiments.py
```

This trains three new models from ImageNet initialization using constant, plateau
and cosine schedules. Change `EXPERIMENT_SETTINGS` and `SCHEDULES` in that script;
other settings are inherited from settings.py. All runs use the same seed and
split. No test evaluation is performed. Each invocation creates a timestamped
folder under outputs/pretrained_experiments containing per-run settings.txt,
history.csv, best_model.pt and a shared validation_comparison.csv. The baseline
outputs are preserved. This does not resume the previously trained 92.44% model.
Remote baseline histories and checkpoints must still be backed up on LiU.

```text
training.py                Start training
compare_models.py          Train all models and summarize validation results
settings.py                All experiment settings and default paths
evaluate.py                Test the saved model
evaluate_compare.py        Test all three saved models and compare metrics
predict.py                 Classify your own images
image_classifier/
    model.py               Small CNN, ResNet18 factory and checkpoint loading
    schedulers.py          Constant, linear, cosine and plateau LR policies
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
Only training uses random horizontal flips. New runs use ImageNet channel means
and standard deviations; legacy checkpoints use the original [-1, 1] scaling.

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

The diagram describes the small CNN, with 101,858 trainable parameters and random
initialization. ResNet18 instead uses residual blocks and a new Linear(512, 2)
classifier. Its backbone starts randomly or from ImageNet weights; the new
classifier always starts randomly. No layers are frozen.

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
| history.csv | training.py | Training/validation loss, accuracy and learning rate per epoch |
| comparison.csv | compare_models.py | Best-validation-epoch results across models |
| evaluation.txt | evaluate.py | Accuracy, per-class precision/recall/F1 and confusion matrix |
| test_predictions.csv | evaluate.py | Each test image's label, prediction and scores |
| predictions.csv | predict.py | Predictions for your own images |

Outputs are created only when you run the corresponding command. Reusing the
same output directory can overwrite previous results; change OUTPUT in settings.py
for a new experiment. There are no bundled old models: run training first.
Checkpoints store settings internally, so no separate config JSON is needed.
A checkpoint is for inference, not exact training resumption: optimizer/RNG state
is not stored. Loading reconstructs the architecture recorded in the checkpoint;
older checkpoints without an architecture name load as the original small CNN.

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
use the trained model locally (new runs use outputs/<model_name>/best_model.pt).
The old pilot datasets and checkpoints were removed.

## Tests

```bash
python -m unittest discover -s tests -v
```
