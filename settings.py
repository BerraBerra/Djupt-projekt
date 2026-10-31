"""Edit training settings here. No command-line arguments or JSON files are required."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data' / 'manifest.csv'
MODEL = 'cnn'  # cnn, resnet18_random, resnet18_pretrained

TRAINING = {
    'image_size': 224,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'learning_rate_type': 'constant',  # constant, linear, cosinedecay, plateau
    'min_learning_rate': 0.000001,
    'lr_factor': 0.5,  # Plateau: multiply LR by this factor.
    'lr_patience': 3,  # Plateau: tolerated epochs without improvement.
    'weight_decay': 0.0001,
    'label_smoothing': 0.05,
    'patience': 10,
    'seed': 42,
    'num_workers': 0,  # Portable default. Increase on a suitable training server.
    'cpu_threads': 4,
}

# Starting points, not optimized settings. All layers are trained in every model.
# Both randomly initialized models inherit the same shared settings.
MODEL_SETTINGS = {
    'cnn': {},
    'resnet18_random': {},
    'resnet18_pretrained': {'learning_rate': 0.0001},
}


def training_config(model_name=None):
    name = MODEL if model_name is None else model_name
    return {**TRAINING, **MODEL_SETTINGS[name], 'model_name': name,
            'normalization': 'imagenet'}


OUTPUT = ROOT / 'outputs' / MODEL
CHECKPOINT = OUTPUT / 'best_model.pt'
