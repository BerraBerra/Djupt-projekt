"""Edit training settings here. No command-line arguments or JSON files are required."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data' / 'manifest.csv'
OUTPUT = ROOT / 'outputs'
CHECKPOINT = OUTPUT / 'best_model.pt'

TRAINING = {
    'image_size': 224,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'label_smoothing': 0.05,
    'patience': 10,
    'seed': 42,
    'num_workers': 0,  # Portable default. Increase on a suitable training server.
    'cpu_threads': 4,
}
