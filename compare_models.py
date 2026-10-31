"""Train all three models on fixed splits; summarize validation, not test results."""
import csv
from settings import MANIFEST, ROOT, MODEL_SETTINGS, training_config
from image_classifier.training import train


if __name__ == '__main__':
    results = []
    for name in MODEL_SETTINGS:
        config = training_config(name)
        output = ROOT / 'outputs' / name
        print(f'\nTraining {name}', flush=True)
        train(MANIFEST, config, output)
        with (output / 'history.csv').open(newline='', encoding='utf-8') as handle:
            best = min(csv.DictReader(handle), key=lambda row: float(row['val_loss']))
        results.append({'model': name, **best})
        with (ROOT / 'outputs' / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
