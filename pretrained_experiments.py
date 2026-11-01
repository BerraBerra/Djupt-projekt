"""Compare LR schedules on validation data; preserve earlier runs automatically."""
import csv
from datetime import datetime
from settings import MANIFEST, ROOT, training_config
from image_classifier.training import train

# Edit these settings for this experiment only. Shared settings still come from
# settings.py. All three runs start from ImageNet weights with all layers trainable.
EXPERIMENT_SETTINGS = {
    'epochs': 100,
    'patience': 15,
    'learning_rate': 0.0001,
    'min_learning_rate': 0.000001,
    'lr_patience': 3,
    'lr_factor': 0.5,
}
SCHEDULES = ('constant', 'plateau', 'cosinedecay')


def run_experiments():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = ROOT / 'outputs' / 'pretrained_experiments' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for schedule in SCHEDULES:
        config = {**training_config('resnet18_pretrained'), **EXPERIMENT_SETTINGS,
                  'learning_rate_type': schedule}
        output = destination / schedule
        output.mkdir()
        # Record the complete effective configuration even if training is interrupted.
        (output / 'settings.txt').write_text(
            '\n'.join(f'{key} = {value!r}' for key, value in sorted(config.items())) + '\n',
            encoding='utf-8')
        print(f'\nPretrained ResNet18: {schedule}; results: {output}', flush=True)
        train(MANIFEST, config, output)
        with (output / 'history.csv').open(newline='', encoding='utf-8') as handle:
            history = list(csv.DictReader(handle))
        best = min(history, key=lambda row: float(row['val_loss']))
        results.append({'schedule': schedule, 'seed': config['seed'],
                        'epochs_completed': len(history), **best})
        with (destination / 'validation_comparison.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda row: float(row['val_loss'])))
    print(f'\nValidation comparison saved in {destination}', flush=True)
    return destination


if __name__ == '__main__':
    run_experiments()
