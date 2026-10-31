"""Evaluate all saved models on the same test split, without training."""
import csv
from pathlib import Path
from settings import MANIFEST, ROOT, MODEL_SETTINGS
from image_classifier.evaluation import evaluate


def evaluate_all(manifest=MANIFEST, output_root=ROOT / 'outputs', model_names=None):
    output_root = Path(output_root)
    names = list(MODEL_SETTINGS if model_names is None else model_names)
    if not names:
        raise ValueError('No models selected')
    missing = [output_root / name / 'best_model.pt' for name in names
               if not (output_root / name / 'best_model.pt').is_file()]
    if missing:
        raise FileNotFoundError('Train all models with python compare_models.py first. '
                                'Missing checkpoints:\n' + '\n'.join(map(str, missing)))
    results = []
    for name in names:
        folder = output_root / name
        print(f'\nEvaluating {name} on the test split', flush=True)
        metrics = evaluate(manifest, folder / 'best_model.pt', folder)
        report = metrics['classification_report']
        results.append({
            'model': name,
            'test_accuracy': metrics['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'photo_recall': report['photo']['recall'],
            'ai_recall': report['ai_generated']['recall'],
            'test_images': metrics['test_images'],
            'checkpoint_epoch': metrics['checkpoint_epoch'],
        })
    results.sort(key=lambda row: row['test_accuracy'], reverse=True)
    destination = output_root / 'test_comparison.csv'
    with destination.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print('\nTest comparison (sorted by accuracy):')
    print(f"{'Model':24} {'Accuracy':>10} {'Macro F1':>10} {'Epoch':>7}")
    for row in results:
        print(f"{row['model']:24} {row['test_accuracy']:10.2%} "
              f"{row['macro_f1']:10.3f} {row['checkpoint_epoch']:7}")
    print(f'\nSaved: {destination}')
    return results


if __name__ == '__main__':
    evaluate_all()
