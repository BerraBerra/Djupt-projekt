"""Evaluate a selected checkpoint once on held-out labeled images."""
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .data import LabeledImages
from .model import load_checkpoint, CLASSES


def evaluate(manifest, checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, saved = load_checkpoint(checkpoint)
    torch.set_num_threads(saved['config']['cpu_threads'])
    dataset = LabeledImages(manifest, 'test', saved['config']['image_size'])
    rows, truth, predicted = [], [], []
    with torch.inference_mode():
        for images, labels, paths in DataLoader(dataset, batch_size=64):
            probabilities = model(images).softmax(1)
            guesses = probabilities.argmax(1)
            for path, label, guess, probs in zip(paths, labels.tolist(), guesses.tolist(), probabilities.tolist()):
                truth.append(label)
                predicted.append(guess)
                rows.append({'path': path, 'true_label': CLASSES[label], 'predicted_label': CLASSES[guess],
                             **{f'p_{name}': p for name, p in zip(CLASSES, probs)}})
    with (output / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metrics = {'accuracy': accuracy_score(truth, predicted), 'test_images': len(truth),
               'checkpoint_epoch': saved['epoch'], 'classes': list(CLASSES),
               'confusion_matrix': confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
               'classification_report': classification_report(truth, predicted, labels=[0, 1, 2],
                    target_names=CLASSES, output_dict=True, zero_division=0)}
    summary = (
        f"Test images: {len(truth)}\n"
        f"Checkpoint epoch: {saved['epoch']}\n"
        f"Accuracy: {metrics['accuracy']:.2%}\n\n"
        + classification_report(truth, predicted, labels=[0, 1, 2],
                                target_names=CLASSES, zero_division=0)
        + "\nConfusion matrix (rows=true, columns=predicted):\n"
        + str(metrics['confusion_matrix']) + "\n"
    )
    (output / 'evaluation.txt').write_text(summary, encoding='utf-8')
    print(summary)
    return metrics
