"""Prediction for one user image or a folder. Never trains or changes labels."""
import csv
from pathlib import Path
import torch
from .data import image_transform, read_rgb, EXTENSIONS
from .model import load_checkpoint, CLASSES


def predict(input_path, checkpoint, output, threshold=.7):
    if not 0 <= threshold <= 1:
        raise ValueError('Review threshold must lie between 0 and 1')
    model, saved = load_checkpoint(checkpoint)
    torch.set_num_threads(saved['config']['cpu_threads'])
    transform = image_transform(saved['config']['image_size'])
    source = Path(input_path)
    files = sorted(p for p in source.rglob('*') if p.suffix.lower() in EXTENSIONS) if source.is_dir() else [source]
    if not files:
        raise ValueError('No supported images found')
    rows = []
    for path in files:
        row = {'path': str(path.resolve()), 'predicted_label': '', 'confidence': '',
               'needs_review': True, **{f'p_{c}': '' for c in CLASSES}, 'error': ''}
        try:
            image = transform(read_rgb(path)).unsqueeze(0)
            with torch.inference_mode():
                probs = model(image).softmax(1)[0].tolist()
            guess = max(range(3), key=probs.__getitem__)
            row.update(predicted_label=CLASSES[guess], confidence=probs[guess],
                       needs_review=probs[guess] < threshold,
                       **{f'p_{c}': p for c, p in zip(CLASSES, probs)})
            print(f'{path.name}: {CLASSES[guess]} ({probs[guess]:.1%})', flush=True)
        except (OSError, ValueError) as error:
            row['error'] = str(error)
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows
