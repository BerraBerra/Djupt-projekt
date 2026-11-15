"""Evaluate one fixed checkpoint on the external dataset without training."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from image_classifier.evaluation import evaluate
from settings import ROOT


def subgroup_results(manifest, predictions):
    metadata = {r['path']: r for r in manifest}
    groups = {}
    for row in predictions:
        original = metadata[row['path']]
        label = original['label']
        key = (label, original['generator'] if label == 'ai_generated' else original['real_source'])
        group = groups.setdefault(key, {'images': 0, 'correct': 0})
        group['images'] += 1
        group['correct'] += row['predicted_label'] == label
    return [{'true_class': label, 'generator_or_photo_source': name, **counts,
             'recall': counts['correct']/counts['images']}
            for (label, name), counts in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data'/'external_community'/'manifest.csv')
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error('Run python prepare_external.py first')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint does not exist; copy your selected LiU checkpoint or run on LiU')
    output = ROOT/'outputs'/'external_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    metrics = evaluate(args.manifest, args.checkpoint, output)
    with args.manifest.open(newline='', encoding='utf-8') as handle:
        manifest = list(csv.DictReader(handle))
    with (output/'test_predictions.csv').open(newline='', encoding='utf-8') as handle:
        predictions = list(csv.DictReader(handle))
    groups = subgroup_results(manifest, predictions)
    with (output/'source_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=groups[0].keys())
        writer.writeheader()
        writer.writerows(groups)
    audit = {'checkpoint': str(args.checkpoint.resolve()),
             'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
             'manifest': str(args.manifest.resolve()),
             'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
             'accuracy': metrics['accuracy'], 'macro_f1': metrics['classification_report']['macro avg']['f1-score']}
    (output/'evaluation_provenance.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'External results saved in {output}')


if __name__ == '__main__':
    main()
