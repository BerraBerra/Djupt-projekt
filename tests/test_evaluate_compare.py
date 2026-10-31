import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluate_compare import evaluate_all


class ComparisonTests(unittest.TestCase):
    def test_missing_checkpoint_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            with patch('evaluate_compare.evaluate') as evaluate:
                with self.assertRaisesRegex(FileNotFoundError, 'Missing checkpoints'):
                    evaluate_all('manifest.csv', temp, ['cnn'])
                evaluate.assert_not_called()

    def test_same_manifest_sorted_results_and_separate_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            names = ['cnn', 'resnet18_random', 'resnet18_pretrained']
            for name in names:
                (root / name).mkdir()
                (root / name / 'best_model.pt').touch()
            def fake_evaluate(manifest, checkpoint, output):
                self.assertEqual(manifest, 'fixed_manifest.csv')
                self.assertEqual(checkpoint, output / 'best_model.pt')
                return dict(accuracy={'cnn': .8, 'resnet18_random': .7,
                                     'resnet18_pretrained': .9}[output.name],
                            checkpoint_epoch=10, test_images=450,
                            classification_report={'macro avg': {'f1-score': .8},
                                                   'photo': {'recall': .7},
                                                   'ai_generated': {'recall': .9}})
            with patch('evaluate_compare.evaluate', side_effect=fake_evaluate) as evaluate:
                results = evaluate_all('fixed_manifest.csv', root, names)
                self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(results[0]['model'], 'resnet18_pretrained')
            with (root / 'test_comparison.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['test_accuracy'], '0.9')

