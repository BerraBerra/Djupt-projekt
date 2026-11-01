"""Check experiment isolation and configuration without downloading weights."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pretrained_experiments as experiments


class ExperimentTests(unittest.TestCase):
    def test_separate_runs_preserve_settings_and_rank_validation(self):
        configurations = []

        def fake_train(manifest, config, output):
            configurations.append(config)
            with (output / 'history.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['epoch', 'val_loss', 'val_accuracy'])
                writer.writeheader()
                writer.writerows([{'epoch': 1, 'val_loss': .5, 'val_accuracy': .8},
                                  {'epoch': 2, 'val_loss': .4, 'val_accuracy': .9}])

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            with patch.object(experiments, 'ROOT', Path(temp)), \
                 patch.object(experiments, 'train', side_effect=fake_train):
                first = experiments.run_experiments()
                second = experiments.run_experiments()
            self.assertNotEqual(first, second)
            self.assertTrue((first / 'constant' / 'settings.txt').exists())
            with (first / 'validation_comparison.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row['epoch'] == '2' for row in rows))
            base = {k: v for k, v in configurations[0].items() if k != 'learning_rate_type'}
            for config in configurations:
                self.assertEqual(base, {k: v for k, v in config.items()
                                        if k != 'learning_rate_type'})
