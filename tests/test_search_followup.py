"""Exercise the real Optuna queue/database with a cheap stand-in for training."""
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import optuna
import search_pretrained as search


class FollowupTests(unittest.TestCase):
    def test_plan_summary_and_resume_skip_completed(self):
        configs = []

        def fake_train(manifest, config, output, epoch_callback):
            configs.append(config)
            record = {'epoch': 1, 'val_macro_f1': .9 + (config['seed'] - 42) / 1000,
                      'val_accuracy': .9, 'val_loss': .3}
            epoch_callback(record, record)
            return record

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp).resolve()
            manifest = root / 'manifest.csv'
            manifest.write_text('fixture')
            with patch.object(search, 'ROOT', root), patch.object(search, 'MANIFEST', manifest), \
                 patch.object(search, 'train', side_effect=fake_train), \
                 patch.object(search.subprocess, 'check_output', return_value='test'), \
                 redirect_stdout(io.StringIO()):
                with patch('sys.argv', ['search_pretrained.py']):
                    search.main()
                folder = next((root / 'outputs' / 'search_pretrained').iterdir())
                self.assertEqual(len(configs), 15)
                for config in configs:
                    self.assertEqual(config['model_name'], 'resnet18_pretrained')
                    self.assertNotEqual(config['finetune_mode'], 'head')
                with (folder / 'variant_summary.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row['completed_seeds'] == '3' for row in rows))
                self.assertAlmostEqual(float(rows[0]['std_val_macro_f1']), .001)
                with patch('sys.argv', ['search_pretrained.py', '--study-dir', str(folder)]):
                    search.main()
                self.assertEqual(len(configs), 15)
                with patch('sys.argv', ['search_pretrained.py', '--study-dir', str(folder), '--report-only']):
                    search.main()
                self.assertEqual(len(configs), 15)
