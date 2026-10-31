"""Check actual LR sequences and architecture-aware checkpoint restoration."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from image_classifier.schedulers import build_scheduler, step_scheduler
from image_classifier.model import build_model, load_checkpoint, CLASSES
from settings import training_config


class SchedulerTests(unittest.TestCase):
    def sequence(self, kind, losses):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        config = dict(learning_rate=.1, min_learning_rate=.01, epochs=len(losses),
                      learning_rate_type=kind, lr_patience=1, lr_factor=.5)
        scheduler = build_scheduler(optimizer, config)
        rates = []
        for loss in losses:
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            step_scheduler(scheduler, loss)
        return rates

    def test_fixed_schedules(self):
        self.assertEqual(self.sequence('constant', [1]*5), [.1]*5)
        for kind in ('linear', 'cosinedecay'):
            rates = self.sequence(kind, [1]*5)
            self.assertAlmostEqual(rates[0], .1)
            self.assertAlmostEqual(rates[-1], .01)
            self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertAlmostEqual(self.sequence('linear', [1]*5)[1], .0775)

    def test_plateau_ignores_single_bad_epoch_and_resets_on_improvement(self):
        rates = self.sequence('plateau', [.5, .6, .4, .5, .6, .7])
        self.assertEqual(rates[:5], [.1]*5)
        self.assertAlmostEqual(rates[5], .05)

    def test_invalid_scheduler_fails(self):
        with self.assertRaises(ValueError):
            self.sequence('typo', [1, 1])

    def test_shared_and_overridden_settings(self):
        cnn = training_config('cnn')
        random = training_config('resnet18_random')
        pretrained = training_config('resnet18_pretrained')
        self.assertEqual(cnn['learning_rate'], random['learning_rate'])
        self.assertEqual(cnn['learning_rate_type'], pretrained['learning_rate_type'])
        self.assertLess(pretrained['learning_rate'], random['learning_rate'])

    def test_resnet_restore_does_not_download(self):
        torch.set_num_threads(2)
        model = build_model('resnet18_random').eval()
        inputs = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            expected = model(inputs)
        self.assertEqual(tuple(expected.shape), (2, 2))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / 'model.pt'
            torch.save({'model_state': model.state_dict(), 'classes': list(CLASSES),
                        'config': {'model_name': 'resnet18_pretrained'}}, path)
            with patch('torchvision.models.resnet18', wraps=__import__(
                    'torchvision').models.resnet18) as factory:
                restored, _ = load_checkpoint(path)
                self.assertIsNone(factory.call_args.kwargs['weights'])
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(inputs))

    def test_pretrained_factory_requests_weights_and_replaces_head(self):
        from torchvision.models import ResNet18_Weights
        backbone = build_model('resnet18_random')
        with patch('torchvision.models.resnet18', return_value=backbone) as factory:
            model = build_model('resnet18_pretrained')
        self.assertEqual(factory.call_args.kwargs['weights'], ResNet18_Weights.DEFAULT)
        self.assertEqual(model.fc.out_features, 2)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
