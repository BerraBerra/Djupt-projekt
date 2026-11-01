import unittest
import torch
from image_classifier.model import build_model
from image_classifier.training import configure_finetuning, training_mode


class FinetuningTests(unittest.TestCase):
    def test_frozen_weights_and_batchnorm_statistics(self):
        torch.set_num_threads(2)
        model = build_model('resnet18_random')
        for mode in ('head', 'last_block', 'all'):
            configure_finetuning(model, mode)
            training_mode(model, mode)
            self.assertEqual(model.conv1.weight.requires_grad, mode == 'all')
            self.assertEqual(model.bn1.training, mode == 'all')
            self.assertEqual(model.layer4[0].bn1.training, mode != 'head')
            self.assertTrue(model.fc.weight.requires_grad)
            before = model.bn1.running_mean.clone()
            model(torch.rand(2, 3, 64, 64)).sum().backward()
            if mode != 'all':
                torch.testing.assert_close(before, model.bn1.running_mean)
            self.assertIsNotNone(model.fc.weight.grad)
