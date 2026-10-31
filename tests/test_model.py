import tempfile
import csv
import unittest
from pathlib import Path
import torch
from PIL import Image
from image_classifier.model import ImageClassifier, CLASSES, load_checkpoint
from image_classifier.losses import build_loss
from image_classifier.data import read_manifest
from image_classifier.inference import predict
from image_classifier.training import train
from image_classifier.evaluation import evaluate


class ModelTests(unittest.TestCase):
    def test_training_evaluation_without_json_configs(self):
        """Run the complete workflow on tiny synthetic fixtures, not benchmark data."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder = Path(temp)
            rows = []
            for split in ('train', 'val', 'test'):
                for index, label in enumerate(CLASSES):
                    name = f'{split}_{label}.png'
                    Image.new('RGB', (32, 32), (index * 80, 50, 100)).save(folder / name)
                    rows.append({'path': name, 'label': label, 'split': split,
                                 'group_id': name, 'pixel_sha256': name})
            manifest = folder / 'manifest.csv'
            with manifest.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            config = dict(image_size=32, epochs=1, batch_size=3, learning_rate=.001,
                          weight_decay=.0001, label_smoothing=.05, patience=1,
                          seed=42, num_workers=0, cpu_threads=2)
            output = folder / 'outputs'
            train(manifest, config, output)
            metrics = evaluate(manifest, output / 'best_model.pt', output)
            self.assertEqual(metrics['test_images'], 3)
            self.assertTrue((output / 'history.csv').exists())
            self.assertTrue((output / 'evaluation.txt').exists())
            self.assertEqual(list(output.glob('*.json')), [])

    def test_high_resolution_shapes(self):
        torch.set_num_threads(2)
        model=ImageClassifier().eval()
        shapes=[]
        handles=[layer.register_forward_hook(lambda module, inputs, output: shapes.append(tuple(output.shape)))
                 for layer in model.features if isinstance(layer,torch.nn.MaxPool2d)]
        with torch.inference_mode():
            output=model(torch.rand(2,3,224,224))
        for handle in handles: handle.remove()
        self.assertEqual(shapes,[(2,32,112,112),(2,64,56,56),(2,128,28,28)])
        self.assertEqual(tuple(output.shape),(2,3))

    def test_logits_loss_and_gradients(self):
        torch.set_num_threads(2)
        model = ImageClassifier()
        logits = model(torch.rand(4, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (4, 3))
        loss = build_loss()(logits, torch.tensor([0, 1, 2, 0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0)

    def test_image_checkpoint_and_inference_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder = Path(temp)
            image = folder/'test.png'
            Image.new('RGB', (100, 60), (20, 120, 40)).save(image)
            model = ImageClassifier().eval()
            checkpoint = folder/'model.pt'
            torch.save({'model_state':model.state_dict(), 'classes':list(CLASSES),
                        'config':{'image_size':32, 'cpu_threads':2}}, checkpoint)
            restored,_ = load_checkpoint(checkpoint)
            x = torch.rand(2,3,32,32)
            torch.testing.assert_close(model(x),restored(x))
            rows = predict(image, checkpoint, folder/'prediction.csv')
            self.assertEqual(rows[0]['error'], '')
            self.assertAlmostEqual(sum(rows[0]['p_'+c] for c in CLASSES),1,places=6)

    def test_duplicate_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder = Path(temp)
            Image.new('RGB', (32, 32)).save(folder/'a.png')
            rows = [{'path':'a.png','label':'real','split':s,'group_id':'same',
                     'pixel_sha256':'same-hash'} for s in ['train','test']]
            with (folder/'manifest.csv').open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'cross'):
                read_manifest(folder/'manifest.csv')


if __name__ == '__main__':
    unittest.main()
