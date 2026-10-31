"""The ONLY model architecture. Random initialization; no downloaded weights."""
import torch
from torch import nn

CLASSES = ('photo', 'ai_generated')


class ImageClassifier(nn.Module):
    """RGB pixels -> convolutional features -> two unnormalized logits.

    At 224x224: 3x224x224 -> 32x112x112 -> 64x56x56 -> 128x28x28 -> 128 -> 64 -> 2.
    BatchNorm learns statistics on training batches. Dropout is disabled in eval().
    """
    def __init__(self):
        super().__init__()
        layers = []
        for in_channels, out_channels in [(3, 32), (32, 64), (64, 128)]:
            layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                        nn.Dropout(.25), nn.Linear(64, len(CLASSES)))

    def forward(self, images):
        return self.classifier(self.features(images))


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if tuple(checkpoint['classes']) != CLASSES:
        raise ValueError('Checkpoint class ordering differs from this project')
    model = ImageClassifier().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint
