"""Read actual RGB image files and labels from a manifest, not precomputed features."""
import csv
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from .model import CLASSES

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def image_transform(size, training=False, normalization='legacy'):
    operations = [transforms.Resize((size, size), antialias=True)]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    if normalization == 'imagenet':
        mean, std = (.485, .456, .406), (.229, .224, .225)
    elif normalization == 'legacy':
        mean, std = (.5,)*3, (.5,)*3
    else:
        raise ValueError(f'Unknown normalization: {normalization}')
    operations += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(operations)


def read_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def read_manifest(path):
    path = Path(path)
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty manifest')
    groups, hashes = {}, {}
    for row in rows:
        if row['label'] not in CLASSES or row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid label/split: {row}')
        resolved = (path.parent / row['path']).resolve()
        if not resolved.is_relative_to(path.parent.resolve()) or not resolved.is_file():
            raise ValueError(f'Invalid/missing image path: {resolved}')
        for lookup, key in [(groups, row['group_id']), (hashes, row['pixel_sha256'])]:
            if key in lookup and lookup[key] != row['split']:
                raise ValueError('Related or identical images cross train/val/test boundaries')
            lookup[key] = row['split']
    return rows


class LabeledImages(Dataset):
    def __init__(self, manifest, split, size, training=False, normalization='legacy'):
        self.base = Path(manifest).parent
        self.rows = [r for r in read_manifest(manifest) if r['split'] == split]
        if set(r['label'] for r in self.rows) != set(CLASSES):
            raise ValueError(f'{split} must contain both classes')
        self.transform = image_transform(size, training, normalization)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.transform(read_rgb(self.base / row['path']))
        return image, CLASSES.index(row['label']), row['path']
