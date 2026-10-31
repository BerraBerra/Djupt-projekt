"""Training loop: train split for gradients, validation only for checkpoint selection."""
import hashlib
import csv
import random
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from .data import LabeledImages
from .model import build_model, CLASSES
from .schedulers import build_scheduler, step_scheduler
from .losses import build_loss


def train(manifest, config, output):
    if config['epochs'] < 1 or config['batch_size'] < 2 or config['patience'] < 1:
        raise ValueError('epochs/patience must be positive and batch_size at least 2')
    if config['image_size'] < 16 or config['learning_rate'] <= 0:
        raise ValueError('image_size must be >=16 and learning_rate positive')
    if not 0 <= config['label_smoothing'] < 1:
        raise ValueError('label_smoothing must be in [0,1)')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; input: {config["image_size"]}x{config["image_size"]}; output: {output}', flush=True)
    datasets = {s: LabeledImages(manifest, s, config['image_size'], s == 'train',
                config.get('normalization', 'legacy')) for s in ['train', 'val']}
    loaders = {s: DataLoader(d, batch_size=config['batch_size'], shuffle=s == 'train',
                            num_workers=config['num_workers']) for s, d in datasets.items()}
    model = build_model(config.get('model_name', 'cnn')).to(device)
    criterion = build_loss(config['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = build_scheduler(optimizer, config)
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, config['epochs'] + 1):
        record = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']}
        for split in ['train', 'val']:
            model.train(split == 'train')
            total_loss, correct, count = 0., 0, 0
            progress = tqdm(loaders[split], desc=f'Epoch {epoch}/{config["epochs"]} | {split}',
                            unit='batch', dynamic_ncols=True, mininterval=1)
            for batch_index, (images, labels, _) in enumerate(progress, 1):
                images, labels = images.to(device), labels.to(device)
                with torch.set_grad_enabled(split == 'train'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
                progress.set_postfix(loss=f'{total_loss/count:.3f}', accuracy=f'{correct/count:.1%}', refresh=False)
            record[split + '_loss'] = total_loss / count
            record[split + '_accuracy'] = correct / count
        history.append(record)
        print(f"Learning rate used: {record['learning_rate']:.6g}", flush=True)
        print(f"Epoch {epoch:02d}: train loss {record['train_loss']:.3f}, val loss {record['val_loss']:.3f}, val accuracy {record['val_accuracy']:.1%}", flush=True)
        if record['val_loss'] < best:
            best, stale = record['val_loss'], 0
            torch.save({'model_state': model.state_dict(), 'classes': list(CLASSES),
                        'config': config, 'epoch': epoch, 'validation_loss': best,
                        'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest()},
                       output / 'best_model.pt')
        else:
            stale += 1
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        if stale >= config['patience']:
            break
        if epoch < config['epochs']:
            step_scheduler(scheduler, record['val_loss'])
    print(f'Saved best validation checkpoint: {output / "best_model.pt"}', flush=True)
