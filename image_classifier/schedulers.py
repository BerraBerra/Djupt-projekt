"""Epoch-level learning rate policies shared by all architectures."""
import torch


def build_scheduler(optimizer, config):
    kind = config.get('learning_rate_type', 'constant')
    minimum = config.get('min_learning_rate', 1e-6)
    initial = config['learning_rate']
    if not 0 <= minimum <= initial:
        raise ValueError('min_learning_rate must be between zero and learning_rate')
    # Epoch 1 uses initial LR; the last planned epoch uses minimum LR.
    duration = max(config['epochs'] - 1, 1)
    if kind == 'constant':
        return None
    if kind == 'linear':
        ratio = minimum / initial
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - (1 - ratio) * min(step / duration, 1))
    if kind == 'cosinedecay':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=duration, eta_min=minimum)
    if kind == 'plateau':
        factor, patience = config.get('lr_factor', .5), config.get('lr_patience', 3)
        if not 0 < factor < 1 or patience < 0:
            raise ValueError('lr_factor must be in (0,1) and lr_patience nonnegative')
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, min_lr=minimum)
    raise ValueError(f'Unknown learning_rate_type: {kind}')


def step_scheduler(scheduler, validation_loss):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
    elif scheduler is not None:
        scheduler.step()
