"""Settings for pretrained ResNet18 hyperparameter search only."""
SEARCH = {
    'trials': 24,  # New trials per invocation.
    'seed': 42,
    'epochs': 100,
    'patience': 15,
    'startup_trials': 6,
    'pruning_warmup_epochs': 15,
}
SPACE = {
    'learning_rate': (1e-5, 1e-3),
    'weight_decay': (1e-6, 1e-2),
    'batch_size': [16, 32, 64],
    'finetune_mode': ['all', 'head', 'last_block'],
    'learning_rate_type': ['constant', 'plateau', 'cosinedecay'],
}
