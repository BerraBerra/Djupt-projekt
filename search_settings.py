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

# Round 2: controlled follow-up of LiU study trial 6. No random parameter search.
SEEDS = [42, 43, 44]
REFERENCE = {
    'learning_rate': 0.0001926496250871329,
    'weight_decay': 1.0567900894501647e-06,
    'batch_size': 64,
    'finetune_mode': 'last_block',
    'learning_rate_type': 'cosinedecay',
}
VARIANTS = {
    'reference': {},
    'batch32': {'batch_size': 32},
    'plateau': {'learning_rate_type': 'plateau'},
    'constant': {'learning_rate_type': 'constant'},
    'all_layers': {'finetune_mode': 'all'},
}
# SEARCH['trials'], startup_trials and pruning_warmup_epochs above belong to
# historical round 1. Round 2 runs len(VARIANTS) * len(SEEDS) planned trials,
# without Optuna pruning. Per-run early stopping remains active.
