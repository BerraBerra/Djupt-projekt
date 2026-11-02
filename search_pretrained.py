"""Search pretrained ResNet18 on validation only; no test split is evaluated."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import statistics
from datetime import datetime
from pathlib import Path
import optuna
import torch
import torchvision
from settings import MANIFEST, ROOT, training_config
from search_settings import SEARCH, SPACE, SEEDS, REFERENCE, VARIANTS
from image_classifier.training import train


def export_results(study, folder):
    rows = []
    for trial in study.trials:
        record = trial.user_attrs.get('best_validation', {})
        rows.append({'trial': trial.number, 'state': trial.state.name,
                     'variant': trial.user_attrs.get('variant', ''),
                     'seed': trial.user_attrs.get('seed', ''),
                     'val_macro_f1': trial.value if trial.state.name == 'COMPLETE' else '',
                     'val_accuracy': record.get('val_accuracy', ''),
                     'val_loss': record.get('val_loss', ''), 'epoch': record.get('epoch', ''),
                     'checkpoint': str(folder / f'trial_{trial.number:04d}' / 'best_model.pt'),
                     **{key: trial.user_attrs.get('config', trial.params).get(key, '') for key in SPACE}})
    rows.sort(key=lambda r: (r['state'] != 'COMPLETE', -float(r['val_macro_f1'] or 0)))
    if rows:
        with (folder / 'validation_ranking.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print('\nCompleted trials, ranked by validation macro F1:')
    for row in [r for r in rows if r['state'] == 'COMPLETE'][:10]:
        print(f"Trial {row['trial']:3}: F1={row['val_macro_f1']:.4f}, "
              f"accuracy={row['val_accuracy']:.2%}, epoch={row['epoch']}")
    print(f'Results: {folder / "validation_ranking.csv"}')
    plan = study.user_attrs.get('plan', {})
    if plan:
        summary = []
        for variant in plan['variants']:
            # One successful result per planned seed; failed retries are not replications.
            completed = {r['seed']: r for r in rows
                         if r['state'] == 'COMPLETE' and r['variant'] == variant}
            scores = [r['val_macro_f1'] for r in completed.values()]
            accuracies = [r['val_accuracy'] for r in completed.values()]
            summary.append({'variant': variant, 'completed_seeds': len(scores),
                'expected_seeds': len(plan['seeds']),
                'mean_val_macro_f1': statistics.mean(scores) if scores else '',
                'std_val_macro_f1': statistics.stdev(scores) if len(scores) > 1 else '',
                'mean_val_accuracy': statistics.mean(accuracies) if accuracies else ''})
        summary.sort(key=lambda r: (r['completed_seeds'] != r['expected_seeds'],
                                   -float(r['mean_val_macro_f1'] or 0)))
        with (folder / 'variant_summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print('\nVariant summary (incomplete groups are provisional):')
        for row in summary:
            print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, help='Continue an existing study directory')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only and args.study_dir is None:
        parser.error('--report-only requires --study-dir')
    folder = (args.study_dir or ROOT / 'outputs' / 'search_pretrained' /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if args.study_dir and not (folder / 'study.db').is_file():
        parser.error('No study.db found in --study-dir')
    folder.mkdir(parents=True, exist_ok=True)
    storage = 'sqlite:///' + (folder / 'study.db').as_posix()
    if args.report_only:
        database = optuna.storages.RDBStorage(storage)
        try:
            export_results(optuna.load_study(study_name='resnet18_pretrained', storage=database), folder)
        finally:
            database.remove_session()
            database.engine.dispose()
        return
    base = {**training_config('resnet18_pretrained'),
            **{k: SEARCH[k] for k in ('epochs', 'patience', 'seed')},
            'selection_metric': 'val_macro_f1'}
    # Protect study comparability when resuming, including changes in training code.
    code_hash = hashlib.sha256()
    for path in sorted((ROOT / 'image_classifier').glob('*.py')) + [Path(__file__)]:
        code_hash.update(path.read_bytes())
    contract = json.loads(json.dumps({'base': base, 'reference': REFERENCE,
        'variants': VARIANTS, 'seeds': SEEDS, 'round': 2,
        'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'code_sha256': code_hash.hexdigest(),
        'pruning': False}))
    contract_file = folder / 'study_settings.json'
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError('Settings, data or code changed. Start a new study instead.')
    contract_file.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = 'unavailable', 'unavailable'
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'torchvision': str(torchvision.__version__), 'optuna': optuna.__version__,
                   'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                   'git_revision': revision, 'git_status': dirty}
    (folder / f'environment_{datetime.now():%Y%m%d_%H%M%S_%f}.json').write_text(
        json.dumps(environment, indent=2), encoding='utf-8')
    database = optuna.storages.RDBStorage(storage)
    study = optuna.create_study(study_name='resnet18_pretrained', storage=database,
        load_if_exists=True, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEARCH['seed'], n_startup_trials=SEARCH['startup_trials']),
        pruner=optuna.pruners.NopPruner())
    study.set_user_attr('plan', {'variants': list(VARIANTS), 'seeds': SEEDS})
    # Single-process continuation: an interrupted attempt is retried from ImageNet.
    for previous in study.get_trials(deepcopy=False):
        if previous.state == optuna.trial.TrialState.RUNNING:
            study.tell(previous.number, state=optuna.trial.TrialState.FAIL)
    for seed in SEEDS:
        for variant in VARIANTS:
            exists = any(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.WAITING)
                         and t.user_attrs.get('variant') == variant
                         and t.user_attrs.get('seed') == seed for t in study.trials)
            if not exists:
                study.enqueue_trial({'variant': variant, 'seed': seed},
                                    user_attrs={'variant': variant, 'seed': seed})

    def objective(trial):
        variant = trial.suggest_categorical('variant', list(VARIANTS))
        seed = trial.suggest_categorical('seed', SEEDS)
        config = {**base, **REFERENCE, **VARIANTS[variant], 'seed': seed}
        trial.set_user_attr('config', config)
        output = folder / f'trial_{trial.number:04d}'
        output.mkdir(exist_ok=False)
        (output / 'settings.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

        def callback(record, best):
            trial.set_user_attr('best_validation', best)
            trial.report(best['val_macro_f1'], record['epoch'])
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            best = train(MANIFEST, config, output, epoch_callback=callback)
            return best['val_macro_f1']
        except Exception as error:
            (output / 'status.txt').write_text(f'{type(error).__name__}: {error}', encoding='utf-8')
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f'Study directory: {folder}', flush=True)
    try:
        remaining = sum(t.state == optuna.trial.TrialState.WAITING for t in study.trials)
        study.optimize(objective, n_trials=remaining,
                       catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True,
                       callbacks=[lambda study, trial: export_results(study, folder)])
    finally:
        try:
            export_results(study, folder)
        finally:
            database.remove_session()
            database.engine.dispose()


if __name__ == '__main__':
    main()
