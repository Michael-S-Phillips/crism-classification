"""
Sequential hyperparameter sweep across all model families.
Each config trains one model variant and logs to wandb.
Restartable: skips configs whose checkpoint already exists.

Usage:
    conda run -n crism python scripts/sweep.py
    conda run -n crism python scripts/sweep.py --dry_run   # print configs only
"""
import argparse
import os
import subprocess
import csv
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(PROJ, 'scripts', 'train.py')
CKPT_DIR = os.path.join(PROJ, 'checkpoints')
LOG_DIR = os.path.join(PROJ, 'logs')

# ---------------------------------------------------------------------------
# Sweep configurations
# Each entry is a dict whose keys map to --arg names in train.py.
# 'run_name' is used for --run_name and determines checkpoint filename.
# Checkpoint: {CKPT_DIR}/{run_name}_best.pt  (sklearn: {run_name}_model.pkl)
# ---------------------------------------------------------------------------
SWEEP_CONFIGS = [
    # --- lgbm: one best-bet config (deeper trees, more estimators) ---
    # Skipped the other 3 lgbm + all 4 xgb: each takes ~60-90 min on 726k samples.
    # Neural models are where the real gains are after adding dropout + pos_weight.
    dict(model='lgbm', run_name='lgbm_sw2',
         n_estimators=500, learning_rate=0.03, num_leaves=127),

    # --- MLP variants (baseline: mAP=0.613, hidden=(256,128), dropout=0.3, lr=1e-3) ---
    dict(model='mlp', run_name='mlp_sw1',
         epochs=200, patience=15, lr=1e-3, batch_size=512,
         hidden_dims='512,256,128', dropout=0.3, use_pos_weight=True),
    dict(model='mlp', run_name='mlp_sw2',
         epochs=200, patience=15, lr=5e-4, batch_size=512,
         hidden_dims='256,128', dropout=0.5, use_pos_weight=True),
    dict(model='mlp', run_name='mlp_sw3',
         epochs=200, patience=15, lr=1e-3, batch_size=256,
         hidden_dims='512,256', dropout=0.3, use_pos_weight=False),
    dict(model='mlp', run_name='mlp_sw4',
         epochs=200, patience=15, lr=2e-3, batch_size=1024,
         hidden_dims='256,128', dropout=0.2, use_pos_weight=True),

    # --- CNN variants (baseline: mAP=0.636, stopped epoch 3, no dropout) ---
    dict(model='cnn', run_name='cnn_sw1', patch_size=7,
         epochs=200, patience=20, lr=5e-4, batch_size=256,
         dropout=0.3, use_pos_weight=True, weight_decay=1e-4),
    dict(model='cnn', run_name='cnn_sw2', patch_size=7,
         epochs=200, patience=20, lr=3e-4, batch_size=256,
         dropout=0.5, use_pos_weight=True, weight_decay=1e-4),
    dict(model='cnn', run_name='cnn_sw3', patch_size=7,
         epochs=200, patience=20, lr=1e-4, batch_size=256,
         dropout=0.3, use_pos_weight=False, weight_decay=1e-3),
    dict(model='cnn', run_name='cnn_sw4', patch_size=7,
         epochs=200, patience=20, lr=5e-4, batch_size=256,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4),

    # --- ViT variants (baseline: mAP=0.634, stopped epoch 1, dropout=0.1) ---
    dict(model='vit', run_name='vit_sw1', patch_size=7,
         epochs=200, patience=20, lr=5e-4, batch_size=256,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.3, use_pos_weight=True, weight_decay=1e-4),
    dict(model='vit', run_name='vit_sw2', patch_size=7,
         epochs=200, patience=20, lr=3e-4, batch_size=256,
         embed_dim=64, n_heads=4, n_layers=4,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4),
    dict(model='vit', run_name='vit_sw3', patch_size=7,
         epochs=200, patience=20, lr=1e-4, batch_size=256,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.3, use_pos_weight=False, weight_decay=1e-4),
]


def ckpt_exists(run_name: str, model: str) -> bool:
    if model in ('logreg', 'svc', 'rf', 'xgb', 'lgbm'):
        return os.path.exists(os.path.join(CKPT_DIR, f'{run_name}_model.pkl'))
    return os.path.exists(os.path.join(CKPT_DIR, f'{run_name}_best.pt'))


def config_to_args(cfg: dict) -> list:
    """Convert config dict to train.py CLI args list."""
    args = ['python', TRAIN]
    for k, v in cfg.items():
        if k == 'use_pos_weight':
            if v:
                args.append('--use_pos_weight')
        elif v is not None:
            args += [f'--{k}', str(v)]
    return args


def main():
    parser = argparse.ArgumentParser(description='Sequential hyperparameter sweep.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configs without running them')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    summary_path = os.path.join(
        LOG_DIR, f'sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    results = []

    total = len(SWEEP_CONFIGS)
    for i, cfg in enumerate(SWEEP_CONFIGS):
        run_name = cfg['run_name']
        model = cfg['model']
        print(f'\n[{i+1}/{total}] {run_name} — {cfg}', flush=True)

        if ckpt_exists(run_name, model):
            print(f'  SKIPPING — checkpoint exists', flush=True)
            continue

        cmd = config_to_args(cfg)
        if args.dry_run:
            print(f'  DRY RUN: {" ".join(cmd)}', flush=True)
            continue

        print(f'  CMD: {" ".join(cmd)}', flush=True)
        result = subprocess.run(
            ['conda', 'run', '-n', 'crism'] + cmd,
            cwd=PROJ,
        )
        exit_code = result.returncode
        status = 'ok' if exit_code == 0 else f'FAILED({exit_code})'
        results.append({
            'run_name': run_name, 'model': model,
            'status': status, 'config': str(cfg),
        })
        print(f'  {status}', flush=True)

    if results and not args.dry_run:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['run_name', 'model', 'status', 'config']
            )
            writer.writeheader()
            writer.writerows(results)
        print(f'\nSweep summary written to {summary_path}')

    print(f'\nDone. {len(results)} configs run, '
          f'{sum(1 for r in results if r["status"] == "ok")} succeeded.')


if __name__ == '__main__':
    main()
