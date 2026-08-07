"""
Re-run the two best ablation configs on the full 1.97M parquet (including Hellas).

Goals:
  - scnn_focal_v2: verify scnn_focal result on new data (dropped balanced/aug)
  - svit_mae_v2:   verify svit_mae on new data with extended patience
  - scnn_focal_v2 + svit_mae_v2 ensemble is the target final model

Usage:
    python scripts/sweep_v4.py
    python scripts/sweep_v4.py --dry_run
"""
import argparse
import csv
import os
import subprocess
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(PROJ, 'scripts', 'train.py')
CKPT_DIR = os.path.join(PROJ, 'checkpoints')
LOG_DIR = os.path.join(PROJ, 'logs')
MAE_CKPT = os.path.join(CKPT_DIR, 'mae_pretrain_128d_4l_best.pt')

SWEEP_CONFIGS = [
    # scnn_focal re-run on new 1.97M parquet — focal loss only, no balanced/aug
    dict(model='spectral_cnn', run_name='scnn_focal_v2',
         epochs=200, patience=30, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         focal_loss=True, focal_gamma=2.0,
         warmup_epochs=0, lr_t_max=50),

    # svit_mae re-run on new 1.97M parquet — MAE pretrain + focal loss
    dict(model='spectral_vit', run_name='svit_mae_v2',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         focal_loss=True, focal_gamma=2.0,
         warmup_epochs=5, lr_t_max=50,
         pretrain_ckpt=MAE_CKPT),
]

BOOL_FLAGS = {'use_pos_weight', 'high_conf_only', 'focal_loss',
              'balanced_sampling', 'spectral_aug'}


def ckpt_exists(run_name: str) -> bool:
    return os.path.exists(os.path.join(CKPT_DIR, f'{run_name}_best.pt'))


def config_to_args(cfg: dict) -> list:
    args = ['python', TRAIN]
    for k, v in cfg.items():
        if k in BOOL_FLAGS:
            if v:
                args.append(f'--{k}')
        elif v is not None:
            args += [f'--{k}', str(v)]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    total = len(SWEEP_CONFIGS)
    results = []

    for i, cfg in enumerate(SWEEP_CONFIGS):
        run_name = cfg['run_name']
        print(f'\n[{i+1}/{total}] {run_name}', flush=True)
        if ckpt_exists(run_name):
            print('  SKIPPING — checkpoint exists', flush=True)
            continue
        if 'pretrain_ckpt' in cfg and not os.path.exists(cfg.get('pretrain_ckpt', '')):
            print(f'  SKIPPING — MAE checkpoint not found: {cfg["pretrain_ckpt"]}', flush=True)
            continue
        cmd = config_to_args(cfg)
        if args.dry_run:
            print(f'  DRY RUN: {" ".join(cmd)}', flush=True)
            continue
        print(f'  CMD: {" ".join(cmd)}', flush=True)
        result = subprocess.run(['conda', 'run', '-n', 'crism'] + cmd, cwd=PROJ)
        status = 'ok' if result.returncode == 0 else f'FAILED({result.returncode})'
        results.append({'run_name': run_name, 'status': status})
        print(f'  {status}', flush=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results and not args.dry_run:
        out = os.path.join(LOG_DIR, f'sweep_v4_{stamp}.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'status'])
            w.writeheader()
            w.writerows(results)
        print(f'\nSweep summary: {out}')

    print(f'\nDone. {len(results)} ran, {sum(1 for r in results if r["status"] == "ok")} ok.')


if __name__ == '__main__':
    main()
