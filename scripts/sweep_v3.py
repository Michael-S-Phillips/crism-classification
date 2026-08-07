"""
Ablation sweep comparing mrral-based spectral models.
Answers: what combination of components gets us to >0.90 mAP?

Ablation groups:
  A. Input data: mrral spectral_cnn baseline (compare to cnn_sw4 mAP=0.652 on mrrsu)
  B. Label quality: all conf vs high_conf_only
  C. Loss function: BCE vs focal loss
  D. Sampler: unweighted vs balanced
  E. Augmentation: no aug vs spectral_aug
  F. Architecture: SpectralTransformer (no pretrain)
  G. Pre-training: SpectralTransformer + MAE pretrain
  H. Kitchen sink: all improvements combined

Usage:
    conda run -n crism python scripts/sweep_v3.py
    conda run -n crism python scripts/sweep_v3.py --dry_run
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
    # --- Group A: raw mrral spectral data vs mrrsu ---
    dict(model='spectral_cnn', run_name='scnn_base',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         warmup_epochs=0, lr_t_max=50),

    # --- Group B: high_conf_only ---
    dict(model='spectral_cnn', run_name='scnn_highconf',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         high_conf_only=True, warmup_epochs=0, lr_t_max=50),

    # --- Group C: focal loss ---
    dict(model='spectral_cnn', run_name='scnn_focal',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         focal_loss=True, focal_gamma=2.0,
         warmup_epochs=0, lr_t_max=50),

    # --- Group D: balanced sampler ---
    dict(model='spectral_cnn', run_name='scnn_balanced',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         balanced_sampling=True, warmup_epochs=0, lr_t_max=50),

    # --- Group E: spectral augmentation ---
    dict(model='spectral_cnn', run_name='scnn_aug',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         spectral_aug=True, warmup_epochs=0, lr_t_max=50),

    # --- Group F: SpectralTransformer (no pretrain) ---
    dict(model='spectral_vit', run_name='svit_base',
         epochs=200, patience=25, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         warmup_epochs=5, lr_t_max=50),

    # --- Group G: SpectralTransformer + MAE pretrain ---
    dict(model='spectral_vit', run_name='svit_mae',
         epochs=200, patience=25, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         warmup_epochs=5, lr_t_max=50,
         pretrain_ckpt=MAE_CKPT),

    # --- Group H: Kitchen sink (all improvements) ---
    dict(model='spectral_cnn', run_name='scnn_best',
         epochs=200, patience=30, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         high_conf_only=True, focal_loss=True, focal_gamma=2.0,
         balanced_sampling=True, spectral_aug=True,
         warmup_epochs=0, lr_t_max=50),
    dict(model='spectral_vit', run_name='svit_best',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         high_conf_only=True, focal_loss=True, focal_gamma=2.0,
         balanced_sampling=True, spectral_aug=True,
         warmup_epochs=5, lr_t_max=50,
         pretrain_ckpt=MAE_CKPT),
]

BOOL_FLAGS = {'use_pos_weight', 'high_conf_only', 'focal_loss', 'balanced_sampling', 'spectral_aug'}


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
        out = os.path.join(LOG_DIR, f'sweep_v3_{stamp}.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'status'])
            w.writeheader()
            w.writerows(results)
        print(f'\nSweep summary: {out}')

    print(f'\nDone. {len(results)} ran, {sum(1 for r in results if r["status"] == "ok")} ok.')


if __name__ == '__main__':
    main()
