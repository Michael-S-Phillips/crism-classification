"""
Sweep v6: ASL + differential LR + combined mrral/mrrsu features.

Changes from v5:
  - AsymmetricLoss (gamma_neg=4, gamma_pos=0) replaces focal loss
  - Differential LR (encoder_lr = lr * 0.1) for MAE-pretrained configs
  - SpectralHybridClassifier: mrral encoder + mrrsu summary param branch
  - spectral_aug re-enabled with moderate settings
  - All configs use n_layers=4 when loading 4-layer MAE checkpoint

Configs:
  A. scnn_asl_v6:          SpectralCNN  + ASL                (CNN baseline)
  B. svit_asl_v6:          SpectralViT  + ASL   6L  no pretrain
  C. svit_asl_diffr_v6:    SpectralViT  + ASL + diff LR + 4L MAE  (mrral only)
  D. shybrid_asl_v6:       Hybrid       + ASL   4L  no pretrain
  E. shybrid_asl_diffr_v6: Hybrid       + ASL + diff LR + 4L MAE  ← primary target

Usage:
    python scripts/sweep_v6.py
    python scripts/sweep_v6.py --dry_run
    python scripts/sweep_v6.py --only scnn_asl_v6 svit_asl_v6
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
MAE4_CKPT = os.path.join(CKPT_DIR, 'mae_pretrain_128d_4l_best.pt')

SWEEP_CONFIGS = [
    # A: SpectralCNN + ASL
    dict(model='spectral_cnn', run_name='scnn_asl_v6',
         epochs=200, patience=30, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         warmup_epochs=0, lr_t_max=50),

    # B: SpectralViT + ASL (6L, no pretrain)
    dict(model='spectral_vit', run_name='svit_asl_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         warmup_epochs=5, lr_t_max=50),

    # C: SpectralViT + ASL + differential LR + 4-layer MAE (n_layers=4 to match checkpoint)
    dict(model='spectral_vit', run_name='svit_asl_diffr_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         encoder_lr_scale=0.1,
         pretrain_ckpt=MAE4_CKPT,
         warmup_epochs=5, lr_t_max=50),

    # D: Hybrid (mrral+mrrsu) + ASL (4L, no pretrain)
    dict(model='spectral_hybrid', run_name='shybrid_asl_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         warmup_epochs=5, lr_t_max=50),

    # E: Hybrid + ASL + differential LR + 4-layer MAE (primary target)
    dict(model='spectral_hybrid', run_name='shybrid_asl_diffr_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         encoder_lr_scale=0.1,
         pretrain_ckpt=MAE4_CKPT,
         warmup_epochs=5, lr_t_max=50),
]

BOOL_FLAGS = {
    'use_pos_weight', 'high_conf_only', 'focal_loss',
    'balanced_sampling', 'spectral_aug', 'asl_loss',
}


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
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these run_names (space-separated)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if checkpoint already exists')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    configs = SWEEP_CONFIGS
    if args.only:
        configs = [c for c in configs if c['run_name'] in args.only]

    total = len(configs)
    results = []

    for i, cfg in enumerate(configs):
        run_name = cfg['run_name']
        print(f'\n[{i+1}/{total}] {run_name}', flush=True)
        if not args.force and ckpt_exists(run_name):
            print('  SKIPPING — checkpoint exists (use --force to rerun)', flush=True)
            continue
        pretrain = cfg.get('pretrain_ckpt')
        if pretrain and not os.path.exists(pretrain):
            print(f'  SKIPPING — MAE checkpoint not found: {pretrain}', flush=True)
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
        out = os.path.join(LOG_DIR, f'sweep_v6_{stamp}.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'status'])
            w.writeheader()
            w.writerows(results)
        print(f'\nSweep summary: {out}')

    print(f'\nDone. {len(results)} ran, {sum(1 for r in results if r["status"] == "ok")} ok.')


if __name__ == '__main__':
    main()
