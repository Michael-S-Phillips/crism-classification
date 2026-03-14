"""
Spatial MAE pre-training on all global CRISM mrral tiles.

Streams random 7×7 patches from all 1,764 mrral tiles. One "epoch" = 1M patches.
Saves best checkpoint (lowest reconstruction loss) and periodic checkpoints.

Usage:
    conda run -n crism python scripts/pretrain_spatial_mae.py

    # Custom config:
    conda run -n crism python scripts/pretrain_spatial_mae.py \\
        --epochs 400 --embed_dim 128 --n_layers 6 --mask_ratio 0.75 \\
        --batch_size 512 --no_wandb

Checkpoint: checkpoints/spatial_mae_{embed_dim}d_{n_layers}l_best.pt
Format:     {'encoder_state': ..., 'mae_state': ..., 'mae_loss': ...,
             'epoch': ..., 'config': {...}}
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Glob candidates tried in order; first that finds files wins.
# Pattern 1: local server layout  mc##/t*mrral*.hdr
# Pattern 2: flat transfer layout  t*mrral*.hdr (all files in one dir)
MRRAL_GLOB_CANDIDATES = [
    '/mnt/crism/MRDR/mc*/t*mrral*.hdr',   # local: mc## subdirs
    None,                                   # filled from config data_root below
]
PATCHES_PER_EPOCH = 100_000
SAVE_EVERY = 50  # save periodic checkpoint every N epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--embed_dim',   type=int,   default=128)
    parser.add_argument('--n_heads',     type=int,   default=4)
    parser.add_argument('--n_layers',    type=int,   default=6)
    parser.add_argument('--decoder_dim', type=int,   default=64)
    parser.add_argument('--mask_ratio',  type=float, default=0.85)
    parser.add_argument('--warmup',      type=int,   default=20)
    parser.add_argument('--num_workers',       type=int,   default=8)
    parser.add_argument('--patches_per_epoch', type=int,   default=PATCHES_PER_EPOCH,
                        help='Patches per epoch (default 1M; use smaller for smoke tests)')
    parser.add_argument('--no_wandb',    action='store_true')
    parser.add_argument('--resume',      type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--ckpt_dir',    type=str,   default=None,
                        help='Checkpoint directory (default: config.yaml checkpoints_dir, '
                             'falls back to <project>/checkpoints)')
    args = parser.parse_args()

    import yaml
    cfg_path = os.path.join(PROJ, 'config.yaml')
    from config_loader import load_config
    cfg = load_config(cfg_path)
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        cfg_ckpt = cfg.get('checkpoints_dir', '')
        # Fall back to project-relative path if config path isn't writable
        if cfg_ckpt and os.access(os.path.dirname(cfg_ckpt) or '.', os.W_OK):
            ckpt_dir = cfg_ckpt
        else:
            ckpt_dir = os.path.join(PROJ, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log.info(f"Checkpoints → {ckpt_dir}")

    run_name = f'spatial_mae_{args.embed_dim}d_{args.n_layers}l'

    # ── Data ──────────────────────────────────────────────────────────────
    # Build glob candidates from config data_root (supports flat and mc##/ layouts)
    data_root = cfg.get('data_root', '')
    globs_to_try = [
        os.path.join(data_root, 'mc*', 't*mrral*.hdr'),   # mc## subdirs
        os.path.join(data_root, 't*mrral*.hdr'),           # flat directory
        '/mnt/crism/MRDR/mc*/t*mrral*.hdr',               # hardcoded fallback
    ] if data_root else ['/mnt/crism/MRDR/mc*/t*mrral*.hdr']

    hdr_files = []
    for pattern in globs_to_try:
        hdr_files = sorted(glob.glob(pattern))
        if hdr_files:
            log.info(f"Found {len(hdr_files)} mrral tiles via {pattern}")
            break
    if not hdr_files:
        raise FileNotFoundError(
            f"No mrral HDR files found. Tried:\n" + "\n".join(f"  {g}" for g in globs_to_try)
        )

    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(hdr_files, patch_size=7, min_valid_frac=0.8)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=args.decoder_dim, mask_ratio=args.mask_ratio,
    ).to(device)

    # ── Optimizer & schedule ──────────────────────────────────────────────
    base_lr = 1.5e-4 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        betas=(0.9, 0.95), weight_decay=0.05,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['mae_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('mae_loss', float('inf'))
        log.info(f"Resumed from {args.resume} at epoch {start_epoch}, loss={best_loss:.6f}")

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb_entity = cfg.get('wandb', {}).get('entity') or None
            wandb.init(project='crism-mineral-classification', entity=wandb_entity,
                       name=run_name, config=vars(args), resume='allow')
        except Exception as e:
            log.warning(f"wandb init failed ({e}), continuing without")
            use_wandb = False

    # ── Training loop ─────────────────────────────────────────────────────
    batches_per_epoch = args.patches_per_epoch // args.batch_size
    data_iter = iter(loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(batches_per_epoch):
            try:
                patches = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                patches = next(data_iter)

            patches = patches.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(patches)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch}/{args.epochs} | mae_loss={mean_loss:.6f} | lr={lr_now:.2e}")

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'mae_loss': mean_loss, 'lr': lr_now})

        # Save best checkpoint
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({
                'encoder_state': model.encoder_state_dict(),
                'mae_state': model.state_dict(),
                'mae_loss': best_loss,
                'epoch': epoch,
                'config': vars(args),
            }, best_path)
            log.info(f"  → New best: {best_loss:.6f}  saved to {best_path}")

        # Save periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            periodic_path = os.path.join(ckpt_dir, f'{run_name}_epoch{epoch}.pt')
            torch.save({
                'encoder_state': model.encoder_state_dict(),
                'mae_state': model.state_dict(),
                'mae_loss': mean_loss,
                'epoch': epoch,
                'config': vars(args),
            }, periodic_path)
            log.info(f"  Periodic checkpoint → {periodic_path}")

    log.info(f"Pre-training complete. Best MAE loss: {best_loss:.6f}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
