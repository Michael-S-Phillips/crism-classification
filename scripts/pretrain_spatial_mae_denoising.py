"""
Denoising MAE pre-training for CRISM spatial-spectral patches.

Same data and training-loop machinery as scripts/pretrain_spatial_mae.py.
Differences:
  - Uses DenoisingSpatialSpectralMAE (corrupts input, recovers clean target)
  - Adds CLI flags for the three noise σ values

Usage (HPC):
    python scripts/pretrain_spatial_mae_denoising.py \\
        --epochs 200 --embed_dim 128 --n_layers 6 --mask_ratio 0.75 \\
        --sigma_gauss 0.0087 --sigma_spike 0.0058 --sigma_column 0.0049
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import get_device
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Schedule
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--warmup',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=1024)
    parser.add_argument('--patches_per_epoch', type=int, default=200_000)
    parser.add_argument('--num_workers', type=int,   default=4)
    # Architecture
    parser.add_argument('--embed_dim',   type=int,   default=128)
    parser.add_argument('--n_heads',     type=int,   default=4)
    parser.add_argument('--n_layers',    type=int,   default=6)
    parser.add_argument('--decoder_dim', type=int,   default=64)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--mask_ratio',  type=float, default=0.75)
    # Noise augmentation
    parser.add_argument('--sigma_gauss',  type=float, default=0.0087)
    parser.add_argument('--sigma_spike',  type=float, default=0.0058)
    parser.add_argument('--sigma_column', type=float, default=0.0049)
    parser.add_argument('--spike_center_band', type=int, default=15)
    parser.add_argument('--spike_fwhm_bands',  type=float, default=3.0)
    # Representation
    parser.add_argument('--continuum_removed', action='store_true',
                        help='CR pipeline: the MAE reconstructs in continuum-removed '
                             'space. Patches are fed UN-z-scored (per-patch z-score '
                             'would rescale away absolute band depth). Off → raw '
                             'pretrain unchanged. Expects global_patch_cache_dir to '
                             'hold a precomputed CR cache (build_global_patch_cache.py '
                             '--continuum_removed) unless --cr_on_read is set.')
    parser.add_argument('--cr_on_read', action='store_true',
                        help='Shards are RAW; continuum-remove each on read (slow — '
                             'per-pixel hull; use only for tests/small caches). '
                             'Requires --continuum_removed. Default: shards are '
                             'already CR, fed as-is.')
    # Run management
    parser.add_argument('--run_name', type=str, default='spatial_mae_denoising_128d_6l')
    parser.add_argument('--config',   type=str, default='config.yaml')
    parser.add_argument('--resume',   type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config
    )
    from config_loader import load_config
    cfg = load_config(cfg_path)

    run_name = args.run_name
    log.info(f"Run name: {run_name}")
    log.info(f"σ_gauss={args.sigma_gauss}, σ_spike={args.sigma_spike}, "
             f"σ_column={args.sigma_column}")

    # ── Data ──────────────────────────────────────────────────────────────
    shard_dir = cfg.get('global_patch_cache_dir')
    if not shard_dir:
        raise KeyError("config.local.yaml must define global_patch_cache_dir")
    log.info(f"Global patch cache: {shard_dir}")

    if args.cr_on_read and not args.continuum_removed:
        parser.error('--cr_on_read requires --continuum_removed')
    if args.continuum_removed:
        log.info("Continuum removal ON: reconstruction target in CR space, fed "
                 f"un-z-scored ({'CR-on-read from raw shards' if args.cr_on_read else 'precomputed CR cache'})")
    from data.cached_patch_dataset import CRISMCachedPatchDataset
    ds = CRISMCachedPatchDataset(
        shard_dir=shard_dir,
        normalize=not args.continuum_removed,      # CR fed un-z-scored (band depth)
        shuffle=True,
        continuum_removed=args.cr_on_read)          # re-CR only if shards are raw
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    device = get_device()
    log.info(f"Using device: {device}")

    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
    model = DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=args.decoder_dim, decoder_layers=args.decoder_layers,
        mask_ratio=args.mask_ratio,
        sigma_gauss=args.sigma_gauss,
        sigma_spike=args.sigma_spike,
        sigma_column=args.sigma_column,
        spike_center_band=args.spike_center_band,
        spike_fwhm_bands=args.spike_fwhm_bands,
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

    ckpt_dir = cfg['checkpoints_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

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
        log.info(f"Epoch {epoch}/{args.epochs} | denoising_loss={mean_loss:.6f} | lr={lr_now:.2e}")

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'denoising_loss': mean_loss, 'lr': lr_now})

        # Save every 50 epochs and at end
        if epoch % 50 == 0 or epoch == args.epochs:
            path = os.path.join(ckpt_dir, f'{run_name}_epoch{epoch}.pt')
            torch.save({
                'mae_state': model.state_dict(),
                'encoder_state': model.encoder_state_dict(),
                'epoch': epoch, 'mae_loss': mean_loss, 'config': vars(args),
            }, path)
            log.info(f"Saved {path}")

        # Save best
        if mean_loss < best_loss:
            best_loss = mean_loss
            path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({
                'mae_state': model.state_dict(),
                'encoder_state': model.encoder_state_dict(),
                'epoch': epoch, 'mae_loss': mean_loss, 'config': vars(args),
            }, path)


if __name__ == '__main__':
    main()
