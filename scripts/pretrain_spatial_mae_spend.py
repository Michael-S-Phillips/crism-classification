"""
SPEND-style spatial-spectral MAE pre-training driver.

Mirrors scripts/pretrain_spatial_mae_denoising.py but with the SPEND
objective and a spectral-mask annealing schedule replacing the synthetic
noise injection.

Usage (HPC):
    python scripts/pretrain_spatial_mae_spend.py \\
        --epochs 200 --embed_dim 128 --n_layers 6 --mask_ratio 0.75 \\
        --spectral_mask_ratio 0.5 \\
        --anneal_start_epoch 161 --anneal_end_epoch 181
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


def main():
    parser = argparse.ArgumentParser()
    # Schedule
    parser.add_argument('--epochs',      type=int, default=200)
    parser.add_argument('--warmup',      type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=1024)
    parser.add_argument('--patches_per_epoch', type=int, default=200_000)
    parser.add_argument('--num_workers', type=int, default=4)
    # Architecture
    parser.add_argument('--embed_dim',      type=int, default=128)
    parser.add_argument('--n_heads',        type=int, default=4)
    parser.add_argument('--n_layers',       type=int, default=6)
    parser.add_argument('--decoder_dim',    type=int, default=64)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--mask_ratio',     type=float, default=0.75)
    # SPEND
    parser.add_argument('--spectral_mask_ratio', type=float, default=0.5,
                        help='Base spectral mask ratio for phase A.')
    parser.add_argument('--anneal_start_epoch', type=int, default=161)
    parser.add_argument('--anneal_end_epoch',   type=int, default=181)
    # Run management
    parser.add_argument('--run_name', type=str, default='spatial_mae_spend_128d_6l')
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
    log.info(f"SPEND base ratio={args.spectral_mask_ratio}, "
             f"anneal {args.anneal_start_epoch}→{args.anneal_end_epoch}")

    # ── Data ──────────────────────────────────────────────────────────────
    data_root = cfg.get('data_root', '/mnt/crism/MRDR')
    globs_to_try = [
        os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
        os.path.join(data_root, 't*mrral*.hdr'),
    ]
    hdr_files = []
    for g in globs_to_try:
        hdr_files = sorted(glob.glob(g))
        if hdr_files:
            break
    if not hdr_files:
        raise FileNotFoundError(
            f"No mrral HDR files found. Tried:\n" + "\n".join(f"  {g}" for g in globs_to_try)
        )
    log.info(f"Found {len(hdr_files)} mrral tiles")

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

    from models.spend_spatial_mae import (
        SpendSpatialSpectralMAE,
        compute_spectral_mask_ratio,
    )
    model = SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=args.decoder_dim, decoder_layers=args.decoder_layers,
        mask_ratio=args.mask_ratio,
        spectral_mask_ratio=args.spectral_mask_ratio,
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

    # Construct scheduler in its default state. We restore its real position
    # via load_state_dict on resume; PyTorch only auto-injects 'initial_lr'
    # into param groups when last_epoch == -1, so we cannot pass last_epoch
    # directly to the constructor here.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume path not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['mae_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        # best_loss persists across resumes so the _best.pt provenance is
        # preserved — fall back to this checkpoint's loss only if missing.
        best_loss = ckpt.get('best_loss', ckpt.get('mae_loss', float('inf')))
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        log.info(
            f"Resumed from {args.resume} at epoch {start_epoch}, "
            f"best_loss={best_loss:.6f}"
        )

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

    ckpt_dir = cfg.get('checkpoints_dir', '/mnt/mrdr/crism_classification/checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Anneal callback: update spectral_mask_ratio for this epoch ────
        model.spectral_mask_ratio = compute_spectral_mask_ratio(
            epoch=epoch,
            anneal_start_epoch=args.anneal_start_epoch,
            anneal_end_epoch=args.anneal_end_epoch,
            base=args.spectral_mask_ratio,
        )

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
            if not torch.isfinite(loss):
                log.warning(f"non-finite loss at epoch {epoch}, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]['lr']
        log.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"spend_loss={mean_loss:.6f} | "
            f"spectral_mask_ratio={model.spectral_mask_ratio:.3f} | "
            f"lr={lr_now:.2e}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'spend_loss': mean_loss,
                'spectral_mask_ratio': model.spectral_mask_ratio,
                'lr': lr_now,
            }, step=epoch)

        # Save every 50 epochs and at end
        if epoch % 50 == 0 or epoch == args.epochs:
            path = os.path.join(ckpt_dir, f'{run_name}_epoch{epoch}.pt')
            torch.save({
                'mae_state': model.state_dict(),
                'encoder_state': model.encoder_state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'mae_loss': mean_loss,
                'best_loss': best_loss,
                'config': vars(args),
            }, path)
            log.info(f"Saved {path}")

        # Save best
        if mean_loss < best_loss:
            best_loss = mean_loss
            path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({
                'mae_state': model.state_dict(),
                'encoder_state': model.encoder_state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'mae_loss': mean_loss,
                'best_loss': best_loss,
                'config': vars(args),
            }, path)


if __name__ == '__main__':
    main()
