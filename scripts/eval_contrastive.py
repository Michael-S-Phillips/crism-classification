"""Evaluate a contrastive-refined encoder via linear probe or full fine-tune.

Two modes:

* **linear** (default, fast): freeze the encoder; train a single ``Linear(D, n_classes)``
  head on the standard val pixels for ``--probe_epochs`` epochs; report per-class AP.
  Quick signal for "did the encoder representation actually improve".

* **finetune** (slow): drop the projection head, attach ``SpatialSpectralClassifier``,
  fine-tune end-to-end with ``encoder_lr_scale=0.001`` and ASL loss. Use this for
  the bottom-line comparison against the current champion.

Usage:

  # linear probe (fast)
  conda run -n crism python scripts/eval_contrastive.py \\
      --ckpt checkpoints/contrastive_plag_v1_best.pt \\
      --mode linear --probe_epochs 5

  # full fine-tune (slow)
  conda run -n crism python scripts/eval_contrastive.py \\
      --ckpt checkpoints/contrastive_plag_v1_best.pt \\
      --mode finetune --epochs 100
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config
from data.dataset import (CRISMSpectralPatchDataset, LABEL_COLS, _collapse_labels,
                          apply_olivine_relabels)
from evaluation.metrics import compute_full_metrics
from models.contrastive_encoder import ContrastiveEncoder
from models.spatial_spectral_transformer import SpatialSpectralClassifier
from training.losses import AsymmetricLoss


def _make_loss(args):
    """Return (loss_fn, name) — ASL by default to match the rest of the
    project's supervised FTs; pass ``--bce_loss`` to switch to BCE."""
    if args.bce_loss:
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        def loss_fn(logits, labels, weights):
            return (bce(logits, labels).mean(dim=-1) * weights).mean()
        return loss_fn, 'bce'
    asl = AsymmetricLoss(
        gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos,
        clip=args.asl_clip,
    )
    def loss_fn(logits, labels, weights):
        return asl(logits, labels, weights)
    return loss_fn, f'asl(g-={args.asl_gamma_neg},g+={args.asl_gamma_pos},clip={args.asl_clip})'


def build_mrral_map(cfg):
    data_root = cfg.get('data_root', '/mnt/mrdr')
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
    return {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


# ---------------------------------------------------------- encoders / loaders
def _build_encoder(args) -> ContrastiveEncoder:
    enc = ContrastiveEncoder(
        n_bands=args.n_bands, patch_size=args.patch_size,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, proj_dim=args.proj_dim,
    )
    return enc


def _load_contrastive_ckpt(model: ContrastiveEncoder, ckpt_path: str):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state') or ck.get('encoder_state') or ck
    if 'proj.0.weight' in state:
        # Full ContrastiveEncoder state
        model.load_state_dict(state, strict=False)
    else:
        # Just the encoder; warm-start only the inner transformer
        model.load_encoder_state_dict(state)
    return model


# ---------------------------------------------------------- linear probe
class LinearProbe(nn.Module):
    """Frozen encoder + a single Linear head."""

    def __init__(self, encoder: ContrastiveEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(encoder.embed_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder.encode(x)
        return self.head(h)


def _make_val_loader(cfg, batch_size: int, apply_relabels: str | None,
                     split: str = 'val', debug_rows: int | None = None):
    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    if apply_relabels:
        df, n = apply_olivine_relabels(df, apply_relabels)
        print(f'applied relabels: {n} pixels updated')
    df = _collapse_labels(df)
    sub = df[df['split'] == split].reset_index(drop=True)
    if debug_rows:
        sub = sub.head(debug_rows).reset_index(drop=True)
    # Pass cache_dir=None when debug-slicing — the on-disk cache is sized for
    # the full split and won't align with the truncated frame.
    cache_dir = None if debug_rows else cfg.get('patch_cache_dir')
    ds = CRISMSpectralPatchDataset(
        sub, build_mrral_map(cfg), patch_size=7,
        cache_dir=cache_dir, split=split,
    )
    tiers = sub['confidence_tier'].astype(str).tolist()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, sub, tiers


def _make_train_loader(cfg, batch_size: int, apply_relabels: str | None,
                       debug_rows: int | None = None,
                       extra_positive_pool_dirs: list[str] | None = None):
    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    if apply_relabels:
        df, n = apply_olivine_relabels(df, apply_relabels)
    df = _collapse_labels(df)
    sub = df[df['split'] == 'train'].reset_index(drop=True)
    if debug_rows:
        sub = sub.head(debug_rows).reset_index(drop=True)
    cache_dir = None if debug_rows else cfg.get('patch_cache_dir')
    base_ds = CRISMSpectralPatchDataset(
        sub, build_mrral_map(cfg), patch_size=7,
        cache_dir=cache_dir, split='train',
    )
    if extra_positive_pool_dirs:
        from torch.utils.data import ConcatDataset
        from data.contrastive_dataset import ExtraPositivesDataset
        extras = []
        for d in extra_positive_pool_dirs:
            try:
                ex = ExtraPositivesDataset(d, positive_class='plagioclase')
                extras.append(ex)
                print(f'  augmenting train loader with {len(ex)} extra positive patches from {d}')
            except FileNotFoundError as e:
                print(f'  skipping extra positive pool {d}: {e}')
        if extras:
            ds = ConcatDataset([base_ds] + extras)
        else:
            ds = base_ds
    else:
        ds = base_ds
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def _score(model: nn.Module, loader: DataLoader, device, tiers):
    model.eval()
    ys, ts = [], []
    with torch.no_grad():
        for feats, labels, _w in loader:
            logits = model(feats.to(device))
            ys.append(torch.sigmoid(logits).cpu().numpy())
            ts.append(labels.numpy())
    y_score = np.concatenate(ys)
    y_true = np.concatenate(ts)
    return compute_full_metrics(y_true, y_score, tiers)


# ---------------------------------------------------------- mode runners
def run_linear_probe(args, cfg, device):
    encoder = _build_encoder(args)
    _load_contrastive_ckpt(encoder, args.ckpt)
    probe = LinearProbe(encoder, n_classes=len(LABEL_COLS)).to(device)

    train_loader = _make_train_loader(cfg, args.batch_size, args.apply_relabels,
                                       debug_rows=args.debug_rows,
                                       extra_positive_pool_dirs=args.extra_positive_pool_dirs)
    val_loader, val_df, val_tiers = _make_val_loader(
        cfg, args.batch_size, args.apply_relabels, split='val',
        debug_rows=args.debug_rows)

    optim = torch.optim.AdamW(probe.head.parameters(), lr=args.probe_lr,
                              weight_decay=1e-4)
    loss_fn, loss_name = _make_loss(args)
    print(f'  probe loss: {loss_name}')

    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        running = 0.0; n = 0
        for step, (feats, labels, weights) in enumerate(train_loader):
            feats = feats.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = probe(feats)
            loss = loss_fn(logits, labels, weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item()) * feats.shape[0]
            n += feats.shape[0]
            if args.debug_steps and step + 1 >= args.debug_steps:
                break
        print(f'  probe epoch {epoch}: train_loss={running/max(n,1):.4f}')

    m = _score(probe, val_loader, device, val_tiers)
    print(f'\nLINEAR PROBE  ckpt={os.path.basename(args.ckpt)}  n_val={len(val_df):,}')
    print(f'  val_mAP = {m["mAP"]:.4f}')
    for cls, apv in m['per_class_ap'].items():
        print(f'  val_AP_{cls:<12s} = {apv:.4f}')
    return m


def run_finetune(args, cfg, device):
    """Full fine-tune: drop projection head, attach classifier, train end-to-end."""
    encoder = _build_encoder(args)
    _load_contrastive_ckpt(encoder, args.ckpt)

    clf = SpatialSpectralClassifier(
        n_bands=args.n_bands, patch_size=args.patch_size,
        n_classes=len(LABEL_COLS),
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    # Transplant the contrastive-refined encoder weights into the classifier
    inner_state = encoder.encoder.state_dict()
    missing, unexpected = clf.load_encoder_state_dict(inner_state)
    print(f'transplanted encoder: missing={len(missing)} unexpected={len(unexpected)}')

    train_loader = _make_train_loader(cfg, args.batch_size, args.apply_relabels,
                                       debug_rows=args.debug_rows,
                                       extra_positive_pool_dirs=args.extra_positive_pool_dirs)
    val_loader, val_df, val_tiers = _make_val_loader(
        cfg, args.batch_size, args.apply_relabels, split='val',
        debug_rows=args.debug_rows)

    optim = torch.optim.AdamW(
        clf.get_param_groups(head_lr=args.lr,
                             encoder_lr=args.lr * args.encoder_lr_scale),
        weight_decay=1e-4,
    )
    loss_fn, loss_name = _make_loss(args)
    print(f'  finetune loss: {loss_name}  (patience={args.patience})')
    best_map = -1.0
    best_metrics = None
    best_epoch = -1
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        clf.train()
        for step, (feats, labels, weights) in enumerate(train_loader):
            feats = feats.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = clf(feats)
            loss = loss_fn(logits, labels, weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if args.debug_steps and step + 1 >= args.debug_steps:
                break
        m = _score(clf, val_loader, device, val_tiers)
        per_class = m['per_class_ap']
        plag = per_class.get('plagioclase', float('nan'))
        hcp = per_class.get('hcp', float('nan'))
        print(f'epoch {epoch}: val_mAP={m["mAP"]:.4f}  plag={plag:.4f}  hcp={hcp:.4f}')
        if m['mAP'] > best_map:
            best_map = m['mAP']
            best_metrics = m
            best_epoch = epoch
            patience_counter = 0
            torch.save({'model_state': clf.state_dict(), 'epoch': epoch,
                        'val_mAP': best_map},
                       os.path.join(args.output_dir,
                                    f'{args.run_name}_finetune_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'early-stopping at epoch {epoch} (no mAP improvement '
                      f'since epoch {best_epoch})')
                break

    print(f'\nFULL FINETUNE  ckpt={os.path.basename(args.ckpt)}  '
          f'n_val={len(val_df):,}  best_epoch={best_epoch}')
    print(f'  val_mAP = {best_map:.4f}')
    if best_metrics is not None:
        for cls, apv in best_metrics['per_class_ap'].items():
            print(f'  val_AP_{cls:<12s} = {apv:.4f}')
    return best_metrics


# ---------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='Contrastive checkpoint (model_state or encoder_state).')
    ap.add_argument('--mode', choices=['linear', 'finetune'], default='linear')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--apply_relabels', default=None,
                    help='Path to olivine_relabels.csv to apply corrected val labels.')
    ap.add_argument('--extra_positive_pool_dirs', nargs='*', default=None,
                    help='Pre-built positive patch pool directories (each contains '
                         'patches.npy + meta.parquet). Concatenated with the standard '
                         'train loader so the linear-probe / fine-tune head sees these '
                         'hand-vetted plag patches in addition to the parquet train split.')
    ap.add_argument('--batch_size', type=int, default=256)
    # linear probe
    ap.add_argument('--probe_epochs', type=int, default=5)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    # finetune
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--encoder_lr_scale', type=float, default=0.001)
    ap.add_argument('--run_name', default='eval_contrastive')
    ap.add_argument('--output_dir', default='checkpoints')
    ap.add_argument('--patience', type=int, default=25,
                    help='Early-stop patience (finetune mode only).')
    # loss config
    ap.add_argument('--bce_loss', action='store_true',
                    help='Use BCE instead of ASL. Default is ASL to match the '
                         'project\'s other supervised FTs.')
    ap.add_argument('--asl_gamma_neg', type=float, default=4.0)
    ap.add_argument('--asl_gamma_pos', type=float, default=0.0)
    ap.add_argument('--asl_clip', type=float, default=0.05)
    # encoder shape
    ap.add_argument('--n_bands', type=int, default=59)
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--proj_dim', type=int, default=64)
    # debugging
    ap.add_argument('--debug_steps', type=int, default=None,
                    help='Only run this many training steps per epoch (smoke test).')
    ap.add_argument('--debug_rows', type=int, default=None,
                    help='Limit train/val to first N rows (smoke test only — '
                         'disables the on-disk patch cache because it is sized '
                         'for the full split).')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg_path)
    cfg = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'linear':
        run_linear_probe(args, cfg, device)
    else:
        run_finetune(args, cfg, device)


if __name__ == '__main__':
    main()
