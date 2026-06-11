"""Train a Linear(128, 5) head on top of a frozen contrastive encoder, and save
the result as a SpatialSpectralClassifier-format checkpoint that
``scripts/classify_tile_supervised.py`` can consume directly.

This is the "deployable linear probe": same numbers as
``eval_contrastive.py --mode linear`` produces, but the model state is saved in
the classifier wrapper's format so existing inference + vectorization scripts
just work.

Local-machine note: ``/mnt/mrdr`` here is a 9p network mount. Random-access
reads over the full ~22 GB patch cache are unusably slow. We therefore
random-sample N indices, sort them for near-sequential memmap reads, load the
selected patches into RAM, and train the head against that fixed tensor.

Usage:
    conda run -n crism python scripts/save_contrastive_deploy_ckpt.py \\
        --ckpt checkpoints/contrastive_plag_v1_best.pt \\
        --out_ckpt checkpoints/contrastive_plag_v1_deploy.pt \\
        --max_train_patches 200000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config
from data.contrastive_dataset import ExtraPositivesDataset
from data.dataset import (LABEL_COLS, _collapse_labels, apply_olivine_relabels)
from models.contrastive_encoder import ContrastiveEncoder
from models.spatial_spectral_transformer import SpatialSpectralClassifier


def _load_contrastive_encoder(ckpt_path, device):
    """Load a ContrastiveEncoder from any compatible state dict format."""
    model = ContrastiveEncoder(
        n_bands=59, patch_size=7, embed_dim=128,
        n_heads=4, n_layers=6, dropout=0.1, proj_dim=64,
    )
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state') or ck.get('encoder_state') or ck
    if isinstance(state, dict) and 'proj.0.weight' in state:
        model.load_state_dict(state, strict=False)
    else:
        model.load_encoder_state_dict(state)
    return model.to(device)


def _sample_into_ram(cfg, *, n_samples: int, seed: int, apply_relabels: str | None):
    """Random-sample ``n_samples`` train patches and load them into RAM.

    Returns ``(patches_tensor, labels_tensor, weights_tensor)`` aligned in
    LABEL_COLS order, alongside diagnostics."""
    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    if apply_relabels:
        df, n_rel = apply_olivine_relabels(df, apply_relabels)
        print(f'  applied relabels: {n_rel} pixels')
    df = _collapse_labels(df)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    n_total = len(train_df)
    print(f'  train rows total: {n_total:,}')

    n = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_total, size=n, replace=False))
    print(f'  sampled {n:,} random indices (seed={seed})')

    cache_file = os.path.join(
        cfg['patch_cache_dir'], 'mrral_train_patches_p7.npy')
    cache = np.memmap(cache_file, dtype='float32', mode='r',
                       shape=(n_total, 7, 7, 59))
    print(f'  memmap-reading {n:,} patches (sorted indices ~ sequential)...')
    t0 = time.time()
    patches_np = np.asarray(cache[chosen], dtype=np.float32)  # forces load
    print(f'    read {patches_np.nbytes / 1e9:.2f} GB in {time.time() - t0:.1f}s')

    labels_np = train_df.loc[chosen, LABEL_COLS].to_numpy(dtype=np.float32)
    weights_np = train_df.loc[chosen, 'confidence_weight'].to_numpy(dtype=np.float32)

    class_sums = labels_np.sum(axis=0)
    print(f'  class positives in sample: '
          f'{dict(zip(LABEL_COLS, class_sums.astype(int)))}')

    return (
        torch.from_numpy(patches_np),
        torch.from_numpy(labels_np),
        torch.from_numpy(weights_np),
    )


class _WeightedTensorDataset(torch.utils.data.Dataset):
    """TensorDataset that yields (patch, label, weight) tuples."""
    def __init__(self, patches, labels, weights):
        self.patches = patches
        self.labels = labels
        self.weights = weights
    def __len__(self):
        return self.patches.shape[0]
    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx], self.weights[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='Contrastive checkpoint (ContrastiveEncoder state).')
    ap.add_argument('--out_ckpt', required=True,
                    help='Output path for the SpatialSpectralClassifier ckpt.')
    ap.add_argument('--head_only_out', default=None,
                    help='Optional: also save head-only state dict (for '
                         'eval_polygon_accuracy --probe_head_path).')
    ap.add_argument('--extra_positive_pool_dir',
                    default='data/contrastive/extra_plag_roi')
    ap.add_argument('--apply_relabels', default=None)
    ap.add_argument('--max_train_patches', type=int, default=200000,
                    help='Random sample size from train set (default 200k).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            cfg_path,
        )
    cfg = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 1. Load contrastive encoder
    contrastive = _load_contrastive_encoder(args.ckpt, device)
    print(f'loaded contrastive encoder from {args.ckpt}')

    # 2. Build SpatialSpectralClassifier; transplant encoder
    clf = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=len(LABEL_COLS),
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.1,
    ).to(device)
    missing, unexpected = clf.load_encoder_state_dict(
        contrastive.encoder.state_dict())
    print(f'transplanted encoder: missing={len(missing)} unexpected={len(unexpected)}')

    for p in clf.encoder.parameters():
        p.requires_grad_(False)
    clf.encoder.eval()

    # 3. Sample training data into RAM
    print('\n--- sampling train data into RAM ---')
    patches_t, labels_t, weights_t = _sample_into_ram(
        cfg, n_samples=args.max_train_patches, seed=args.seed,
        apply_relabels=args.apply_relabels)

    base_ds = _WeightedTensorDataset(patches_t, labels_t, weights_t)

    # 4. Optional: append ROI augmentation
    extra_npy = os.path.join(args.extra_positive_pool_dir, 'patches.npy')
    if os.path.exists(extra_npy):
        extra_ds = ExtraPositivesDataset(args.extra_positive_pool_dir,
                                          positive_class='plagioclase')
        train_ds = ConcatDataset([base_ds, extra_ds])
        print(f'  + {len(extra_ds):,} ROI patches = {len(train_ds):,} total')
    else:
        train_ds = base_ds
        print(f'  no ROI augmentation at {args.extra_positive_pool_dir}')

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    # 5. Train head
    optim = torch.optim.AdamW(clf.head.parameters(), lr=args.lr,
                              weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    print('\n--- training head ---')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0; n = 0
        for feats, labels, weights in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            logits = clf(feats)
            loss = (bce(logits, labels).mean(dim=-1) * weights).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item()) * feats.shape[0]
            n += feats.shape[0]
        print(f'  epoch {epoch:>2d}/{args.epochs}: '
              f'train_loss={running / max(n, 1):.4f}  '
              f'({time.time() - t0:.1f}s)')

    # 6. Save
    os.makedirs(os.path.dirname(args.out_ckpt) or '.', exist_ok=True)
    torch.save({
        'model_state': clf.state_dict(),
        'source_contrastive_ckpt': args.ckpt,
        'epochs_head_trained': args.epochs,
        'lr': args.lr,
        'max_train_patches': args.max_train_patches,
        'seed': args.seed,
        'roi_pool_used': extra_npy if os.path.exists(extra_npy) else None,
    }, args.out_ckpt)
    print(f'\nsaved deploy checkpoint to {args.out_ckpt}')

    if args.head_only_out:
        os.makedirs(os.path.dirname(args.head_only_out) or '.', exist_ok=True)
        torch.save({'head_state': clf.head.state_dict()}, args.head_only_out)
        print(f'saved head-only state to {args.head_only_out}')


if __name__ == '__main__':
    main()
