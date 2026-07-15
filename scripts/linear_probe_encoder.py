"""
Frozen-encoder linear probe for CR encoder-size selection (Task 6).

Encoder-size decision for the CR-native denoising MAE is made by a frozen linear
probe on the honest val split, NOT by MAE reconstruction loss (recon loss rewards
encoding brightness/texture — the nuisance CR removes). This script:

  1. loads a pretrained encoder checkpoint (``encoder_state``) and FREEZES it;
  2. extracts the center-token feature for every labeled patch (train + val),
     continuum-removing patches identically to fine-tuning;
  3. fits a linear head (multi-label logistic regression, gradient-trained) on
     the train features only;
  4. reports the frozen-encoder ``val_mAP_core`` — mean per-class AP EXCLUDING
     the junk class (identical to plain val_mAP for 5/6-class heads).

Run once per encoder (128-dim and 256-dim) and keep the larger encoder only if
it wins the probe; otherwise ship the 128-dim default.

Usage:
    conda run -n crism python scripts/linear_probe_encoder.py \\
        --encoder_ckpt checkpoints/spatial_mae_cr_denoising_128d_6l_best.pt \\
        --mrral_parquets data/mrral_pixels_7cls.parquet \\
        --patch_cache_dir data/patch_cache_cr \\
        --embed_dim 128 --n_layers 6 --seven_class \\
        --continuum_removed --cache_is_cr [--brightness_aux]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def load_frozen_encoder(encoder_ckpt: str, *, embed_dim: int = 128, n_heads: int = 4,
                        n_layers: int = 6, patch_size: int = 7, n_bands: int = 59,
                        device='cpu') -> nn.Module:
    """Load a SpatialSpectralTransformer encoder from a pretrain checkpoint and
    freeze it (requires_grad=False, eval mode). Accepts a dict with an
    ``encoder_state`` key (MAE pretrain checkpoint) or a bare encoder state_dict."""
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'encoder_state' in ckpt:
        state = ckpt['encoder_state']
    else:
        state = ckpt
    encoder = SpatialSpectralTransformer(
        n_bands=n_bands, patch_size=patch_size, embed_dim=embed_dim,
        n_heads=n_heads, n_layers=n_layers, dropout=0.0).to(device)
    missing, unexpected = encoder.load_encoder_state_dict(state)
    if missing or unexpected:
        log.info(f'encoder load: missing={missing}, unexpected={unexpected}')
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


@torch.no_grad()
def _extract_features(encoder: nn.Module, dataset, device, *, brightness_aux: bool,
                      batch_size: int = 512):
    """Return (X (N, D), Y (N, C)) center-token features + labels for a dataset.

    D = embed_dim (+1 when brightness_aux, brightness concatenated). The dataset
    yields (patch, label, weight) or (patch, brightness(1,), label, weight) when
    it was built with return_brightness=True.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    center_idx = encoder.n_tokens // 2 + 1  # +1 for CLS
    feats, labels = [], []
    for batch in loader:
        if brightness_aux:
            patch, bright, label, _w = batch
        else:
            patch, label, _w = batch
        patch = patch.to(device)
        out = encoder(patch)                     # (B, n_tokens+1, embed_dim)
        f = out[:, center_idx]                   # (B, embed_dim)
        if brightness_aux:
            f = torch.cat([f, bright.to(device)], dim=-1)
        feats.append(f.cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _fit_linear_head(X_tr, Y_tr, *, n_classes, epochs=300, lr=1e-2,
                     weight_decay=1e-4, device='cpu', seed=0):
    """Train a single Linear(D, C) multi-label logistic-regression head on
    standardized train features. Returns (head, mu, sd)."""
    torch.manual_seed(seed)
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    Xz = torch.from_numpy(((X_tr - mu) / sd).astype(np.float32)).to(device)
    Yt = torch.from_numpy((Y_tr > 0.4).astype(np.float32)).to(device)
    head = nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(Xz), Yt)
        loss.backward()
        opt.step()
    head.eval()
    return head, mu, sd


def linear_probe(encoder_ckpt: str, df, mrral_map: Optional[dict] = None,
                 cache_dir: Optional[str] = None, *,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 6,
                 patch_size: int = 7, continuum_removed: bool = True,
                 cache_is_cr: bool = False, brightness_aux: bool = False,
                 epochs: int = 300, lr: float = 1e-2, device: Optional[str] = None,
                 seed: int = 0) -> Dict[str, object]:
    """Freeze the encoder, fit a linear head on train center-token features, and
    return frozen-encoder val metrics. Key result: ``val_mAP_core`` (mean per-class
    AP excluding junk)."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    from data.dataset import CRISMSpectralPatchDataset, _collapse_labels, LABEL_COLS
    from evaluation.metrics import compute_map, compute_per_class_ap

    df = _collapse_labels(df)
    n_classes = len(LABEL_COLS)

    encoder = load_frozen_encoder(
        encoder_ckpt, embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
        patch_size=patch_size, device=device)

    def _ds(split):
        sub = df[df['split'] == split]
        return CRISMSpectralPatchDataset(
            sub, mrral_map or {}, patch_size=patch_size,
            cache_dir=cache_dir, split=split,
            continuum_removed=continuum_removed,
            return_brightness=brightness_aux, cache_is_cr=cache_is_cr)

    X_tr, Y_tr = _extract_features(encoder, _ds('train'), device,
                                   brightness_aux=brightness_aux)
    X_val, Y_val = _extract_features(encoder, _ds('val'), device,
                                     brightness_aux=brightness_aux)
    log.info(f'features: train {X_tr.shape}, val {X_val.shape}, n_classes={n_classes}')

    head, mu, sd = _fit_linear_head(
        X_tr, Y_tr, n_classes=n_classes, epochs=epochs, lr=lr,
        device=device, seed=seed)

    with torch.no_grad():
        Xz = torch.from_numpy(((X_val - mu) / sd).astype(np.float32)).to(device)
        y_score = torch.sigmoid(head(Xz)).cpu().numpy()

    val_map_core = compute_map(Y_val, y_score, exclude=('junk',))
    per_class = compute_per_class_ap(Y_val, y_score)
    return {
        'val_mAP_core': float(val_map_core),
        'per_class_ap': per_class,
        'n_train': int(len(X_tr)),
        'n_val': int(len(X_val)),
        'feature_dim': int(X_tr.shape[1]),
        'n_classes': int(n_classes),
    }


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder_ckpt', required=True,
                        help='Pretrain checkpoint (encoder_state) to probe.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--mrral_parquets', nargs='+', type=str, default=None,
                        help='Labeled mrral parquet(s); concatenated in order '
                             '(default: cfg.output_dir/mrral_pixels.parquet).')
    parser.add_argument('--patch_cache_dir', type=str, default=None,
                        help='CR/raw labeled patch cache dir (default cfg.patch_cache_dir).')
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=300,
                        help='Linear-head training epochs (frozen encoder).')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--continuum_removed', action='store_true',
                        help='CR patches on read (match CR fine-tuning).')
    parser.add_argument('--cache_is_cr', action='store_true',
                        help='The patch cache already holds CR patches.')
    parser.add_argument('--brightness_aux', action='store_true',
                        help='Concatenate the brightness scalar to the probe '
                             'features (match the brightness-aux classifier).')
    parser.add_argument('--seven_class', action='store_true')
    parser.add_argument('--with_alteration', action='store_true')
    args = parser.parse_args()

    if args.seven_class and args.with_alteration:
        parser.error('--seven_class and --with_alteration are mutually exclusive.')
    if args.cache_is_cr and not args.continuum_removed:
        parser.error('--cache_is_cr requires --continuum_removed.')
    if args.brightness_aux and not args.continuum_removed:
        parser.error('--brightness_aux requires --continuum_removed.')

    import data.dataset
    if args.seven_class:
        data.dataset.LABEL_COLS = list(data.dataset.LABEL_COLS_7CLASS)
    elif args.with_alteration:
        data.dataset.LABEL_COLS = list(data.dataset.LABEL_COLS_WITH_ALTERATION)
    log.info(f'LABEL_COLS = {data.dataset.LABEL_COLS}')

    from config_loader import load_config
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    cfg = load_config(cfg_path)

    paths = args.mrral_parquets or [os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    log.info(f'loaded {len(df):,} labeled rows from {paths}')

    cache_dir = args.patch_cache_dir or cfg.get('patch_cache_dir')

    # Build mrral_map only when there is no complete cache (on-the-fly fallback).
    mrral_map = {}
    cache_complete = cache_dir and all(
        os.path.exists(os.path.join(cache_dir, f'mrral_{s}_patches_p{args.patch_size}.npy'))
        for s in ('train', 'val'))
    if not cache_complete:
        import glob as _glob
        data_root = cfg.get('data_root', '/mnt/mrdr')
        hdrs = sorted(set(_glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                          + _glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
        mrral_map = {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
                     for h in hdrs}

    result = linear_probe(
        encoder_ckpt=args.encoder_ckpt, df=df, mrral_map=mrral_map,
        cache_dir=cache_dir, embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, patch_size=args.patch_size,
        continuum_removed=args.continuum_removed, cache_is_cr=args.cache_is_cr,
        brightness_aux=args.brightness_aux, epochs=args.epochs, lr=args.lr)

    print('\n=== Frozen linear probe ===')
    print(f'  encoder_ckpt: {args.encoder_ckpt}')
    print(f'  embed_dim={args.embed_dim} n_layers={args.n_layers} '
          f'feature_dim={result["feature_dim"]}')
    print(f'  n_train={result["n_train"]} n_val={result["n_val"]}')
    print(f'  val_mAP_core: {result["val_mAP_core"]:.4f}')
    for cls, ap in result['per_class_ap'].items():
        print(f'    AP[{cls}]: {ap:.4f}')


if __name__ == '__main__':
    main()
