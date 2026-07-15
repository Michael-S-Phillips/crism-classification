"""Task 6: frozen-encoder linear-probe selector.

scripts/linear_probe_encoder.py freezes a pretrained CR encoder, extracts the
center-token feature for each labeled patch, fits a linear head on the train
split, and reports the frozen-encoder val_mAP_core (mean per-class AP EXCLUDING
junk) on the val split — the metric used to pick the encoder size (128 vs 256).

Synthetic test: random encoder + random labels → a finite val_mAP_core in [0,1].
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _make_labeled_cache(tmp_path, n=180, patch_size=7, seed=0):
    """Build a synthetic labeled df + raw mrral patch cache (train/val/test)."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = (rng.random(n) > 0.7).astype(np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    data['tile_id'] = 't0001'
    data['polygon_id'] = 0
    data['pixel_row'] = 0
    data['pixel_col'] = 0
    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    data['split'] = ['train'] * n_tr + ['val'] * n_val + ['test'] * (n - n_tr - n_val)
    df = pd.DataFrame(data)

    cache_dir = str(tmp_path / 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        fp = np.memmap(os.path.join(cache_dir, f'mrral_{split}_patches_p{patch_size}.npy'),
                       dtype='float32', mode='w+', shape=(len(sub), patch_size, patch_size, 59))
        fp[:] = rng.uniform(0.0, 0.5, size=(len(sub), patch_size, patch_size, 59)).astype(np.float32)
        fp.flush(); del fp
    return df, cache_dir


def _make_encoder_ckpt(tmp_path, embed_dim=32, n_heads=2, n_layers=2):
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
    mae = DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7, embed_dim=embed_dim,
        n_heads=n_heads, n_layers=n_layers, decoder_dim=16, decoder_layers=1)
    ckpt = tmp_path / 'enc.pt'
    torch.save({'encoder_state': mae.encoder_state_dict()}, str(ckpt))
    return str(ckpt)


def test_linear_probe_returns_finite_val_map_core(tmp_path):
    from scripts.linear_probe_encoder import linear_probe

    df, cache_dir = _make_labeled_cache(tmp_path, n=180, seed=1)
    ckpt = _make_encoder_ckpt(tmp_path, embed_dim=32, n_heads=2, n_layers=2)

    result = linear_probe(
        encoder_ckpt=ckpt, df=df, cache_dir=cache_dir,
        embed_dim=32, n_heads=2, n_layers=2, patch_size=7,
        continuum_removed=True, cache_is_cr=False, brightness_aux=False,
        device='cpu', seed=0)

    assert 'val_mAP_core' in result
    v = result['val_mAP_core']
    assert isinstance(v, float)
    assert np.isfinite(v)
    assert 0.0 <= v <= 1.0


def test_linear_probe_freezes_encoder(tmp_path):
    """The probe must not mutate the loaded encoder weights (frozen)."""
    from scripts.linear_probe_encoder import linear_probe, load_frozen_encoder

    df, cache_dir = _make_labeled_cache(tmp_path, n=120, seed=2)
    ckpt = _make_encoder_ckpt(tmp_path, embed_dim=32, n_heads=2, n_layers=2)

    enc = load_frozen_encoder(ckpt, embed_dim=32, n_heads=2, n_layers=2,
                              patch_size=7, device='cpu')
    assert all(not p.requires_grad for p in enc.parameters())
    before = torch.cat([p.detach().flatten() for p in enc.parameters()]).clone()

    linear_probe(encoder_ckpt=ckpt, df=df, cache_dir=cache_dir,
                 embed_dim=32, n_heads=2, n_layers=2, patch_size=7,
                 continuum_removed=True, device='cpu', seed=0)

    enc2 = load_frozen_encoder(ckpt, embed_dim=32, n_heads=2, n_layers=2,
                               patch_size=7, device='cpu')
    after = torch.cat([p.detach().flatten() for p in enc2.parameters()])
    torch.testing.assert_close(before, after)
