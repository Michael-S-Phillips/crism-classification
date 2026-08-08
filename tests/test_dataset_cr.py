"""Tests for continuum-removal + brightness options on CRISMSpectralPatchDataset.

NB: the plan names this "CRISMPatchDataset"; the 59-band mrral patch dataset that
the CR module (59 good bands) and the Task-2 (n,P,P,59) cache actually feed is
CRISMSpectralPatchDataset. CRISMPatchDataset serves 60-band mrrsu summary params
and is not a CR target. See implementation report.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import CRISMSpectralPatchDataset
from data.continuum_removal import continuum_removed, brightness_scalar


def _make_mrral_tile(path_prefix, H=30, W=30, n_bands=59, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 0.4, size=(n_bands, H, W)).astype(np.float32)
    img_path = path_prefix + '.img'
    profile = {'driver': 'ENVI', 'dtype': 'float32', 'count': n_bands,
               'height': H, 'width': W, 'interleave': 'bsq'}
    with rasterio.open(img_path, 'w', **profile) as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)
    return img_path


def _make_df(tile_id, rows, cols, n):
    return pd.DataFrame({
        'tile_id': [tile_id] * n,
        'pixel_row': rows, 'pixel_col': cols,
        'olivine_t1': [0.0] * n, 'olivine_t2': [0.0] * n,
        'lcp': [1.0] * n, 'hcp': [0.0] * n,
        'plagioclase': [0.0] * n, 'other': [0.0] * n,
        'confidence_weight': [1.0] * n,
        'confidence_tier': ['High'] * n,
        'split': ['train'] * n,
    })


def test_on_the_fly_cr_matches_module(tmp_path):
    """No cache: CR patch equals continuum_removed(raw patch); brightness is the
    center-pixel scalar; label/weight unchanged; raw mode still returns raw."""
    img = _make_mrral_tile(str(tmp_path / 't0001_mrral_00n000_0327_4'), H=30, W=30, seed=5)
    mrral_map = {'t0001': img}
    df = _make_df('t0001', rows=[15, 10], cols=[15, 20], n=2)

    ds_raw = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7)
    ds_cr = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7,
                                      continuum_removed=True, return_brightness=True)

    raw_patch, raw_label, raw_weight = ds_raw[0]
    assert raw_patch.shape == (7, 7, 59)

    out = ds_cr[0]
    assert len(out) == 4, "return_brightness=True must yield (patch, brightness, label, weight)"
    cr_patch_t, bright_t, label_t, weight_t = out

    assert cr_patch_t.shape == (7, 7, 59)
    assert bright_t.shape == (1,)
    np.testing.assert_allclose(
        cr_patch_t.numpy(), continuum_removed(raw_patch.numpy()), rtol=0, atol=1e-6)
    # Brightness aux = center-pixel mean good-band reflectance (pre-CR).
    expected_b = brightness_scalar(raw_patch.numpy())[3, 3]
    np.testing.assert_allclose(bright_t.item(), expected_b, rtol=0, atol=1e-6)
    # Labels/weights untouched by CR.
    assert torch.equal(label_t, raw_label)
    assert torch.equal(weight_t, raw_weight)
    assert cr_patch_t.max().item() <= 1.0001

    # continuum_removed without return_brightness → 3-tuple, CR applied.
    ds_cr_nob = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7,
                                          continuum_removed=True)
    out3 = ds_cr_nob[0]
    assert len(out3) == 3
    np.testing.assert_allclose(
        out3[0].numpy(), continuum_removed(raw_patch.numpy()), rtol=0, atol=1e-6)


def test_cache_cr_on_read(tmp_path):
    """A raw (n,P,P,59) cache is CR-transformed on read when cache_is_cr=False."""
    n, P = 6, 7
    rng = np.random.default_rng(1)
    raw = (rng.uniform(0.0, 0.5, size=(n, P, P, 59))).astype(np.float32)
    cache_file = tmp_path / f'mrral_train_patches_p{P}.npy'
    fp = np.memmap(str(cache_file), dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = raw
    fp.flush(); del fp

    df = _make_df('t0001', rows=[0] * n, cols=[0] * n, n=n)

    ds_raw = CRISMSpectralPatchDataset(df, {}, patch_size=P,
                                       cache_dir=str(tmp_path), split='train')
    ds_cr = CRISMSpectralPatchDataset(df, {}, patch_size=P,
                                      cache_dir=str(tmp_path), split='train',
                                      continuum_removed=True, return_brightness=True)

    for i in range(n):
        raw_patch = ds_raw[i][0].numpy()
        np.testing.assert_allclose(raw_patch, raw[i], rtol=0, atol=1e-6)
        cr_patch_t, bright_t, _, _ = ds_cr[i]
        np.testing.assert_allclose(
            cr_patch_t.numpy(), continuum_removed(raw[i]), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            bright_t.item(), brightness_scalar(raw[i])[3, 3], rtol=0, atol=1e-6)


def test_cache_is_cr_uses_patch_asis_with_brightness_sidecar(tmp_path):
    """cache_is_cr=True returns the cached (already-CR) patch as-is and reads the
    brightness scalar from the *_brightness.npy sidecar."""
    n, P = 5, 7
    rng = np.random.default_rng(2)
    cr_cache = rng.uniform(0.0, 1.0, size=(n, P, P, 59)).astype(np.float32)
    bright = rng.uniform(0.0, 0.4, size=(n, P, P)).astype(np.float32)
    cache_file = tmp_path / f'mrral_train_patches_p{P}.npy'
    fp = np.memmap(str(cache_file), dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = cr_cache; fp.flush(); del fp
    np.save(tmp_path / f'mrral_train_patches_p{P}_brightness.npy', bright)

    df = _make_df('t0001', rows=[0] * n, cols=[0] * n, n=n)
    ds = CRISMSpectralPatchDataset(df, {}, patch_size=P, cache_dir=str(tmp_path),
                                   split='train', continuum_removed=True,
                                   return_brightness=True, cache_is_cr=True)
    for i in range(n):
        patch_t, bright_t, _, _ = ds[i]
        np.testing.assert_allclose(patch_t.numpy(), cr_cache[i], rtol=0, atol=1e-6)
        np.testing.assert_allclose(bright_t.item(), bright[i, 3, 3], rtol=0, atol=1e-6)


def test_cache_is_cr_missing_split_cache_raises(tmp_path):
    """A CR cache that is missing the split being asked for must fail loudly at
    construction, naming the split.

    Regression: an incomplete patch_cache_base_cr (train built, val never
    written) let the val dataset silently fall through to the on-the-fly
    rasterio path, which — because cache_is_cr suppressed CR — produced a
    brightness of None and died mid-epoch with an unattributable
    'NoneType' object is not subscriptable. Runs pyxalt_cr_{0,1,2} 2026-08-07.
    """
    n, P = 4, 7
    rng = np.random.default_rng(3)
    cr_cache = rng.uniform(0.0, 1.0, size=(n, P, P, 59)).astype(np.float32)
    fp = np.memmap(str(tmp_path / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = cr_cache; fp.flush(); del fp
    np.save(tmp_path / f'mrral_train_patches_p{P}_brightness.npy',
            rng.uniform(0.0, 0.4, size=(n, P, P)).astype(np.float32))
    # 'val' was never built — only 'train' exists in this cache dir.

    img = _make_mrral_tile(str(tmp_path / 't0001_mrral_00n000_0327_4'), H=30, W=30, seed=7)
    df = _make_df('t0001', rows=[15] * n, cols=[15] * n, n=n)

    with pytest.raises(FileNotFoundError, match='val'):
        CRISMSpectralPatchDataset(df, {'t0001': img}, patch_size=P,
                                  cache_dir=str(tmp_path), split='val',
                                  continuum_removed=True,
                                  return_brightness=True, cache_is_cr=True)


def test_cache_is_cr_without_cache_still_applies_cr(tmp_path):
    """cache_is_cr describes the cache, not the request: with no cache in play
    the on-the-fly path must still CR the patch and compute brightness rather
    than serving a raw patch to a CR-trained model."""
    img = _make_mrral_tile(str(tmp_path / 't0001_mrral_00n000_0327_4'), H=30, W=30, seed=9)
    mrral_map = {'t0001': img}
    df = _make_df('t0001', rows=[15, 12], cols=[15, 18], n=2)

    ds_raw = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7)
    ds = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7,
                                   continuum_removed=True,
                                   return_brightness=True, cache_is_cr=True)

    raw_patch = ds_raw[0][0].numpy()
    patch_t, bright_t, _, _ = ds[0]
    np.testing.assert_allclose(
        patch_t.numpy(), continuum_removed(raw_patch), rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        bright_t.item(), brightness_scalar(raw_patch)[3, 3], rtol=0, atol=1e-6)
