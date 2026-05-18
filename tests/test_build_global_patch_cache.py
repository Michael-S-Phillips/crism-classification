"""Tests for scripts/build_global_patch_cache.py."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def _make_fake_tile(path_prefix: str, H: int = 50, W: int = 50, n_bands: int = 59,
                    nodata_value: float = 65535, nodata_frac: float = 0.0,
                    seed: int = 0):
    """Write a minimal ENVI/.img+.hdr pair we can read with rasterio."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 0.3, size=(n_bands, H, W)).astype(np.float32)
    if nodata_frac > 0:
        nodata_mask = rng.uniform(0, 1, size=(H, W)) < nodata_frac
        data[:, nodata_mask] = nodata_value

    img_path = path_prefix + '.img'
    hdr_path = path_prefix + '.hdr'

    profile = {
        'driver': 'ENVI',
        'dtype': 'float32',
        'count': n_bands,
        'height': H,
        'width': W,
        'nodata': nodata_value,
        'interleave': 'bsq',
    }
    with rasterio.open(img_path, 'w', **profile) as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)
    return hdr_path, img_path


def test_extract_patches_from_tile_returns_correct_shape(tmp_path):
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50)
    patches, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr,
        n_target=30,
        patch_size=7,
        min_valid_frac=0.8,
        clip_max=0.5,
        nodata_value=65535,
        seed=42,
    )
    assert patches.shape == (30, 7, 7, 59)
    assert patches.dtype == np.float32
    assert n_skipped_short == 0


def test_extract_patches_clipped_to_clip_max(tmp_path):
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50)
    patches, _ = extract_patches_from_tile(
        hdr_path=hdr, n_target=10, patch_size=7,
        min_valid_frac=0.8, clip_max=0.5, nodata_value=65535, seed=0,
    )
    assert (patches >= 0).all()
    assert (patches <= 0.5).all()


def test_extract_skips_short_when_fewer_valid_centers(tmp_path):
    """A tile with nearly all nodata returns fewer than n_target patches."""
    from build_global_patch_cache import extract_patches_from_tile

    # 80% nodata → most centers fail min_valid_frac=0.8
    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_b'), H=30, W=30, nodata_frac=0.8)
    patches, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr, n_target=100, patch_size=7,
        min_valid_frac=0.8, clip_max=0.5, nodata_value=65535, seed=0,
    )
    # Should return fewer patches than requested
    assert patches.shape[0] < 100
    assert n_skipped_short == 100 - patches.shape[0]
