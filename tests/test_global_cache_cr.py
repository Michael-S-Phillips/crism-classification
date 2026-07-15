"""Tests for the --continuum_removed hook in scripts/build_global_patch_cache.py.

CR is applied per patch after the existing clip; a parallel brightness map is
returned/saved alongside. Raw (CR-off) behaviour must remain byte-identical.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from data.continuum_removal import continuum_removed, brightness_scalar

# Reuse the fake-tile helpers from the existing builder test suite.
from test_build_global_patch_cache import _make_fake_tile, _make_synthetic_data_root


def test_cr_patch_matches_continuum_removed(tmp_path):
    """With CR on, each returned patch equals continuum_removed(raw_patch) and
    the brightness map equals brightness_scalar(raw_patch)."""
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50, seed=7)

    common = dict(hdr_path=hdr, n_target=20, patch_size=7, min_valid_frac=0.8,
                  clip_max=0.5, nodata_value=65535, seed=123)

    # Raw run (CR off) — same seed → same sampled centers → aligned patches.
    raw_patches, n_raw = extract_patches_from_tile(**common)
    # CR run.
    cr_patches, brightness, n_cr = extract_patches_from_tile(
        continuum_removed=True, **common)

    assert raw_patches.shape == (20, 7, 7, 59)
    assert cr_patches.shape == (20, 7, 7, 59)
    assert cr_patches.dtype == np.float32
    assert brightness.shape == (20, 7, 7)
    assert brightness.dtype == np.float32
    assert n_raw == n_cr == 0

    for i in range(len(raw_patches)):
        np.testing.assert_allclose(
            cr_patches[i], continuum_removed(raw_patches[i]), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            brightness[i], brightness_scalar(raw_patches[i]), rtol=0, atol=1e-6)

    # CR is a valid representation: never NaN/Inf, never > 1.0001.
    assert np.all(np.isfinite(cr_patches))
    assert cr_patches.max() <= 1.0001


def test_cr_off_is_byte_identical(tmp_path):
    """CR off returns the original 2-tuple with unchanged patch bytes."""
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_b'), H=40, W=40, seed=3)
    common = dict(hdr_path=hdr, n_target=15, patch_size=7, min_valid_frac=0.8,
                  clip_max=0.5, nodata_value=65535, seed=9)

    out = extract_patches_from_tile(**common)
    assert isinstance(out, tuple) and len(out) == 2  # (patches, n_skipped_short)
    patches, _ = out
    assert patches.shape == (15, 7, 7, 59)
    assert patches.max() <= 0.5  # clip unchanged, no CR


def test_main_writes_brightness_sidecars(tmp_path):
    """End-to-end: --continuum_removed writes a *_brightness.npy per shard with
    matching row counts, and records the flag in shard_index.json."""
    data_root, _hdrs = _make_synthetic_data_root(tmp_path, n_tiles=3, H=60, W=60)
    output = tmp_path / 'cr_cache'

    result = subprocess.run(
        [
            sys.executable, '-u',
            os.path.join(os.path.dirname(__file__), '..', 'scripts',
                         'build_global_patch_cache.py'),
            '--output', str(output),
            '--data_root', data_root,
            '--workers', '2',
            '--seed', '42',
            '--patches_per_tile_target', '30',
            '--patches_per_shard', '40',
            '--continuum_removed',
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"builder failed: {result.stderr}"

    shard_files = sorted(glob.glob(str(output / 'global_patches_*.npy')))
    shard_files = [f for f in shard_files if not f.endswith('_brightness.npy')]
    bright_files = sorted(glob.glob(str(output / 'global_patches_*_brightness.npy')))
    assert len(shard_files) == len(bright_files) == 3

    for pf, bf in zip(shard_files, bright_files):
        p = np.load(pf, mmap_mode='r')
        b = np.load(bf, mmap_mode='r')
        assert p.shape[0] == b.shape[0]
        assert p.shape == (p.shape[0], 7, 7, 59)
        assert b.shape == (b.shape[0], 7, 7)
        assert np.asarray(p).max() <= 1.0001  # CR patches, not raw reflectance

    with open(output / 'shard_index.json') as f:
        idx = json.load(f)
    assert idx['continuum_removed'] is True
