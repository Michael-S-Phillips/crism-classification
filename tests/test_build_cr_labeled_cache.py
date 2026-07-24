"""Test the raw→CR labeled-cache converter matches data.continuum_removal."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.continuum_removal import continuum_removed, brightness_scalar  # noqa: E402
from scripts.build_cr_labeled_cache import convert_split  # noqa: E402


def test_convert_split_matches_module(tmp_path):
    raw_dir = tmp_path / 'raw'
    out_dir = tmp_path / 'cr'
    raw_dir.mkdir()
    rng = np.random.default_rng(0)
    n, P = 37, 7
    patches = (0.05 + 0.3 * rng.random((n, P, P, 59))).astype(np.float32)
    # Write the RAW cache the way cache_mrral_patches.py does: a headerless
    # memmap (NOT np.save — that would add a .npy header the real reader/converter
    # never expect). This is what makes the test reflect production.
    raw_path = raw_dir / f'mrral_train_patches_p{P}.npy'
    mm = np.memmap(raw_path, dtype='float32', mode='w+', shape=(n, P, P, 59))
    mm[:] = patches
    mm.flush(); del mm

    count = convert_split(str(raw_dir), str(out_dir), 'train', P, chunk=8)
    assert count == n

    cr_path = out_dir / f'mrral_train_patches_p{P}.npy'
    # CR patches must be a RAW headerless memmap of exactly n*P*P*59*4 bytes
    # (the fine-tune reads them via np.memmap with a byte-count guard).
    assert os.path.getsize(cr_path) == n * P * P * 59 * 4
    cr = np.memmap(cr_path, dtype='float32', mode='r', shape=(n, P, P, 59))
    # Brightness sidecar is a real .npy (fine-tune reads it via np.load).
    br = np.load(out_dir / f'mrral_train_patches_p{P}_brightness.npy')
    assert br.shape == (n, P, P)
    for i in (0, 5, 19, n - 1):
        assert np.allclose(cr[i], continuum_removed(patches[i]), atol=1e-5)
        assert np.allclose(br[i], brightness_scalar(patches[i]), atol=1e-5)


def test_missing_split_is_skipped(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    count = convert_split(str(raw_dir), str(tmp_path / 'cr'), 'val', 7, chunk=8)
    assert count == 0
