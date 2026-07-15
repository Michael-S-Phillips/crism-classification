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
    np.save(raw_dir / f'mrral_train_patches_p{P}.npy', patches)

    count = convert_split(str(raw_dir), str(out_dir), 'train', P, chunk=8)
    assert count == n

    cr = np.load(out_dir / f'mrral_train_patches_p{P}.npy')
    br = np.load(out_dir / f'mrral_train_patches_p{P}_brightness.npy')
    assert cr.shape == (n, P, P, 59)
    assert br.shape == (n, P, P)
    # every patch matches the reference transform
    for i in (0, 5, 19, n - 1):
        assert np.allclose(cr[i], continuum_removed(patches[i]), atol=1e-5)
        assert np.allclose(br[i], brightness_scalar(patches[i]), atol=1e-5)


def test_missing_split_is_skipped(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    count = convert_split(str(raw_dir), str(tmp_path / 'cr'), 'val', 7, chunk=8)
    assert count == 0
