"""load_tile's masking must match the dataset the classifier was trained through.

The classifier's training patches come from CRISMSpectralPatchDataset, which
treats reflectance above PHYS_MAX as corrupt and zeroes it. classify_tile_supervised
originally applied only the ==NODATA and non-finite tests, so a blue-edge value of
~3900 I/F was CLIPPED to 0.5 and survived as plausible-looking data. The model was
therefore trained on 0.0 and deployed on 0.5 for those pixels, and because band 0
anchors the continuum-removal hull the error propagates across the whole spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.dataset import CRISMSpectralPatchDataset
import scripts.classify_tile_supervised as cts


def test_phys_max_constant_matches_the_training_dataset():
    """If these drift, the model is trained and deployed on different inputs."""
    assert cts.PHYS_MAX == CRISMSpectralPatchDataset.PHYS_MAX
    assert cts.NODATA == CRISMSpectralPatchDataset.NODATA
    assert cts.CLIP_MAX == CRISMSpectralPatchDataset.CLIP_MAX


def _mask_like_load_tile(data, tmp_path):
    """Call the REAL load_tile on a real raster.

    An earlier version of this file reimplemented the masking locally and
    asserted against the copy. It passed even with PHYS_MAX deleted from
    load_tile, because it never called load_tile at all -- the seventeenth test
    in this project that could not fail for its stated reason. Write the array
    to disk and go through the actual function.
    """
    import rasterio
    path = str(tmp_path / 'tile.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[1],
                       width=data.shape[2], count=data.shape[0],
                       dtype='float32') as dst:
        dst.write(data.astype(np.float32))
    cube, valid, _t, _c = cts.load_tile(path)
    return cube.transpose(2, 0, 1), valid


def _mask_like_dataset(patch):
    p = patch.copy()
    p[(p == CRISMSpectralPatchDataset.NODATA) | ~np.isfinite(p)
      | (p > CRISMSpectralPatchDataset.PHYS_MAX)] = 0.0
    return np.clip(p, 0.0, CRISMSpectralPatchDataset.CLIP_MAX)


def test_blue_edge_is_masked_to_zero_not_clipped_to_half(tmp_path):
    """The specific failure: 3913 I/F must become 0.0, never 0.5."""
    data = np.full((cts.N_SRC_BANDS, 2, 2), 0.12, dtype=np.float32)
    data[0, 0, 0] = 3913.33                      # real t1030 band-0 maximum
    out, valid = _mask_like_load_tile(data, tmp_path)
    assert out[0, 0, 0] == 0.0, (
        f'blue-edge pixel became {out[0, 0, 0]} — clipped rather than masked')
    assert not valid[0, 0], 'blue-edge pixel was still counted valid'
    assert valid[1, 1], 'an unaffected pixel was wrongly invalidated'


def test_masking_agrees_with_the_dataset_on_random_spectra(tmp_path):
    """Whole-array agreement, not just the one crafted case."""
    rng = np.random.default_rng(0)
    data = (rng.random((cts.N_SRC_BANDS, 8, 8)) * 0.4).astype(np.float32)
    data[0, ::3, ::3] = rng.uniform(1.5, 4000.0, size=data[0, ::3, ::3].shape)
    data[5, 1, 1] = cts.NODATA
    data[7, 2, 2] = np.inf
    out, _ = _mask_like_load_tile(data, tmp_path)
    np.testing.assert_array_equal(out, _mask_like_dataset(data))


def test_a_value_between_clip_and_phys_max_is_kept_and_clipped(tmp_path):
    """0.5 < v <= 1.0 is legitimate-but-bright: clipped, NOT masked. This is the
    boundary the fix must not overreach past."""
    data = np.full((cts.N_SRC_BANDS, 1, 1), 0.12, dtype=np.float32)
    data[0, 0, 0] = 0.8
    out, valid = _mask_like_load_tile(data, tmp_path)
    assert out[0, 0, 0] == pytest.approx(cts.CLIP_MAX)
    assert valid[0, 0], 'a bright but physical pixel was wrongly invalidated'
