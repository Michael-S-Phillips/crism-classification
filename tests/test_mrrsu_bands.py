"""Tests for ``data/mrrsu_bands.py``: band-name registry and cube reader for
mrrsu summary-parameter tiles.

Uses a real mrrsu tile when available (skips cleanly otherwise) because the
whole point of resolving indices from the header is to catch a tile whose
band order doesn't match what CLAUDE.md documents -- a synthetic header
could never exercise that."""
import glob
import os

import numpy as np
import pytest

from config_loader import load_config
from data.mrrsu_bands import (read_band_names, band_index, read_mrrsu_cube,
                              CORE_INDICES)


def _a_real_mrrsu_hdr():
    root = load_config()['data_root']
    hits = sorted(glob.glob(os.path.join(root, 'mc*', 't*mrrsu*.hdr'))
                  + glob.glob(os.path.join(root, 't*mrrsu*.hdr')))
    if not hits:
        pytest.skip('no mrrsu tile available locally')
    return hits[0]


def test_core_indices_match_documented_values():
    """CLAUDE.md documents these; a tile with a different band order must not
    silently shift them."""
    names = read_band_names(_a_real_mrrsu_hdr())
    assert len(names) == 60
    assert band_index(names, 'OLINDEX3') == 15
    assert band_index(names, 'BD1300') == 17
    assert band_index(names, 'LCPINDEX2') == 18
    assert band_index(names, 'HCPINDEX2') == 19
    assert CORE_INDICES == {'OLINDEX3': 15, 'BD1300': 17,
                            'LCPINDEX2': 18, 'HCPINDEX2': 19}


def test_unknown_param_raises_naming_it():
    names = read_band_names(_a_real_mrrsu_hdr())
    with pytest.raises(KeyError, match='NOSUCHPARAM'):
        band_index(names, 'NOSUCHPARAM')


def test_nodata_becomes_nan_not_65535():
    """65535 left as a number would poison every threshold comparison."""
    hdr = _a_real_mrrsu_hdr()
    cube, names = read_mrrsu_cube(hdr.replace('.hdr', '.img'))
    assert cube.dtype == np.float32
    assert cube.shape[-1] == 60
    assert not np.any(cube == 65535.0)
    assert np.isnan(cube).any()   # real tiles always have some nodata


def test_read_mrrsu_cube_returns_band_names_matching_read_band_names():
    """The names returned alongside the cube must be the same list
    ``read_band_names`` would produce for that tile's header -- otherwise a
    caller indexing the cube via ``band_index(names, ...)`` could silently
    desync from the array it's paired with."""
    hdr = _a_real_mrrsu_hdr()
    img_path = hdr.replace('.hdr', '.img')
    cube, names_from_cube = read_mrrsu_cube(img_path)
    names_from_hdr = read_band_names(hdr)
    assert names_from_cube == names_from_hdr


def test_cube_shape_matches_raster_dimensions_not_transposed():
    """(H, W, 60) is the documented return shape. mrrsu tiles are not square
    (e.g. 1630 samples x 1636 lines), so a (width, height) vs (height, width)
    transposition bug would be invisible to a bare ``shape[-1] == 60`` check
    but is exactly the kind of silent-shift bug this module exists to avoid."""
    import rasterio

    hdr = _a_real_mrrsu_hdr()
    img_path = hdr.replace('.hdr', '.img')
    with rasterio.open(img_path) as src:
        expected_h, expected_w = src.height, src.width
    cube, _ = read_mrrsu_cube(img_path)
    assert cube.shape[:2] == (expected_h, expected_w)
