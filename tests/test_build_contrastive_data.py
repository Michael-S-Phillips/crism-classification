"""Unit test for the patch-extraction helper in build_contrastive_data.py.

Builds a tiny synthetic mrral-shaped raster and a single polygon over an
interior region, then verifies that the extractor returns the right number of
patches with the right NODATA / clipping behaviour.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rasterio = pytest.importorskip('rasterio')
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from scripts.build_contrastive_data import (extract_patches_from_geom,
                                              N_BANDS, PATCH_SIZE, NODATA,
                                              CLIP_MAX)


def _write_synthetic_cube(path: str, h: int, w: int, n_bands: int = 59,
                          nodata_corner_size: int = 2) -> str:
    """Write a tiny ENVI-style cube: uniform reflectance 0.25 everywhere except
    a NODATA corner. Returns the .img path. Saved as GeoTIFF since we just need
    rasterio to open it; the schema doesn't need to be ENVI for these tests.
    """
    arr = np.full((n_bands, h, w), 0.25, dtype=np.float32)
    # Inject a NODATA corner so the center-pixel NODATA filter has work to do
    arr[:, :nodata_corner_size, :nodata_corner_size] = NODATA
    # And one OOB-large value to exercise clipping
    arr[0, h - 1, w - 1] = 1.0

    transform = from_origin(0, h, 1, 1)            # 1×1 pixel size, identity-ish
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=h, width=w, count=n_bands,
        dtype='float32',
        transform=transform,
    ) as dst:
        dst.write(arr)
    return path


def test_extract_patches_from_geom_basic(tmp_path):
    img = _write_synthetic_cube(str(tmp_path / 'tile.tif'), h=50, w=50)
    with rasterio.open(img) as src:
        # Polygon covering an interior 10x10 region (pixels 20..29 in both dims).
        # In pixel space, transform is 1px=1unit with origin at top-left (0, h).
        # Build polygon in raster CRS coordinates.
        # Using rasterio.transform: row 20..30, col 20..30 maps to
        # (x: 20..30, y: 30..20) since y is inverted (origin top).
        poly = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
        # Transform y back through the transform.
        patches = extract_patches_from_geom(
            src, poly, src.transform, src.height, src.width,
            max_per_polygon=200, seed=0,
        )
    # We expect at least PATCH_SIZE**2 patches before max_per_polygon caps anything
    assert len(patches) >= PATCH_SIZE * PATCH_SIZE
    for patch, r, c in patches:
        assert patch.shape == (PATCH_SIZE, PATCH_SIZE, N_BANDS)
        assert patch.dtype == np.float32
        # All values are within [0, CLIP_MAX] (clipped)
        assert patch.min() >= 0.0 - 1e-6
        assert patch.max() <= CLIP_MAX + 1e-6


def test_small_polygon_skipped(tmp_path):
    img = _write_synthetic_cube(str(tmp_path / 'tile.tif'), h=30, w=30)
    with rasterio.open(img) as src:
        # 3x3 polygon → 9 pixels < 49; should be skipped
        poly = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])
        patches = extract_patches_from_geom(
            src, poly, src.transform, src.height, src.width,
            max_per_polygon=200, seed=0,
        )
    assert patches == [], f'expected skip for small polygon, got {len(patches)}'


def test_max_per_polygon_caps_count(tmp_path):
    img = _write_synthetic_cube(str(tmp_path / 'tile.tif'), h=80, w=80)
    with rasterio.open(img) as src:
        # Large interior polygon -> many pixels
        poly = Polygon([(10, 10), (60, 10), (60, 60), (10, 60)])
        patches = extract_patches_from_geom(
            src, poly, src.transform, src.height, src.width,
            max_per_polygon=25, seed=0,
        )
    assert len(patches) == 25, f'expected exactly 25 patches, got {len(patches)}'
