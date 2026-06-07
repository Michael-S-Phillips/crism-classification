import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from scripts.review.loader import PixelBundle, load_polygon_pixels


def _write_synthetic_mrral(path, height=20, width=20, n_bands=59, nodata=65535):
    """Write a tiny ENVI-style float32 tile.
    - all values = 0.1 + 0.01 * band_index
    - column 0 and row 0 are NODATA (to test masking)
    """
    arr = np.zeros((n_bands, height, width), dtype=np.float32)
    for b in range(n_bands):
        arr[b] = 0.1 + 0.01 * b
    arr[:, 0, :] = nodata
    arr[:, :, 0] = nodata
    transform = from_origin(0, height, 1, 1)  # 1 m px, north-up
    profile = dict(
        driver='ENVI', dtype='float32', count=n_bands,
        height=height, width=width, transform=transform, crs='+proj=eqc +R=3396190',
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr)


def test_loader_returns_interior_pixels(tmp_path):
    tile_dir = tmp_path
    img = tile_dir / 't0001_mrral_x.img'
    _write_synthetic_mrral(str(img))
    # Polygon covering (row 5-9, col 5-9) → 5x5 = 25 interior pixels
    geom = Polygon([(5, 11), (10, 11), (10, 16), (5, 16)])
    bundle = load_polygon_pixels(geometry=geom, tile_id='t0001', mrral_dir=str(tile_dir))
    assert isinstance(bundle, PixelBundle)
    assert bundle.spectra.shape == (25, 59)
    assert bundle.rows.shape == (25,)
    assert bundle.cols.shape == (25,)
    # Spectra are uniform across pixels (synthetic) → std == 0
    assert np.allclose(bundle.std, 0.0, atol=1e-6)
    # Mean matches the per-band fill rule
    assert bundle.mean[0] == pytest.approx(0.1, abs=1e-6)
    assert bundle.mean[58] == pytest.approx(0.1 + 0.01 * 58, abs=1e-6)


def test_loader_masks_nodata(tmp_path):
    tile_dir = tmp_path
    img = tile_dir / 't0001_mrral_x.img'
    _write_synthetic_mrral(str(img))
    # Polygon covering (row 0-4, col 0-4) — row 0 and col 0 are NODATA
    geom = Polygon([(0, 15), (5, 15), (5, 20), (0, 20)])
    bundle = load_polygon_pixels(geometry=geom, tile_id='t0001', mrral_dir=str(tile_dir))
    # 5x5=25 raw pixels, but row 0 (5 px) + col 0 in remaining rows (4 px) = 9 NODATA → 16 left
    assert bundle.spectra.shape[0] == 16


def test_loader_returns_empty_for_polygon_outside_tile(tmp_path):
    tile_dir = tmp_path
    img = tile_dir / 't0001_mrral_x.img'
    _write_synthetic_mrral(str(img))
    geom = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    bundle = load_polygon_pixels(geometry=geom, tile_id='t0001', mrral_dir=str(tile_dir))
    assert bundle.spectra.shape == (0, 59)


def test_loader_raises_if_tile_missing(tmp_path):
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(FileNotFoundError):
        load_polygon_pixels(geometry=geom, tile_id='t9999', mrral_dir=str(tmp_path))
