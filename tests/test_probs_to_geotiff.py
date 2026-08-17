"""probs .npz -> GeoTIFF: georeferencing and masking must survive the round trip.

Georeferencing errors here are silent and expensive: a raster that lands in the
wrong place still opens, still looks plausible, and misleads whoever is reviewing
polygons against it. So these tests assert the transform, CRS, band names, and
nodata all come back, by reading the file rasterio wrote.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio

from scripts.probs_to_geotiff import convert, load_probs

CLASSES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
# Mars 2000 geographic, as the classifier writes it.
WKT = ('GEOGCS["GCS_Mars_2000",DATUM["D_Mars_2000",'
       'SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
       'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]]')
# Affine constructor order (a, b, c, d, e, f) = x_scale, x_shear, x_origin,
# y_shear, y_scale, y_origin -- this is what the npz stores, because
# classify_tile_supervised does Affine(*data['transform']). It is NOT gdal order,
# which is (c, a, b, f, d, e); conflating the two rotates a raster onto the wrong
# hemisphere while still opening cleanly, so the test below pins a corner
# coordinate rather than only comparing tuples.
TRANSFORM = (0.001, 0.0, 72.5, 0.0, -0.001, 25.0)


def _npz(tmp_path, h=6, w=5, valid=None, probs=None):
    rng = np.random.default_rng(0)
    if probs is None:
        probs = rng.random((h, w, len(CLASSES))).astype(np.float32)
    if valid is None:
        valid = np.ones((h, w), dtype=bool)
        valid[0, 0] = False
    p = tmp_path / 't9999_probs.npz'
    np.savez(p, probs=probs, valid_mask=valid,
             transform=np.array(TRANSFORM), crs_wkt=WKT,
             class_names=np.array(CLASSES, dtype=object))
    return str(p)


def test_band_names_reach_the_file(tmp_path):
    """QGIS shows band descriptions in the layer panel; without them a 7-band
    float raster is unreadable ('Band 1'..'Band 7')."""
    out = str(tmp_path / 'o.tif')
    convert(_npz(tmp_path), out)
    with rasterio.open(out) as ds:
        assert list(ds.descriptions) == CLASSES


def test_transform_and_crs_survive(tmp_path):
    out = str(tmp_path / 'o.tif')
    convert(_npz(tmp_path, h=6, w=5), out)
    with rasterio.open(out) as ds:
        assert tuple(ds.transform)[:6] == pytest.approx(TRANSFORM)
        assert 'Mars_2000' in ds.crs.to_wkt()
        assert ds.count == len(CLASSES)
        # Pin real coordinates: origin at 72.5E/25.0N, 0.001 deg/px, y counting
        # DOWN. A transposed or gdal-ordered transform fails here even when the
        # six numbers all appear somewhere in the tuple.
        assert ds.xy(0, 0) == pytest.approx((72.5005, 24.9995))
        assert ds.xy(5, 4) == pytest.approx((72.5045, 24.9945))


def test_invalid_pixels_become_nan_not_zero(tmp_path):
    """0.0 would paint a black frame AND drag the display stretch down. The
    distinction is invisible in a summary but obvious on screen."""
    probs = np.full((4, 4, len(CLASSES)), 0.7, dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    valid[2, 3] = False
    out = str(tmp_path / 'o.tif')
    convert(_npz(tmp_path, 4, 4, valid=valid, probs=probs), out)
    with rasterio.open(out) as ds:
        a = ds.read(1)
        assert np.isnan(a[2, 3])
        assert a[0, 0] == pytest.approx(0.7)
        assert np.isnan(ds.nodata)


def test_probability_values_are_not_rescaled(tmp_path):
    """Byte or scaled output would destroy the >=0.999 rungs the ladder relies on."""
    probs = np.zeros((3, 3, len(CLASSES)), dtype=np.float32)
    probs[1, 1, 1] = 0.99947
    out = str(tmp_path / 'o.tif')
    convert(_npz(tmp_path, 3, 3, valid=np.ones((3, 3), bool), probs=probs), out)
    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == 'float32'
        assert ds.read(2)[1, 1] == pytest.approx(0.99947, abs=1e-6)


def test_split_writes_one_named_file_per_class(tmp_path):
    out = str(tmp_path / 'o.tif')
    written = convert(_npz(tmp_path), out, split=True)
    assert len(written) == 1 + len(CLASSES)
    for nm in CLASSES:
        p = str(tmp_path / f'o_{nm}.tif')
        assert p in written
        with rasterio.open(p) as ds:
            assert ds.count == 1
            assert ds.descriptions[0] == nm


def test_channel_count_mismatch_is_rejected(tmp_path):
    """A probs cube whose channels do not match class_names means the vocab is
    wrong; naming bands from it would mislabel every layer."""
    p = tmp_path / 'bad_probs.npz'
    np.savez(p, probs=np.zeros((3, 3, 4), dtype=np.float32),
             valid_mask=np.ones((3, 3), bool), transform=np.array(TRANSFORM),
             crs_wkt=WKT, class_names=np.array(CLASSES, dtype=object))
    with pytest.raises(ValueError, match='class names'):
        load_probs(str(p))
