"""Tests for data.continuum_removal — upper-hull CR over the 59-band good-band window."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import (  # noqa: E402
    good_band_mask_59,
    continuum_removed,
    brightness_scalar,
    cr_patch,
    WAVELENGTHS_59,
)


def test_good_band_mask_excludes_only_1um_overlap():
    m = good_band_mask_59()
    assert m.shape == (59,)
    assert m.dtype == bool
    # exactly the four 1 µm detector-overlap bands (1021-1056 nm) are excluded
    assert list(np.where(~m)[0]) == [16, 17, 18, 19]
    assert m.sum() == 55


def test_wavelengths_loaded():
    assert len(WAVELENGTHS_59) == 59
    assert abs(WAVELENGTHS_59[0] - 410.1) < 0.5
    assert abs(WAVELENGTHS_59[-1] - 2456.8) < 0.5


def test_cr_recovers_a_planted_absorption():
    # Bright sloped continuum with a Gaussian absorption near 1900 nm (Band II).
    wl = np.asarray(WAVELENGTHS_59)
    cont = 0.20 + 0.00002 * (wl - wl[0])            # gentle positive slope
    band = 0.06 * np.exp(-0.5 * ((wl - 1900) / 120) ** 2)
    spec = (cont - band).astype(np.float32)
    cr = continuum_removed(spec)
    # deepest point of CR should sit at the planted band center, depth ~0.06/cont
    depth = 1.0 - cr.min()
    assert 0.15 < depth < 0.45                       # 0.06 on a ~0.2 continuum
    center = wl[np.argmin(cr)]
    assert 1750 < center < 2050


def test_cr_never_exceeds_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        spec = (0.05 + 0.3 * rng.random(59)).astype(np.float32)
        cr = continuum_removed(spec)
        assert np.nanmax(cr) <= 1.0001
        assert np.all(np.isfinite(cr))


def test_flat_spectrum_is_all_ones():
    spec = np.full(59, 0.15, dtype=np.float32)
    cr = continuum_removed(spec)
    assert np.allclose(cr, 1.0, atol=1e-4)


def test_zero_nodata_pixel_is_safe():
    spec = np.zeros(59, dtype=np.float32)            # NODATA→0 pixel
    cr = continuum_removed(spec)
    assert np.all(np.isfinite(cr))
    assert np.allclose(cr, 1.0)


def test_excluded_bands_are_one():
    wl = np.asarray(WAVELENGTHS_59)
    cont = 0.20 + 0.00002 * (wl - wl[0])
    cr = continuum_removed(cont.astype(np.float32))
    assert np.allclose(cr[16:20], 1.0, atol=1e-4)


def test_brightness_scalar_is_good_band_mean():
    spec = (0.05 + 0.3 * np.random.default_rng(1).random(59)).astype(np.float32)
    m = good_band_mask_59()
    assert abs(float(brightness_scalar(spec)) - float(spec[m].mean())) < 1e-5


def test_cr_patch_shapes_and_consistency():
    rng = np.random.default_rng(2)
    patch = (0.05 + 0.3 * rng.random((7, 7, 59))).astype(np.float32)
    cr, bright = cr_patch(patch)
    assert cr.shape == (7, 7, 59)
    assert bright.shape == (7, 7)
    # center pixel CR matches the 1-D transform
    assert np.allclose(cr[3, 3], continuum_removed(patch[3, 3]), atol=1e-5)
    assert abs(float(bright[3, 3]) - float(brightness_scalar(patch[3, 3]))) < 1e-5


def test_batched_input():
    rng = np.random.default_rng(3)
    specs = (0.05 + 0.3 * rng.random((10, 59))).astype(np.float32)
    cr = continuum_removed(specs)
    assert cr.shape == (10, 59)
    assert np.allclose(cr[0], continuum_removed(specs[0]), atol=1e-5)
