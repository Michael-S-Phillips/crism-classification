"""Unit tests for sam_analysis.sam — pure-numpy SAM angle computation."""
from __future__ import annotations

import numpy as np
import pytest

from sam_analysis.sam import sam_raster, spectral_angle


def test_identical_vector_zero_angle():
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    assert spectral_angle(v, v) == pytest.approx(0.0, abs=1e-10)


def test_orthogonal_pi_over_two():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert spectral_angle(a, b) == pytest.approx(np.pi / 2, abs=1e-10)


def test_anti_parallel_pi():
    a = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    b = -a
    assert spectral_angle(a, b) == pytest.approx(np.pi, abs=1e-10)


def test_all_nan_target_returns_nan():
    t = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    ref = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    out = spectral_angle(t, ref)
    assert np.isnan(out)


def test_zero_norm_returns_nan():
    t = np.zeros(5, dtype=np.float64)
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    assert np.isnan(spectral_angle(t, ref))


def test_batched_matches_scalar():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.02, 0.4, size=59).astype(np.float64)
    pixels = rng.uniform(0.02, 0.4, size=(5, 59)).astype(np.float64)
    batched = spectral_angle(pixels, ref)
    scalar = np.array([spectral_angle(p, ref) for p in pixels])
    np.testing.assert_allclose(batched, scalar, atol=1e-12)


def test_pairwise_nan_masking():
    """Bands that are NaN in either target or ref must be ignored on BOTH sides."""
    t = np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float64)
    ref = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    # Should equal angle between [0.1, 0.3, 0.4] and [0.1, 0.3, 0.4] = 0
    angle = spectral_angle(t, ref)
    assert angle == pytest.approx(0.0, abs=1e-10)


def test_sam_raster_shape_and_nan_propagation():
    rng = np.random.default_rng(1)
    cube = rng.uniform(0.02, 0.4, size=(4, 5, 10)).astype(np.float32)
    ref = rng.uniform(0.02, 0.4, size=10).astype(np.float64)
    # Knock out one pixel entirely.
    cube[2, 3, :] = np.nan
    angles = sam_raster(cube, ref)
    assert angles.shape == (4, 5)
    assert angles.dtype == np.float32
    assert np.isnan(angles[2, 3])
    assert np.isfinite(angles[0, 0])


def test_sam_raster_self_reference_zero():
    """A pixel that equals the reference should get a zero angle."""
    rng = np.random.default_rng(2)
    ref = rng.uniform(0.02, 0.4, size=8).astype(np.float64)
    cube = np.broadcast_to(ref.astype(np.float32), (3, 3, 8)).copy()
    out = sam_raster(cube, ref)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_band_count_mismatch_raises():
    t = np.zeros((4, 10))
    ref = np.zeros(11)
    with pytest.raises(ValueError):
        spectral_angle(t, ref)


def test_1d_returns_scalar():
    t = np.array([0.1, 0.2, 0.3])
    ref = np.array([0.1, 0.2, 0.3])
    out = spectral_angle(t, ref)
    assert isinstance(out, float)
    assert out == pytest.approx(0.0, abs=1e-12)
