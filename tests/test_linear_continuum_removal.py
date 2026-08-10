"""Tests for linear continuum removal.

Linear CR divides by a per-spectrum least-squares line over the good bands. It
removes level and slope -- the albedo shortcut, which spans 1.76x across classes
and is why a raw-fed model generalises badly -- but CANNOT remove curvature,
because a line has no curvature. That is the whole point: upper-hull CR destroys
alteration's 1-2um convex arch (41% retained) because a broad convex arch IS
approximately the hull.
"""
import numpy as np
import pytest

from data.continuum_removal import (
    linear_continuum_removed, good_band_mask_59, WAVELENGTHS_59)

W = WAVELENGTHS_59
G = good_band_mask_59()


def _line(level, slope):
    """A pure straight line in reflectance: no curvature to preserve."""
    x = (W - W.min()) / (W.max() - W.min())
    return (level + slope * x).astype(np.float32)


def _arch(y, amp):
    """A line plus a convex bump peaking mid-spectrum (alteration-like)."""
    x = (W - W.min()) / (W.max() - W.min())
    return (y + amp * np.sin(np.pi * x)).astype(np.float32)


def test_a_pure_line_flattens_to_one():
    """Level and slope are exactly what linear CR must remove."""
    out = linear_continuum_removed(_line(0.20, 0.10))
    assert np.allclose(out[G], 1.0, atol=1e-4)


def test_level_invariance():
    """Two spectra differing ONLY in brightness must map to the same output."""
    a = linear_continuum_removed(_arch(_line(0.10, 0.05), 0.02))
    b = linear_continuum_removed(_arch(_line(0.30, 0.15), 0.06))  # 3x brighter
    np.testing.assert_allclose(a[G], b[G], atol=1e-3)


def test_convex_arch_survives():
    """The feature hull-CR destroys must be preserved with the right SIGN."""
    out = linear_continuum_removed(_arch(_line(0.20, 0.0), 0.03))
    mid = int(np.argmin(np.abs(W - 1600)))
    assert out[mid] > 1.02, 'a convex arch must sit ABOVE the linear continuum'


def test_absorption_goes_below_one():
    """Concave (absorption) features must land below 1.0 -- opposite sign to an
    arch. That signed contrast is what separates alteration from bland."""
    y = _line(0.20, 0.0).copy()
    lo = int(np.argmin(np.abs(W - 1900)))
    y[lo - 2:lo + 3] *= 0.85
    out = linear_continuum_removed(y)
    assert out[lo] < 0.98


def test_excluded_bands_are_one():
    """Same convention as hull CR: the 1021-1056nm overlap window is not data."""
    out = linear_continuum_removed(_arch(_line(0.2, 0.05), 0.02))
    assert np.allclose(out[~G], 1.0)


def test_clipped_to_range():
    out = linear_continuum_removed(_line(1e-7, 0.0))
    assert np.all(out >= 0.0) and np.all(out <= 2.0)


def test_batch_shape_and_nan_safety():
    rng = np.random.default_rng(0)
    batch = rng.uniform(0.05, 0.35, size=(4, 7, 7, 59)).astype(np.float32)
    out = linear_continuum_removed(batch)
    assert out.shape == batch.shape
    assert np.isfinite(out).all()

    degenerate = np.zeros(59, dtype=np.float32)
    assert np.allclose(linear_continuum_removed(degenerate), 1.0)


def test_wrong_band_count_raises():
    with pytest.raises(ValueError, match='59'):
        linear_continuum_removed(np.zeros(40, dtype=np.float32))


def test_batched_call_matches_per_row_calls():
    """A batched call must equal calling the function on each row separately.

    Guards against a batch-shared continuum fit, which would silently break
    level invariance: np.linalg.lstsq with a multi-column RHS solves each column
    independently, but a refactor that reduced y across rows before fitting
    would pass every other test in this file.
    """
    rows = np.stack([
        _line(0.10, 0.05),                      # dim, gentle positive slope
        _arch(_line(0.30, -0.10), 0.04),        # 3x brighter, NEGATIVE slope, arch
        _arch(_line(0.15, 0.0), -0.03),         # concave (negative amplitude)
        _line(0.22, 0.0),                       # flat line
    ])
    batched = linear_continuum_removed(rows)
    for i in range(len(rows)):
        np.testing.assert_allclose(
            batched[i], linear_continuum_removed(rows[i]), rtol=0, atol=1e-6,
            err_msg=f'row {i} differs between batched and single-row call — '
                    f'the continuum fit is being shared across rows')
