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
    linear_continuum_removed, good_band_mask_59, WAVELENGTHS_59, LIN_CR_CLIP)

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
    """The brief's original version of this test fed _line(1e-7, 0.0): a flat
    spectrum whose max-abs (1e-7) falls below the degeneracy threshold at
    continuum_removal.py:130 (`> 1e-6`), so it hits the all-ones early-out and
    the clip is never reached -- deleting LIN_CR_CLIP entirely left this test
    (and all 51 others) passing. This spectrum instead clears the degeneracy
    threshold (max-abs 0.30 >> 1e-6) but is shaped so the fitted least-squares
    line goes through zero near the high-wavelength edge while a handful of
    real bands there are pulled up well above the line -- exactly the
    situation LIN_CR_CLIP exists to bound. Unclipped, this specific
    construction reaches ~2.56 (verified against _linear_continuum before the
    clip is applied), so the clip at 2.0 genuinely engages here."""
    y = _line(0.30, -0.29)
    last5 = np.where(G)[0][-5:]
    y = y.copy()
    y[last5] = 0.05
    out = linear_continuum_removed(y)
    assert np.all(out >= LIN_CR_CLIP[0]) and np.all(out <= LIN_CR_CLIP[1])
    # The clip must actually have engaged (not just happen to be satisfied):
    # at least one of the bumped bands should be pinned at the upper bound.
    assert np.any(np.isclose(out[last5], LIN_CR_CLIP[1])), (
        'expected at least one value pinned at the upper clip bound; '
        f'got {out[last5]}')


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
