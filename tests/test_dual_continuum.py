"""Tests for dual-channel (hull-CR + linear-CR) assembly.

Channel order is load-bearing: 0-58 hull, 59-117 linear. Every producer and
consumer in the pipeline assumes it, so it is locked by test.
"""
import numpy as np
import pytest

import os

from data.continuum_removal import (
    dual_continuum, continuum_removed, linear_continuum_removed,
    CR_SCALES, N_BANDS)

# The variance invariant only holds on real spectra (see that test's docstring).
SPECTRA_NPZ = os.environ.get('CRISM_SPECTRA_NPZ', '')


def _spec(seed=0, n=32):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.35, size=(n, N_BANDS)).astype(np.float32)


def test_shape_is_118_channels():
    out = dual_continuum(_spec())
    assert out.shape == (32, 2 * N_BANDS)


def test_channel_order_is_hull_then_linear():
    """If this flips, the caches and the encoder silently disagree."""
    s = _spec()
    out = dual_continuum(s, standardize=False)
    np.testing.assert_allclose(out[:, :N_BANDS], continuum_removed(s), atol=1e-6)
    np.testing.assert_allclose(out[:, N_BANDS:], linear_continuum_removed(s),
                               atol=1e-6)


@pytest.mark.skipif(not os.path.exists(SPECTRA_NPZ),
                    reason='needs the sampled-spectra npz; run '
                           'scripts/sample_class_spectra.py')
def test_standardisation_equalises_channel_variance():
    """The 2.45x variance ratio is what would skew a pooled MAE loss.

    Asserted on the REAL sampled spectra, not synthetic noise: CR_SCALES is
    computed from real data, so only real data is guaranteed to standardise to
    ~1.0. Synthetic uniform noise has different hull/linear stds and a correct
    implementation could fail such a test.
    """
    d = np.load(SPECTRA_NPZ)
    keys = [k for k in d.files if k not in ('wav', 'good')]
    raw = np.concatenate([d[k] for k in keys]).astype(np.float32)
    raw[(raw > 1.0) | (raw == 65535) | (~np.isfinite(raw))] = np.nan
    raw = np.clip(raw, 0.0, 0.5)
    raw = raw[np.isfinite(raw).all(axis=1)]

    out = dual_continuum(raw, standardize=True)
    ratio = out[:, :N_BANDS].std() / out[:, N_BANDS:].std()
    assert 0.8 < ratio < 1.25, (
        f'channels still differ by {ratio:.2f}x after standardisation; '
        f'CR_SCALES may be stale relative to the transform definition')


def test_standardize_divides_each_block_by_its_own_constant():
    """standardize=True must divide hull by hull_std and linear by linear_std.

    A swap would amplify the 2.45x imbalance this task exists to remove, and
    every other test in this file would still pass -- the variance test skips
    without CRISM_SPECTRA_NPZ, and the ordering test uses standardize=False.
    This runs on synthetic data so it executes on every CI run.
    """
    s = _spec(n=16)
    plain = dual_continuum(s, standardize=False)
    scaled = dual_continuum(s, standardize=True)

    np.testing.assert_allclose(scaled[:, :N_BANDS],
                               plain[:, :N_BANDS] / CR_SCALES['hull_std'],
                               rtol=1e-5, atol=0)
    np.testing.assert_allclose(scaled[:, N_BANDS:],
                               plain[:, N_BANDS:] / CR_SCALES['linear_std'],
                               rtol=1e-5, atol=0)

    # And prove the constants are distinct enough that a swap is detectable --
    # if they were equal the assertions above would not constrain the mapping.
    assert abs(CR_SCALES['hull_std'] - CR_SCALES['linear_std']) > 0.05


def test_scales_are_loaded_not_hardcoded():
    assert set(CR_SCALES) >= {'hull_std', 'linear_std'}
    assert CR_SCALES['hull_std'] > 0 and CR_SCALES['linear_std'] > 0
    # Provenance must travel with the numbers.
    assert 'source' in CR_SCALES


def test_preserves_patch_dims():
    rng = np.random.default_rng(1)
    patch = rng.uniform(0.05, 0.35, size=(5, 7, 7, N_BANDS)).astype(np.float32)
    out = dual_continuum(patch)
    assert out.shape == (5, 7, 7, 2 * N_BANDS)


def test_nan_safe():
    s = _spec()
    s[0, 5] = np.nan
    out = dual_continuum(s)
    assert np.isfinite(out).all()
