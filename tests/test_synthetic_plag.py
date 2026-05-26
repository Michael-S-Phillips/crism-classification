# tests/test_synthetic_plag.py
import numpy as np
import pytest

from data.synthetic_plag import interp_to_mrral_wavelengths


def test_interp_basic_linear():
    # library wavelengths 400..500, reflectance = wl/1000 (so 0.4..0.5)
    lib_wl = np.array([400.0, 450.0, 500.0])
    lib_refl = np.array([0.40, 0.45, 0.50])
    target_wl = np.array([425.0, 475.0])
    out = interp_to_mrral_wavelengths(lib_wl, lib_refl, target_wl)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [0.425, 0.475], atol=1e-6)


def test_interp_drops_sentinel_and_nan():
    # 65535 wavelength sentinel and NaN reflectance bands must be ignored
    lib_wl = np.array([400.0, 450.0, 65535.0, 500.0])
    lib_refl = np.array([0.40, np.nan, 0.99, 0.50])
    target_wl = np.array([450.0])
    out = interp_to_mrral_wavelengths(lib_wl, lib_refl, target_wl)
    # only (400,0.40) and (500,0.50) are valid → interp at 450 = 0.45
    np.testing.assert_allclose(out, [0.45], atol=1e-6)


from data.synthetic_plag import synthesize_patches


def test_synthesize_patches_shape_and_clip():
    rng = np.random.default_rng(0)
    spectrum = np.full(59, 0.2, dtype=np.float32)
    patches = synthesize_patches(spectrum, n_aug=8, rng=rng)
    assert patches.shape == (8, 7, 7, 59)
    assert patches.dtype == np.float32
    assert patches.min() >= 0.0 and patches.max() <= 0.5  # clipped to [0, CLIP_MAX]


def test_synthesize_patches_not_flat():
    # per-pixel noise must make neighbours differ (no flat-tile shortcut)
    rng = np.random.default_rng(1)
    spectrum = np.full(59, 0.2, dtype=np.float32)
    patches = synthesize_patches(spectrum, n_aug=4, rng=rng)
    # within a single patch, the 49 center-band values should not be identical
    band0 = patches[0, :, :, 0].ravel()
    assert band0.std() > 1e-4


def test_synthesize_patches_centered_on_spectrum():
    # mean over many augmentations/pixels should track the source spectrum
    rng = np.random.default_rng(2)
    spectrum = np.linspace(0.1, 0.4, 59).astype(np.float32)
    patches = synthesize_patches(spectrum, n_aug=200, rng=rng,
                                 noise_sigma=0.005, jitter_sigma=0.003,
                                 continuum_scale_range=(0.97, 1.03))
    mean_spec = patches.mean(axis=(0, 1, 2))
    np.testing.assert_allclose(mean_spec, spectrum, atol=0.02)
