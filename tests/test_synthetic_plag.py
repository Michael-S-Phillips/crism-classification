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
