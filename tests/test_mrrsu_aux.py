# tests/test_mrrsu_aux.py
import numpy as np
import pytest

from data.mrrsu_aux import (
    BAND_VALID_RANGES,
    NODATA,
    apply_invalid_to_nan,
    mean_pool_nodata,
    physically_valid_mask,
)


def test_mean_excludes_nodata():
    # 3x3 all ones except one NODATA; 3x3 window mean at center = mean of 8 ones = 1.0
    r = np.ones((3, 3), dtype=np.float32)
    r[0, 0] = 65535.0
    out = mean_pool_nodata(r, patch_size=3, nodata=65535.0)
    assert abs(float(out[1, 1]) - 1.0) < 1e-6


def test_all_nodata_window_is_nan():
    r = np.full((3, 3), 65535.0, dtype=np.float32)
    out = mean_pool_nodata(r, patch_size=3, nodata=65535.0)
    assert np.isnan(out[1, 1])


def test_uniform_value_preserved():
    r = np.full((9, 9), 0.73, dtype=np.float32)
    out = mean_pool_nodata(r, patch_size=7, nodata=65535.0)
    assert np.allclose(out[4, 4], 0.73, atol=1e-6)


def test_physically_valid_mask_rpeak1_rejects_outliers():
    lo, hi = BAND_VALID_RANGES["RPEAK1"]
    arr = np.array(
        [
            lo - 0.01,       # below range -> invalid
            lo,              # boundary -> valid (inclusive)
            (lo + hi) / 2,   # mid-range -> valid
            hi,              # boundary -> valid
            hi + 0.01,       # above range -> invalid
            NODATA,          # sentinel -> invalid
            np.nan,          # not finite -> invalid
            np.inf,          # not finite -> invalid
            -np.inf,         # not finite -> invalid
        ],
        dtype=np.float32,
    )
    mask = physically_valid_mask(arr, "RPEAK1")
    expected = np.array([False, True, True, True, False, False, False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_physically_valid_mask_bd1300_allows_negative_in_range():
    lo, hi = BAND_VALID_RANGES["BD1300"]
    assert lo < 0 < hi, "BD1300 range should straddle zero (band depth)"
    arr = np.array([lo - 1e-3, lo, 0.0, hi, hi + 1e-3, NODATA, np.nan],
                   dtype=np.float32)
    mask = physically_valid_mask(arr, "BD1300")
    expected = np.array([False, True, True, True, False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_physically_valid_mask_unknown_band_raises():
    with pytest.raises(KeyError):
        physically_valid_mask(np.zeros(3, dtype=np.float32), "BOGUS")


def test_apply_invalid_to_nan_preserves_valid_and_nans_rest():
    arr = np.array([NODATA, 0.75, 0.4, np.inf, 0.85], dtype=np.float32)
    out = apply_invalid_to_nan(arr, "RPEAK1")
    # Valid entries preserved
    assert out[1] == np.float32(0.75)
    assert out[4] == np.float32(0.85)
    # Invalid entries (sentinel, below range, non-finite) -> NaN
    assert np.isnan(out[0])
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    # Original array not mutated
    assert arr[0] == NODATA
    assert arr[3] == np.inf


def test_apply_invalid_to_nan_returns_float32():
    arr = np.array([65535, 100, 50], dtype=np.int32)
    out = apply_invalid_to_nan(arr, "RPEAK1")  # all out of [0.5, 1.0] anyway
    assert out.dtype == np.float32
    assert np.isnan(out).all()
