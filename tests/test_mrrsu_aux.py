# tests/test_mrrsu_aux.py
import numpy as np

from data.mrrsu_aux import mean_pool_nodata


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
