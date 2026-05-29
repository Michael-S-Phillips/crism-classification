# tests/test_build_mrrsu_aux.py
"""Tests for ``scripts/build_mrrsu_aux.py``: focuses on the pure-function
``compute_stats`` for each norm mode (the file-IO main() is exercised via the
local CPU smoke step, not in unit tests)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _train_arr_with_some_invalid():
    """Synth (n, 2) RPEAK1/BD1300 with a mix of valid + invalid rows.

    Invalid rows mix sentinels, NaN, and out-of-range values; the build
    script should ignore all of them when computing stats.
    """
    valid = np.array([
        [0.70, -0.02],
        [0.72,  0.00],
        [0.74,  0.01],
        [0.76,  0.02],
        [0.78,  0.04],
        [0.80, -0.01],
    ], dtype=np.float32)
    invalid = np.array([
        [65535.0, 0.01],         # sentinel
        [0.75, 65535.0],         # sentinel
        [np.nan, 0.0],           # nan
        [0.4, 0.0],              # RPEAK1 below physical range (< 0.5)
        [1.1, 0.0],              # RPEAK1 above physical range (> 1.0)
        [0.75, 0.7],             # BD1300 above physical range (> 0.5)
    ], dtype=np.float32)
    return valid, invalid, np.concatenate([valid, invalid], axis=0)


def test_compute_stats_zscore():
    from scripts.build_mrrsu_aux import compute_stats

    valid, _, full = _train_arr_with_some_invalid()
    st = compute_stats(full, "zscore", min_valid_per_tile=1000)

    assert st["mode"] == "zscore"
    assert st["version"] == 2
    assert "physical_ranges" in st and st["physical_ranges"]["RPEAK1"] == [0.5, 1.0]
    assert st["band_order"] == ["RPEAK1", "BD1300"]
    assert st["n_valid_train_rows"] == len(valid)
    assert st["n_train_rows"] == len(full)
    # mean / std computed on the valid subset only
    np.testing.assert_allclose(st["mean"], valid.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        np.array(st["std"]) - 1e-8, valid.std(axis=0), atol=1e-6
    )


def test_compute_stats_minmax():
    from scripts.build_mrrsu_aux import compute_stats

    valid, _, full = _train_arr_with_some_invalid()
    st = compute_stats(full, "minmax", min_valid_per_tile=1000)

    assert st["mode"] == "minmax"
    assert st["version"] == 2
    # min / max computed on the valid subset only
    np.testing.assert_allclose(st["min"], valid.min(axis=0), atol=1e-6)
    np.testing.assert_allclose(st["max"], valid.max(axis=0), atol=1e-6)


def test_compute_stats_pertile_zscore():
    from scripts.build_mrrsu_aux import compute_stats

    valid, _, full = _train_arr_with_some_invalid()
    st = compute_stats(full, "pertile_zscore", min_valid_per_tile=42)

    assert st["mode"] == "pertile_zscore"
    assert st["version"] == 2
    # No global min/max/mean/std exposed -- only the fallback fields and threshold.
    assert "fallback_mean" in st and "fallback_std" in st
    assert st["min_valid_per_tile"] == 42
    np.testing.assert_allclose(st["fallback_mean"], valid.mean(axis=0), atol=1e-6)


def test_compute_stats_empty_raises():
    from scripts.build_mrrsu_aux import compute_stats

    # All invalid: every row is sentinel / out-of-range
    arr = np.array([
        [65535.0, 0.0],
        [0.4, 0.0],
        [np.nan, np.nan],
    ], dtype=np.float32)
    with pytest.raises(ValueError, match="no physically-valid rows"):
        compute_stats(arr, "zscore", min_valid_per_tile=1000)


def test_compute_stats_unknown_mode_raises():
    from scripts.build_mrrsu_aux import compute_stats

    valid, _, full = _train_arr_with_some_invalid()
    with pytest.raises(ValueError, match="unknown mode"):
        compute_stats(full, "robust_iqr", min_valid_per_tile=1000)


def test_build_split_filters_invalid_pixels(tmp_path, monkeypatch):
    """End-to-end mini test for ``build_split``: synthesizes a fake mrrsu tile
    where one corner has a clean RPEAK1 + BD1300 patch and the rest is NODATA
    (sentinel). The 7x7 mean of the center of the valid patch should propagate
    correctly; rows outside the valid patch should be NaN."""
    import pandas as pd
    import rasterio as _rasterio
    from scripts.build_mrrsu_aux import build_split

    H, W = 21, 21
    n_bands = 20  # at least up to band index 18 (BD1300, 1-indexed)
    rpeak = np.full((H, W), 65535.0, dtype=np.float32)
    bd = np.full((H, W), 65535.0, dtype=np.float32)
    rpeak[5:16, 5:16] = 0.78
    bd[5:16, 5:16] = 0.01

    class _Src:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self, band_index):
            if band_index == 9:
                return rpeak
            if band_index == 18:
                return bd
            raise ValueError(band_index)

    monkeypatch.setattr(_rasterio, "open", lambda *a, **k: _Src())

    # Two rows: one at the center of the valid region (10, 10), one at (1, 1) which is
    # surrounded by NODATA.
    df = pd.DataFrame({
        "tile_id": ["tFAKE", "tFAKE"],
        "pixel_row": [10, 1],
        "pixel_col": [10, 1],
    })
    mrrsu_map = {"tFAKE": "fake.img"}
    out = build_split(df, mrrsu_map, patch_size=7)
    assert out.shape == (2, 2)
    # Row 0 (interior of valid patch): pool is exactly 0.78 / 0.01
    np.testing.assert_allclose(out[0], [0.78, 0.01], atol=1e-5)
    # Row 1 (in NODATA region): all NaN
    assert np.isnan(out[1]).all()
