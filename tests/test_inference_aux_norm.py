# tests/test_inference_aux_norm.py
"""Tests for ``scripts/classify_tile_supervised.load_mrrsu_aux_rasters``.

These cover the three normalization modes (zscore / minmax / pertile_zscore)
including the per-tile fallback path when valid-pixel count is below threshold.
The mrrsu raster is faked via ``monkeypatch`` of ``rasterio.open`` to keep these
tests fully offline and CPU-only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


# Ensure project root + scripts/ are on sys.path so we can import the
# classify_tile_supervised module directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _make_rpeak_bd_raster(H=21, W=21):
    """Return (rpeak_band, bd_band) numpy arrays for a fake mrrsu tile.

    All entries are physically plausible (RPEAK1 in [0.7, 0.85], BD1300 small).
    """
    rng = np.random.default_rng(0)
    rpeak = rng.uniform(0.72, 0.83, size=(H, W)).astype(np.float32)
    bd = rng.uniform(-0.02, 0.02, size=(H, W)).astype(np.float32)
    return rpeak, bd


class _FakeRasterio:
    """Context-manager + .read() shim mimicking the rasterio API used here."""
    def __init__(self, rpeak, bd):
        self._rpeak = rpeak
        self._bd = bd

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, band_index):
        # The code reads RPEAK1_BAND+1 (= 9) then BD1300_BAND+1 (= 18).
        if band_index == 9:
            return self._rpeak
        if band_index == 18:
            return self._bd
        raise ValueError(f"unexpected band {band_index}")


def _patch_rasterio(monkeypatch, rpeak, bd):
    import rasterio
    monkeypatch.setattr(rasterio, "open",
                        lambda *a, **k: _FakeRasterio(rpeak, bd))


def _write_stats(tmp_path, mode, **fields):
    base = {
        "version": 2,
        "mode": mode,
        "physical_ranges": {"RPEAK1": [0.5, 1.0], "BD1300": [-0.5, 0.5]},
        "band_order": ["RPEAK1", "BD1300"],
    }
    base.update(fields)
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(base))
    return str(p)


def test_zscore_mode(monkeypatch, tmp_path):
    from scripts.classify_tile_supervised import load_mrrsu_aux_rasters
    rpeak, bd = _make_rpeak_bd_raster()
    _patch_rasterio(monkeypatch, rpeak, bd)
    stats = _write_stats(tmp_path, "zscore", mean=[0.775, 0.0], std=[0.05, 0.02])
    out = load_mrrsu_aux_rasters("ignored.img", stats)
    assert out.shape == (rpeak.shape[0], rpeak.shape[1], 2)
    assert out.dtype == np.float32
    # The center pixel (away from the border so the 7x7 pool is fully populated)
    # should have nonzero z value -- not exactly the train mean (== 0).
    center = out[10, 10]
    assert np.isfinite(center).all()
    # Values are bounded: 7x7 mean of values in [0.72, 0.83] lies in [0.72, 0.83]
    # so z = (mean - 0.775)/0.05 lies in [-1.1, 1.1].
    assert -2.0 < float(center[0]) < 2.0
    assert -2.0 < float(center[1]) < 2.0


def test_minmax_mode_in_unit_interval(monkeypatch, tmp_path):
    from scripts.classify_tile_supervised import load_mrrsu_aux_rasters
    rpeak, bd = _make_rpeak_bd_raster()
    _patch_rasterio(monkeypatch, rpeak, bd)
    stats = _write_stats(tmp_path, "minmax", min=[0.7, -0.03], max=[0.85, 0.03])
    out = load_mrrsu_aux_rasters("ignored.img", stats)
    # All entries lie in [0, 1] (clipped if outside the train range).
    assert (out >= 0.0).all() and (out <= 1.0).all()
    # And the center 7x7-mean pixel is strictly inside (not clipped).
    center = out[10, 10]
    assert 0.0 < float(center[0]) < 1.0


def test_pertile_zscore_above_threshold(monkeypatch, tmp_path):
    """If the tile has plenty of valid pixels, the per-tile branch is taken
    (not the fallback). We verify this by setting fallback to grossly wrong
    values and checking the result is *not* equal to the fallback transform."""
    from scripts.classify_tile_supervised import load_mrrsu_aux_rasters
    rpeak, bd = _make_rpeak_bd_raster(H=21, W=21)  # 441 px -> well above threshold
    _patch_rasterio(monkeypatch, rpeak, bd)
    # Fallback is intentionally far from the truth so we can detect if it was
    # used by accident.
    bad_fallback_mean = [10.0, 10.0]
    bad_fallback_std = [0.001, 0.001]
    stats = _write_stats(tmp_path, "pertile_zscore",
                         fallback_mean=bad_fallback_mean,
                         fallback_std=bad_fallback_std,
                         min_valid_per_tile=100)
    out = load_mrrsu_aux_rasters("ignored.img", stats)
    # Per-tile z-score on a uniform-distributed tile -> values in roughly
    # +-3 stddevs, NOT thousands (which is what the bad fallback would produce).
    finite = out[np.isfinite(out)]
    assert np.abs(finite).max() < 5.0, (
        f"per-tile z-score should bound values; got max abs {np.abs(finite).max()}"
    )


def test_pertile_zscore_below_threshold_uses_fallback(monkeypatch, tmp_path):
    """Tile has only a handful of valid pixels -> must fall back to global
    fallback_mean / fallback_std."""
    from scripts.classify_tile_supervised import load_mrrsu_aux_rasters
    # Make almost everything NODATA, leaving only a small valid patch.
    H, W = 21, 21
    rpeak = np.full((H, W), 65535.0, dtype=np.float32)
    bd = np.full((H, W), 65535.0, dtype=np.float32)
    rpeak[0:3, 0:3] = 0.80
    bd[0:3, 0:3] = 0.01
    _patch_rasterio(monkeypatch, rpeak, bd)
    fallback_mean = [0.5, 0.0]
    fallback_std = [0.1, 0.1]
    stats = _write_stats(tmp_path, "pertile_zscore",
                         fallback_mean=fallback_mean, fallback_std=fallback_std,
                         min_valid_per_tile=1000)
    out = load_mrrsu_aux_rasters("ignored.img", stats)
    # Pixel (1, 1) is at the center of the only valid 3x3 region -- its 7x7
    # pool drags in invalid neighbours which are masked, leaving only ~9 valid
    # neighbours, and the resulting pool ~= [0.80, 0.01].
    pooled = out[1, 1]
    # With fallback applied: (0.80 - 0.5) / 0.1 = 3.0 ; (0.01 - 0.0) / 0.1 = 0.1
    np.testing.assert_allclose(pooled, [3.0, 0.1], atol=1e-4)


def test_inference_rejects_legacy_v1(monkeypatch, tmp_path):
    from scripts.classify_tile_supervised import load_mrrsu_aux_rasters
    rpeak, bd = _make_rpeak_bd_raster()
    _patch_rasterio(monkeypatch, rpeak, bd)
    # Legacy stats (no version field)
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0]}))
    with pytest.raises(ValueError, match="version"):
        load_mrrsu_aux_rasters("ignored.img", str(p))


def test_inference_unknown_mode_raises(monkeypatch, tmp_path):
    from scripts.classify_tile_supervised import load_mrrsu_aux_rasters
    rpeak, bd = _make_rpeak_bd_raster()
    _patch_rasterio(monkeypatch, rpeak, bd)
    stats = _write_stats(tmp_path, "robust_iqr")
    with pytest.raises(ValueError, match="unsupported norm mode"):
        load_mrrsu_aux_rasters("ignored.img", stats)
