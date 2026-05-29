# tests/test_mrrsu_aux_dataset.py
import json
import numpy as np
import pandas as pd
import pytest
import torch


def _make_inner(n):
    """A stub inner dataset that returns deterministic (patch,label,weight)."""
    class _Inner:
        def __len__(self): return n

        def __getitem__(self, i):
            return (torch.zeros(7, 7, 59), torch.zeros(5), torch.tensor(1.0))
    return _Inner()


def _write_stats_v2(path, mode: str, **fields):
    base = {
        "version": 2,
        "mode": mode,
        "physical_ranges": {"RPEAK1": [0.5, 1.0], "BD1300": [-0.5, 0.5]},
        "band_order": ["RPEAK1", "BD1300"],
    }
    base.update(fields)
    path.write_text(json.dumps(base))


def test_aux_dataset_zscore_and_tuple(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    n = 4
    aux = np.array([[0.77, 0.01], [0.75, 0.00], [0.80, 0.02], [0.74, -0.01]],
                   dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    _write_stats_v2(tmp_path / "stats.json", "zscore",
                    mean=[0.765, 0.005], std=[0.02, 0.01])

    # Patch the inner dataset construction to avoid needing real tiles/cache.
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(n))

    d = MrrsuAuxPatchDataset(
        df=pd.DataFrame({"x": range(n)}), mrral_map={}, patch_size=7,
        aux_npy=str(tmp_path / "aux.npy"), stats_json=str(tmp_path / "stats.json"),
        cache_dir=None, split="train",
    )
    assert len(d) == n
    assert d.mode == "zscore"
    patch, aux2, label, weight = d[0]
    assert patch.shape == (7, 7, 59)
    assert aux2.shape == (2,)
    # z-scored: (0.77-0.765)/0.02 = 0.25 ; (0.01-0.005)/0.01 = 0.5
    assert abs(float(aux2[0]) - 0.25) < 1e-4
    assert abs(float(aux2[1]) - 0.5) < 1e-4


def test_aux_nan_becomes_zero(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset
    aux = np.array([[np.nan, np.nan]], dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    _write_stats_v2(tmp_path / "stats.json", "zscore",
                    mean=[0.765, 0.005], std=[0.02, 0.01])
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(1))
    d = MrrsuAuxPatchDataset(df=pd.DataFrame({"x": [0]}), mrral_map={}, patch_size=7,
                             aux_npy=str(tmp_path / "aux.npy"),
                             stats_json=str(tmp_path / "stats.json"),
                             cache_dir=None, split="train")
    _, aux2, _, _ = d[0]
    # NaN aux -> z-scored 0.0 (the train mean), i.e. "no information"
    assert float(aux2[0]) == 0.0 and float(aux2[1]) == 0.0


def test_aux_dataset_minmax(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    n = 3
    aux = np.array([[0.7, -0.05], [0.8, 0.05], [0.9, 0.0]], dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    _write_stats_v2(tmp_path / "stats.json", "minmax",
                    min=[0.7, -0.05], max=[0.9, 0.05])
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(n))

    d = MrrsuAuxPatchDataset(
        df=pd.DataFrame({"x": range(n)}), mrral_map={}, patch_size=7,
        aux_npy=str(tmp_path / "aux.npy"), stats_json=str(tmp_path / "stats.json"),
        cache_dir=None, split="train",
    )
    assert d.mode == "minmax"
    expected = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]], dtype=np.float32)
    for i in range(n):
        _, aux2, _, _ = d[i]
        np.testing.assert_allclose(aux2.numpy(), expected[i], atol=1e-5)


def test_aux_dataset_minmax_clips_out_of_range(tmp_path, monkeypatch):
    """A value outside [min, max] (e.g. val-split outlier) is clipped to [0, 1]."""
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    aux = np.array([[1.2, -0.3]], dtype=np.float32)  # both outside training range
    np.save(tmp_path / "aux.npy", aux)
    _write_stats_v2(tmp_path / "stats.json", "minmax",
                    min=[0.7, -0.05], max=[0.9, 0.05])
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(1))

    d = MrrsuAuxPatchDataset(df=pd.DataFrame({"x": [0]}), mrral_map={}, patch_size=7,
                             aux_npy=str(tmp_path / "aux.npy"),
                             stats_json=str(tmp_path / "stats.json"),
                             cache_dir=None, split="train")
    _, aux2, _, _ = d[0]
    # 1.2 -> 1.0 (clip), -0.3 -> 0.0 (clip)
    assert float(aux2[0]) == 1.0
    assert float(aux2[1]) == 0.0


def test_aux_dataset_pertile_zscore_uses_tile_stats(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    # Two tiles, each with 4 rows. Both above min_valid_per_tile threshold so
    # we exercise the per-tile branch (not fallback).
    aux = np.array([
        # tile A: mean ~ [0.80, 0.01], std small
        [0.78, 0.005], [0.80, 0.010], [0.82, 0.015], [0.80, 0.010],
        # tile B: mean ~ [0.72, -0.01], distinctly different
        [0.71, -0.012], [0.72, -0.010], [0.73, -0.008], [0.72, -0.010],
    ], dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    _write_stats_v2(tmp_path / "stats.json", "pertile_zscore",
                    fallback_mean=[0.5, 0.0], fallback_std=[1.0, 1.0],
                    min_valid_per_tile=3)
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(8))

    df = pd.DataFrame({
        "tile_id": ["tA"] * 4 + ["tB"] * 4,
    })
    d = MrrsuAuxPatchDataset(df=df, mrral_map={}, patch_size=7,
                             aux_npy=str(tmp_path / "aux.npy"),
                             stats_json=str(tmp_path / "stats.json"),
                             cache_dir=None, split="train")
    assert d.mode == "pertile_zscore"

    # Collect transformed aux rows and verify each tile's transformed vectors
    # have ~zero mean (within numerical noise) -- the defining property of per-tile
    # z-score.
    rows = np.stack([d[i][1].numpy() for i in range(8)])
    np.testing.assert_allclose(rows[:4].mean(axis=0), np.zeros(2), atol=1e-5)
    np.testing.assert_allclose(rows[4:].mean(axis=0), np.zeros(2), atol=1e-5)
    # And that the two tiles' rows are not identical (proves per-tile stats took effect)
    assert not np.allclose(rows[:4], rows[4:])


def test_aux_dataset_pertile_zscore_falls_back_below_threshold(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    # Single tile with just 2 valid rows, threshold is 3 -> fallback path.
    aux = np.array([[0.80, 0.01], [0.84, 0.02]], dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    _write_stats_v2(tmp_path / "stats.json", "pertile_zscore",
                    fallback_mean=[0.76, 0.0], fallback_std=[0.02, 0.01],
                    min_valid_per_tile=3)
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(2))

    df = pd.DataFrame({"tile_id": ["tA", "tA"]})
    d = MrrsuAuxPatchDataset(df=df, mrral_map={}, patch_size=7,
                             aux_npy=str(tmp_path / "aux.npy"),
                             stats_json=str(tmp_path / "stats.json"),
                             cache_dir=None, split="train")
    # Row 0: (0.80 - 0.76)/0.02 = 2.0 ; (0.01 - 0)/0.01 = 1.0
    np.testing.assert_allclose(d[0][1].numpy(), [2.0, 1.0], atol=1e-4)
    # Row 1: (0.84 - 0.76)/0.02 = 4.0 ; (0.02 - 0)/0.01 = 2.0
    np.testing.assert_allclose(d[1][1].numpy(), [4.0, 2.0], atol=1e-4)


def test_aux_dataset_rejects_legacy_v1_stats(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    aux = np.zeros((1, 2), dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    # Legacy v1 (no version, no mode): must raise.
    (tmp_path / "stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0]}))
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(1))

    with pytest.raises(ValueError, match="version"):
        MrrsuAuxPatchDataset(df=pd.DataFrame({"x": [0]}), mrral_map={}, patch_size=7,
                             aux_npy=str(tmp_path / "aux.npy"),
                             stats_json=str(tmp_path / "stats.json"),
                             cache_dir=None, split="train")
