# tests/test_mrrsu_aux_dataset.py
import json
import numpy as np
import pandas as pd
import torch


def _make_inner(monkeypatch, n):
    """A stub inner dataset that returns deterministic (patch,label,weight)."""
    class _Inner:
        def __len__(self): return n
        def __getitem__(self, i):
            return (torch.zeros(7, 7, 59), torch.zeros(5), torch.tensor(1.0))
    return _Inner()


def test_aux_dataset_zscore_and_tuple(tmp_path, monkeypatch):
    from data import dataset as ds_mod
    from data.dataset import MrrsuAuxPatchDataset

    n = 4
    aux = np.array([[0.77, 0.01], [0.75, 0.00], [0.80, 0.02], [0.74, -0.01]],
                   dtype=np.float32)
    np.save(tmp_path / "aux.npy", aux)
    stats = {"mean": [0.765, 0.005], "std": [0.02, 0.01]}
    (tmp_path / "stats.json").write_text(json.dumps(stats))

    # Patch the inner dataset construction to avoid needing real tiles/cache
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(monkeypatch, n))

    d = MrrsuAuxPatchDataset(
        df=pd.DataFrame({"x": range(n)}), mrral_map={}, patch_size=7,
        aux_npy=str(tmp_path / "aux.npy"), stats_json=str(tmp_path / "stats.json"),
        cache_dir=None, split="train",
    )
    assert len(d) == n
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
    (tmp_path / "stats.json").write_text(json.dumps({"mean": [0.765, 0.005], "std": [0.02, 0.01]}))
    monkeypatch.setattr(ds_mod, "CRISMSpectralPatchDataset",
                        lambda *a, **k: _make_inner(monkeypatch, 1))
    d = MrrsuAuxPatchDataset(df=pd.DataFrame({"x": [0]}), mrral_map={}, patch_size=7,
                             aux_npy=str(tmp_path / "aux.npy"),
                             stats_json=str(tmp_path / "stats.json"),
                             cache_dir=None, split="train")
    _, aux2, _, _ = d[0]
    # NaN aux → z-scored 0.0 (the train mean), i.e. "no information"
    assert float(aux2[0]) == 0.0 and float(aux2[1]) == 0.0
