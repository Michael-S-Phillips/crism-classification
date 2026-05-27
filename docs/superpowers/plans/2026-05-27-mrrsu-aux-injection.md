# mrrsu RPEAK1/BD1300 Aux-Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the mineral classifier a spatially-smoothed RPEAK1/BD1300 discriminant (mrrsu bands 8 & 17, 7×7-mean, z-scored) via a late-fusion aux head, to push plagioclase AP above the ~0.14 encoder ceiling.

**Architecture:** A new `SpatialSpectralClassifierAux` wraps the unchanged `SpatialSpectralTransformer` encoder and concatenates a small MLP embedding of the 2 smoothed params to the center-token embedding before the classification head. A build step precomputes an aligned per-pixel aux cache from the paired mrrsu tiles; a dataset wrapper serves `(patch, aux2, label, weight)`; training and inference are extended to feed the aux vector. The 59-band encoder stays warm-startable from any MAE checkpoint.

**Tech Stack:** PyTorch, numpy, pandas, rasterio, scipy (for windowed mean), pytest 9.

**Spec:** `docs/superpowers/specs/2026-05-27-mrrsu-aux-injection-design.md`

---

## Conventions (read once)

- Run from repo root `/mnt/mrdr/crism_classification` in the `crism` conda env: prefix python/pytest with `conda run -n crism`.
- mrrsu bands (0-indexed, = parquet `b{i}` and rasterio band `i+1`): **RPEAK1 = 8, BD1300 = 17**. NODATA = 65535.0.
- Label order `LABEL_COLS = ['olivine','lcp','hcp','plagioclase','other']` (`data/dataset.py:14`).
- The plain classifier (`models/spatial_spectral_transformer.py`) uses the center-pixel token: `out[:, n_tokens//2 + 1]`. The aux model mirrors this.
- Aux feature vector order: `[mean_7x7(RPEAK1), mean_7x7(BD1300)]` (RPEAK1 first).
- Commit only files named in each task, by explicit path (the tree has unrelated user changes).

## File Structure

| File | Responsibility |
|---|---|
| `models/spatial_spectral_classifier_aux.py` (new) | `SpatialSpectralClassifierAux` — encoder + aux MLP + widened head. |
| `data/mrrsu_aux.py` (new) | `mean_pool_nodata` 7×7 windowed-mean helper (NODATA-aware). |
| `scripts/build_mrrsu_aux.py` (new) | Build aligned `mrrsu_aux_{split}.npy` + `mrrsu_aux_stats.json`. |
| `data/dataset.py` (modify) | `MrrsuAuxPatchDataset` wrapper. |
| `training/train_torch.py` (modify) | aux dataset construction + 4-tuple batch unpacking. |
| `scripts/train.py` (modify) | `--model spatial_vit_aux` branch + aux CLI args. |
| `scripts/classify_tile_supervised.py` (modify) | paired-mrrsu read + aux feed at inference. |
| `scripts/hpc_finetune_mrrsu_aux.slurm` (new) | HPC fine-tune. |
| tests: `tests/test_spatial_spectral_classifier_aux.py`, `tests/test_mrrsu_aux.py`, `tests/test_mrrsu_aux_dataset.py` (new) | |

---

## Task 1: `SpatialSpectralClassifierAux` model

**Files:**
- Create: `models/spatial_spectral_classifier_aux.py`
- Test: `tests/test_spatial_spectral_classifier_aux.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spatial_spectral_classifier_aux.py
import torch

from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux


def _model():
    return SpatialSpectralClassifierAux(
        n_bands=59, patch_size=7, n_classes=5, embed_dim=128,
        n_heads=4, n_layers=6, aux_dim=2, aux_hidden=16,
    )


def test_forward_shape():
    m = _model()
    x = torch.randn(4, 7, 7, 59) * 0.1
    aux = torch.randn(4, 2)
    out = m(x, aux)
    assert out.shape == (4, 5)


def test_encoder_loads_from_mae_checkpoint():
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    m = _model()
    enc = SpatialSpectralTransformer(n_bands=59, patch_size=7, embed_dim=128,
                                     n_heads=4, n_layers=6)
    missing, unexpected = m.load_encoder_state_dict(enc.state_dict())
    assert missing == [] and unexpected == []


def test_param_groups_split_encoder_vs_head():
    m = _model()
    groups = m.get_param_groups(head_lr=1e-3, encoder_lr=1e-6)
    assert len(groups) == 2
    # aux_mlp + head params live in the head group (lr 1e-3), encoder in the other
    enc_ids = {id(p) for p in m.encoder.parameters()}
    head_group = [g for g in groups if g['lr'] == 1e-3][0]
    head_ids = {id(p) for p in head_group['params']}
    assert enc_ids.isdisjoint(head_ids)
    aux_ids = {id(p) for p in m.aux_mlp.parameters()} | {id(p) for p in m.head.parameters()}
    assert aux_ids == head_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_spatial_spectral_classifier_aux.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.spatial_spectral_classifier_aux'`

- [ ] **Step 3: Write minimal implementation**

```python
# models/spatial_spectral_classifier_aux.py
"""Mineral classifier with a late-fusion auxiliary head for smoothed mrrsu params.

Identical to SpatialSpectralClassifier except the center-token embedding is
concatenated with a small MLP embedding of an auxiliary feature vector
([mean_7x7 RPEAK1, mean_7x7 BD1300]) before the classification head. The encoder
is unchanged and loads from any SpatialSpectralMAE checkpoint.

Spec: docs/superpowers/specs/2026-05-27-mrrsu-aux-injection-design.md
"""
import torch
import torch.nn as nn

from models.spatial_spectral_transformer import SpatialSpectralTransformer


class SpatialSpectralClassifierAux(nn.Module):
    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        aux_dim: int = 2,
        aux_hidden: int = 16,
    ):
        super().__init__()
        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, aux_hidden),
        )
        self.head = nn.Linear(embed_dim + aux_hidden, n_classes)
        self._center_idx = self.encoder.n_tokens // 2 + 1  # +1 for CLS

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # x: (B, patch, patch, n_bands); aux: (B, aux_dim)
        out = self.encoder(x)                       # (B, n_tokens+1, embed_dim)
        center = out[:, self._center_idx]           # (B, embed_dim)
        aux_emb = self.aux_mlp(aux)                  # (B, aux_hidden)
        return self.head(torch.cat([center, aux_emb], dim=-1))

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        head_params = list(self.aux_mlp.parameters()) + list(self.head.parameters())
        head_param_ids = {id(p) for p in head_params}
        encoder_params = [p for p in self.parameters() if id(p) not in head_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        return self.encoder.load_encoder_state_dict(state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_spatial_spectral_classifier_aux.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add models/spatial_spectral_classifier_aux.py tests/test_spatial_spectral_classifier_aux.py
git commit -m "feat(model): SpatialSpectralClassifierAux with late-fusion aux head"
```

---

## Task 2: NODATA-aware 7×7 mean helper

**Files:**
- Create: `data/mrrsu_aux.py`
- Test: `tests/test_mrrsu_aux.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_mrrsu_aux.py -v`
Expected: FAIL with `ImportError: cannot import name 'mean_pool_nodata'`

- [ ] **Step 3: Write minimal implementation**

```python
# data/mrrsu_aux.py
"""Helpers for building the smoothed mrrsu auxiliary features (RPEAK1, BD1300).

The plag-vs-olivine discriminant (RPEAK1) is regional, not per-pixel, so we feed
the classifier a 7x7-mean of the mrrsu parameter rasters. NODATA pixels are
excluded from each window mean.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

NODATA = 65535.0
RPEAK1_BAND = 8      # 0-indexed mrrsu band
BD1300_BAND = 17     # 0-indexed mrrsu band


def mean_pool_nodata(raster: np.ndarray, patch_size: int = 7,
                     nodata: float = NODATA) -> np.ndarray:
    """KxK windowed mean of a 2-D raster, excluding NODATA / non-finite pixels.

    Returns a float32 array the same shape as `raster`. Windows with no valid
    pixels are NaN. Uses two box filters (sum of values / count of valid) so the
    cost is O(H*W) regardless of window size.
    """
    r = raster.astype(np.float64)
    valid = np.isfinite(r) & (r != nodata)
    vals = np.where(valid, r, 0.0)
    counts = valid.astype(np.float64)
    # uniform_filter computes the MEAN over the window; multiply by area to get sums
    area = patch_size * patch_size
    sum_vals = uniform_filter(vals, size=patch_size, mode='constant', cval=0.0) * area
    sum_cnt = uniform_filter(counts, size=patch_size, mode='constant', cval=0.0) * area
    out = np.full(r.shape, np.nan, dtype=np.float32)
    nz = sum_cnt > 0
    out[nz] = (sum_vals[nz] / sum_cnt[nz]).astype(np.float32)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_mrrsu_aux.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add data/mrrsu_aux.py tests/test_mrrsu_aux.py
git commit -m "feat(mrrsu-aux): NODATA-aware 7x7 windowed-mean helper"
```

---

## Task 3: Aligned aux-cache + stats builder

**Files:**
- Create: `scripts/build_mrrsu_aux.py`
- (uses `data/mrrsu_aux.py` from Task 2)

This is a CLI with a local smoke test (no unit test — it composes Task 2's tested helper with rasterio I/O).

- [ ] **Step 1: Write the script**

```python
# scripts/build_mrrsu_aux.py
"""Build the aligned mrrsu auxiliary cache: per labeled pixel, the 7x7-mean of
RPEAK1 (band 8) and BD1300 (band 17) from the paired mrrsu tile.

Writes, into <output_dir> (default data/patch_cache):
  mrrsu_aux_{train,val,test}.npy   (n_split, 2) float32, parquet-row order,
                                   column 0 = RPEAK1 mean, column 1 = BD1300 mean
  mrrsu_aux_stats.json             {"mean": [r,b], "std": [r,b]} computed on the
                                   TRAIN split's finite rows (pre-z-score)

Row order matches mrral_pixels.parquet within each split (same as the patch cache).

Usage:
  conda run -n crism python scripts/build_mrrsu_aux.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
from data.mrrsu_aux import mean_pool_nodata, RPEAK1_BAND, BD1300_BAND, NODATA


def build_mrrsu_map(cfg) -> dict:
    data_root = cfg.get('data_root', '/mnt/mrdr')
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrrsu*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrrsu*.hdr'))))
    return {os.path.basename(h).split('_mrrsu_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def build_split(df_split, mrrsu_map, patch_size):
    """Return (n,2) float32 array of [RPEAK1_mean, BD1300_mean] per row."""
    out = np.full((len(df_split), 2), np.nan, dtype=np.float32)
    # group by tile so each mrrsu raster is read + pooled once
    for tid, grp in df_split.groupby('tile_id', sort=False):
        if str(tid).startswith('SYNTH_') or tid not in mrrsu_map:
            continue  # synthetic rows / tiles without a paired mrrsu stay NaN
        with rasterio.open(mrrsu_map[tid]) as src:
            rpeak = src.read(RPEAK1_BAND + 1).astype(np.float32)   # rasterio is 1-indexed
            bd = src.read(BD1300_BAND + 1).astype(np.float32)
        rpeak_m = mean_pool_nodata(rpeak, patch_size=patch_size, nodata=NODATA)
        bd_m = mean_pool_nodata(bd, patch_size=patch_size, nodata=NODATA)
        rows = grp.index.to_numpy()
        rr = grp['pixel_row'].to_numpy().astype(int)
        cc = grp['pixel_col'].to_numpy().astype(int)
        H, W = rpeak_m.shape
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        out[rows[inb], 0] = rpeak_m[rr[inb], cc[inb]]
        out[rows[inb], 1] = bd_m[rr[inb], cc[inb]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.config)
    cfg = load_config(cfg_path)
    out_dir = args.output_dir or cfg['patch_cache_dir']
    os.makedirs(out_dir, exist_ok=True)

    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df = pd.read_parquet(parquet, columns=['tile_id', 'pixel_row', 'pixel_col', 'split'])
    mrrsu_map = build_mrrsu_map(cfg)
    print(f'mrrsu tiles found: {len(mrrsu_map)}')

    arrays = {}
    for split in args.splits:
        sub = df[df['split'] == split].reset_index(drop=True)
        arr = build_split(sub, mrrsu_map, args.patch_size)
        path = os.path.join(out_dir, f'mrrsu_aux_{split}.npy')
        np.save(path, arr)
        n_nan = int(np.isnan(arr).any(axis=1).sum())
        print(f'  {split}: {len(arr):,} rows -> {path}  ({n_nan:,} NaN rows)')
        arrays[split] = arr

    # Train-split stats over finite rows (per feature), pre-z-score
    tr = arrays['train']
    finite = np.isfinite(tr).all(axis=1)
    mean = tr[finite].mean(axis=0).tolist()
    std = (tr[finite].std(axis=0) + 1e-8).tolist()
    stats_path = os.path.join(out_dir, 'mrrsu_aux_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=2)
    print(f'wrote {stats_path}  mean={mean} std={std}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `conda run -n crism python -c "import ast; ast.parse(open('scripts/build_mrrsu_aux.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: Smoke-test on the val split only (fastest) if data is present locally**

Run: `conda run -n crism python scripts/build_mrrsu_aux.py --splits val --output_dir /tmp/aux_smoke`
Expected: prints `mrrsu tiles found: <N>` and `val: <rows> rows -> /tmp/aux_smoke/mrrsu_aux_val.npy (<k> NaN rows)` and writes `mrrsu_aux_stats.json` (NOTE: with only val built, the stats line still runs but uses `arrays['train']` — so for the smoke, instead run with `--splits train val` OR accept a KeyError on stats and ignore; for the real build use all splits). For a clean smoke that includes stats, run `--splits train val test`. If local RAM is tight, `--splits val` and ignore the stats KeyError is acceptable — the parse + per-tile pooling is what's being verified.

> If `mrral_pixels.parquet` or the mrrsu tiles are not present locally, this build runs on the HPC instead (it's cheap, ~minutes). Report that as expected rather than forcing it.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_mrrsu_aux.py
git commit -m "feat(mrrsu-aux): aligned aux cache + stats builder"
```

---

## Task 4: `MrrsuAuxPatchDataset`

**Files:**
- Modify: `data/dataset.py` (add the class after `SyntheticPatchDataset`)
- Test: `tests/test_mrrsu_aux_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_mrrsu_aux_dataset.py -v`
Expected: FAIL with `ImportError: cannot import name 'MrrsuAuxPatchDataset'`

- [ ] **Step 3: Add `MrrsuAuxPatchDataset` to `data/dataset.py`**

Insert after the `SyntheticPatchDataset` class (uses already-imported `np`, `torch`, `Dataset`, `json` — add `import json` at the top of `data/dataset.py` if not present):

```python
class MrrsuAuxPatchDataset(Dataset):
    """Wraps a CRISMSpectralPatchDataset and appends a z-scored mrrsu aux vector.

    Yields (patch (7,7,59), aux2 (2,), label (5,), weight). aux2 is the z-scored
    [mean_7x7 RPEAK1, mean_7x7 BD1300] from the aligned mrrsu_aux_{split}.npy cache.
    NaN aux rows (tiles without a paired mrrsu, or all-NODATA windows) map to 0.0
    after z-scoring — i.e. the train mean, contributing no information.
    """

    def __init__(self, df, mrral_map, patch_size, aux_npy, stats_json,
                 cache_dir=None, split='train'):
        self.inner = CRISMSpectralPatchDataset(
            df, mrral_map, patch_size=patch_size, cache_dir=cache_dir, split=split)
        aux = np.load(aux_npy).astype(np.float32)
        assert len(aux) == len(self.inner), (
            f"aux rows {len(aux)} != patch rows {len(self.inner)}")
        with open(stats_json) as f:
            st = json.load(f)
        mean = np.asarray(st['mean'], dtype=np.float32)
        std = np.asarray(st['std'], dtype=np.float32)
        z = (aux - mean) / std
        z[~np.isfinite(z)] = 0.0          # NaN/inf → 0 (== train mean)
        self.aux = torch.from_numpy(z)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        patch, label, weight = self.inner[idx]
        return patch, self.aux[idx], label, weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_mrrsu_aux_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add data/dataset.py tests/test_mrrsu_aux_dataset.py
git commit -m "feat(dataset): MrrsuAuxPatchDataset (z-scored aux + mrral patch)"
```

---

## Task 5: Training wiring (`train_torch.py` + `train.py`)

**Files:**
- Modify: `training/train_torch.py`
- Modify: `scripts/train.py`

No new unit test; verification is `--help` + a CPU forward smoke (Step 6).

- [ ] **Step 1: Add aux params to `train_torch_model` signature**

In `training/train_torch.py`, after the `synth_train_parquet: Optional[str] = None,` line in the signature, add:

```python
    mrrsu_aux_dir: Optional[str] = None,
    is_aux_model: bool = False,
```

- [ ] **Step 2: Wrap datasets when aux is requested**

In `training/train_torch.py`, replace the `train_ds = make_dataset(train_df, 'train')` line and the `val_ds = make_dataset(val_df, 'val')` line region with aux-aware construction. Find:

```python
    train_ds = make_dataset(train_df, 'train')
    if synth_train_cache and synth_train_parquet:
```

and insert an aux branch so the block reads:

```python
    if mrrsu_aux_dir is not None:
        import os as _os
        from data.dataset import MrrsuAuxPatchDataset
        stats = _os.path.join(mrrsu_aux_dir, 'mrrsu_aux_stats.json')
        train_ds = MrrsuAuxPatchDataset(
            train_df, mrral_map, patch_size,
            aux_npy=_os.path.join(mrrsu_aux_dir, 'mrrsu_aux_train.npy'),
            stats_json=stats, cache_dir=cache_dir, split='train')
        val_ds = MrrsuAuxPatchDataset(
            val_df, mrral_map, patch_size,
            aux_npy=_os.path.join(mrrsu_aux_dir, 'mrrsu_aux_val.npy'),
            stats_json=stats, cache_dir=cache_dir, split='val')
    else:
        train_ds = make_dataset(train_df, 'train')
        if synth_train_cache and synth_train_parquet:
            from data.dataset import SyntheticPatchDataset
            from torch.utils.data import ConcatDataset
            synth_ds = SyntheticPatchDataset(synth_train_cache, synth_train_parquet)
            logger.info(f"Concatenating {len(synth_ds)} synthetic plag patches into train set")
            train_ds = ConcatDataset([train_ds, synth_ds])
        val_ds = make_dataset(val_df, 'val')
```

(Delete the old standalone `train_ds = ...`, the synth `if`, and `val_ds = make_dataset(val_df, 'val')` lines that this block replaces — they now live inside the `else`.)

- [ ] **Step 3: Unpack the 4-tuple + call `model(features, aux2)` in the train loop**

In `training/train_torch.py`, the train loop begins `for features, labels, weights in train_loader:`. Replace that loop header and the plain-model forward with aux-aware handling. Change:

```python
        for features, labels, weights in train_loader:
            features = features.to(device)
            if augment is not None:
                augment.train()
                features = augment(features)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
```

to:

```python
        for batch in train_loader:
            if is_aux_model:
                features, aux2, labels, weights = batch
                aux2 = aux2.to(device)
            else:
                features, labels, weights = batch
            features = features.to(device)
            if augment is not None:
                augment.train()
                features = augment(features)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
```

And in the same loop's `else:` forward branch (the plain-model path), change:

```python
            else:
                logits = model(features)
```

to:

```python
            else:
                logits = model(features, aux2) if is_aux_model else model(features)
```

- [ ] **Step 4: Same unpacking + forward in the val loop**

In `training/train_torch.py`, the val loop begins `for features, labels, weights in val_loader:`. Change its header the same way:

```python
            for batch in val_loader:
                if is_aux_model:
                    features, aux2, labels, weights = batch
                    aux2 = aux2.to(device)
                else:
                    features, labels, weights = batch
                features = features.to(device)
```

and the plain-model val forward (the final `else` that computes `logits = model(features)` in the val block) to:

```python
                else:
                    logits = model(features, aux2) if is_aux_model else model(features)
```

(Leave the `is_decomp` / `is_decomp_adv` branches unchanged — aux is mutually exclusive with those.)

- [ ] **Step 5: Add the `spatial_vit_aux` branch + CLI args to `scripts/train.py`**

5a. Register the model name. Find `TORCH_MODELS = {...}` near the top and add `'spatial_vit_aux'`:

```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit',
                'spectral_hybrid', 'spatial_vit', 'spatial_vit_aux',
                'decomp_spatial_vit', 'decomp_spatial_vit_adv'}
```

5b. Add CLI args near `--synth_train_parquet`:

```python
    parser.add_argument('--mrrsu_aux_dir', type=str, default=None,
                        help='Dir with mrrsu_aux_{split}.npy + mrrsu_aux_stats.json '
                             '(enables the spatial_vit_aux model).')
```

5c. Add a `spatial_vit_aux` branch. Locate the `elif args.model == 'spatial_vit':` block; directly AFTER it (before the next `elif`), add a parallel branch. It mirrors `spatial_vit` but builds `SpatialSpectralClassifierAux`, requires `--mrrsu_aux_dir`, and passes `mrrsu_aux_dir` + `is_aux_model=True`:

```python
        elif args.model == 'spatial_vit_aux':
            if not args.mrrsu_aux_dir:
                parser.error('--model spatial_vit_aux requires --mrrsu_aux_dir')
            import glob as _glob
            data_root = cfg.get('data_root', '/mnt/mrdr')
            mrral_hdrs = sorted(set(
                _glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                + _glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
            mrral_map = {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
                         for h in mrral_hdrs}
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')
            mrral_parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
            df_mrral = pd.read_parquet(mrral_parquet)
            dropout = args.dropout if args.dropout is not None else 0.1
            from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
            model = SpatialSpectralClassifierAux(
                n_bands=59, patch_size=args.patch_size, n_classes=args.n_classes,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(f'Loaded encoder from {args.pretrain_ckpt}. '
                             f'Missing: {missing}, Unexpected: {unexpected}')
            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrral_map=mrral_map, patch_size=args.patch_size,
                cache_dir=mrral_cache_dir,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs, lr_t_max=args.lr_t_max,
                use_focal_loss=args.focal_loss, focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss, asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos, asl_clip=args.asl_clip,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                min_delta=args.min_delta,
                mrrsu_aux_dir=args.mrrsu_aux_dir,
                is_aux_model=True,
            )
```

(Use `run_name` exactly as the `spatial_vit` branch derives it — it is already defined above the branch in `train.py`. Do not redefine it.)

- [ ] **Step 6: Verify args + a CPU forward smoke**

Run: `conda run -n crism python scripts/train.py --help 2>&1 | grep -E "spatial_vit_aux|mrrsu_aux_dir"`
Expected: `spatial_vit_aux` appears in the model choices line and `--mrrsu_aux_dir` in the options.

Then a no-data forward smoke (model + loss path) — create `/tmp/verify_aux_train.py`:

```python
import sys; sys.path.insert(0, '/mnt/mrdr/crism_classification')
import torch
from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
from training.losses import AsymmetricLoss
m = SpatialSpectralClassifierAux(n_bands=59, patch_size=7, n_classes=5,
                                 embed_dim=128, n_heads=4, n_layers=6)
asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
x = torch.rand(8, 7, 7, 59) * 0.3
aux = torch.randn(8, 2)
y = torch.zeros(8, 5); y[:, 3] = 1.0
w = torch.ones(8)
logits = m(x, aux)
loss = asl(logits, y, w)
loss.backward()
assert logits.shape == (8, 5) and torch.isfinite(loss)
# param groups usable by AdamW
g = m.get_param_groups(head_lr=5e-4, encoder_lr=5e-7)
torch.optim.AdamW(g)
print('aux train smoke OK')
```

Run: `conda run -n crism python /tmp/verify_aux_train.py`
Expected: `aux train smoke OK`

- [ ] **Step 7: Commit**

```bash
git add training/train_torch.py scripts/train.py
git commit -m "feat(train): spatial_vit_aux model + aux dataset/batch wiring"
```

---

## Task 6: Inference (`classify_tile_supervised.py`)

**Files:**
- Modify: `scripts/classify_tile_supervised.py`

Verification is parse + a single-tile run if data is local (else HPC-deferred).

- [ ] **Step 1: Add aux CLI args + imports**

Near the top of `main()`'s argparse in `scripts/classify_tile_supervised.py`, add:

```python
    parser.add_argument('--mrrsu_aux', action='store_true',
                        help='Use the SpatialSpectralClassifierAux model with smoothed '
                             'mrrsu RPEAK1/BD1300 features.')
    parser.add_argument('--mrrsu_tile', type=str, default=None,
                        help='Paired mrrsu .img (default: derive from --tile path).')
    parser.add_argument('--mrrsu_aux_stats', type=str,
                        default='data/patch_cache/mrrsu_aux_stats.json',
                        help='z-score stats json from build_mrrsu_aux.py.')
```

- [ ] **Step 2: Add an aux-raster builder function**

Add this helper near `load_tile` in `scripts/classify_tile_supervised.py`:

```python
def load_mrrsu_aux_rasters(mrrsu_path, stats_json, patch_size=PATCH_SIZE):
    """Return a (H, W, 2) float32 array of z-scored [mean7x7 RPEAK1, mean7x7 BD1300].
    NaN windows → 0.0 (train mean). H,W match the mrrsu tile (== mrral tile grid)."""
    import json
    import rasterio
    from data.mrrsu_aux import mean_pool_nodata, RPEAK1_BAND, BD1300_BAND, NODATA as ND
    with rasterio.open(mrrsu_path) as src:
        rpeak = src.read(RPEAK1_BAND + 1).astype(np.float32)
        bd = src.read(BD1300_BAND + 1).astype(np.float32)
    rpeak_m = mean_pool_nodata(rpeak, patch_size=patch_size, nodata=ND)
    bd_m = mean_pool_nodata(bd, patch_size=patch_size, nodata=ND)
    with open(stats_json) as f:
        st = json.load(f)
    mean = np.asarray(st['mean'], dtype=np.float32)
    std = np.asarray(st['std'], dtype=np.float32)
    aux = np.stack([rpeak_m, bd_m], axis=-1)            # (H, W, 2)
    z = (aux - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def derive_mrrsu_path(mrral_path):
    """t..._mrral_..._.img -> t..._mrrsu_..._.img in the same directory."""
    base = os.path.basename(mrral_path)
    return os.path.join(os.path.dirname(mrral_path), base.replace('_mrral_', '_mrrsu_'))
```

- [ ] **Step 3: Build the model and feed aux in `run_inference`**

Find where `run_inference(tile, model, device, ...)` extracts patches and calls `model(x)`. Add an optional `aux_rasters` arg and feed the per-pixel aux. Change the signature and the model call:

```python
def run_inference(tile, model, device, batch_size=4096, aux_rasters=None):
    ...
    # inside the batched loop, alongside `patches` for pixel indices (r, c):
    #   build aux_batch = aux_rasters[r, c]  (B, 2) for the batch's center pixels
    #   logits = model(x, torch.from_numpy(aux_batch).to(device)) if aux_rasters is not None
    #            else model(x)
```

Concretely, the existing `extract_patches_batched` yields patches in row-major pixel order; mirror that ordering to gather `aux_rasters.reshape(-1, 2)` in the same batches:

```python
    aux_flat = aux_rasters.reshape(-1, 2) if aux_rasters is not None else None
    ...
    for start in range(0, n_pixels, batch_size):
        ...
        x = torch.from_numpy(patches).to(device)
        if aux_flat is not None:
            ab = torch.from_numpy(aux_flat[start:start + x.shape[0]]).to(device)
            logits = model(x, ab)
        else:
            logits = model(x)
```

- [ ] **Step 4: Wire model construction in `main()`**

Where `main()` builds `SpatialSpectralClassifier` and loads the checkpoint, branch on `args.mrrsu_aux`:

```python
    if args.mrrsu_aux:
        from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
        model = SpatialSpectralClassifierAux(
            n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
            embed_dim=128, n_heads=4, n_layers=6)
        mrrsu_path = args.mrrsu_tile or derive_mrrsu_path(args.tile)
        aux_rasters = load_mrrsu_aux_rasters(mrrsu_path, args.mrrsu_aux_stats)
    else:
        model = SpatialSpectralClassifier(
            n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
            embed_dim=128, n_heads=4, n_layers=6)
        aux_rasters = None
```

Load the full classifier state dict as the existing code does (the checkpoint is a full fine-tuned `model_state` — load with `model.load_state_dict(state)`; if the existing code loads `ckpt['model_state']`, keep that). Pass `aux_rasters=aux_rasters` into the `run_inference(...)` call.

- [ ] **Step 5: Verify parse + (if local) one-tile run**

Run: `conda run -n crism python -c "import ast; ast.parse(open('scripts/classify_tile_supervised.py').read()); print('parse ok')"`
Expected: `parse ok`

If a fine-tuned aux checkpoint + a local tile are available, run a single classify and confirm a probability map is produced without shape errors. Otherwise this is HPC/post-training-deferred — report as expected.

- [ ] **Step 6: Commit**

```bash
git add scripts/classify_tile_supervised.py
git commit -m "feat(inference): paired-mrrsu aux feed for spatial_vit_aux"
```

---

## Task 7: HPC fine-tune slurm

**Files:**
- Create: `scripts/hpc_finetune_mrrsu_aux.slurm`

- [ ] **Step 1: Write the slurm script**

```bash
#!/bin/bash
#SBATCH --job-name=crism_ft_mrrsu_aux
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_mrrsu_aux_%j.log
#SBATCH --error=logs/ft_mrrsu_aux_%j.log

# Fine-tune the aux classifier (encoder + smoothed RPEAK1/BD1300 head) from the best
# available encoder. Ablation partner: ft_plag_aware_real_only (same encoder, no aux).
# Requires the aux cache (build_mrrsu_aux.py) to exist in data/patch_cache/.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
AUX_DIR="${WORKDIR}/data/patch_cache"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
EOF
fi

# Build the aux cache if missing (cheap: reads mrrsu tiles, 7x7-mean, sample).
if [ ! -f "${AUX_DIR}/mrrsu_aux_train.npy" ] || [ ! -f "${AUX_DIR}/mrrsu_aux_stats.json" ]; then
    echo "=== building mrrsu aux cache ==="
    python -u scripts/build_mrrsu_aux.py
fi

if [ ! -f "$ENCODER" ]; then
    echo "ERROR: encoder missing: $ENCODER"; exit 1
fi

python -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name ft_mrrsu_aux \
    --pretrain_ckpt "${ENCODER}" \
    --mrrsu_aux_dir "${AUX_DIR}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== mrrsu-aux fine-tune done ==="
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/hpc_finetune_mrrsu_aux.slurm && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc_finetune_mrrsu_aux.slurm
git commit -m "feat(hpc): slurm for mrrsu-aux fine-tune (+ auto aux-cache build)"
```

---

## Execution Order & Dependencies

1. Task 1 (model) and Task 2 (helper) are independent.
2. Task 3 (builder) depends on Task 2.
3. Task 4 (dataset) is independent of 1–3 (uses a stubbed inner dataset in tests).
4. Task 5 (training wiring) depends on Tasks 1 and 4.
5. Task 6 (inference) depends on Tasks 1 and 2.
6. Task 7 (slurm) depends on Tasks 3 and 5.

## Notes for the implementer

- Local box: 15 GiB RAM; the labeled patch cache is mmap'd, the parquet is ~870 MB. Keep smoke tests tiny / single-split. The aux cache build and the fine-tune run on HPC.
- The aux model's checkpoint is a full `model_state` (encoder + aux_mlp + head). At inference, load it directly into `SpatialSpectralClassifierAux` (NOT via `load_encoder_state_dict`).
- The training ablation is `ft_mrrsu_aux` (this) vs `ft_plag_aware_real_only` (same encoder, no aux) — compare plag val_AP on the official split via wandb.
