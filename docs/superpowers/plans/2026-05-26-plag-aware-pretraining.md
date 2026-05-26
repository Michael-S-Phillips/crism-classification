# Plag-Aware Multi-Task Pretraining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift plagioclase val_AP above the ~0.13 encoder-representation ceiling by warm-starting the denoising MAE encoder with a supervised 5-class auxiliary head, plus folding the 30-spectrum plagioclase library into train-only fine-tuning data.

**Architecture:** A thin `MultiTaskDenoisingMAE` wraps the existing `DenoisingSpatialSpectralMAE` and adds one `Linear(128, 5)` aux head reading the full-visibility center-token embedding. A dual-stream pretraining loop computes reconstruction loss on unlabeled global-cache patches plus labeled mrral patches, and a 5-class ASL aux loss on the labeled patches only. Synthetic plag patches (tiled + per-pixel-noised mean spectra) are added to the fine-tuning train split via a separate cache.

**Tech Stack:** PyTorch, numpy, pandas/pyarrow, `spectral.io.envi` (ENVI spectral library reading), rasterio, pytest 9.

**Spec:** `docs/superpowers/specs/2026-05-26-plag-aware-pretraining-design.md`

---

## Conventions (read once before starting)

- All commands run from repo root `/mnt/mrdr/crism_classification` in the `crism` conda env: prefix with `conda run -n crism`.
- Patches are `(N, 7, 7, 59)` float32, reflectance clipped to `[0, 0.5]` (matches `CRISMSpectralPatchDataset.CLIP_MAX`).
- Label order is `LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']` (see `data/dataset.py:14`). Plagioclase is index 3.
- The encoder consumes `(B, 7, 7, 59)`; `MAE.encode(x)` returns the full-visibility center-token embedding `(B, 128)` (see `models/spatial_mae.py:140`).
- ASL loss: `AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)`, `forward(logits, targets, weights, pos_weight=None, class_weights=None)` (see `training/losses.py:85`).
- Target mrral wavelengths = the first 59 band centers of any mrral `.hdr` (72 total bands; first 59 span 410.12–2456.79 nm). Read with `spectral.io.envi.open(hdr).bands.centers[:59]`.
- New tests go in `tests/`, run with `conda run -n crism python -m pytest tests/<file>::<test> -v`.

## File Structure

| File | Responsibility |
|---|---|
| `data/synthetic_plag.py` (new) | Pure functions: read ENVI libraries → interpolate to 59 mrral bands; synthesize augmented `(N,7,7,59)` patches from a mean spectrum. |
| `scripts/build_synthetic_plag_patches.py` (new) | CLI: wire the above, emit `synth_plag_patches_p7.npy` + `synth_plag_rows.parquet`. |
| `models/multitask_denoising_mae.py` (new) | `MultiTaskDenoisingMAE` wrapper + `aux_head` + `forward_aux`. |
| `scripts/pretrain_plag_aware_mae.py` (new) | Dual-stream pretraining loop. |
| `data/dataset.py` (modify) | Add `SyntheticPatchDataset`. |
| `training/train_torch.py` (modify) | Accept `synth_train_cache` + `synth_train_parquet`; ConcatDataset into the train loader. |
| `scripts/hpc_pretrain_plag_aware.slurm` (new) | HPC submission. |
| `scripts/eval_plag_aware.py` (new) | 3-way eval table from wandb / checkpoints. |
| `tests/test_synthetic_plag.py` (new) | Unit tests for `data/synthetic_plag.py`. |
| `tests/test_multitask_denoising_mae.py` (new) | Unit tests for the wrapper. |

---

## Task 1: ENVI library → 59-band interpolated spectra

**Files:**
- Create: `data/synthetic_plag.py`
- Test: `tests/test_synthetic_plag.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_synthetic_plag.py -v`
Expected: FAIL with `ImportError: cannot import name 'interp_to_mrral_wavelengths'`

- [ ] **Step 3: Write minimal implementation**

```python
# data/synthetic_plag.py
"""Build synthetic plagioclase training patches from ENVI mean-spectra libraries.

The 30 plagioclase spectra in /mnt/mrdr/plagioclase-targeted/ are mean spectra per
ROI (545 bands, 364-3937 nm) with no spatial info. This module (a) resamples each
spectrum to the 59 mrral bands and (b) synthesizes spatial 7x7x59 patches by tiling
the spectrum and adding per-pixel noise — so they can train the spatial encoder.
"""
from __future__ import annotations

import numpy as np

WL_SENTINEL = 65535.0   # invalid-wavelength marker in the ENVI band centers
CLIP_MAX = 0.5          # matches CRISMSpectralPatchDataset.CLIP_MAX


def interp_to_mrral_wavelengths(
    lib_wl: np.ndarray,
    lib_refl: np.ndarray,
    target_wl: np.ndarray,
) -> np.ndarray:
    """Linearly resample one library spectrum onto the target mrral wavelengths.

    Drops invalid library samples (wavelength == 65535 sentinel, or non-finite
    wavelength/reflectance) before interpolating. Linear interp; values outside
    the valid library range are held at the nearest endpoint (np.interp default).
    """
    lib_wl = np.asarray(lib_wl, dtype=np.float64)
    lib_refl = np.asarray(lib_refl, dtype=np.float64)
    valid = (
        np.isfinite(lib_wl) & np.isfinite(lib_refl) & (lib_wl < WL_SENTINEL)
    )
    if valid.sum() < 2:
        raise ValueError("Need >=2 valid library samples to interpolate")
    wl = lib_wl[valid]
    refl = lib_refl[valid]
    order = np.argsort(wl)
    return np.interp(np.asarray(target_wl, dtype=np.float64), wl[order], refl[order])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_synthetic_plag.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add data/synthetic_plag.py tests/test_synthetic_plag.py
git commit -m "feat(synthetic-plag): wavelength resampling for ENVI plag spectra"
```

---

## Task 2: Synthesize augmented spatial patches from a mean spectrum

**Files:**
- Modify: `data/synthetic_plag.py`
- Test: `tests/test_synthetic_plag.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_synthetic_plag.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_synthetic_plag.py -v`
Expected: FAIL with `ImportError: cannot import name 'synthesize_patches'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to data/synthetic_plag.py
def synthesize_patches(
    spectrum: np.ndarray,
    n_aug: int,
    rng: np.random.Generator,
    patch_size: int = 7,
    noise_sigma: float = 0.005,
    jitter_sigma: float = 0.003,
    continuum_scale_range: tuple[float, float] = (0.97, 1.03),
) -> np.ndarray:
    """Tile one 59-band spectrum into n_aug augmented (patch_size, patch_size, 59) patches.

    Each augmentation applies, independently per patch:
      - a global continuum scale (multiplicative, uniform in continuum_scale_range)
      - per-band jitter (additive, same across the 49 pixels — a spectral wobble)
      - per-pixel Gaussian noise (additive, independent per pixel & band)
    Per-pixel noise is what prevents a flat-tile shortcut: the 49 pixels differ.
    Output is clipped to [0, CLIP_MAX] to match the real patch normalization.
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    n_bands = spectrum.shape[0]
    P = patch_size
    out = np.empty((n_aug, P, P, n_bands), dtype=np.float32)
    for i in range(n_aug):
        scale = rng.uniform(*continuum_scale_range)
        band_jitter = rng.normal(0.0, jitter_sigma, size=n_bands).astype(np.float32)
        base = spectrum * scale + band_jitter                      # (59,)
        tile = np.broadcast_to(base, (P, P, n_bands)).copy()       # (7,7,59)
        tile += rng.normal(0.0, noise_sigma, size=tile.shape).astype(np.float32)
        out[i] = np.clip(tile, 0.0, CLIP_MAX)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_synthetic_plag.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add data/synthetic_plag.py tests/test_synthetic_plag.py
git commit -m "feat(synthetic-plag): augmented patch synthesis from mean spectrum"
```

---

## Task 3: Builder CLI — emit synthetic patch cache + parquet fragment

**Files:**
- Create: `scripts/build_synthetic_plag_patches.py`
- Test: `tests/test_synthetic_plag.py`

- [ ] **Step 1: Write the failing test (for the row-builder helper)**

```python
# append to tests/test_synthetic_plag.py
from data.synthetic_plag import build_synth_rows


def test_build_synth_rows_schema():
    import pandas as pd
    rng = np.random.default_rng(3)
    spectra = {
        "FRT00008842_07_#1_Plagioclase": np.full(59, 0.2, dtype=np.float32),
        "FRT000092B4_07_#1_Plagioclase": np.full(59, 0.25, dtype=np.float32),
    }
    patches, df = build_synth_rows(spectra, n_aug=5, rng=rng,
                                   confidence_tier="High")
    assert patches.shape == (10, 7, 7, 59)              # 2 spectra * 5 aug
    assert len(df) == 10
    # schema must match mrral_pixels.parquet label/meta columns
    for col in ["tile_id", "polygon_id", "pixel_row", "pixel_col",
                "olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase",
                "other", "confidence_tier", "split"]:
        assert col in df.columns
    assert (df["plagioclase"] == 1).all()
    assert (df["other"] == 0).all()
    assert (df["lcp"] == 0).all() and (df["hcp"] == 0).all()
    assert (df["olivine_t1"] == 0).all() and (df["olivine_t2"] == 0).all()
    assert (df["split"] == "train").all()              # train-only, never val/test
    assert df["tile_id"].str.startswith("SYNTH_PLAG_").all()
    assert [f"m{i}" for i in range(59)] == [c for c in df.columns if c.startswith("m")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_synthetic_plag.py::test_build_synth_rows_schema -v`
Expected: FAIL with `ImportError: cannot import name 'build_synth_rows'`

- [ ] **Step 3: Implement `build_synth_rows` in `data/synthetic_plag.py`**

```python
# append to data/synthetic_plag.py
import pandas as pd

N_BANDS = 59


def build_synth_rows(
    spectra: dict[str, np.ndarray],
    n_aug: int,
    rng: np.random.Generator,
    confidence_tier: str,
) -> tuple[np.ndarray, "pd.DataFrame"]:
    """Turn {spectrum_name: 59-band spectrum} into a patch array + parquet fragment.

    Returns (patches (M,7,7,59) float32, df with M rows). Rows carry plagioclase=1
    and all other labels 0, split='train', and a SYNTH_PLAG_<name>_<i> tile_id.
    Band columns m0..m58 hold the clean source spectrum (center reference).
    """
    band_cols = [f"m{i}" for i in range(N_BANDS)]
    patch_chunks = []
    records = []
    for name, spec in spectra.items():
        spec = np.asarray(spec, dtype=np.float32)
        patches = synthesize_patches(spec, n_aug=n_aug, rng=rng)
        patch_chunks.append(patches)
        safe = name.replace("#", "").replace(" ", "")
        for i in range(n_aug):
            rec = {
                "tile_id": f"SYNTH_PLAG_{safe}_{i}",
                "polygon_id": -1,
                "pixel_row": -1,
                "pixel_col": -1,
                "olivine_t1": 0.0, "olivine_t2": 0.0,
                "lcp": 0.0, "hcp": 0.0, "plagioclase": 1.0, "other": 0.0,
                "confidence_tier": confidence_tier,
                "split": "train",
            }
            rec.update({band_cols[b]: float(spec[b]) for b in range(N_BANDS)})
            records.append(rec)
    patches_all = np.concatenate(patch_chunks, axis=0).astype(np.float32)
    df = pd.DataFrame.from_records(records)
    # Order columns: meta, bands, labels, split (band block contiguous)
    ordered = (["tile_id", "polygon_id", "pixel_row", "pixel_col"]
               + band_cols
               + ["olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase",
                  "other", "confidence_tier", "split"])
    return patches_all, df[ordered]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_synthetic_plag.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Write the CLI driver**

```python
# scripts/build_synthetic_plag_patches.py
"""Build the synthetic plagioclase patch cache + parquet fragment.

Reads the two ENVI plag libraries, resamples each spectrum to the 59 mrral bands,
synthesizes augmented 7x7x59 patches (train-split only), and writes:
  <output_dir>/synth_plag_patches_p7.npy   (M, 7, 7, 59) float32
  <output_dir>/synth_plag_rows.parquet     M rows, mrral_pixels schema subset

Usage:
  conda run -n crism python scripts/build_synthetic_plag_patches.py \\
    --n_aug 300 \\
    --mrral_hdr /mnt/mrdr/mc26/t0505_mrral_35s313_0327_4.hdr
"""
import argparse
import glob
import os
import sys

import numpy as np
import spectral.io.envi as envi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_plag import build_synth_rows, interp_to_mrral_wavelengths

LIBS = [
    ("/mnt/mrdr/plagioclase-targeted/unratioed_plag_highconfidence.hdr", "High"),
    ("/mnt/mrdr/plagioclase-targeted/unratioed_plag_moderateconfidence_w_2micron.hdr",
     "Moderate"),
]


def load_target_wavelengths(mrral_hdr: str) -> np.ndarray:
    img = envi.open(mrral_hdr)
    return np.asarray(img.bands.centers, dtype=np.float64)[:59]


def load_library_resampled(hdr: str, target_wl: np.ndarray) -> dict:
    lib = envi.open(hdr)
    lib_wl = np.asarray(lib.bands.centers, dtype=np.float64)
    spectra = np.asarray(lib.spectra, dtype=np.float64)   # (n_spectra, n_bands)
    names = list(lib.names)
    out = {}
    for name, refl in zip(names, spectra):
        out[name] = interp_to_mrral_wavelengths(lib_wl, refl, target_wl).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_aug", type=int, default=300,
                    help="augmented patches per source spectrum")
    ap.add_argument("--mrral_hdr", type=str, default=None,
                    help="any mrral .hdr to read the 59 target wavelengths from")
    ap.add_argument("--output_dir", type=str,
                    default="/mnt/mrdr/crism_classification/data/patch_cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mrral_hdr = args.mrral_hdr or sorted(glob.glob("/mnt/mrdr/mc*/t*_mrral_*.hdr"))[0]
    target_wl = load_target_wavelengths(mrral_hdr)
    print(f"target wavelengths: {len(target_wl)} bands, "
          f"{target_wl[0]:.1f}-{target_wl[-1]:.1f} nm (from {mrral_hdr})")

    rng = np.random.default_rng(args.seed)
    all_patches, all_dfs = [], []
    for hdr, tier in LIBS:
        spectra = load_library_resampled(hdr, target_wl)
        patches, df = build_synth_rows(spectra, n_aug=args.n_aug, rng=rng,
                                       confidence_tier=tier)
        print(f"  {os.path.basename(hdr)}: {len(spectra)} spectra -> "
              f"{len(df)} rows ({tier})")
        all_patches.append(patches)
        all_dfs.append(df)

    import pandas as pd
    patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    df = pd.concat(all_dfs, ignore_index=True)
    assert len(df) == len(patches)

    os.makedirs(args.output_dir, exist_ok=True)
    npy_path = os.path.join(args.output_dir, "synth_plag_patches_p7.npy")
    pq_path = os.path.join(args.output_dir, "synth_plag_rows.parquet")
    # Save as a plain .npy (loadable via np.load mmap_mode='r')
    np.save(npy_path, patches)
    df.to_parquet(pq_path, index=False)
    print(f"wrote {npy_path}  shape={patches.shape}")
    print(f"wrote {pq_path}   rows={len(df)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the builder on a tiny n_aug to smoke-test end-to-end**

Run: `conda run -n crism python scripts/build_synthetic_plag_patches.py --n_aug 4 --output_dir /tmp/synth_smoke`
Expected: prints `30 spectra -> ...`, writes `/tmp/synth_smoke/synth_plag_patches_p7.npy` shape `(120, 7, 7, 59)` and `synth_plag_rows.parquet` rows=120. (13+17 spectra × 4 aug = 120.)

- [ ] **Step 7: Commit**

```bash
git add data/synthetic_plag.py scripts/build_synthetic_plag_patches.py tests/test_synthetic_plag.py
git commit -m "feat(synthetic-plag): builder CLI for patch cache + parquet fragment"
```

---

## Task 4: `MultiTaskDenoisingMAE` wrapper + 5-class aux head

**Files:**
- Create: `models/multitask_denoising_mae.py`
- Test: `tests/test_multitask_denoising_mae.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multitask_denoising_mae.py
import torch

from models.multitask_denoising_mae import MultiTaskDenoisingMAE


def _model():
    return MultiTaskDenoisingMAE(
        n_bands=59, patch_size=7, embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2, mask_ratio=0.75, n_classes=5,
        sigma_gauss=0.0087, sigma_spike=0.0058, sigma_column=0.0049,
    )


def test_recon_path_unchanged():
    m = _model()
    x = torch.randn(4, 7, 7, 59) * 0.1
    loss, recon, mask = m(x)
    assert loss.ndim == 0 and recon.shape == (4, 49, 59) and mask.shape == (4, 49)


def test_aux_forward_shape():
    m = _model()
    x = torch.randn(3, 7, 7, 59) * 0.1
    logits = m.forward_aux(x)
    assert logits.shape == (3, 5)


def test_aux_uses_full_visibility_center_token():
    # forward_aux must be deterministic given fixed weights (no random masking)
    m = _model().eval()
    x = torch.randn(2, 7, 7, 59) * 0.1
    a = m.forward_aux(x)
    b = m.forward_aux(x)
    torch.testing.assert_close(a, b)


def test_encoder_state_dict_loads_into_classifier():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    m = _model()
    clf = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=5, embed_dim=128, n_heads=4, n_layers=6,
    )
    missing, unexpected = clf.load_encoder_state_dict(m.encoder_state_dict())
    assert missing == [] and unexpected == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_multitask_denoising_mae.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.multitask_denoising_mae'`

- [ ] **Step 3: Write minimal implementation**

```python
# models/multitask_denoising_mae.py
"""Denoising MAE + supervised 5-class auxiliary head (multi-task pretraining).

Inherits the full denoising-recon path from DenoisingSpatialSpectralMAE unchanged.
Adds a single Linear(embed_dim, n_classes) head reading the full-visibility
center-token embedding (MAE.encode) — the exact token the downstream classifier
uses. forward() is the recon path; forward_aux() runs the supervised path.

Checkpoints save encoder_state in the standard format so the encoder loads into
SpatialSpectralClassifier with no changes.

Spec: docs/superpowers/specs/2026-05-26-plag-aware-pretraining-design.md
"""
from typing import Tuple

import torch
import torch.nn as nn

from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE


class MultiTaskDenoisingMAE(DenoisingSpatialSpectralMAE):
    def __init__(self, *args, n_classes: int = 5, embed_dim: int = 128, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        self.n_classes = n_classes
        self.aux_head = nn.Linear(embed_dim, n_classes)

    def forward_aux(self, x_labeled: torch.Tensor) -> torch.Tensor:
        """Full-visibility center-token classification logits. Shape: (B, n_classes)."""
        center = self.encode(x_labeled)        # (B, embed_dim), inherited, no masking
        return self.aux_head(center)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_multitask_denoising_mae.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add models/multitask_denoising_mae.py tests/test_multitask_denoising_mae.py
git commit -m "feat(model): MultiTaskDenoisingMAE wrapper with 5-class aux head"
```

---

## Task 5: Dual-stream plag-aware pretraining loop

**Files:**
- Create: `scripts/pretrain_plag_aware_mae.py`

This task is a runtime script with a smoke-test rather than unit tests (it composes already-tested pieces). The smoke test in Step 3 is the verification gate.

- [ ] **Step 1: Write the script**

```python
# scripts/pretrain_plag_aware_mae.py
"""Plag-aware multi-task pretraining: denoising MAE recon + 5-class ASL aux.

Dual stream per step:
  - Stream U: unlabeled global patch cache -> recon loss only
  - Stream L: labeled mrral patches        -> recon loss + aux ASL loss
Total: recon(U) + recon(L) + lambda * ASL(aux_logits_L, labels_L)
lambda ramps 0 -> lambda_target over --aux_warmup epochs.

Warm-starts encoder+decoder from a denoising-MAE checkpoint (--init).

Usage (HPC):
  python scripts/pretrain_plag_aware_mae.py \\
    --init checkpoints/spatial_mae_denoising_128d_6l_best.pt \\
    --epochs 40 --aux_warmup 5 --lambda_target 1.0 \\
    --plag_class_weight 5.0 --run_name plag_aware_mae_128d_6l
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, required=True,
                    help="denoising-MAE checkpoint to warm-start from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5, help="LR warmup epochs")
    ap.add_argument("--aux_warmup", type=int, default=5, help="lambda ramp epochs")
    ap.add_argument("--lambda_target", type=float, default=1.0)
    ap.add_argument("--plag_class_weight", type=float, default=5.0,
                    help="ASL class_weight on plagioclase (index 3); others 1.0")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--labeled_batch_size", type=int, default=256)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--monitor_frac", type=float, default=0.03,
                    help="fraction of train split held out for plag-AP checkpoint selection")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--run_name", type=str, default="plag_aware_mae_128d_6l")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    from config_loader import load_config
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.config)
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}")

    # ── Stream U: unlabeled global cache (recon only) ────────────────────────
    shard_dir = cfg.get("global_patch_cache_dir")
    if not shard_dir:
        raise KeyError("config must define global_patch_cache_dir")
    from data.cached_patch_dataset import CRISMCachedPatchDataset
    ds_u = CRISMCachedPatchDataset(shard_dir=shard_dir, normalize=True, shuffle=True)
    loader_u = DataLoader(ds_u, batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=args.num_workers > 0,
                          prefetch_factor=4 if args.num_workers > 0 else None)

    # ── Stream L: labeled mrral patches (recon + aux) ────────────────────────
    # Carve a small monitoring slice OUT of the train split (never the official
    # val split — that stays clean for the 3-way fine-tuning comparison). The
    # monitoring slice is used only for checkpoint selection by plag AP.
    from data.dataset import CRISMSpectralPatchDataset, LABEL_COLS
    parquet = os.path.join(cfg["output_dir"], "mrral_pixels.parquet")
    df = pd.read_parquet(parquet)
    train_all = df[df["split"] == "train"].reset_index(drop=True)
    mon_rng = np.random.default_rng(42)
    mon_mask = mon_rng.random(len(train_all)) < args.monitor_frac
    train_mon = train_all[mon_mask].reset_index(drop=True)
    train_core = train_all[~mon_mask].reset_index(drop=True)
    mrral_map = _build_mrral_map(cfg)
    cache_dir = cfg.get("patch_cache_dir")
    # NOTE: the patch-cache memmap is row-aligned to the FULL train split, so the
    # boolean-masked sub-frames must NOT use the cache (indices would misalign).
    # Pass cache_dir=None so these read patches live from tiles via rasterio.
    ds_l = CRISMSpectralPatchDataset(train_core, mrral_map, patch_size=7,
                                     cache_dir=None, split="train")
    ds_mon = CRISMSpectralPatchDataset(train_mon, mrral_map, patch_size=7,
                                       cache_dir=None, split="train")
    loader_l = DataLoader(ds_l, batch_size=args.labeled_batch_size, shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=args.num_workers > 0,
                          prefetch_factor=4 if args.num_workers > 0 else None,
                          drop_last=True)
    loader_mon = DataLoader(ds_mon, batch_size=512, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())
    log.info(f"labeled train-core rows: {len(ds_l):,}; monitor rows: {len(ds_mon):,}; "
             f"core plag positives {int((train_core['plagioclase'] > 0).sum()):,}")

    # ── Model (warm-start) ───────────────────────────────────────────────────
    from models.multitask_denoising_mae import MultiTaskDenoisingMAE
    model = MultiTaskDenoisingMAE(
        n_bands=59, patch_size=7, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=64, decoder_layers=2, mask_ratio=args.mask_ratio, n_classes=5,
    ).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    # warm-start the inherited MAE weights; aux_head stays random
    missing, unexpected = model.load_state_dict(ckpt["mae_state"], strict=False)
    log.info(f"warm-start from {args.init}: missing={missing} unexpected={unexpected}")
    assert all(k.startswith("aux_head") for k in missing), \
        f"unexpected missing keys beyond aux_head: {missing}"

    # ── Loss / optim ─────────────────────────────────────────────────────────
    from training.losses import AsymmetricLoss
    asl = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos,
                         clip=args.asl_clip)
    class_weights = torch.ones(5, device=device)
    class_weights[LABEL_COLS.index("plagioclase")] = args.plag_class_weight

    base_lr = 1.5e-4 * args.batch_size / 256
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95),
                            weight_decay=0.05)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="crism-mineral-classification",
                       entity=cfg.get("wandb", {}).get("entity") or None,
                       name=args.run_name, config=vars(args))
        except Exception as e:
            log.warning(f"wandb off ({e})"); use_wandb = False

    from sklearn.metrics import average_precision_score
    PLAG = LABEL_COLS.index("plagioclase")

    @torch.no_grad()
    def monitor_plag_ap():
        """Plag AP on the held-out train monitoring slice (full-visibility aux head)."""
        model.eval()
        scores, targets = [], []
        for xb, yb, _wb in loader_mon:
            logits = model.forward_aux(xb.to(device))
            scores.append(torch.sigmoid(logits[:, PLAG]).cpu().numpy())
            targets.append(yb[:, PLAG].numpy())
        model.train()
        y = np.concatenate(targets); p = np.concatenate(scores)
        if y.sum() == 0:
            return float("nan")
        return float(average_precision_score(y, p))

    ckpt_dir = cfg.get("checkpoints_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    it_u, it_l = iter(loader_u), iter(loader_l)
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = args.lambda_target * min(1.0, epoch / max(1, args.aux_warmup))
        rec_losses, aux_losses = [], []
        for _ in range(args.steps_per_epoch):
            try:
                xu = next(it_u)
            except StopIteration:
                it_u = iter(loader_u); xu = next(it_u)
            try:
                xl, yl, wl = next(it_l)
            except StopIteration:
                it_l = iter(loader_l); xl, yl, wl = next(it_l)
            xu = xu.to(device); xl = xl.to(device)
            yl = yl.to(device); wl = wl.to(device)

            opt.zero_grad()
            recon_u, _, _ = model(xu)
            recon_l, _, _ = model(xl)
            recon = recon_u + recon_l
            aux_logits = model.forward_aux(xl)
            aux = asl(aux_logits, yl, wl, class_weights=class_weights)
            loss = recon + lam * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rec_losses.append(float(recon)); aux_losses.append(float(aux))

        sched.step()
        mrec, maux = float(np.mean(rec_losses)), float(np.mean(aux_losses))
        mon_ap = monitor_plag_ap()
        lr_now = opt.param_groups[0]["lr"]
        log.info(f"epoch {epoch}/{args.epochs} | recon={mrec:.6f} | aux={maux:.6f} "
                 f"| monitor_plag_AP={mon_ap:.4f} | lambda={lam:.3f} | lr={lr_now:.2e}")
        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "recon": mrec, "aux": maux,
                       "monitor_plag_AP": mon_ap, "lambda": lam, "lr": lr_now})

        # Select best by held-out monitoring plag AP (recon logged as a guardrail).
        if mon_ap == mon_ap and mon_ap > best_ap:   # not NaN and improved
            best_ap = mon_ap
            path = os.path.join(ckpt_dir, f"{args.run_name}_best.pt")
            torch.save({"mae_state": model.state_dict(),
                        "encoder_state": model.encoder_state_dict(),
                        "epoch": epoch, "recon": mrec, "aux": maux,
                        "monitor_plag_AP": mon_ap, "config": vars(args)}, path)
            log.info(f"saved best (monitor_plag_AP={mon_ap:.4f}) -> {path}")


def _build_mrral_map(cfg):
    import glob
    data_root = cfg.get("data_root", "/mnt/mrdr")
    hdrs = sorted(set(glob.glob(os.path.join(data_root, "mc*", "t*mrral*.hdr"))
                      + glob.glob(os.path.join(data_root, "t*mrral*.hdr"))))
    return {os.path.basename(h).split("_mrral_")[0]: h.replace(".hdr", ".img")
            for h in hdrs}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses and imports cleanly**

Run: `conda run -n crism python -c "import ast; ast.parse(open('scripts/pretrain_plag_aware_mae.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: Smoke-test 1 epoch on tiny settings (local GPU)**

Run:
```bash
conda run -n crism python scripts/pretrain_plag_aware_mae.py \
  --init checkpoints/spatial_mae_denoising_128d_6l_best.pt \
  --epochs 1 --aux_warmup 1 --lambda_target 1.0 \
  --batch_size 64 --labeled_batch_size 64 --steps_per_epoch 5 \
  --monitor_frac 0.001 --num_workers 0 \
  --run_name plag_aware_smoke --no_wandb
```
Expected: logs one `epoch 1/1 | recon=... | aux=... | monitor_plag_AP=... | lambda=1.000` line with finite recon and aux; saves `checkpoints/plag_aware_smoke_best.pt`. The warm-start log must show `missing=['aux_head.weight', 'aux_head.bias']` and `unexpected=[]`. (`--monitor_frac 0.001` keeps the monitor slice tiny so the smoke test's per-epoch AP pass is fast.)

> **I/O note:** the labeled core + monitor streams read patches live from tiles via rasterio (`cache_dir=None`), NOT the memmap cache. This is deliberate: the cache is row-aligned to the *full* train split, so the boolean core/monitor split would misalign cache indices. Live reads are correct and keep the monitor cleanly held out, at the cost of speed (~102K windowed reads/epoch at the default `--steps_per_epoch 400 --labeled_batch_size 256`). On HPC with `--num_workers 6` this fits the 1-day budget. A future optimization (out of scope) is to add integer index-remap support to `CRISMSpectralPatchDataset` so a masked subset can still use the cache.
>
> Local box has 15 GiB RAM; keep `--num_workers 0` and small batches for the smoke test. Real training runs on HPC (Task 7).

- [ ] **Step 4: Verify the smoke checkpoint loads into the classifier**

Run:
```bash
conda run -n crism python -c "
import torch
from models.spatial_spectral_transformer import SpatialSpectralClassifier
ck = torch.load('checkpoints/plag_aware_smoke_best.pt', map_location='cpu', weights_only=False)
clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5, embed_dim=128, n_heads=4, n_layers=6)
m,u = clf.load_encoder_state_dict(ck['encoder_state'])
print('missing', m, 'unexpected', u)
assert m==[] and u==[]
print('OK')
"
```
Expected: `missing [] unexpected []` then `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/pretrain_plag_aware_mae.py
git commit -m "feat(pretrain): dual-stream plag-aware multi-task MAE loop"
```

---

## Task 6: `SyntheticPatchDataset` + train-loader concat

**Files:**
- Modify: `data/dataset.py` (add `SyntheticPatchDataset` near `CRISMSpectralPatchDataset`)
- Modify: `training/train_torch.py` (accept synth paths; ConcatDataset on train only)
- Test: `tests/test_dataset.py` (add a synthetic-dataset test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dataset.py
def test_synthetic_patch_dataset(tmp_path):
    import numpy as np
    import pandas as pd
    import torch
    from data.dataset import SyntheticPatchDataset

    n = 6
    patches = (np.random.rand(n, 7, 7, 59) * 0.5).astype("float32")
    npy = tmp_path / "synth.npy"
    np.save(npy, patches)
    band_cols = [f"m{i}" for i in range(59)]
    rows = []
    for i in range(n):
        r = {"tile_id": f"SYNTH_PLAG_x_{i}", "polygon_id": -1,
             "pixel_row": -1, "pixel_col": -1,
             "olivine_t1": 0.0, "olivine_t2": 0.0, "lcp": 0.0, "hcp": 0.0,
             "plagioclase": 1.0, "other": 0.0,
             "confidence_tier": "High", "split": "train"}
        r.update({c: 0.2 for c in band_cols})
        rows.append(r)
    pq = tmp_path / "synth.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False)

    ds = SyntheticPatchDataset(str(npy), str(pq))
    assert len(ds) == n
    patch, label, weight = ds[0]
    assert patch.shape == (7, 7, 59)
    assert label.shape == (5,)              # LABEL_COLS order
    assert float(label[3]) == 1.0           # plagioclase index
    assert float(label.sum()) == 1.0
    assert weight.ndim == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_dataset.py::test_synthetic_patch_dataset -v`
Expected: FAIL with `ImportError: cannot import name 'SyntheticPatchDataset'`

- [ ] **Step 3: Implement `SyntheticPatchDataset` in `data/dataset.py`**

Add after the `CRISMSpectralPatchDataset` class (uses module-level `LABEL_COLS` and `_collapse_labels` already defined in this file):

```python
class SyntheticPatchDataset(Dataset):
    """Serves pre-synthesized plagioclase patches from a .npy + parquet fragment.

    Mirrors CRISMSpectralPatchDataset's __getitem__ contract:
    returns (patch (7,7,59) float32 tensor, label (5,) tensor in LABEL_COLS order,
    weight scalar tensor). Patches are read from a memmap'd .npy aligned row-for-row
    with the parquet fragment.
    """

    def __init__(self, npy_path: str, parquet_path: str, patch_size: int = 7):
        df = _collapse_labels(pd.read_parquet(parquet_path)).reset_index(drop=True)
        self._n = len(df)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        self._cache = np.load(npy_path, mmap_mode='r')
        assert self._cache.shape[0] == self._n, (
            f"patch count {self._cache.shape[0]} != parquet rows {self._n}")
        assert self._cache.shape[1:] == (patch_size, patch_size, 59)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        patch = torch.from_numpy(np.asarray(self._cache[idx], dtype=np.float32).copy())
        return patch, self.labels[idx], self.weights[idx]
```

> Note: `_collapse_labels` derives `confidence_weight` from `confidence_tier`, so the parquet fragment does not need a `confidence_weight` column.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_dataset.py::test_synthetic_patch_dataset -v`
Expected: PASS

- [ ] **Step 5: Wire ConcatDataset into the train loader**

In `training/train_torch.py`: add two parameters to `train_torch_model` (place them next to `class_weights`/`pos_weight` in the signature):

```python
    synth_train_cache: Optional[str] = None,
    synth_train_parquet: Optional[str] = None,
```

Then find where the train dataset is built (the `make_dataset(train_df, 'train')` call) and wrap it:

```python
    train_dataset = make_dataset(train_df, 'train')
    if synth_train_cache and synth_train_parquet:
        from data.dataset import SyntheticPatchDataset
        from torch.utils.data import ConcatDataset
        synth_ds = SyntheticPatchDataset(synth_train_cache, synth_train_parquet)
        logger.info(f"Concatenating {len(synth_ds)} synthetic plag patches into train set")
        train_dataset = ConcatDataset([train_dataset, synth_ds])
```

(Use the existing `train_dataset` variable name if present; otherwise rename the existing train-dataset construction to assign to `train_dataset` and pass that into the train `DataLoader`.)

- [ ] **Step 6: Thread the args through `scripts/train.py`**

Add CLI args (near `--class_weights`, around `scripts/train.py:117`):

```python
    parser.add_argument('--synth_train_cache', type=str, default=None,
                        help='Path to synth_plag_patches_p7.npy to add to the train split.')
    parser.add_argument('--synth_train_parquet', type=str, default=None,
                        help='Path to synth_plag_rows.parquet (row-aligned with the cache).')
```

And pass them into the `spatial_vit` `train_torch_model(...)` call (next to `class_weights=class_weights_tensor`, around `scripts/train.py:495`):

```python
                synth_train_cache=args.synth_train_cache,
                synth_train_parquet=args.synth_train_parquet,
```

- [ ] **Step 7: Verify train.py still parses and the new args exist**

Run:
```bash
conda run -n crism python scripts/train.py --help 2>&1 | grep -E "synth_train_(cache|parquet)"
```
Expected: both `--synth_train_cache` and `--synth_train_parquet` appear in the help output.

- [ ] **Step 8: Run the full dataset test file to confirm no regression**

Run: `conda run -n crism python -m pytest tests/test_dataset.py -v`
Expected: all tests pass (existing + the new `test_synthetic_patch_dataset`).

- [ ] **Step 9: Commit**

```bash
git add data/dataset.py training/train_torch.py scripts/train.py tests/test_dataset.py
git commit -m "feat(dataset): SyntheticPatchDataset + train-split concat wiring"
```

---

## Task 7: HPC slurm script

**Files:**
- Create: `scripts/hpc_pretrain_plag_aware.slurm`

This is config, not code — verification is a parse/dry check, not a unit test.

- [ ] **Step 1: Write the slurm script**

Model it on `scripts/hpc_finetune_bland_v1_binaryplag.slurm` (same micromamba activation + WORKDIR/CKPT layout). Content:

```bash
#!/bin/bash
#SBATCH --job-name=crism_plag_aware_mae
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/plag_aware_mae_%j.log
#SBATCH --error=logs/plag_aware_mae_%j.log

# Plag-aware multi-task pretraining: warm-start the denoising MAE encoder with a
# 5-class ASL aux head (dual stream: unlabeled global cache for recon, labeled
# mrral patches for recon + aux). Produces an encoder checkpoint whose encoder_state
# loads straight into SpatialSpectralClassifier.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
CKPT_DIR="${WORKDIR}/checkpoints"
INIT="${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f "$INIT" ]; then
    echo "ERROR: warm-start checkpoint missing: $INIT"; exit 1
fi

python -u scripts/pretrain_plag_aware_mae.py \
    --init "$INIT" \
    --epochs 40 \
    --warmup 5 \
    --aux_warmup 5 \
    --lambda_target 1.0 \
    --plag_class_weight 5.0 \
    --batch_size 512 \
    --labeled_batch_size 256 \
    --steps_per_epoch 400 \
    --num_workers 6 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --mask_ratio 0.75 \
    --asl_gamma_neg 4.0 --asl_gamma_pos 0.0 --asl_clip 0.05 \
    --run_name plag_aware_mae_128d_6l

echo "=== plag-aware pretraining done ==="
```

- [ ] **Step 2: Verify the script is valid bash**

Run: `bash -n scripts/hpc_pretrain_plag_aware.slurm && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc_pretrain_plag_aware.slurm
git commit -m "feat(hpc): slurm for plag-aware multi-task pretraining"
```

---

## Task 8: 3-way evaluation harness

**Files:**
- Create: `scripts/eval_plag_aware.py`

Produces the comparison table from the spec. The three fine-tuning runs themselves are launched on HPC (commands documented below); this script collates per-class APs from wandb.

- [ ] **Step 1: Write the collation script**

```python
# scripts/eval_plag_aware.py
"""Collate the 3-way plag-aware evaluation from wandb.

Pulls per-class val APs for the baseline, encoder-only, and encoder+synthetic
fine-tuning runs and prints the spec's comparison table.

Usage:
  conda run -n crism python scripts/eval_plag_aware.py \\
    --baseline ft_bland_v3_lrscale0001_cont1 \\
    --enc_only ft_plag_aware_real_only \\
    --enc_synth ft_plag_aware_real_plus_synth
"""
import argparse

import wandb

PROJECT = "space-imagery-center/crism-mineral-classification"
CLASSES = ["olivine", "lcp", "hcp", "plagioclase", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--enc_only", required=True)
    ap.add_argument("--enc_synth", required=True)
    args = ap.parse_args()

    api = wandb.Api()
    runs = {r.name: r for r in api.runs(PROJECT, per_page=200, order="-created_at")}

    rows = [("baseline", args.baseline), ("encoder-only", args.enc_only),
            ("encoder+synth", args.enc_synth)]
    hdr = f'{"run":>16s}  {"mAP":>7s}  ' + "  ".join(f"{c[:5]:>6s}" for c in CLASSES)
    print(hdr); print("-" * len(hdr))
    plag_baseline = None
    for label, name in rows:
        r = runs.get(name)
        if r is None:
            print(f"{label:>16s}  (run '{name}' not found)"); continue
        s = r.summary
        def g(k):
            return s.get(k, float("nan"))
        plag = g("val_AP_plagioclase")
        if label == "baseline":
            plag_baseline = plag
        cells = "  ".join(f"{g('val_AP_'+c):>6.3f}" for c in CLASSES)
        print(f"{label:>16s}  {g('val_mAP'):>7.4f}  {cells}")
    if plag_baseline == plag_baseline:  # not NaN
        print(f"\nplag baseline = {plag_baseline:.3f}; "
              f"publishable target = 0.60; signal gate = 0.20")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `conda run -n crism python -c "import ast; ast.parse(open('scripts/eval_plag_aware.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: Document the three HPC fine-tuning commands inside the script's module docstring is sufficient; also record them here**

The three fine-tuning runs (launched on HPC after `plag_aware_mae_128d_6l_best.pt` exists), all identical to the cont1 config except the noted variable:

```bash
# Run A — encoder-only (plag-aware encoder, real data)
python -u scripts/train.py --model spatial_vit \
  --run_name ft_plag_aware_real_only \
  --pretrain_ckpt checkpoints/plag_aware_mae_128d_6l_best.pt \
  --encoder_lr_scale 0.001 --epochs 100 --patience 25 \
  --batch_size 256 --lr 5e-4 --embed_dim 128 --n_heads 4 --n_layers 6 \
  --patch_size 7 --asl_loss

# Run B — encoder + synthetic plag data
python -u scripts/train.py --model spatial_vit \
  --run_name ft_plag_aware_real_plus_synth \
  --pretrain_ckpt checkpoints/plag_aware_mae_128d_6l_best.pt \
  --encoder_lr_scale 0.001 --epochs 100 --patience 25 \
  --batch_size 256 --lr 5e-4 --embed_dim 128 --n_heads 4 --n_layers 6 \
  --patch_size 7 --asl_loss \
  --synth_train_cache data/patch_cache/synth_plag_patches_p7.npy \
  --synth_train_parquet data/patch_cache/synth_plag_rows.parquet

# Baseline is the existing ft_bland_v3_lrscale0001_cont1 run (already in wandb).
```

> `--pretrain_ckpt` loads `encoder_state` (encoder only, fresh classifier head) — the correct flag for evaluating a new encoder. Confirm `val_AP_<class>` keys exist in the run summary; if the training logs per-class APs under different keys, adjust `eval_plag_aware.py` accordingly.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_plag_aware.py
git commit -m "feat(eval): 3-way plag-aware comparison collator"
```

---

## Execution Order & Dependencies

1. Tasks 1→2→3 (synthetic data) are independent of Tasks 4→5 (model+pretrain). They can be built in either order.
2. Task 6 (dataset concat) depends on Task 3's output schema.
3. Task 7 (slurm) depends on Task 5.
4. Task 8 (eval) depends on Tasks 5, 6, 7 having run on HPC.
5. Building the real synthetic cache (`--n_aug 300`) and launching HPC jobs are runtime actions performed after the code lands.

## Notes for the implementer

- The local box has 15 GiB RAM; the labeled patch cache is 22.5 GB mmap'd. Do all smoke tests with tiny batch sizes / `--num_workers 0`. Real pretraining and fine-tuning run on HPC.
- `--n_aug 300` over 30 spectra yields 9,000 synthetic plag rows — roughly one labeled tile's worth, a deliberately modest supplement. Tune later if Run B underperforms Run A.
- Do not let synthetic rows leak into val/test: `build_synth_rows` hard-codes `split='train'`, and the concat happens only on the train dataset. Keep it that way.
