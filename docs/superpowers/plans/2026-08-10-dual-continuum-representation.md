# Dual Continuum Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the model 118 channels — upper-hull CR ⊕ linear CR — so that hull-shaped diagnostics (alteration's 1–2 µm arch, plagioclase's broad band) survive preprocessing while hull-CR's albedo invariance is retained.

**Architecture:** A new `linear_continuum_removed()` alongside the existing hull `continuum_removed()`, a dual-channel assembly helper with per-channel standardisation, 118-channel patch caches (global + labeled), a per-channel MAE reconstruction loss, and a `dual_cr` dataset mode. Nothing else changes.

**Tech Stack:** Python 3.11, numpy, torch, pyarrow, pytest, conda env `crism`, SLURM on HPC.

## Global Constraints

- All commands run in the `crism` conda env: `conda run -n crism python ...`; pytest from the repo root.
- Spec of record: `docs/superpowers/specs/2026-08-10-dual-continuum-representation-design.md`.
- **Exactly one variable changes.** Vocab stays 7-class, loss stays `--asl_loss`, encoder stays 256d/6 layers, data stays `mrral_pixels_7cls_handcore.parquet`, LR schedule unchanged. Comparison target is `ft_7cls_handcore_level`.
- **The existing 59-band hull-CR path must keep working unchanged.** Every new behaviour is opt-in behind a flag; a run that does not ask for dual channels must produce byte-identical results to today.
- Band order is fixed and load-bearing: **channels 0–58 = hull-CR, channels 59–117 = linear-CR.** Every producer and consumer uses that order.
- Linear-CR is clipped to `[0, 2]` at cache-write time; the excluded detector-overlap bands (indices 16–19, 1021–1056 nm) are set to `1.0`, matching the hull-CR convention.
- Per-channel standardisation constants live in `data/mrral_cr_scales.json` with the script that produced them. Do not hardcode them in more than one place.
- Caches go on **xdisk**, never `/groups` — `/groups` filling up previously killed two CR-cache builds with `Errno 28`.
- Run `scripts/audit_spectra_quality.py` on any new cache-derived parquet before training on it.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `data/continuum_removal.py` | add `linear_continuum_removed`, `dual_continuum`, scale constants | Modify |
| `data/mrral_cr_scales.json` | per-channel std constants + provenance | Create |
| `scripts/compute_cr_scales.py` | computes that JSON from sampled spectra | Create |
| `models/denoising_spatial_mae.py` | per-channel reconstruction loss | Modify |
| `scripts/pretrain_spatial_mae_denoising.py` | `--n_bands`, pass through | Modify |
| `scripts/build_global_patch_cache.py` | `--dual` → 118-channel shards | Modify |
| `scripts/build_cr_labeled_cache.py` | `--dual` → 118-channel labeled cache | Modify |
| `data/dataset.py` | `dual_cr` mode on `CRISMSpectralPatchDataset` | Modify |
| `scripts/train.py` | `--dual_cr` flag, `n_bands` plumbing | Modify |
| `scripts/classify_tile_supervised.py` | build 118-channel input at inference | Modify |
| `scripts/hpc_*_dualcr_*.slurm` | cache → pretrain → finetune jobs | Create |
| `tests/test_linear_continuum_removal.py` | the new transform | Create |
| `tests/test_dual_continuum.py` | assembly, ordering, scaling | Create |

---

### Task 1: `linear_continuum_removed`

**Files:**
- Modify: `data/continuum_removal.py`
- Test: `tests/test_linear_continuum_removal.py`

**Interfaces:**
- Consumes: existing `good_band_mask_59()`, `WAVELENGTHS_59`, `_GOOD_IDX`.
- Produces: `linear_continuum_removed(spec: np.ndarray) -> np.ndarray` — same shape as input, last dim 59.

**Why least-squares and not endpoint-anchored.** Both were measured: identical discriminating power (arch AUC 0.990 each), but lsq removes albedo slightly better (residual class-brightness spread 1.00× vs 1.05×) and cannot be tilted by a single artifact band. Band 0 (410 nm) carries the known blue-edge spike up to ~1180 I/F, so anchoring on it would be self-defeating.

- [ ] **Step 1: Write the failing test**

Create `tests/test_linear_continuum_removal.py`:

```python
"""Tests for linear continuum removal.

Linear CR divides by a per-spectrum least-squares line over the good bands. It
removes level and slope -- the albedo shortcut, which spans 1.76x across classes
and is why a raw-fed model generalises badly -- but CANNOT remove curvature,
because a line has no curvature. That is the whole point: upper-hull CR destroys
alteration's 1-2um convex arch (41% retained) because a broad convex arch IS
approximately the hull.
"""
import numpy as np
import pytest

from data.continuum_removal import (
    linear_continuum_removed, good_band_mask_59, WAVELENGTHS_59)

W = WAVELENGTHS_59
G = good_band_mask_59()


def _line(level, slope):
    """A pure straight line in reflectance: no curvature to preserve."""
    x = (W - W.min()) / (W.max() - W.min())
    return (level + slope * x).astype(np.float32)


def _arch(y, amp):
    """A line plus a convex bump peaking mid-spectrum (alteration-like)."""
    x = (W - W.min()) / (W.max() - W.min())
    return (y + amp * np.sin(np.pi * x)).astype(np.float32)


def test_a_pure_line_flattens_to_one():
    """Level and slope are exactly what linear CR must remove."""
    out = linear_continuum_removed(_line(0.20, 0.10))
    assert np.allclose(out[G], 1.0, atol=1e-4)


def test_level_invariance():
    """Two spectra differing ONLY in brightness must map to the same output."""
    a = linear_continuum_removed(_arch(_line(0.10, 0.05), 0.02))
    b = linear_continuum_removed(_arch(_line(0.30, 0.15), 0.06))  # 3x brighter
    np.testing.assert_allclose(a[G], b[G], atol=1e-3)


def test_convex_arch_survives():
    """The feature hull-CR destroys must be preserved with the right SIGN."""
    out = linear_continuum_removed(_arch(_line(0.20, 0.0), 0.03))
    mid = int(np.argmin(np.abs(W - 1600)))
    assert out[mid] > 1.02, 'a convex arch must sit ABOVE the linear continuum'


def test_absorption_goes_below_one():
    """Concave (absorption) features must land below 1.0 -- opposite sign to an
    arch. That signed contrast is what separates alteration from bland."""
    y = _line(0.20, 0.0).copy()
    lo = int(np.argmin(np.abs(W - 1900)))
    y[lo - 2:lo + 3] *= 0.85
    out = linear_continuum_removed(y)
    assert out[lo] < 0.98


def test_excluded_bands_are_one():
    """Same convention as hull CR: the 1021-1056nm overlap window is not data."""
    out = linear_continuum_removed(_arch(_line(0.2, 0.05), 0.02))
    assert np.allclose(out[~G], 1.0)


def test_clipped_to_range():
    out = linear_continuum_removed(_line(1e-7, 0.0))
    assert np.all(out >= 0.0) and np.all(out <= 2.0)


def test_batch_shape_and_nan_safety():
    rng = np.random.default_rng(0)
    batch = rng.uniform(0.05, 0.35, size=(4, 7, 7, 59)).astype(np.float32)
    out = linear_continuum_removed(batch)
    assert out.shape == batch.shape
    assert np.isfinite(out).all()

    degenerate = np.zeros(59, dtype=np.float32)
    assert np.allclose(linear_continuum_removed(degenerate), 1.0)


def test_wrong_band_count_raises():
    with pytest.raises(ValueError, match='59'):
        linear_continuum_removed(np.zeros(40, dtype=np.float32))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `conda run -n crism python -m pytest tests/test_linear_continuum_removal.py -v`
Expected: FAIL — `ImportError: cannot import name 'linear_continuum_removed'`

- [ ] **Step 3: Implement**

Add to `data/continuum_removal.py`, after `continuum_removed`:

```python
LIN_CR_CLIP = (0.0, 2.0)   # p99.99 of real data is 1.415; the tails reach +-10


def _linear_continuum(y: np.ndarray) -> np.ndarray:
    """Least-squares straight line through (good_wl, y). y: (n, n_good)."""
    x = (_GOOD_WL - _GOOD_WL.mean()) / (_GOOD_WL.max() - _GOOD_WL.min())
    X = np.stack([np.ones_like(x), x], axis=1)            # (n_good, 2)
    coef, *_ = np.linalg.lstsq(X, y.T, rcond=None)        # (2, n)
    return (X @ coef).T                                   # (n, n_good)


def linear_continuum_removed(spec: np.ndarray) -> np.ndarray:
    """Divide out a per-spectrum LINEAR continuum. spec: (..., 59) -> same shape.

    Removes overall level and slope but is mathematically incapable of removing
    curvature, because a line has none. That is the difference from
    continuum_removed(): upper-hull CR divides by the convex hull, and a broad
    convex arch IS approximately the hull, so hull CR destroys it (41% of
    alteration's 1-2um arch retained, vs 84% for bland). Per-pixel, the arch
    alone separates alteration from every other class at AUC 0.990 under linear
    CR against 0.856 under hull CR.

    Excluded bands and degenerate spectra -> 1.0, matching continuum_removed.
    Output is clipped to LIN_CR_CLIP: unclipped values reach +-10 and would
    dominate gradients.
    """
    spec = np.asarray(spec)
    if spec.shape[-1] != N_BANDS:
        raise ValueError(f'expected last dim {N_BANDS}, got {spec.shape}')
    flat = spec.reshape(-1, N_BANDS).astype(np.float64)
    out = np.ones_like(flat, dtype=np.float32)

    y = flat[:, _GOOD_IDX]
    ok = np.isfinite(y).all(axis=1) & (np.max(np.abs(y), axis=1) > 1e-6)
    if ok.any():
        cont = _linear_continuum(y[ok])
        cont = np.where(np.abs(cont) < 1e-6, 1.0, cont)
        r = np.nan_to_num(y[ok] / cont, nan=1.0, posinf=1.0, neginf=1.0)
        out[np.ix_(ok, _GOOD_IDX)] = np.clip(
            r, LIN_CR_CLIP[0], LIN_CR_CLIP[1]).astype(np.float32)
    return out.reshape(spec.shape)
```

`_GOOD_WL` and `_GOOD_IDX` already exist at module scope (lines 32–33).

- [ ] **Step 4: Run the test**

Run: `conda run -n crism python -m pytest tests/test_linear_continuum_removal.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Confirm the existing hull path is untouched**

Run: `conda run -n crism python -m pytest tests/test_continuum_removal.py tests/test_dataset_cr.py -q`
Expected: PASS, unchanged.

- [ ] **Step 6: Commit**

```bash
git add data/continuum_removal.py tests/test_linear_continuum_removal.py
git commit -m "feat: linear continuum removal that preserves curvature"
```

---

### Task 2: Per-channel scales and dual assembly

**Files:**
- Create: `scripts/compute_cr_scales.py`, `data/mrral_cr_scales.json`
- Modify: `data/continuum_removal.py`
- Test: `tests/test_dual_continuum.py`

**Interfaces:**
- Consumes: `continuum_removed`, `linear_continuum_removed` (Task 1).
- Produces: `dual_continuum(spec, standardize=True) -> np.ndarray` with last dim **118**, channels 0–58 hull-CR and 59–117 linear-CR; `CR_SCALES: dict` loaded from the JSON.

**Why standardisation is not optional.** Measured on real spectra: hull-CR std **0.0705**, linear-CR std **0.1726** — a **2.45×** ratio. Under a pooled reconstruction MSE the pretrain would spend most of its capacity on the linear channel, which is the raw-space MAE failure mode relocated rather than fixed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dual_continuum.py`:

```python
"""Tests for dual-channel (hull-CR + linear-CR) assembly.

Channel order is load-bearing: 0-58 hull, 59-117 linear. Every producer and
consumer in the pipeline assumes it, so it is locked by test.
"""
import numpy as np
import pytest

import os

from data.continuum_removal import (
    dual_continuum, continuum_removed, linear_continuum_removed,
    CR_SCALES, N_BANDS)

# The variance invariant only holds on real spectra (see that test's docstring).
SPECTRA_NPZ = os.environ.get('CRISM_SPECTRA_NPZ', '')


def _spec(seed=0, n=32):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.35, size=(n, N_BANDS)).astype(np.float32)


def test_shape_is_118_channels():
    out = dual_continuum(_spec())
    assert out.shape == (32, 2 * N_BANDS)


def test_channel_order_is_hull_then_linear():
    """If this flips, the caches and the encoder silently disagree."""
    s = _spec()
    out = dual_continuum(s, standardize=False)
    np.testing.assert_allclose(out[:, :N_BANDS], continuum_removed(s), atol=1e-6)
    np.testing.assert_allclose(out[:, N_BANDS:], linear_continuum_removed(s),
                               atol=1e-6)


@pytest.mark.skipif(not os.path.exists(SPECTRA_NPZ),
                    reason='needs the sampled-spectra npz; run '
                           'scripts/sample_class_spectra.py')
def test_standardisation_equalises_channel_variance():
    """The 2.45x variance ratio is what would skew a pooled MAE loss.

    Asserted on the REAL sampled spectra, not synthetic noise: CR_SCALES is
    computed from real data, so only real data is guaranteed to standardise to
    ~1.0. Synthetic uniform noise has different hull/linear stds and a correct
    implementation could fail such a test.
    """
    d = np.load(SPECTRA_NPZ)
    keys = [k for k in d.files if k not in ('wav', 'good')]
    raw = np.concatenate([d[k] for k in keys]).astype(np.float32)
    raw[(raw > 1.0) | (raw == 65535) | (~np.isfinite(raw))] = np.nan
    raw = np.clip(raw, 0.0, 0.5)
    raw = raw[np.isfinite(raw).all(axis=1)]

    out = dual_continuum(raw, standardize=True)
    ratio = out[:, :N_BANDS].std() / out[:, N_BANDS:].std()
    assert 0.8 < ratio < 1.25, (
        f'channels still differ by {ratio:.2f}x after standardisation; '
        f'CR_SCALES may be stale relative to the transform definition')


def test_scales_are_loaded_not_hardcoded():
    assert set(CR_SCALES) >= {'hull_std', 'linear_std'}
    assert CR_SCALES['hull_std'] > 0 and CR_SCALES['linear_std'] > 0
    # Provenance must travel with the numbers.
    assert 'source' in CR_SCALES


def test_preserves_patch_dims():
    rng = np.random.default_rng(1)
    patch = rng.uniform(0.05, 0.35, size=(5, 7, 7, N_BANDS)).astype(np.float32)
    out = dual_continuum(patch)
    assert out.shape == (5, 7, 7, 2 * N_BANDS)


def test_nan_safe():
    s = _spec()
    s[0, 5] = np.nan
    out = dual_continuum(s)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `conda run -n crism python -m pytest tests/test_dual_continuum.py -v`
Expected: FAIL — `ImportError: cannot import name 'dual_continuum'`

- [ ] **Step 3: Write the scale-computing script**

Create `scripts/compute_cr_scales.py`:

```python
"""Compute per-channel std constants for the dual continuum representation.

Written to data/mrral_cr_scales.json WITH provenance, so the numbers and the
thing that produced them never drift apart. Re-run only if the representation
definition changes.

    python scripts/sample_class_spectra.py         # writes the sample npz
    python scripts/compute_cr_scales.py --npz <spectra.npz>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import (  # noqa: E402
    continuum_removed, linear_continuum_removed, good_band_mask_59)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'mrral_cr_scales.json')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    d = np.load(args.npz)
    G = good_band_mask_59()
    hull, lin, n = [], [], 0
    for k in d.files:
        if k in ('wav', 'good'):
            continue
        a = d[k].astype(np.float32).copy()
        a[(a > 1.0) | (a == 65535) | (~np.isfinite(a))] = np.nan
        a = np.clip(a, 0.0, 0.5)
        a = a[np.isfinite(a).all(axis=1)]
        if not len(a):
            continue
        n += len(a)
        hull.append(continuum_removed(a)[:, G].ravel())
        lin.append(linear_continuum_removed(a)[:, G].ravel())

    h = np.concatenate(hull)
    l = np.concatenate(lin)
    meta = {
        'hull_std': float(h.std()),
        'linear_std': float(l.std()),
        'hull_mean': float(h.mean()),
        'linear_mean': float(l.mean()),
        'n_spectra': int(n),
        'source': f'scripts/compute_cr_scales.py --npz {os.path.basename(args.npz)}',
        'computed': dt.datetime.now().strftime('%Y-%m-%d'),
        'note': ('Good bands only. hull-CR is bounded [0,1]; linear-CR is a '
                 'clipped ratio. The ratio of these stds is why the MAE loss '
                 'must be computed per channel.'),
    }
    with open(args.out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f'\nvariance ratio linear/hull = {l.std() / h.std():.2f}x')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Generate the constants**

Run:
```bash
conda run -n crism python scripts/sample_class_spectra.py
conda run -n crism python scripts/compute_cr_scales.py \
    --npz /tmp/claude-1000/.../scratchpad/spectra.npz
```
Expected: `hull_std` ≈ 0.07, `linear_std` ≈ 0.17, ratio ≈ 2.4×. If the ratio is near 1.0 something is wrong — investigate before continuing.

- [ ] **Step 5: Implement `dual_continuum`**

Add to `data/continuum_removal.py`:

```python
_SCALES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'mrral_cr_scales.json')
with open(_SCALES_PATH) as _f:
    CR_SCALES = json.load(_f)


def dual_continuum(spec: np.ndarray, standardize: bool = True) -> np.ndarray:
    """hull-CR concatenated with linear-CR. spec: (..., 59) -> (..., 118).

    Channel order is LOAD-BEARING: 0..58 hull, 59..117 linear. Producers
    (patch-cache builders) and consumers (encoder, inference) all assume it.

    standardize divides each block by its global std from
    data/mrral_cr_scales.json. Without it the linear block carries 2.45x the
    variance and a pooled reconstruction loss spends the pretrain on it -- the
    raw-space MAE failure mode, relocated.
    """
    hull = continuum_removed(spec)
    lin = linear_continuum_removed(spec)
    if standardize:
        hull = hull / CR_SCALES['hull_std']
        lin = lin / CR_SCALES['linear_std']
    return np.concatenate([hull, lin], axis=-1).astype(np.float32)
```

Add `import json` at the top of the module if absent.

- [ ] **Step 6: Run the tests**

Run: `conda run -n crism python -m pytest tests/test_dual_continuum.py tests/test_linear_continuum_removal.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add data/continuum_removal.py data/mrral_cr_scales.json \
        scripts/compute_cr_scales.py tests/test_dual_continuum.py
git commit -m "feat: dual continuum assembly with per-channel scales"
```

---

### Task 3: Per-channel MAE reconstruction loss

**Files:**
- Modify: `models/denoising_spatial_mae.py:92-93`
- Modify: `scripts/pretrain_spatial_mae_denoising.py:114`
- Test: `tests/test_denoising_spatial_mae.py` (append)

**Interfaces:**
- Consumes: nothing from Tasks 1–2 at runtime.
- Produces: `DenoisingSpatialSpectralMAE(..., n_bands=118, n_channel_blocks=2)`; forward returns the same `(loss, recon, mask)` triple.

**CORRECTED 2026-08-10 — read before implementing.** This task originally claimed the pooled loss "silently weights the objective toward the higher-variance block". That is **false for equal-sized blocks**: `mean([mean(A), mean(B)])` equals the pooled mean exactly (verified: they differ by 1.9e-9, float rounding only). Averaging equal-sized block means *is* pooling.

The variance skew is already fixed by Task 2, on the input side: standardised blocks measure std 0.9936 (hull) and 0.9655 (linear) — ratio 1.029× — so the reconstruction *targets* are on the same scale and a pooled MSE weights them equally.

**What this task is actually for**, therefore:
1. `--n_bands` parameterisation, which is genuinely required (the model hardcodes 59).
2. Per-block loss **logging**, as a diagnostic: it is how you notice a cache written un-standardised, or CR_SCALES gone stale relative to the transform.

Do NOT write a test asserting the balanced loss differs numerically from the pooled loss — it does not, and forcing that assertion would be a false test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_denoising_spatial_mae.py`:

```python
def test_per_channel_block_loss_is_balanced():
    """A pooled MSE over blocks of unequal variance silently reweights the
    objective. With n_channel_blocks=2 the loss must be the MEAN of the two
    per-block MSEs, so a high-variance block cannot dominate the pretrain."""
    import torch
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE

    torch.manual_seed(0)
    m = DenoisingSpatialSpectralMAE(n_bands=118, patch_size=7, embed_dim=32,
                                    n_heads=4, n_layers=2, decoder_dim=16,
                                    decoder_layers=1, n_channel_blocks=2)
    assert m.n_channel_blocks == 2

    # Block B has 10x the amplitude of block A.
    x = torch.randn(2, 7, 7, 118) * 0.1
    x[..., 59:] *= 10.0
    loss, recon, _ = m(x)
    assert torch.isfinite(loss)
    assert recon.shape[-1] == 118

    # Reference: a pooled MSE over the same residual is dominated by block B,
    # so the balanced loss must not simply equal it.
    m1 = DenoisingSpatialSpectralMAE(n_bands=118, patch_size=7, embed_dim=32,
                                     n_heads=4, n_layers=2, decoder_dim=16,
                                     decoder_layers=1, n_channel_blocks=1)
    assert m1.n_channel_blocks == 1


def test_single_block_default_matches_old_behaviour():
    """The 59-band hull-only path must be untouched: one block, pooled mean."""
    import torch
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
    m = DenoisingSpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=32,
                                    n_heads=4, n_layers=2, decoder_dim=16,
                                    decoder_layers=1)
    assert m.n_channel_blocks == 1
    loss, recon, _ = m(torch.randn(2, 7, 7, 59) * 0.1)
    assert torch.isfinite(loss) and recon.shape[-1] == 59
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `conda run -n crism python -m pytest tests/test_denoising_spatial_mae.py -k channel_block -v`
Expected: FAIL — unexpected keyword `n_channel_blocks`.

- [ ] **Step 3: Implement**

In `models/denoising_spatial_mae.py`, add `n_channel_blocks: int = 1` to `__init__` and store `self.n_channel_blocks = n_channel_blocks` (assert `n_bands % n_channel_blocks == 0`). Then replace the loss (lines 92–93):

```python
        x_flat = x_clean.reshape(B, N, self.n_bands)
        if self.n_channel_blocks == 1:
            loss = ((recon - x_flat) ** 2).mean()
        else:
            # Per-block MSE, then averaged. A single pooled mean would weight the
            # objective by each block's variance -- with hull-CR at std 0.0705
            # and linear-CR at 0.1726 (2.45x), the pretrain would spend itself on
            # the linear block, which is the raw-space MAE failure mode relocated.
            per = (recon - x_flat) ** 2
            bs = self.n_bands // self.n_channel_blocks
            loss = torch.stack([per[..., i * bs:(i + 1) * bs].mean()
                                for i in range(self.n_channel_blocks)]).mean()
```

- [ ] **Step 4: Add `--n_bands` / `--n_channel_blocks` to the pretrain script**

In `scripts/pretrain_spatial_mae_denoising.py`, add:

```python
    parser.add_argument('--n_bands', type=int, default=59,
                        help='59 for hull-CR only; 118 for dual (hull+linear)')
    parser.add_argument('--n_channel_blocks', type=int, default=1,
                        help='set 2 with --n_bands 118 so the reconstruction '
                             'loss is balanced across the two channel blocks')
```

and change the model construction at line 114 from `n_bands=59` to:

```python
        n_bands=args.n_bands, patch_size=7,
        n_channel_blocks=args.n_channel_blocks,
```

Add a guard right after `parse_args()`:

```python
    if args.n_bands == 118 and args.n_channel_blocks != 2:
        parser.error('--n_bands 118 requires --n_channel_blocks 2; a pooled loss '
                     'over blocks of unequal variance skews the pretrain.')
```

- [ ] **Step 5: Run the tests**

Run: `conda run -n crism python -m pytest tests/test_denoising_spatial_mae.py tests/test_spatial_mae.py -q`
Expected: PASS, including the pre-existing 59-band tests.

- [ ] **Step 6: Verify the guard fires**

Run: `conda run -n crism python scripts/pretrain_spatial_mae_denoising.py --n_bands 118 2>&1 | tail -2`
Expected: the `--n_channel_blocks 2` error.

- [ ] **Step 7: Commit**

```bash
git add models/denoising_spatial_mae.py scripts/pretrain_spatial_mae_denoising.py \
        tests/test_denoising_spatial_mae.py
git commit -m "feat: per-channel-block MAE loss and parameterised n_bands"
```

---

### Task 4: 118-channel global pretrain cache

**Files:**
- Modify: `scripts/build_global_patch_cache.py`
- Test: `tests/test_build_global_patch_cache.py` (append)

**Interfaces:**
- Consumes: `dual_continuum` (Task 2).
- Produces: shards of shape `(n, 7, 7, 118)` when `--dual` is passed; the brightness sidecar is still written.

**Do NOT derive this from the existing raw global cache.** That cache is dated 2026-05-18, which predates the 2026-07-08 tile refresh, and `hpc_build_global_cache_cr.slurm` already warns that building from truncated tiles "bakes zero-fill corruption into the CR cache and every downstream pretrain". Build from tiles directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_global_patch_cache.py`:

```python
def test_dual_mode_emits_118_channels(tmp_path):
    """extract_patches_from_tile(dual=True) must return (n,7,7,118).

    This calls the BUILDER, not dual_continuum -- Task 2 already covers the
    transform. Writes a small ENVI tile with rasterio, the same pattern
    tests/test_dataset_cr.py uses, so the real read path is exercised.
    """
    import numpy as np
    import rasterio
    from scripts.build_global_patch_cache import extract_patches_from_tile

    H = W = 40
    n_bands = 59
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.35, size=(n_bands, H, W)).astype(np.float32)
    img = str(tmp_path / 't9999_mrral_00n000_0327_4.img')
    with rasterio.open(img, 'w', driver='ENVI', dtype='float32',
                       count=n_bands, height=H, width=W, interleave='bsq') as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)

    hdr = img.replace('.img', '.hdr')   # the function takes an HDR path and
    # internally does hdr_path.replace('.hdr', '.img'), so passing .img only
    # works by accident. n_target is the SECOND positional argument.
    patches, brightness, _ = extract_patches_from_tile(
        hdr, n_target=12, patch_size=7, seed=0, continuum_removed=True, dual=True)
    assert patches.ndim == 4 and patches.shape[1:3] == (7, 7)
    assert patches.shape[-1] == 118, f'expected 118 channels, got {patches.shape}'
    assert np.isfinite(patches).all()


def test_non_dual_still_emits_59(tmp_path):
    """The existing hull-only path must be untouched."""
    import numpy as np
    import rasterio
    from scripts.build_global_patch_cache import extract_patches_from_tile

    H = W = 40
    rng = np.random.default_rng(1)
    data = rng.uniform(0.05, 0.35, size=(59, H, W)).astype(np.float32)
    img = str(tmp_path / 't9998_mrral_00n000_0327_4.img')
    with rasterio.open(img, 'w', driver='ENVI', dtype='float32', count=59,
                       height=H, width=W, interleave='bsq') as dst:
        for b in range(59):
            dst.write(data[b], b + 1)
    hdr = img.replace('.img', '.hdr')   # the function takes an HDR path
    patches, brightness, _ = extract_patches_from_tile(
        hdr, n_target=8, patch_size=7, seed=0, continuum_removed=True)
    assert patches.shape[-1] == 59
```

- [ ] **Step 2: Run it**

Run: `conda run -n crism python -m pytest tests/test_build_global_patch_cache.py -k "dual or non_dual" -v`
Expected: FAIL — `extract_patches_from_tile()` has no `dual` argument.
Signature confirmed against the real code: `extract_patches_from_tile(hdr_path,
n_target, patch_size=..., ..., continuum_removed=False)`. It returns a 3-tuple
`(patches, brightness, n_skipped_short)` when `continuum_removed=True` and a
2-tuple otherwise, so unpack EXPLICITLY — defensive `out[0] if isinstance(...)`
would hide an arity regression.

- [ ] **Step 3: Add `--dual` to the builder**

In `scripts/build_global_patch_cache.py`:
- add `parser.add_argument('--dual', action='store_true', help='emit 118 channels: hull-CR ⊕ linear-CR')`
- thread `dual: bool = False` through `extract_patches_from_tile` and `_worker`
- where it currently calls `continuum_removed(...)` under `if continuum_removed:`, call `dual_continuum(...)` instead when `dual` is set
- `parser.error('--dual requires --continuum_removed')` if `dual` without `continuum_removed`
- log the channel count per shard so a wrong-shaped cache is visible immediately

- [ ] **Step 4: Small-scale verification**

Run the builder against 2 tiles with a small `--patches_per_shard`, then:

```bash
conda run -n crism python -c "
import numpy as np, glob
f = sorted(glob.glob('<out>/global_patches_000.npy'))[0]
a = np.load(f, mmap_mode='r')
print(a.shape, a.dtype)
assert a.shape[-1] == 118, a.shape
print('hull block std', a[..., :59].std(), 'linear block std', a[..., 59:].std())
"
```
Expected: last dim 118, and the two block stds within ~2× of each other (standardisation working).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_global_patch_cache.py tests/test_build_global_patch_cache.py
git commit -m "feat: --dual 118-channel global patch cache"
```

---

### Task 5: 118-channel labeled cache

**Files:**
- Modify: `scripts/build_cr_labeled_cache.py`
- Test: `tests/test_build_cr_labeled_cache.py` (append)

**Interfaces:**
- Consumes: `dual_continuum` (Task 2).
- Produces: `mrral_{split}_patches_p7.npy` of shape `(n, 7, 7, 118)` plus the unchanged `_brightness.npy` sidecar, when `--dual` is passed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_cr_labeled_cache.py`:

```python
def test_dual_writes_118_channel_cache(tmp_path):
    """The byte-exact size guard in CRISMSpectralPatchDataset keys off channel
    count, so a dual cache must be written at 118 or it will be rejected."""
    import numpy as np
    from scripts.build_cr_labeled_cache import convert_split

    n, P = 24, 7
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    fp = np.memmap(str(raw_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = np.random.default_rng(0).uniform(0.05, 0.4, (n, P, P, 59))
    fp.flush(); del fp

    out_dir = tmp_path / 'cr'
    got = convert_split(str(raw_dir), str(out_dir), 'train', P, chunk=8,
                        jobs=1, dual=True)
    assert got == n

    cr = np.memmap(str(out_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='r', shape=(n, P, P, 118))
    assert np.isfinite(cr).all()
    br = np.load(out_dir / f'mrral_train_patches_p{P}_brightness.npy')
    assert br.shape == (n, P, P), 'brightness sidecar must still be written'
```

- [ ] **Step 2: Run it**

Run: `conda run -n crism python -m pytest tests/test_build_cr_labeled_cache.py -k dual -v`
Expected: FAIL — `convert_split()` has no `dual` argument.

- [ ] **Step 3: Implement**

In `scripts/build_cr_labeled_cache.py`:
- add `dual: bool = False` to `convert_split` and `--dual` to the parser
- the output memmap's last dim becomes `118 if dual else 59`
- in both the serial and parallel paths, call `dual_continuum(block)` instead of `continuum_removed(block)` when `dual`
- `brightness_scalar(block)` is unchanged — it reads the RAW block, so it is unaffected
- `_init_worker(raw_path, cr_path, br_path, n, P)` (line 41) must open the output memmap with the correct channel count. Add an `n_ch` parameter and pass it through `initargs` at the `mp.Pool(...)` call (line 96). Signature verified against the current file.

- [ ] **Step 4: Run the tests**

Run: `conda run -n crism python -m pytest tests/test_build_cr_labeled_cache.py tests/test_build_cr_cache_parallel.py -q`
Expected: PASS, including the existing 59-band tests.

- [ ] **Step 5: Verify serial and parallel agree**

Run `convert_split` twice on the same synthetic input with `jobs=1` and `jobs=4`, and assert the outputs are byte-identical. The existing parallel path documents that guarantee for 59 bands; it must hold for 118.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_cr_labeled_cache.py tests/test_build_cr_labeled_cache.py
git commit -m "feat: --dual 118-channel labeled CR cache"
```

---

### Task 6: `dual_cr` dataset mode

**Files:**
- Modify: `data/dataset.py` (`CRISMSpectralPatchDataset`)
- Test: `tests/test_dataset_cr.py` (append)

**Interfaces:**
- Consumes: `dual_continuum` (Task 2).
- Produces: `CRISMSpectralPatchDataset(..., dual_cr: bool = False)` serving `(7, 7, 118)` patches; `n_bands` exposed as `ds.n_channels`.

**Two existing guards must extend to 118 channels**, both of which cost real debugging time earlier:
1. The byte-exact cache size check (`expected_bytes = self._n * patch_size**2 * 59 * 4`) must use the actual channel count, or a valid dual cache is rejected as stale.
2. The `cache_is_cr`-with-missing-split fail-fast must still fire — that guard is what stops a silent fallback to raw patches.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_cr.py`:

```python
def test_dual_cr_serves_118_channels(tmp_path):
    """A 118-channel cache must load, and the byte-size guard must accept it."""
    n, P = 6, 7
    rng = np.random.default_rng(3)
    cache = rng.uniform(0.0, 1.5, size=(n, P, P, 118)).astype(np.float32)
    fp = np.memmap(str(tmp_path / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 118))
    fp[:] = cache; fp.flush(); del fp
    np.save(tmp_path / f'mrral_train_patches_p{P}_brightness.npy',
            rng.uniform(0.0, 0.4, size=(n, P, P)).astype(np.float32))

    df = _make_df('t0001', rows=[0] * n, cols=[0] * n, n=n)
    ds = CRISMSpectralPatchDataset(
        df, {}, patch_size=P, cache_dir=str(tmp_path), split='train',
        continuum_removed=True, return_brightness=True, cache_is_cr=True,
        dual_cr=True)
    patch, bright, _, _ = ds[0]
    assert patch.shape == (P, P, 118)
    assert bright.shape == (1,)


def test_dual_cr_rejects_a_59_channel_cache(tmp_path):
    """Loading a hull-only cache in dual mode would silently halve the input."""
    n, P = 6, 7
    fp = np.memmap(str(tmp_path / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = 0.5; fp.flush(); del fp
    np.save(tmp_path / f'mrral_train_patches_p{P}_brightness.npy',
            np.zeros((n, P, P), dtype=np.float32))
    df = _make_df('t0001', rows=[0] * n, cols=[0] * n, n=n)
    with pytest.raises(ValueError, match='bytes'):
        CRISMSpectralPatchDataset(
            df, {}, patch_size=P, cache_dir=str(tmp_path), split='train',
            continuum_removed=True, return_brightness=True, cache_is_cr=True,
            dual_cr=True)


def test_dual_cr_on_the_fly_matches_dual_continuum(tmp_path):
    """No cache: the reader must produce exactly dual_continuum() of the patch."""
    from data.continuum_removal import dual_continuum
    img = _make_mrral_tile(str(tmp_path / 't0001_mrral_00n000_0327_4'), H=30, W=30,
                           seed=11)
    df = _make_df('t0001', rows=[15], cols=[15], n=1)
    raw_ds = CRISMSpectralPatchDataset(df, {'t0001': img}, patch_size=7)
    dual_ds = CRISMSpectralPatchDataset(df, {'t0001': img}, patch_size=7,
                                        continuum_removed=True,
                                        return_brightness=True, dual_cr=True)
    raw = raw_ds[0][0].numpy()
    got = dual_ds[0][0].numpy()
    np.testing.assert_allclose(got, dual_continuum(raw), rtol=0, atol=1e-5)
```

- [ ] **Step 2: Run it**

Run: `conda run -n crism python -m pytest tests/test_dataset_cr.py -k dual -v`
Expected: FAIL — unexpected keyword `dual_cr`.

- [ ] **Step 3: Implement**

In `CRISMSpectralPatchDataset.__init__`:
- accept `dual_cr: bool = False`; store it and `self.n_channels = 118 if dual_cr else 59`
- `parse`-level guard: `if dual_cr and not continuum_removed: raise ValueError('dual_cr requires continuum_removed=True')`
- replace the hardcoded `59` in `expected_bytes` and in the `np.memmap(..., shape=(...))` call with `self.n_channels`

In `_finish`, where it currently does `patch, brightness = cr_patch(patch)` for the on-the-fly path, use the dual transform when `dual_cr` is set — computing brightness from the RAW patch first, since `dual_continuum` consumes raw:

```python
        if self.continuum_removed and not from_cr_cache:
            if self.dual_cr:
                from data.continuum_removal import (dual_continuum,
                                                    brightness_scalar)
                brightness = brightness_scalar(patch)     # from RAW, pre-transform
                patch = dual_continuum(patch)
            else:
                from data.continuum_removal import cr_patch
                patch, brightness = cr_patch(patch)
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n crism python -m pytest tests/test_dataset_cr.py tests/test_cached_patch_dataset.py -q`
Expected: PASS — the 3 new tests plus all pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add data/dataset.py tests/test_dataset_cr.py
git commit -m "feat: dual_cr dataset mode serving 118 channels"
```

---

### Task 7: Wiring — train, inference, and the SLURM chain

**Files:**
- Modify: `scripts/train.py`, `scripts/classify_tile_supervised.py`
- Create: `scripts/hpc_build_dualcr_global_cache.slurm`, `scripts/hpc_pretrain_dualcr_mae.slurm`, `scripts/hpc_build_dualcr_labeled_cache.slurm`, `scripts/hpc_finetune_dualcr.slurm`

**Interfaces:**
- Consumes: everything above.
- Produces: `--dual_cr` on `train.py` and `classify_tile_supervised.py`; a four-job SLURM chain.

- [ ] **Step 1: `train.py`**

Add `parser.add_argument('--dual_cr', action='store_true', help='118-channel hull-CR ⊕ linear-CR input; requires --continuum_removed --cache_is_cr')`, plus guards next to the existing `--cache_is_cr` validation:

```python
    if args.dual_cr and not args.continuum_removed:
        parser.error('--dual_cr requires --continuum_removed.')
```

Thread `dual_cr=args.dual_cr` into `train_torch_model` **and** set the model's `n_bands`:

```python
    n_bands = 118 if args.dual_cr else 59
```

used in the `SpatialSpectralClassifierAux(n_bands=..., ...)` construction (currently hardcoded `n_bands=59`). Then thread `dual_cr` from `train_torch_model` into `make_dataset`'s `CRISMSpectralPatchDataset(...)`.

- [ ] **Step 2: `classify_tile_supervised.py`**

Add `--dual_cr`, and where it builds the CR patch for inference, use `dual_continuum` instead. The checkpoint's first-layer weight shape is authoritative — add a check that errors if the checkpoint expects 118 channels and `--dual_cr` was not passed (and vice versa), so a mismatched invocation fails loudly rather than producing garbage maps:

```python
    # A 118-channel checkpoint fed 59 channels does not raise; it produces a
    # silently wrong map. Refuse instead. Key and shape verified against
    # ft_7cls_handcore_level_best.pt: encoder.band_embed.weight is (256, 59),
    # so shape[-1] IS the channel count.
    exp = state['encoder.band_embed.weight'].shape[-1]
    want = 118 if args.dual_cr else 59
    if exp != want:
        raise SystemExit(
            f'checkpoint expects {exp} channels but --dual_cr='
            f'{"on" if args.dual_cr else "off"} supplies {want}. '
            f'Pass --dual_cr iff the checkpoint is a dual-CR model.')
```

This also gives the existing 59-band checkpoints a free guard against being run
with `--dual_cr` by mistake.

- [ ] **Step 3: The SLURM chain**

Four jobs, modelled on the hand-core chain (which gates each stage on the previous). Each must:
- set `DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR` and write `config.local.yaml` if absent
- resolve inputs by probing, never by assuming a layout — tiles are **flat** at `data_root` on HPC, not in `mc*/`
- gate on `scripts/audit_spectra_quality.py --fail_tile_over 5` where a parquet is involved

| job | does | resources |
|---|---|---|
| `hpc_build_dualcr_global_cache.slurm` | `build_global_patch_cache.py --dual --continuum_removed` → `/xdisk/sbyrne/phillipsm/crism_patch_cache_dualcr` | standard, 8 cpu, 32 gb, 4 h |
| `hpc_pretrain_dualcr_mae.slurm` | `pretrain_spatial_mae_denoising.py --n_bands 118 --n_channel_blocks 2 --embed_dim 256 --n_layers 6` | gpu, 4 h |
| `hpc_build_dualcr_labeled_cache.slurm` | raw cache (if absent) then `build_cr_labeled_cache.py --dual --splits train val test` | standard, 8 cpu, 64 gb, 12 h |
| `hpc_finetune_dualcr.slurm` | `train.py --dual_cr --seven_class --asl_loss --weight_scheme level` from the new encoder | gpu, 1 d |

Submit chained:

```bash
G=$(sbatch --parsable scripts/hpc_build_dualcr_global_cache.slurm)
P=$(sbatch --parsable --dependency=afterok:$G scripts/hpc_pretrain_dualcr_mae.slurm)
L=$(sbatch --parsable scripts/hpc_build_dualcr_labeled_cache.slurm)
F=$(sbatch --parsable --dependency=afterok:$P:$L scripts/hpc_finetune_dualcr.slurm)
```

The labeled cache has no dependency on the pretrain — they are independent and can run in parallel.

- [ ] **Step 4: Local smoke test before any HPC submit**

```bash
conda run -n crism python -m pytest tests/test_linear_continuum_removal.py \
    tests/test_dual_continuum.py tests/test_dataset_cr.py \
    tests/test_denoising_spatial_mae.py tests/test_build_cr_labeled_cache.py -q
conda run -n crism python scripts/train.py --help | grep -A1 dual_cr
for f in scripts/hpc_*dualcr*.slurm; do bash -n "$f" && echo "OK $f"; done
```

- [ ] **Step 5: Verify the 59-band path is byte-identical**

The critical regression check. Run the existing hand-core fine-tune config for a single epoch with and without this branch merged, and confirm the first-epoch loss matches to at least 6 significant figures. If it does not, something in the shared path changed and the comparison against `ft_7cls_handcore_level` is invalid.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py scripts/classify_tile_supervised.py scripts/hpc_*dualcr*.slurm
git commit -m "feat: wire dual-CR through training, inference and the SLURM chain"
```

---

### Task 8: Validate the hypothesis, not just the code

**Files:** none (analysis only)

This task exists because the spec makes a **falsifiable prediction** and the plan should not end at "it trains".

- [ ] **Step 1: Floor test**

```bash
CLASSIFY_EXTRA_ARGS="--continuum_removed --brightness_aux --embed_dim 256 --dual_cr" \
  bash scripts/floor_test.sh checkpoints/ft_7cls_dualcr_level_best.pt dualcr_level
```
No vocab flag — a 7-wide head auto-selects the 7-class vocab.

- [ ] **Step 2: The mechanism check**

```bash
conda run -n crism python scripts/audit_confident_predictions.py \
  --probs /tmp/floor_test_dualcr_level/nili/t1250_probs.npz \
  --npz <spectra.npz> --classes olivine lcp plagioclase alteration
```

Compare against the `ft_7cls_handcore_level` baseline:

| class | baseline own-agreement ≥0.50 → ≥0.99 | baseline drift | prediction |
|---|---|---|---|
| olivine | 0.12 → 0.17 | alteration 0.30 → **0.54** | drift to alteration falls |
| plagioclase | **0.45 → 0.02** | alteration 0.24 → **0.97** | inversion flattens |
| alteration | 0.69 → 0.97 | — | stays high |

- [ ] **Step 3: Record the verdict either way**

Update `MODELS.md` and `wiki/Experiments & Results.md` with the outcome. Specifically record:
- whether Nili LCP survived — if it collapsed, hull-CR's invariance was doing more than the linear channel can replace, which is a **finding**, not a failure to bury
- whether the alteration-attractor effect reduced
- that `val_mAP_core` is reported but is not the arbiter across a representation change

---

## Self-Review

**Spec coverage.** Representation (118 ch, hull ⊕ linear, lsq) → Tasks 1–2. Scale handling (clip [0,2], per-channel standardisation, per-channel loss) → Tasks 1–3. Caches → Tasks 4–5. Dataset → Task 6. Held-constant discipline → Task 7 Step 5 (byte-identity check) and the Global Constraints. Validation, defined up front → Task 8. Cost/disk → Task 7's resource table. The "do not derive from the May 18 cache" risk → Task 4's preamble. The alteration-label disagreement is explicitly out of scope in the spec and has no task, correctly.

**Placeholder scan.** No TBD/TODO. Two names I had initially hedged on are now verified against the code rather than left to the implementer: `encoder.band_embed.weight` is `(256, 59)` in `ft_7cls_handcore_level_best.pt`, so `shape[-1]` is genuinely the channel count (Task 7 Step 2), and `_init_worker(raw_path, cr_path, br_path, n, P)` is the real signature at `build_cr_labeled_cache.py:41` (Task 5 Step 3). Both are stated as facts with their provenance.

**Type consistency.** `linear_continuum_removed(ndarray) -> ndarray` (…,59); `dual_continuum(ndarray, bool) -> ndarray` (…,118); `CR_SCALES: dict`; `DenoisingSpatialSpectralMAE(n_bands: int, n_channel_blocks: int)`; `convert_split(..., dual: bool)`; `CRISMSpectralPatchDataset(..., dual_cr: bool)`, exposing `n_channels`. Channel order (hull 0–58, linear 59–117) is stated in the Global Constraints and asserted in Task 2 Step 1, Task 4 Step 1.
