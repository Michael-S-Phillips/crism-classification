# Dust Hard Negatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine bright-dusty pixels that current models confidently call mafic, label them `bland`, and merge them into the 7-class training parquet so a retrain stops firing minerals on dust.

**Architecture:** Two scripts with a clean split of responsibility. `mine_dust_hard_negatives.py` runs **locally** (the `mrral`/`mrrsu` tiles and the 183-tile deployment probs are here) and emits a schema-light parquet of candidate pixels. `merge_hard_negatives.py` runs **on HPC** (where the training parquet lives), matches that parquet's schema, and delegates split assignment to the existing `split_units` machinery. Nothing invents split logic.

**Tech Stack:** Python 3.11, numpy, pandas, pyarrow, rasterio, conda env `crism`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-17-dust-hard-negatives-design.md`. Read it first.
- All commands run under `conda run -n crism` from the repo root.
- **Never mine from the eight floor-test tiles**: `t1249 t1250 t1321 t1322 t0434 t0435 t1086 t1087`. Training on them turns the floor test into partial train-on-test.
- **All mafic/alteration/dust thresholds are tile-relative percentiles, never absolute.** t1249's whole-tile LCPINDEX2 median (0.0299) exceeds t1321's 90th percentile — absolute cuts provably do not transfer.
- `mrrsu` band indices, 0-based, verified present in the headers: `R770=0`, `RBR=1`, `RPEAK1=8`, `OLINDEX3=15`, `BD1300=17`, `LCPINDEX2=18`, `HCPINDEX2=19`, `BD1900_2=27`, `BD2210_2=34`, `D2300=41`. `rasterio` band numbers are these **+1**.
- Nodata for `mrrsu` and `mrral` is `65535`. `PHYS_MAX = 1.0`, `CLIP_MAX = 0.5`, `N_BANDS = 59`.
- Do NOT run the full pytest suite (~50 min, 13 known pre-existing failures). Run only the named test files.
- Never use `git checkout --` to revert; use `cp` backup/restore.

---

### Task 1: Mining script

**Files:**
- Create: `scripts/mine_dust_hard_negatives.py`
- Test: `tests/test_mine_dust_hard_negatives.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `FLOOR_TEST_TILES: frozenset[str]`
  - `MRRSU_IDX: dict[str, int]` (0-based)
  - `select_dust_negatives(mrrsu: np.ndarray, probs: np.ndarray, class_names: list[str], valid: np.ndarray) -> np.ndarray` → bool mask `(H, W)`
  - `thin_mask(mask: np.ndarray, min_sep: int, max_per_tile: int, seed: int) -> np.ndarray` → bool mask
  - output parquet columns: `tile_id, pixel_row, pixel_col, band_00..band_58, RBR, R770, RPEAK1, OLINDEX3, LCPINDEX2, HCPINDEX2, BD1900_2, BD2210_2, D2300`

- [ ] **Step 1: Write the failing tests**

```python
"""Mining dust hard negatives: the criteria must be tile-relative and exclusive."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.mine_dust_hard_negatives import (
    FLOOR_TEST_TILES, MRRSU_IDX, select_dust_negatives, thin_mask,
)

CLASSES = ['olivine', 'pyx', 'plagioclase', 'bland', 'alteration', 'junk']


def _mrrsu(h, w, **overrides):
    """60-band mrrsu cube; every parameter mid-range unless overridden."""
    cube = np.full((60, h, w), 0.02, dtype=np.float32)
    for name, val in overrides.items():
        cube[MRRSU_IDX[name]] = val
    return cube


def _probs(h, w, mineral_p=0.99):
    p = np.zeros((h, w, len(CLASSES)), dtype=np.float32)
    p[:, :, CLASSES.index('pyx')] = mineral_p
    return p


def test_floor_test_tiles_are_the_eight_from_floor_test_sh():
    assert FLOOR_TEST_TILES == frozenset(
        {'t1249', 't1250', 't1321', 't1322', 't0434', 't0435', 't1086', 't1087'})


def test_a_dusty_indexless_confident_pixel_is_selected():
    h = w = 20
    # Half the tile dark+mafic, half bright+indexless, so tile percentiles split.
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:, :10] = 0.06   # mafic half
    cube[MRRSU_IDX['OLINDEX3']][:, :10] = 0.06
    cube[MRRSU_IDX['HCPINDEX2']][:, :10] = 0.06
    cube[MRRSU_IDX['LCPINDEX2']][:, 10:] = 0.000  # dust half: no mafic
    cube[MRRSU_IDX['OLINDEX3']][:, 10:] = 0.000
    cube[MRRSU_IDX['HCPINDEX2']][:, 10:] = 0.000
    cube[MRRSU_IDX['RBR']][:, :10] = 3.0
    cube[MRRSU_IDX['RBR']][:, 10:] = 6.0          # dust half: red
    cube[MRRSU_IDX['R770']][:, :10] = 0.13
    cube[MRRSU_IDX['R770']][:, 10:] = 0.27        # dust half: bright
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert mask[:, 10:].mean() > 0.9, 'dusty indexless half should be mined'
    assert mask[:, :10].sum() == 0, 'mafic half must never be mined'


def test_alteration_signature_blocks_selection():
    """Without this the miner harvests real alteration and teaches the model to
    miss it -- alteration is a mineral in the 7-class vocab."""
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['OLINDEX3']][:, 10:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['RBR']][:, 10:] = 6.0
    cube[MRRSU_IDX['R770']][:, 10:] = 0.27
    cube[MRRSU_IDX['D2300']][:, 10:] = 0.09       # strong alteration
    cube[MRRSU_IDX['BD1900_2']][:, 10:] = 0.09
    cube[MRRSU_IDX['BD2210_2']][:, 10:] = 0.09
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert mask.sum() == 0


def test_pixels_no_model_is_confident_about_are_not_hard_negatives():
    """Easy negatives carry no gradient; only pixels that fool a model qualify."""
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['OLINDEX3']][:, 10:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['RBR']][:, 10:] = 6.0
    cube[MRRSU_IDX['R770']][:, 10:] = 0.27
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w, mineral_p=0.10), CLASSES, valid)
    assert mask.sum() == 0


def test_invalid_pixels_are_never_selected():
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:] = 0.0
    cube[MRRSU_IDX['OLINDEX3']][:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:] = 0.0
    cube[MRRSU_IDX['RBR']][:] = 6.0
    cube[MRRSU_IDX['R770']][:] = 0.27
    valid = np.ones((h, w), bool)
    valid[5, 5] = False
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert not mask[5, 5]


def test_thinning_enforces_separation_and_cap():
    """One dust mantle must not supply the whole negative set."""
    mask = np.ones((40, 40), bool)
    out = thin_mask(mask, min_sep=4, max_per_tile=1000, seed=0)
    ys, xs = np.nonzero(out)
    assert out.sum() < mask.sum()
    d2 = (ys[:, None] - ys[None, :]) ** 2 + (xs[:, None] - xs[None, :]) ** 2
    np.fill_diagonal(d2, 10 ** 9)
    assert d2.min() >= 16, 'two kept pixels closer than min_sep'


def test_thinning_respects_max_per_tile():
    mask = np.ones((60, 60), bool)
    out = thin_mask(mask, min_sep=1, max_per_tile=25, seed=0)
    assert out.sum() == 25


def test_thinning_is_deterministic():
    mask = np.random.default_rng(1).random((30, 30)) > 0.5
    a = thin_mask(mask, min_sep=2, max_per_tile=50, seed=7)
    b = thin_mask(mask, min_sep=2, max_per_tile=50, seed=7)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_mine_dust_hard_negatives.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.mine_dust_hard_negatives'`

- [ ] **Step 3: Implement the selection and thinning core**

Create `scripts/mine_dust_hard_negatives.py`:

```python
"""Mine bright-dusty pixels that current models confidently call mafic.

Spec: docs/superpowers/specs/2026-08-17-dust-hard-negatives-design.md

A pixel qualifies when all five hold:
  1. no mafic signature   OLINDEX3/LCPINDEX2/HCPINDEX2 below tile p40
  2. no alteration        BD1900_2/D2300/BD2210_2 below tile p60
  3. dusty                RBR AND R770 above tile p60
  4. hard                 some model fires >= 0.90 for a mineral there
  5. physically valid     passes the PHYS_MAX/nodata test

Every threshold is a TILE-RELATIVE percentile. Absolute cuts do not transfer:
t1249's whole-tile LCPINDEX2 median (0.0299) exceeds t1321's 90th percentile.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config  # noqa: E402

NODATA = 65535
PHYS_MAX = 1.0
CLIP_MAX = 0.5
N_BANDS = 59
PATCH = 7

# 0-based mrrsu band indices; rasterio band number is index + 1.
MRRSU_IDX = {
    'R770': 0, 'RBR': 1, 'RPEAK1': 8, 'OLINDEX3': 15, 'BD1300': 17,
    'LCPINDEX2': 18, 'HCPINDEX2': 19, 'BD1900_2': 27, 'BD2210_2': 34,
    'D2300': 41,
}
MAFIC = ('OLINDEX3', 'LCPINDEX2', 'HCPINDEX2')
ALTERATION = ('BD1900_2', 'D2300', 'BD2210_2')
DUSTY = ('RBR', 'R770')

# scripts/floor_test.sh — training on these would make the floor test
# partly train-on-test, destroying the one comparator MODELS.md relies on.
FLOOR_TEST_TILES = frozenset(
    {'t1249', 't1250', 't1321', 't1322', 't0434', 't0435', 't1086', 't1087'})

MAFIC_PCTL = 40.0
ALTERATION_PCTL = 60.0
DUSTY_PCTL = 60.0
HARD_P = 0.90
NON_MINERAL = frozenset({'bland', 'other', 'junk'})


def _pctl(band: np.ndarray, valid: np.ndarray, q: float) -> float:
    v = band[valid]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else np.inf


def select_dust_negatives(mrrsu, probs, class_names, valid) -> np.ndarray:
    """Bool (H, W) mask of dust hard negatives. Tile-relative throughout."""
    keep = valid.copy()
    for name in MAFIC:
        b = mrrsu[MRRSU_IDX[name]]
        keep &= np.isfinite(b) & (b < _pctl(b, valid, MAFIC_PCTL))
    for name in ALTERATION:
        b = mrrsu[MRRSU_IDX[name]]
        keep &= np.isfinite(b) & (b < _pctl(b, valid, ALTERATION_PCTL))
    for name in DUSTY:
        b = mrrsu[MRRSU_IDX[name]]
        keep &= np.isfinite(b) & (b > _pctl(b, valid, DUSTY_PCTL))
    mineral_cols = [i for i, c in enumerate(class_names) if c not in NON_MINERAL]
    if not mineral_cols:
        raise ValueError(f'no mineral classes among {class_names}')
    keep &= probs[:, :, mineral_cols].max(axis=2) >= HARD_P
    return keep


def thin_mask(mask, min_sep: int, max_per_tile: int, seed: int) -> np.ndarray:
    """Greedy spatial thinning: no two kept pixels within min_sep, capped.

    Without this one large dust mantle supplies the whole negative set and the
    model learns a location rather than a spectral class.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask)
    order = np.random.default_rng(seed).permutation(len(ys))
    out = np.zeros_like(mask)
    kept_y: list[int] = []
    kept_x: list[int] = []
    sep2 = min_sep * min_sep
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if kept_y:
            dy = np.asarray(kept_y) - y
            dx = np.asarray(kept_x) - x
            if (dy * dy + dx * dx).min() < sep2:
                continue
        out[y, x] = True
        kept_y.append(y)
        kept_x.append(x)
        if len(kept_y) >= max_per_tile:
            break
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_mine_dust_hard_negatives.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Verify the tests can fail (mutation)**

```bash
cp scripts/mine_dust_hard_negatives.py /tmp/mine_backup.py
python3 - <<'PY'
p='scripts/mine_dust_hard_negatives.py'; s=open(p).read()
s=s.replace("    keep &= probs[:, :, mineral_cols].max(axis=2) >= HARD_P\n", "")
open(p,'w').write(s)
PY
conda run -n crism python -m pytest tests/test_mine_dust_hard_negatives.py -q
cp /tmp/mine_backup.py scripts/mine_dust_hard_negatives.py
```
Expected: at least `test_pixels_no_model_is_confident_about_are_not_hard_negatives` FAILS with the mutation, then all pass again after restore. If nothing fails, the test is worthless — fix it before continuing.

- [ ] **Step 6: Add the tile driver and CLI**

Append to `scripts/mine_dust_hard_negatives.py`:

```python
def _load_tile(mrral_path):
    with rasterio.open(mrral_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
    nodata = (data == NODATA) | ~np.isfinite(data) | (data > PHYS_MAX)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata] = 0.0
    return data, ~nodata.any(axis=0)


def _patch_valid(valid, pad=PATCH // 2):
    """True where the whole 7x7 neighbourhood is valid. The classifier reads a
    patch, so a mined centre with a nodata neighbour teaches the padding."""
    from scipy.ndimage import minimum_filter
    ok = minimum_filter(valid.astype(np.uint8), size=PATCH).astype(bool)
    ok[:pad, :] = False; ok[-pad:, :] = False
    ok[:, :pad] = False; ok[:, -pad:] = False
    return ok


def mine_tile(tid, mrral_path, mrrsu_path, probs_path, min_sep, max_per_tile, seed):
    cube, valid = _load_tile(mrral_path)
    with rasterio.open(mrrsu_path) as s:
        mrrsu = s.read().astype(np.float32)
    mrrsu[(mrrsu == NODATA) | ~np.isfinite(mrrsu)] = np.nan
    d = np.load(probs_path, allow_pickle=True)
    probs = d['probs'].astype(np.float32)
    names = [str(x) for x in d['class_names']]
    valid = valid & d['valid_mask'].astype(bool) & _patch_valid(valid)
    mask = select_dust_negatives(mrrsu, probs, names, valid)
    mask = thin_mask(mask, min_sep, max_per_tile, seed)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    out = {'tile_id': tid, 'pixel_row': ys, 'pixel_col': xs}
    for b in range(N_BANDS):
        out[f'band_{b:02d}'] = cube[b][ys, xs]
    for name, idx in MRRSU_IDX.items():
        out[name] = mrrsu[idx][ys, xs]
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs_dir', default='data/mc_deploy_pyx_physmax/probs')
    ap.add_argument('--out', default='data/hard_negatives_dust.parquet')
    ap.add_argument('--min_sep', type=int, default=5)
    ap.add_argument('--max_per_tile', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = load_config()['data_root']
    frames, skipped = [], []
    for p in sorted(glob.glob(os.path.join(args.probs_dir, '*', '*_probs.npz'))):
        tid = os.path.basename(p).replace('_probs.npz', '')
        if tid in FLOOR_TEST_TILES:
            skipped.append(tid)
            continue
        mr = sorted(glob.glob(os.path.join(root, 'mc*', f'{tid}_mrral_*.img')))
        su = sorted(glob.glob(os.path.join(root, 'mc*', f'{tid}_mrrsu_*.img')))
        if not mr or not su:
            print(f'  WARNING: missing mrral/mrrsu for {tid}', file=sys.stderr)
            continue
        df = mine_tile(tid, mr[0], su[0], p, args.min_sep, args.max_per_tile,
                       args.seed)
        n = 0 if df is None else len(df)
        print(f'  {tid}: {n:,} mined', flush=True)
        if df is not None:
            frames.append(df)
    print(f'excluded {len(skipped)} floor-test tiles: {sorted(skipped)}')
    if not frames:
        raise SystemExit('nothing mined')
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f'wrote {args.out}: {len(out):,} rows from {out.tile_id.nunique()} tiles')
    print('\nmined-population mrrsu medians (audit the worst tiles before merging):')
    for name in MRRSU_IDX:
        print(f'  {name:<11} {out[name].median():.4f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 7: Run the miner for real and audit the output**

```bash
conda run --no-capture-output -n crism python scripts/mine_dust_hard_negatives.py 2>&1 | tail -30
```
Expected: a per-tile count table, the line `excluded 8 floor-test tiles: ['t0434', 't0435', 't1086', 't1087', 't1249', 't1250', 't1321', 't1322']`, and a written parquet.

**Acceptance check — do not skip.** The printed medians must show LCPINDEX2/OLINDEX3/HCPINDEX2 near zero and RBR/R770 high. If the mafic medians are NOT near zero, the percentile logic is inverted and the miner is harvesting real minerals; stop and fix before merging.

- [ ] **Step 8: Commit**

```bash
git add scripts/mine_dust_hard_negatives.py tests/test_mine_dust_hard_negatives.py
git commit -m "mine dust hard negatives: bright, index-free pixels models call mafic

Tile-relative percentiles throughout -- absolute cuts do not transfer (t1249's
whole-tile LCPINDEX2 median exceeds t1321's p90). Excludes the eight floor-test
tiles, which would otherwise make the floor test partly train-on-test.

Requires the whole 7x7 patch valid, not just the centre: the classifier reads a
patch, so a mined centre with a nodata neighbour teaches the padding."
```

---

### Task 2: Merge script

**Files:**
- Create: `scripts/merge_hard_negatives.py`
- Test: `tests/test_merge_hard_negatives.py`

**Interfaces:**
- Consumes: the parquet from Task 1 (`tile_id, pixel_row, pixel_col, band_00..band_58, <mrrsu columns>`).
- Produces: `bland_column_of(columns) -> str`, `build_negative_rows(neg_df, target_columns, bland_col, start_id) -> pd.DataFrame`, and a merged parquet with a `split` column.

- [ ] **Step 1: Write the failing tests**

```python
"""Merging mined negatives into the training parquet: schema and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.merge_hard_negatives import bland_column_of, build_negative_rows

SEVEN = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col', 'olivine', 'lcp',
         'hcp', 'plagioclase', 'bland', 'alteration', 'junk',
         'confidence_weight', 'confidence_tier', 'split'] + \
        [f'band_{i:02d}' for i in range(59)]
FIVE = [c.replace('bland', 'other') for c in SEVEN]


def _neg(n=3):
    d = {'tile_id': ['t9001'] * n,
         'pixel_row': np.arange(n), 'pixel_col': np.arange(n)}
    for b in range(59):
        d[f'band_{b:02d}'] = np.full(n, 0.1, dtype=np.float32)
    d['RBR'] = np.full(n, 6.0)
    return pd.DataFrame(d)


def test_bland_column_is_detected_in_both_vocabularies():
    """The 7-class build calls it 'bland'; older parquets call it 'other'.
    Hard-coding either silently mislabels every mined pixel."""
    assert bland_column_of(SEVEN) == 'bland'
    assert bland_column_of(FIVE) == 'other'


def test_missing_bland_column_raises():
    with pytest.raises(ValueError, match='bland'):
        bland_column_of(['tile_id', 'olivine', 'lcp'])


def test_rows_are_labelled_bland_and_nothing_else():
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0)
    assert (out['bland'] == 1).all()
    for c in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration', 'junk'):
        assert (out[c] == 0).all(), f'{c} must be 0 on a dust negative'


def test_output_columns_match_the_target_schema_exactly():
    """A column order or set mismatch makes the concat produce NaN columns that
    train silently as zeros."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0)
    assert list(out.columns) == SEVEN


def test_polygon_ids_are_unique_and_offset():
    """Each mined pixel needs its own synthetic polygon so polygon_units can
    place it geographically; colliding with a real polygon_id would merge a dust
    pixel into a labelled unit."""
    out = build_negative_rows(_neg(4), SEVEN, 'bland', start_id=100)
    assert out['polygon_id'].nunique() == 4
    assert all(str(v).startswith('dustneg_') for v in out['polygon_id'])
    assert 'dustneg_100' in set(out['polygon_id'])


def test_bands_are_carried_through_unchanged():
    neg = _neg(2)
    neg['band_07'] = [0.31, 0.42]
    out = build_negative_rows(neg, SEVEN, 'bland', start_id=0)
    assert out['band_07'].tolist() == pytest.approx([0.31, 0.42])


def test_split_is_not_assigned_here():
    """Splits come from assign_unit_balanced_splits over the CONCATENATED frame.
    Writing 'train' here would put dust from val terrain into train."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0)
    assert out['split'].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_merge_hard_negatives.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.merge_hard_negatives'`

- [ ] **Step 3: Implement**

Create `scripts/merge_hard_negatives.py`:

```python
"""Merge mined dust hard negatives into the 7-class training parquet.

Runs on HPC, where mrral_pixels_7cls_handcore.parquet lives. Reads that file's
schema rather than assuming it, labels every mined pixel bland, and delegates
split assignment to split_units.assign_unit_balanced_splits over the CONCATENATED
frame -- so a mined negative near a val unit is absorbed into that unit and
follows its split. Writes a NEW parquet; the input stays an input.

Spec: docs/superpowers/specs/2026-08-17-dust-hard-negatives-design.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.split_units import assign_unit_balanced_splits  # noqa: E402

BLAND_CANDIDATES = ('bland', 'other')
MINERAL_COLS = ('olivine', 'olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                'plagioclase', 'alteration', 'junk')


def bland_column_of(columns) -> str:
    for c in BLAND_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(
        f'no bland column: tried {BLAND_CANDIDATES}, parquet has {list(columns)}')


def build_negative_rows(neg_df, target_columns, bland_col, start_id: int):
    """Mined pixels as rows matching `target_columns` exactly, labelled bland."""
    n = len(neg_df)
    out = pd.DataFrame(index=range(n))
    for col in target_columns:
        if col in ('tile_id', 'pixel_row', 'pixel_col') or col.startswith('band_'):
            out[col] = neg_df[col].to_numpy() if col in neg_df.columns else 0.0
        elif col == 'polygon_id':
            out[col] = [f'dustneg_{start_id + i}' for i in range(n)]
        elif col == bland_col:
            out[col] = np.ones(n, dtype=np.float32)
        elif col in MINERAL_COLS:
            out[col] = np.zeros(n, dtype=np.float32)
        elif col == 'split':
            out[col] = pd.Series([pd.NA] * n, dtype='object')
        elif col == 'confidence_weight':
            out[col] = np.ones(n, dtype=np.float32)
        elif col == 'confidence_tier':
            out[col] = 'dust_hard_negative'
        else:
            out[col] = np.zeros(n, dtype=np.float32)
    return out[list(target_columns)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', required=True, help='training parquet (input, untouched)')
    ap.add_argument('--negatives', required=True, help='parquet from mine_dust_hard_negatives')
    ap.add_argument('--out', required=True, help='NEW parquet to write')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    base = pd.read_parquet(args.labels)
    neg = pd.read_parquet(args.negatives)
    bland_col = bland_column_of(base.columns)
    print(f'base {len(base):,} rows; bland column is {bland_col!r}; '
          f'{len(neg):,} mined negatives')

    rows = build_negative_rows(neg, base.columns, bland_col, start_id=0)
    merged = pd.concat([base, rows], ignore_index=True)

    label_cols = [c for c in base.columns
                  if c in MINERAL_COLS or c == bland_col]
    merged['split'] = assign_unit_balanced_splits(merged, label_cols, seed=args.seed)
    print('split distribution after reassignment:')
    print(merged['split'].value_counts())
    print('mined-negative split distribution:')
    print(merged.iloc[len(base):]['split'].value_counts())

    merged.to_parquet(args.out, index=False)
    print(f'wrote {args.out}: {len(merged):,} rows')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_merge_hard_negatives.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Verify the tests can fail (mutation)**

```bash
cp scripts/merge_hard_negatives.py /tmp/merge_backup.py
python3 - <<'PY'
p='scripts/merge_hard_negatives.py'; s=open(p).read()
s=s.replace("    for c in BLAND_CANDIDATES:\n        if c in columns:\n            return c\n",
            "    return 'bland'\n")
open(p,'w').write(s)
PY
conda run -n crism python -m pytest tests/test_merge_hard_negatives.py -q
cp /tmp/merge_backup.py scripts/merge_hard_negatives.py
```
Expected: `test_bland_column_is_detected_in_both_vocabularies` and `test_missing_bland_column_raises` FAIL, then all pass after restore.

- [ ] **Step 6: Commit**

```bash
git add scripts/merge_hard_negatives.py tests/test_merge_hard_negatives.py
git commit -m "merge dust hard negatives into the training parquet

Reads the target parquet's schema instead of assuming it -- the 7-class build
calls the bland class 'bland', older parquets call it 'other', and hard-coding
either silently mislabels every mined pixel.

Splits come from assign_unit_balanced_splits over the CONCATENATED frame, never
assigned by hand: polygon_units links centroids at 0.25 deg and unions anything
sharing a pixel, so a mined negative near a val unit is absorbed into that unit
and follows its split. Writing split='train' directly would put dust from val
terrain into train."
```

---

### Task 3: Retrain job

**Files:**
- Create: `scripts/hpc_finetune_dualcr_hardneg.slurm`

**Interfaces:**
- Consumes: the merged parquet from Task 2.
- Produces: checkpoints named `ft_7cls_handcore_dualcr_hardneg*`.

- [ ] **Step 1: Copy the e87 job and rewrite the five anchors**

```bash
cp scripts/hpc_finetune_dualcr.slurm scripts/hpc_finetune_dualcr_hardneg.slurm
python3 - <<'PY'
p='scripts/hpc_finetune_dualcr_hardneg.slurm'; s=open(p).read()
subs=[('#SBATCH --job-name=dualcr_ft','#SBATCH --job-name=dualcr_ft_hardneg'),
      ('#SBATCH --output=logs/dualcr_ft_%j.log','#SBATCH --output=logs/dualcr_ft_hardneg_%j.log'),
      ('#SBATCH --error=logs/dualcr_ft_%j.log','#SBATCH --error=logs/dualcr_ft_hardneg_%j.log'),
      ('RUN_NAME=ft_7cls_handcore_dualcr_level','RUN_NAME=ft_7cls_handcore_dualcr_hardneg'),
      ('PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet',
       'PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore_hardneg.parquet'),
      ('CACHE_DIR=${DATA_DIR}/patch_cache_handcore_dualcr',
       'CACHE_DIR=${DATA_DIR}/patch_cache_handcore_dualcr_hardneg')]
for a,b in subs:
    assert s.count(a)==1, (a, s.count(a))
    s=s.replace(a,b)
open(p,'w').write(s)
print('rewrote 6 anchors')
PY
diff scripts/hpc_finetune_dualcr.slurm scripts/hpc_finetune_dualcr_hardneg.slurm
```
Expected: a six-line diff and nothing else.

- [ ] **Step 2: Verify the arm differs from e87 only in data**

```bash
grep -n "asl_loss\|weight_scheme\|encoder_lr_scale\|--lr \|--epochs\|--patience" scripts/hpc_finetune_dualcr_hardneg.slurm | tail -6
```
Expected: identical values to `scripts/hpc_finetune_dualcr.slurm` — `--asl_loss` with no clip override, `--weight_scheme level`, `--lr 5e-4`, `--encoder_lr_scale 0.001`, `--epochs 150`, `--patience 40`. If any differ, the arm confounds data with hyperparameters; fix before committing.

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc_finetune_dualcr_hardneg.slurm
git commit -m "hard-negative retrain arm: e87 job with the merged parquet

Differs from scripts/hpc_finetune_dualcr.slurm in six lines -- job name, log
paths, run name, parquet, cache dir -- and nothing else, so the only variable
against e87 is the training data."
```

---

## Notes for the executor

- The cache chain between Task 2 and Task 3 is existing tooling and is NOT part of this plan: `cache_mrral_patches.py` → raw cache → `build_cr_labeled_cache.py --dual` → `patch_cache_handcore_dualcr_hardneg`. Run it on HPC before submitting Task 3's job. The job's own preflight will fail loudly if the cache is missing or the wrong width.
- Evaluation criteria live in the spec and are shared with the gate plan: t1321 false share 35% → target <10%; t1249 confident-lcp must not fall >15%; floor test compared in **pixels retained**, not polygon counts.
- HPC data goes on xdisk (`/xdisk/sbyrne/phillipsm/CRISM_MRDR`); `/groups` is code + env only. rsync via the DTN: `rsync -avh --progress --partial <src> phillipsm@filexfer.hpc.arizona.edu:<dest>/`.
