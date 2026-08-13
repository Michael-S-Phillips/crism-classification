# Floor-Test Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the deep model a comparator on the floor tiles — an expert
band-parameter ruleset and two classical ML models — by making each of them
another producer of the probs npz the floor test already consumes.

**Architecture:** Everything converges on the `<tid>_probs.npz` contract written
by `scripts/classify_tile_supervised.py:456`. Nothing downstream changes:
`floor_test.sh`, the threshold ladder, the median smoothing, and the summary
tables are all reused unchanged, so any difference in polygon counts is
attributable to the method rather than the plumbing.

**Tech Stack:** numpy, rasterio, pandas/pyarrow, scikit-learn (already present —
`RandomForestClassifier`, `HistGradientBoostingClassifier`), joblib.

**Spec:** `docs/superpowers/specs/2026-08-13-floor-test-baselines-design.md`

## Global Constraints

- Conda env `crism`. Tests: `conda run -n crism python -m pytest <file> -v`.
- `tests/test_train_torch.py` has **3 pre-existing failures** at HEAD (5-vs-6
  class label-width `ValueError`). Do not fix them; confirm the count is
  unchanged.
- The vocabularies accepted downstream are exactly
  `['olivine','lcp','hcp','plagioclase','bland','alteration','junk']` (7-class)
  and `['olivine','pyx','plagioclase','bland','alteration','junk']` (pyx).
  Anything else must raise.
- **The vocabulary is MULTI-LABEL.** A pixel can be olivine AND hcp. No rule may
  force exclusivity between co-occurring minerals. Hard vetoes are for artifacts
  only; cross-response between indices is handled as a tier modifier.
- `valid_mask` MUST come from `classify_tile_supervised.load_tile`, imported, not
  reimplemented.
- CRISM nodata is `65535`. mrrsu values equal to it become `NaN`.
- Reuse `data/mrrsu_aux.py`'s `BAND_VALID_RANGES` and `NODATA`; do not redefine.
- Calibration reads the **train** split only.
- Every test must be seen FAILING under a mutation of the code it covers before
  being accepted. Eleven tests in this project have shipped unable to fail for
  their stated reason.

---

## File Structure

| file | responsibility |
|---|---|
| `data/mrrsu_bands.py` | band-name→index registry, cube reader, nodata→NaN |
| `data/expert_rules.py` | pure rule evaluation: cube + config → per-class score |
| `scripts/extract_mrrsu_features.py` | 60 params at each labeled pixel → sidecar parquet |
| `scripts/fit_expert_rules.py` | calibrate thresholds + precision ladder → JSON |
| `scripts/fit_ml_baseline.py` | train RF + HistGB → joblib |
| `scripts/classify_tile_baseline.py` | any artifact + tile → probs npz |
| `scripts/atmos_diagnostic.py` | hcp detection rate by elevation / air-mass decile |
| `scripts/floor_test.sh` | one `CLASSIFY_CMD` env hook |

---

### Task 1: mrrsu band registry and cube reader

**Files:**
- Create: `data/mrrsu_bands.py`
- Test: `tests/test_mrrsu_bands.py`

**Interfaces:**
- Consumes: `data/mrrsu_aux.py` → `NODATA` (65535.0), `BAND_VALID_RANGES`
- Produces:
  - `read_band_names(hdr_path: str) -> list[str]`
  - `band_index(names: list[str], param: str) -> int`
  - `read_mrrsu_cube(img_path: str) -> tuple[np.ndarray, list[str]]` — returns
    `(H, W, 60) float32` with nodata as `NaN`, and the band-name list
  - `CORE_INDICES: dict[str, int]` — the four documented in `CLAUDE.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mrrsu_bands.py
import glob, os
import numpy as np
import pytest
from config_loader import load_config
from data.mrrsu_bands import (read_band_names, band_index, read_mrrsu_cube,
                              CORE_INDICES)

def _a_real_mrrsu_hdr():
    root = load_config()['data_root']
    hits = sorted(glob.glob(os.path.join(root, 'mc*', 't*mrrsu*.hdr'))
                  + glob.glob(os.path.join(root, 't*mrrsu*.hdr')))
    if not hits:
        pytest.skip('no mrrsu tile available locally')
    return hits[0]

def test_core_indices_match_documented_values():
    """CLAUDE.md documents these; a tile with a different band order must not
    silently shift them."""
    names = read_band_names(_a_real_mrrsu_hdr())
    assert len(names) == 60
    assert band_index(names, 'OLINDEX3') == 15
    assert band_index(names, 'BD1300') == 17
    assert band_index(names, 'LCPINDEX2') == 18
    assert band_index(names, 'HCPINDEX2') == 19
    assert CORE_INDICES == {'OLINDEX3': 15, 'BD1300': 17,
                            'LCPINDEX2': 18, 'HCPINDEX2': 19}

def test_unknown_param_raises_naming_it():
    names = read_band_names(_a_real_mrrsu_hdr())
    with pytest.raises(KeyError, match='NOSUCHPARAM'):
        band_index(names, 'NOSUCHPARAM')

def test_nodata_becomes_nan_not_65535():
    """65535 left as a number would poison every threshold comparison."""
    hdr = _a_real_mrrsu_hdr()
    cube, names = read_mrrsu_cube(hdr.replace('.hdr', '.img'))
    assert cube.dtype == np.float32
    assert cube.shape[-1] == 60
    assert not np.any(cube == 65535.0)
    assert np.isnan(cube).any()   # real tiles always have some nodata
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_mrrsu_bands.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'data.mrrsu_bands'`

- [ ] **Step 3: Implement**

```python
# data/mrrsu_bands.py
"""Band-name registry and cube reader for mrrsu summary-parameter tiles.

Indices are resolved from the tile's OWN header rather than hardcoded: a tile
written with a different band order would otherwise silently shift every
parameter, and a band-depth threshold applied to the wrong band produces a
plausible map with no error.
"""
from __future__ import annotations

import re

import numpy as np
import rasterio

from data.mrrsu_aux import NODATA

N_MRRSU_BANDS = 60

# Documented in CLAUDE.md; asserted against a real header in the tests so a
# reordered product is caught rather than absorbed.
CORE_INDICES = {'OLINDEX3': 15, 'BD1300': 17, 'LCPINDEX2': 18, 'HCPINDEX2': 19}


def read_band_names(hdr_path: str) -> list[str]:
    txt = open(hdr_path).read()
    m = re.search(r'band names\s*=\s*\{(.*?)\}', txt, re.S)
    if not m:
        raise ValueError(f'{hdr_path}: no "band names" block')
    return [n.strip() for n in m.group(1).replace('\n', ' ').split(',') if n.strip()]


def band_index(names: list[str], param: str) -> int:
    try:
        return names.index(param)
    except ValueError:
        raise KeyError(
            f'{param} not among the {len(names)} mrrsu band names') from None


def read_mrrsu_cube(img_path: str) -> tuple[np.ndarray, list[str]]:
    """(H, W, 60) float32 with nodata as NaN, plus the band-name list."""
    names = read_band_names(img_path.replace('.img', '.hdr'))
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_MRRSU_BANDS + 1))).astype(np.float32)
    data[(data == NODATA) | ~np.isfinite(data)] = np.nan
    return data.transpose(1, 2, 0), names
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_mrrsu_bands.py -v`
Expected: 3 passed

- [ ] **Step 5: Mutation-verify**

Change `data[(data == NODATA) | ...] = np.nan` to `data[~np.isfinite(data)] = np.nan`.
Run the tests. Expected: `test_nodata_becomes_nan_not_65535` FAILS on
`assert not np.any(cube == 65535.0)`. Revert with `cp` from a backup.

- [ ] **Step 6: Commit**

```bash
git add data/mrrsu_bands.py tests/test_mrrsu_bands.py
git commit -m "Add mrrsu band registry and cube reader with nodata->NaN"
```

---

### Task 2: Extract mrrsu features at labeled pixels

**Files:**
- Create: `scripts/extract_mrrsu_features.py`
- Test: `tests/test_extract_mrrsu_features.py`

**Interfaces:**
- Consumes: `data.mrrsu_bands.read_mrrsu_cube`
- Produces: a parquet whose feature columns are the **real parameter names**
  (`OLINDEX3`, `BD1300`, …) plus `tile_id`, `pixel_row`, `pixel_col`, `split`,
  **row-aligned with the input parquet**, and
  `extract_features(df, mrrsu_map, smooth=False, reader=None) -> pd.DataFrame`.
  Naming columns by parameter rather than positionally means no downstream stage
  needs a header to decode them, and a reordered product cannot silently shift
  which index a threshold is applied to.

**Why row alignment gets a real test:** the MTRDR plag caches had exactly this
bug class. A misaligned sidecar attaches every label to the wrong pixel's
parameters and yields a plausible but meaningless baseline, with no error.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract_mrrsu_features.py
import numpy as np
import pandas as pd
import pytest
from scripts.extract_mrrsu_features import extract_features

class FakeCube:
    """Cube whose value encodes its own (row, col) so misalignment is provable."""
    def __init__(self, h=40, w=50):
        self.h, self.w = h, w
    def read(self, tile_id):
        cube = np.zeros((self.h, self.w, 60), dtype=np.float32)
        rr, cc = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing='ij')
        cube[..., 0] = rr * 1000 + cc          # unique, recoverable fingerprint
        return cube, [f'B{i}' for i in range(60)]

def _df():
    rows = [dict(tile_id='t0001', pixel_row=r, pixel_col=c, split='train')
            for r, c in [(3, 7), (11, 2), (0, 0), (39, 49), (20, 20)]]
    return pd.DataFrame(rows)

def test_rows_align_with_the_input_parquet(monkeypatch):
    df = _df()
    out = extract_features(df, {'t0001': 'fake.img'}, reader=FakeCube().read)
    assert len(out) == len(df)
    # p0 encodes row*1000+col; every row must recover ITS OWN coordinates.
    for i, r in df.iterrows():
        assert out['B0'].iloc[i] == r['pixel_row'] * 1000 + r['pixel_col'], (
            f'row {i} got another pixel\'s parameters')

def test_out_of_bounds_pixel_is_nan_not_wrapped(monkeypatch):
    """Negative or oversized indices must not silently wrap to a valid pixel."""
    df = pd.DataFrame([dict(tile_id='t0001', pixel_row=999, pixel_col=0,
                            split='train')])
    out = extract_features(df, {'t0001': 'fake.img'}, reader=FakeCube().read)
    assert np.isnan(out['B0'].iloc[0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_extract_mrrsu_features.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/extract_mrrsu_features.py
"""Extract the 60 mrrsu summary parameters at each labeled pixel.

Row order is preserved EXACTLY: the output is aligned row-for-row with the input
parquet, because downstream code joins them positionally. A reorder or a dropped
row attaches labels to the wrong pixel's parameters and produces a plausible but
meaningless baseline.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mrrsu_bands import N_MRRSU_BANDS, read_mrrsu_cube  # noqa: E402

# Columns are named by PARAMETER, not position: no downstream stage then needs
# a header to decode them, and a reordered product cannot silently shift which
# index a threshold is applied to.


def _smooth_nanmean(cube: np.ndarray, size: int = 7) -> np.ndarray:
    """7x7 mean ignoring NaN. RPEAK1 is documented as a REGIONAL discriminant
    in data/mrrsu_aux.py, so a per-pixel read understates it."""
    filled = np.nan_to_num(cube, nan=0.0)
    valid = np.isfinite(cube).astype(np.float32)
    num = uniform_filter(filled, size=(size, size, 1), mode='nearest')
    den = uniform_filter(valid, size=(size, size, 1), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den == 0] = np.nan
    return out.astype(np.float32)


def extract_features(df: pd.DataFrame, mrrsu_map: dict[str, str],
                     smooth: bool = False, reader=None) -> pd.DataFrame:
    reader = reader or (lambda tid: read_mrrsu_cube(mrrsu_map[tid]))
    out = np.full((len(df), N_MRRSU_BANDS), np.nan, dtype=np.float32)
    col_names: list[str] | None = None
    rows = df['pixel_row'].to_numpy(np.int64)
    cols = df['pixel_col'].to_numpy(np.int64)
    for tid, idx in df.groupby('tile_id', sort=False).indices.items():
        if tid not in mrrsu_map and reader is None:
            continue
        cube, names = reader(tid)
        if col_names is None:
            col_names = list(names)
        elif list(names) != col_names:
            raise ValueError(
                f'{tid}: band order differs from earlier tiles — a threshold '
                f'would be applied to the wrong parameter')
        if smooth:
            cube = _smooth_nanmean(cube)
        h, w = cube.shape[:2]
        r, c = rows[idx], cols[idx]
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        # Positional assignment into `idx` preserves input row order exactly.
        out[idx[ok]] = cube[r[ok], c[ok], :]
    if col_names is None:
        raise ValueError('no tile was read; cannot name the feature columns')
    return pd.DataFrame(out, columns=col_names, index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smooth', action='store_true',
                    help='7x7 nan-aware mean; matches the deep model receptive '
                         'field and the RPEAK1 regional note in mrrsu_aux.py')
    ap.add_argument('--data_root', default=None)
    args = ap.parse_args()

    from config_loader import load_config
    root = args.data_root or load_config()['data_root']
    hdrs = sorted(set(glob.glob(os.path.join(root, 'mc*', 't*mrrsu*.hdr'))
                      + glob.glob(os.path.join(root, 't*mrrsu*.hdr'))))
    mrrsu_map = {os.path.basename(h).split('_mrrsu_')[0]: h.replace('.hdr', '.img')
                 for h in hdrs}

    df = pd.read_parquet(args.parquet,
                         columns=['tile_id', 'pixel_row', 'pixel_col', 'split'])
    print(f'{len(df):,} rows, {df.tile_id.nunique()} tiles, '
          f'{sum(t in mrrsu_map for t in df.tile_id.unique())} with an mrrsu tile')
    feats = extract_features(df, mrrsu_map, smooth=args.smooth)
    result = pd.concat([df.reset_index(drop=True),
                        feats.reset_index(drop=True)], axis=1)
    assert len(result) == len(df), 'row count changed — alignment broken'
    result.to_parquet(args.out, index=False)
    n_all_nan = int(np.isnan(feats.to_numpy()).all(axis=1).sum())
    print(f'wrote {args.out}  ({n_all_nan:,} rows all-NaN — no mrrsu coverage)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_extract_mrrsu_features.py -v`
Expected: 2 passed

- [ ] **Step 5: Mutation-verify**

Replace `out[idx[ok]] = cube[r[ok], c[ok], :]` with
`out[idx[ok]] = cube[c[ok], r[ok], :]` (row/col transposed).
Expected: `test_rows_align_with_the_input_parquet` FAILS. Revert via `cp`.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_mrrsu_features.py tests/test_extract_mrrsu_features.py
git commit -m "Extract 60 mrrsu parameters at labeled pixels, row-aligned"
```

---

### Task 3: Expert rule engine

**Files:**
- Create: `data/expert_rules.py`
- Test: `tests/test_expert_rules.py`

**Interfaces:**
- Consumes: `data.mrrsu_bands.band_index`
- Produces:
  - `evaluate_rules(cube, names, config) -> dict[str, np.ndarray]` — per-class
    `(H, W) float32` score in [0, 1]
  - `DEFAULT_RULES: dict` — the ruleset structure with thresholds as `None`
    until calibrated

**This is the heart of the spec.** The regression it exists to prevent is
exclusivity: a pixel satisfying both olivine and hcp must receive **both**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expert_rules.py
import numpy as np
import pytest
from data.expert_rules import evaluate_rules, DEFAULT_RULES

NAMES = ['R770','RBR','BD530_2','SH600_2','SH770','BD640_2','BD860_2','BD920_2',
         'RPEAK1','BDI1000VIS','R440','IRR1','R530','R600','BDI1000IR','OLINDEX3',
         'R1330','BD1300','LCPINDEX2','HCPINDEX2','VAR','ISLOPE1','BD1400',
         'BD1435','BD1500_2','ICER1_2','BD1750_2','BD1900_2','BD1900R2','BDI2000',
         'BD2100_2','BD2165','BD2190','MIN2200','BD2210_2','D2200','BD2230',
         'BD2250','MIN2250','BD2265','BD2290','D2300','BD2355','SINDEX2','ICER2_2',
         'MIN2295_2480','MIN2345_2537','BD2500_2','BD3000','BD3100','BD3200',
         'BD3400_2','CINDEX2','BD2600','IRR2','IRR3','R1080','R1506','R2529','R3920']

def _cfg():
    """Calibrated config with simple round thresholds, one pixel per scenario."""
    import copy
    c = copy.deepcopy(DEFAULT_RULES)
    for cls in ('olivine', 'lcp', 'hcp'):
        c['classes'][cls]['primary']['threshold'] = 0.05
        c['classes'][cls]['ladder'] = [[0.05, 0.6], [0.10, 0.9]]
    c['classes']['plagioclase']['rpeak1_window'] = [0.70, 0.80]
    c['classes']['plagioclase']['primary']['threshold'] = 0.01
    c['classes']['plagioclase']['hydration_veto'] = 0.10
    c['classes']['plagioclase']['ladder'] = [[0.01, 0.7]]
    c['junk']['icer_high'] = 0.5
    c['junk']['var_high'] = 1e6
    c['junk']['r770_max'] = 1.0
    for g in c['classes']['alteration']['groups']:
        g['thresholds'] = {k: 0.05 for k in g['requires']}
    c['classes']['alteration']['ladder'] = [[0.05, 0.8]]
    return c

def _blank(n=1):
    return np.zeros((1, n, 60), dtype=np.float32)

def _set(cube, param, value, i=0):
    cube[0, i, NAMES.index(param)] = value

def test_olivine_and_hcp_can_BOTH_fire():
    """THE multi-label guarantee. Olivine-bearing basalt is ordinary; an
    exclusive veto would suppress it silently."""
    cube = _blank()
    _set(cube, 'OLINDEX3', 0.20)
    _set(cube, 'HCPINDEX2', 0.20)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['olivine'][0, 0] > 0, 'olivine suppressed by pyroxene presence'
    assert out['hcp'][0, 0] > 0, 'hcp suppressed by olivine presence'

def test_dominance_raises_the_tier_but_does_not_gate():
    """Both pyroxene labels fire when both indices are high; the dominant one
    scores higher. It must NOT zero the weaker one."""
    cube = _blank()
    _set(cube, 'LCPINDEX2', 0.20)
    _set(cube, 'HCPINDEX2', 0.08)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['hcp'][0, 0] > 0, 'non-dominant pyroxene was gated to zero'
    assert out['lcp'][0, 0] > out['hcp'][0, 0], 'dominance did not raise the tier'

def test_carbonate_without_hydration_is_still_alteration():
    """Carbonates are anhydrous. A blanket hydration requirement would silently
    reject Mg-carbonate — the Nili Fossae terrain this project exists for."""
    cube = _blank()
    _set(cube, 'BD2500_2', 0.20)
    _set(cube, 'D2300', 0.20)
    _set(cube, 'BD1900R2', 0.0)      # explicitly NOT hydrated
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['alteration'][0, 0] > 0

def test_ice_is_junk_and_not_alteration():
    cube = _blank()
    _set(cube, 'ICER1_2', 0.9)
    _set(cube, 'D2300', 0.20)
    _set(cube, 'BD2290', 0.20)
    _set(cube, 'BD1900R2', 0.20)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['junk'][0, 0] > 0
    assert out['alteration'][0, 0] == 0, 'frost registered as alteration'

def test_plagioclase_needs_rpeak1_inside_the_window_not_merely_high():
    """RPEAK1 is a WAVELENGTH (~0.7-0.8 um for plag), not an amplitude. A
    one-sided 'high' test would admit everything above 0.8."""
    cube = _blank(2)
    for i in (0, 1):
        _set(cube, 'BD1300', 0.05, i)
        _set(cube, 'R770', 0.2, i)
    _set(cube, 'RPEAK1', 0.75, 0)    # inside
    _set(cube, 'RPEAK1', 0.95, 1)    # above the window
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['plagioclase'][0, 0] > 0
    assert out['plagioclase'][0, 1] == 0

def test_bland_is_the_residual():
    cube = _blank()
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['bland'][0, 0] > 0
    for c in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration'):
        assert out[c][0, 0] == 0

def test_all_nan_pixel_scores_zero_everywhere():
    cube = np.full((1, 1, 60), np.nan, dtype=np.float32)
    out = evaluate_rules(cube, NAMES, _cfg())
    for c, v in out.items():
        assert v[0, 0] == 0, f'{c} scored on an all-NaN pixel'
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_expert_rules.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'data.expert_rules'`

- [ ] **Step 3: Implement**

```python
# data/expert_rules.py
"""Expert band-parameter rules over the CRISM summary parameters.

Structure is fixed by mineralogy and never fitted; only the cut points are
calibrated (scripts/fit_expert_rules.py). Two kinds of gate:

  VETO (hard)      artifacts and genuinely incompatible conditions only --
                   ice, saturation, non-physical values, dust for plagioclase.
  DOMINANCE (soft) cross-responding index pairs raise the TIER of the dominant
                   label without zeroing the other.

The vocabulary is MULTI-LABEL: a pixel can be olivine AND hcp. Exclusive gates
would fight the label structure and silently suppress real assemblages, so there
are none between co-occurring minerals.
"""
from __future__ import annotations

import copy

import numpy as np

CLASSES_7 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
CLASSES_PYX = ['olivine', 'pyx', 'plagioclase', 'bland', 'alteration', 'junk']

DEFAULT_RULES = {
    'vocab': CLASSES_7,
    'junk': {'icer_high': None, 'var_high': None, 'r770_max': None},
    'classes': {
        'olivine':     {'primary': {'param': 'OLINDEX3',  'threshold': None},
                        'ladder': None},
        'lcp':         {'primary': {'param': 'LCPINDEX2', 'threshold': None},
                        'dominance_over': 'HCPINDEX2', 'ladder': None},
        'hcp':         {'primary': {'param': 'HCPINDEX2', 'threshold': None},
                        'dominance_over': 'LCPINDEX2', 'ladder': None},
        'plagioclase': {'primary': {'param': 'BD1300', 'threshold': None},
                        'rpeak1_window': None, 'hydration_veto': None,
                        'ladder': None},
        'alteration':  {'groups': [
                            {'name': 'femg_phyllosilicate',
                             'requires': ['D2300', 'BD2290', 'BD1900R2']},
                            {'name': 'al_phyllosilicate',
                             'requires': ['BD2210_2', 'BD1900R2']},
                            {'name': 'hydrated_silica',
                             'requires': ['MIN2200', 'BD1900R2']},
                            {'name': 'sulfate',
                             'requires': ['SINDEX2', 'BD1900R2']},
                            # Anhydrous: NO BD1900R2. Requiring hydration here
                            # would silently reject Mg-carbonate.
                            {'name': 'carbonate',
                             'requires': ['BD2500_2', 'D2300']},
                        ], 'ladder': None},
    },
}


def _p(cube: np.ndarray, names: list[str], param: str) -> np.ndarray:
    return cube[..., names.index(param)]


def _tier_score(value: np.ndarray, fires: np.ndarray, ladder) -> np.ndarray:
    """Map a firing pixel to the precision of the highest rung it clears."""
    out = np.zeros(value.shape, dtype=np.float32)
    for thresh, precision in sorted(ladder, key=lambda r: r[0]):
        out = np.where(fires & (value >= thresh), np.float32(precision), out)
    return out


def evaluate_rules(cube: np.ndarray, names: list[str], config: dict
                   ) -> dict[str, np.ndarray]:
    cfg = config['classes']
    shape = cube.shape[:2]
    finite = np.isfinite(cube).any(axis=-1)

    # ── junk: artifacts and ice ──────────────────────────────────────────────
    jc = config['junk']
    icer = np.fmax(np.nan_to_num(_p(cube, names, 'ICER1_2'), nan=0.0),
                   np.nan_to_num(_p(cube, names, 'ICER2_2'), nan=0.0))
    co2_ice = np.fmax(np.nan_to_num(_p(cube, names, 'BD1435'), nan=0.0),
                      np.nan_to_num(_p(cube, names, 'BD3200'), nan=0.0))
    r770 = _p(cube, names, 'R770')
    var = np.nan_to_num(_p(cube, names, 'VAR'), nan=0.0)
    is_junk = ((icer >= jc['icer_high']) | (co2_ice >= jc['icer_high'])
               | (np.nan_to_num(r770, nan=0.0) > jc['r770_max'])
               | (var >= jc['var_high'])) & finite
    ok = finite & ~is_junk

    out: dict[str, np.ndarray] = {}

    # ── mafic minerals: own evidence only, no exclusivity ────────────────────
    for cls in ('olivine', 'lcp', 'hcp'):
        if cls not in cfg:
            continue
        c = cfg[cls]
        v = np.nan_to_num(_p(cube, names, c['primary']['param']), nan=-np.inf)
        fires = ok & (v >= c['primary']['threshold'])
        score = _tier_score(v, fires, c['ladder'])
        # Dominance raises the tier; it never gates. A basalt with both LCP and
        # HCP must receive both labels.
        dom = c.get('dominance_over')
        if dom is not None:
            other = np.nan_to_num(_p(cube, names, dom), nan=-np.inf)
            score = np.where(fires & (v > other), score,
                             np.where(fires, score * np.float32(0.5), score))
        out[cls] = score.astype(np.float32)

    # ── plagioclase: RPEAK1 window + BD1300, dust vetoed ─────────────────────
    if 'plagioclase' in cfg:
        c = cfg['plagioclase']
        lo, hi = c['rpeak1_window']
        rp = _p(cube, names, 'RPEAK1')
        bd = np.nan_to_num(_p(cube, names, 'BD1300'), nan=-np.inf)
        hyd = np.nan_to_num(_p(cube, names, 'BD1900R2'), nan=0.0)
        in_window = np.isfinite(rp) & (rp >= lo) & (rp <= hi)
        fires = ok & in_window & (bd >= c['primary']['threshold']) \
            & (hyd < c['hydration_veto'])
        out['plagioclase'] = _tier_score(bd, fires, c['ladder']).astype(np.float32)

    # ── alteration: disjunction of specific groups, ice vetoed ───────────────
    if 'alteration' in cfg:
        c = cfg['alteration']
        any_group = np.zeros(shape, dtype=bool)
        strength = np.zeros(shape, dtype=np.float32)
        for g in c['groups']:
            hit = ok.copy()
            gmin = np.full(shape, np.inf, dtype=np.float32)
            for param in g['requires']:
                v = np.nan_to_num(_p(cube, names, param), nan=-np.inf)
                hit &= v >= g['thresholds'][param]
                gmin = np.minimum(gmin, v)
            any_group |= hit
            strength = np.where(hit, np.maximum(strength, gmin), strength)
        out['alteration'] = _tier_score(strength, any_group,
                                        c['ladder']).astype(np.float32)

    out['junk'] = is_junk.astype(np.float32)

    mineral_keys = [k for k in out if k not in ('junk',)]
    any_mineral = np.zeros(shape, dtype=bool)
    for k in mineral_keys:
        any_mineral |= out[k] > 0
    out['bland'] = (ok & ~any_mineral).astype(np.float32)

    return {k: out[k] for k in config['vocab']}
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_expert_rules.py -v`
Expected: 7 passed

- [ ] **Step 5: Mutation-verify the multi-label guarantee**

Add `& (np.nan_to_num(_p(cube, names, 'HCPINDEX2'), nan=0.0) < 0.05)` to the
olivine `fires` expression (reintroducing the exclusive veto).
Expected: `test_olivine_and_hcp_can_BOTH_fire` FAILS.

Then change the dominance line to `np.where(fires & (v > other), score, 0.0)`
(gating instead of demoting).
Expected: `test_dominance_raises_the_tier_but_does_not_gate` FAILS.

Then add `'BD1900R2'` to the carbonate `requires` list.
Expected: `test_carbonate_without_hydration_is_still_alteration` FAILS.

Revert each via `cp` from a backup; verify with `diff`.

- [ ] **Step 6: Commit**

```bash
git add data/expert_rules.py tests/test_expert_rules.py
git commit -m "Add expert band-parameter rule engine, multi-label by construction"
```

---

### Task 4: Calibrate the ruleset

**Files:**
- Create: `scripts/fit_expert_rules.py`
- Test: `tests/test_fit_expert_rules.py`

**Interfaces:**
- Consumes: `data.expert_rules.DEFAULT_RULES`, the Task-2 sidecar parquet
- Produces: `config/expert_rules_7cls.json`, `config/expert_rules_pyx.json`; and
  `calibrate(feat_df, labels, vocab, retention=0.90) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit_expert_rules.py
import numpy as np
import pandas as pd
import pytest
from scripts.fit_expert_rules import calibrate, precision_ladder

def test_precision_ladder_is_non_decreasing_on_separable_data():
    """A higher threshold sees a purer subset, so precision should not fall.
    A violation means the index is badly behaved and must be REPORTED."""
    score = np.concatenate([np.linspace(0, 0.5, 500), np.linspace(0.5, 1, 500)])
    y = (score > 0.5).astype(int)
    ladder = precision_ladder(score, y, np.linspace(0.1, 0.9, 9))
    precisions = [p for _t, p in ladder]
    assert precisions == sorted(precisions), f'non-monotonic: {precisions}'

def test_veto_retention_floor_is_respected():
    """A veto must not silently annihilate its own class."""
    rng = np.random.default_rng(0)
    n = 1000
    feat = pd.DataFrame({'OLINDEX3': rng.random(n), 'ICER1_2': rng.random(n)})
    y = (feat['OLINDEX3'] > 0.7).astype(int).to_numpy()
    cfg = calibrate(feat, {'olivine': y}, vocab=['olivine'], retention=0.90)
    veto = cfg['junk']['icer_high']
    kept = (feat.loc[y == 1, 'ICER1_2'] < veto).mean()
    assert kept >= 0.90, f'veto retained only {kept:.2%} of olivine positives'

def test_calibration_uses_only_the_rows_it_is_given():
    """Fitting must not reach past its argument into a global split."""
    rng = np.random.default_rng(1)
    feat = pd.DataFrame({'OLINDEX3': rng.random(200), 'ICER1_2': np.zeros(200)})
    y = (feat['OLINDEX3'] > 0.5).astype(int).to_numpy()
    a = calibrate(feat.iloc[:100], {'olivine': y[:100]}, ['olivine'])
    b = calibrate(feat.iloc[:100], {'olivine': y[:100]}, ['olivine'])
    assert a == b, 'calibration is not deterministic on identical input'
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_fit_expert_rules.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/fit_expert_rules.py
"""Calibrate the expert ruleset on the TRAIN split.

Expert structure, data-fitted cut points. The logical form is never searched
over, so this cannot overfit into an uninterpretable rule; only thresholds move.

Each veto is placed so it RETAINS a specified fraction of that class's own
positives (default 90%), which makes it structurally unable to silently
annihilate a class. Each ladder rung carries its empirical precision, and that
precision is the probability written to the probs npz -- so a ladder position
means "this rule at this strictness is right p% of the time".
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.expert_rules import CLASSES_7, CLASSES_PYX, DEFAULT_RULES  # noqa: E402

LADDER_QUANTILES = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]


def precision_ladder(score: np.ndarray, y: np.ndarray,
                     thresholds) -> list[list[float]]:
    """[(threshold, precision), ...] — precision of the rule at each strictness."""
    out = []
    for t in thresholds:
        sel = score >= t
        n = int(sel.sum())
        prec = float(y[sel].mean()) if n else 0.0
        out.append([float(t), round(prec, 4)])
    return out


def _veto_threshold(values: np.ndarray, retention: float) -> float:
    """Place the veto so `retention` of this class's positives survive it."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float('inf')
    return float(np.quantile(v, retention))


def calibrate(feat: pd.DataFrame, labels: dict[str, np.ndarray],
              vocab: list[str], retention: float = 0.90) -> dict:
    cfg = copy.deepcopy(DEFAULT_RULES)
    cfg['vocab'] = list(vocab)
    cfg['retention'] = retention

    pos_any = np.zeros(len(feat), dtype=bool)
    for y in labels.values():
        pos_any |= y.astype(bool)

    # Vetoes: retain `retention` of the positives of any class.
    for name, param in (('icer_high', 'ICER1_2'), ('var_high', 'VAR'),
                        ('r770_max', 'R770')):
        if param in feat.columns and pos_any.any():
            cfg['junk'][name] = _veto_threshold(
                feat.loc[pos_any, param].to_numpy(), retention)
        else:
            cfg['junk'][name] = float('inf')

    for cls in vocab:
        if cls in ('bland', 'junk') or cls not in cfg['classes']:
            continue
        y = labels.get(cls)
        if y is None or y.sum() == 0:
            cfg['classes'][cls]['ladder'] = [[0.0, 0.0]]
            continue
        c = cfg['classes'][cls]
        if cls == 'alteration':
            for g in c['groups']:
                g['thresholds'] = {
                    p: float(np.nanquantile(feat.loc[y == 1, p], 1 - retention))
                    for p in g['requires'] if p in feat.columns}
            strength = feat[[p for p in feat.columns
                             if p in {q for g in c['groups']
                                      for q in g['requires']}]].min(axis=1).to_numpy()
            score = np.nan_to_num(strength, nan=-np.inf)
        else:
            param = c['primary']['param']
            score = np.nan_to_num(feat[param].to_numpy(), nan=-np.inf)
            c['primary']['threshold'] = float(
                np.nanquantile(feat.loc[y == 1, param], 1 - retention))
            if cls == 'plagioclase':
                rp = feat.loc[y == 1, 'RPEAK1']
                c['rpeak1_window'] = [float(np.nanquantile(rp, 0.05)),
                                      float(np.nanquantile(rp, 0.95))]
                c['hydration_veto'] = _veto_threshold(
                    feat.loc[y == 1, 'BD1900R2'].to_numpy(), retention)
        qs = [float(np.nanquantile(score[np.isfinite(score)], q))
              for q in LADDER_QUANTILES]
        ladder = precision_ladder(score, y, sorted(set(qs)))
        precisions = [p for _t, p in ladder]
        if precisions != sorted(precisions):
            print(f'  WARNING {cls}: precision is NON-MONOTONIC along the '
                  f'ladder {precisions} — the index is badly behaved here')
        c['ladder'] = ladder
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True, help='Task-2 sidecar parquet')
    ap.add_argument('--labels', required=True, help='labeled parquet')
    ap.add_argument('--vocab', choices=('7cls', 'pyx'), default='7cls')
    ap.add_argument('--retention', type=float, default=0.90)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    from data.dataset import _collapse_labels

    feat = pd.read_parquet(args.features)
    lab = _collapse_labels(pd.read_parquet(args.labels))
    assert len(feat) == len(lab), (
        f'feature rows {len(feat):,} != label rows {len(lab):,} — the sidecar '
        f'is not aligned with the labels')
    train = (lab['split'] == 'train').to_numpy()
    print(f'calibrating on {train.sum():,} TRAIN rows (of {len(lab):,})')

    vocab = CLASSES_7 if args.vocab == '7cls' else CLASSES_PYX
    # Feature columns already carry real parameter names (Task 2), so the
    # emitted JSON is human-auditable without a decoding step.
    fx = feat.loc[train]
    labels = {c: lab.loc[train, c].to_numpy() for c in vocab
              if c in lab.columns}
    cfg = calibrate(fx, labels, vocab, args.retention)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_fit_expert_rules.py -v`
Expected: 3 passed

- [ ] **Step 5: Mutation-verify**

Change `_veto_threshold` to `return float(np.quantile(v, 1 - retention))`.
Expected: `test_veto_retention_floor_is_respected` FAILS with a retention well
below 0.90. Revert via `cp`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fit_expert_rules.py tests/test_fit_expert_rules.py
git commit -m "Calibrate expert rules: veto retention floors and precision ladders"
```

---

### Task 5: Classical ML baselines

**Files:**
- Create: `scripts/fit_ml_baseline.py`
- Test: `tests/test_fit_ml_baseline.py`

**Interfaces:**
- Consumes: the Task-2 sidecar parquet
- Produces: `<out>/rf.joblib`, `<out>/histgb.joblib`, `<out>/meta.json` with
  `{'vocab': [...], 'feature_cols': [...], 'model': 'rf'|'histgb'}`; and
  `predict_proba_multilabel(model, X, n_classes) -> np.ndarray (N, C)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit_ml_baseline.py
import numpy as np
import pytest
from scripts.fit_ml_baseline import fit_models, predict_proba_multilabel

def _separable(n=400):
    rng = np.random.default_rng(0)
    X = rng.random((n, 6)).astype(np.float32)
    Y = np.zeros((n, 3), dtype=int)
    Y[:, 0] = X[:, 0] > 0.6
    Y[:, 1] = X[:, 1] > 0.6
    Y[:, 2] = (X[:, 0] > 0.6) & (X[:, 1] > 0.6)   # genuine co-occurrence
    return X, Y

def test_both_models_learn_a_separable_multilabel_problem():
    X, Y = _separable()
    models = fit_models(X, Y, seed=0)
    for name, m in models.items():
        P = predict_proba_multilabel(m, X, Y.shape[1])
        assert P.shape == (len(X), 3), f'{name} wrong shape {P.shape}'
        assert ((P >= 0) & (P <= 1)).all(), f'{name} produced non-probabilities'
        acc = ((P > 0.5).astype(int) == Y).mean()
        assert acc > 0.9, f'{name} accuracy {acc:.2f} on a separable problem'

def test_co_occurring_labels_are_both_predicted():
    """Multi-label, not multi-class: a pixel positive for two classes must get
    both, not argmax."""
    X, Y = _separable()
    both = np.flatnonzero(Y[:, 2] == 1)[:20]
    models = fit_models(X, Y, seed=0)
    for name, m in models.items():
        P = predict_proba_multilabel(m, X[both], Y.shape[1])
        assert (P[:, 0] > 0.5).mean() > 0.8, f'{name} dropped class 0'
        assert (P[:, 2] > 0.5).mean() > 0.8, f'{name} dropped class 2'

def test_nan_features_do_not_crash_histgb():
    """mrrsu carries 65535 nodata -> NaN. HistGB handles NaN natively; that is
    why it was chosen over an imputer that would bias the comparison."""
    X, Y = _separable()
    X = X.copy(); X[::10, 0] = np.nan
    models = fit_models(X, Y, seed=0)
    P = predict_proba_multilabel(models['histgb'], X, Y.shape[1])
    assert np.isfinite(P).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_fit_ml_baseline.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/fit_ml_baseline.py
"""Classical ML baselines on the 60 mrrsu summary parameters.

RandomForest is multi-output natively. HistGradientBoosting is single-target, so
it runs one-vs-rest per class -- which also preserves MULTI-LABEL semantics: a
pixel positive for two classes gets both, never an argmax.

HistGB is the second model specifically because it handles NaN natively. mrrsu
carries 65535 nodata; imputing it would bias the comparison against the deep
model, which sees the same gaps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fit_models(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> dict:
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    # RF cannot consume NaN; median-impute for RF ONLY and record it.
    med = np.nanmedian(X, axis=0)
    Xr = np.where(np.isfinite(X), X, med)
    rf.fit(Xr, Y)

    hist = []
    for j in range(Y.shape[1]):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        if len(np.unique(Y[:, j])) < 2:
            m = None            # a class with no positives cannot be fitted
        else:
            m.fit(X, Y[:, j])   # NaN handled natively
        hist.append(m)
    return {'rf': {'model': rf, 'impute_median': med}, 'histgb': hist}


def predict_proba_multilabel(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    if isinstance(model, dict) and 'impute_median' in model:
        Xr = np.where(np.isfinite(X), X, model['impute_median'])
        proba = model['model'].predict_proba(Xr)
        out = np.zeros((len(X), n_classes), dtype=np.float32)
        for j, p in enumerate(proba):        # list of (N,2) per output
            out[:, j] = p[:, 1] if p.shape[1] == 2 else 0.0
        return out
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    for j, m in enumerate(model):
        if m is None:
            continue
        out[:, j] = m.predict_proba(X)[:, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--vocab', choices=('7cls', 'pyx'), default='7cls')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import joblib
    from data.dataset import _collapse_labels
    from data.expert_rules import CLASSES_7, CLASSES_PYX

    feat = pd.read_parquet(args.features)
    lab = _collapse_labels(pd.read_parquet(args.labels))
    assert len(feat) == len(lab), 'feature/label row mismatch'
    vocab = CLASSES_7 if args.vocab == '7cls' else CLASSES_PYX
    key_cols = {'tile_id', 'pixel_row', 'pixel_col', 'split'}
    cols = [c for c in feat.columns if c not in key_cols]
    train = (lab['split'] == 'train').to_numpy()
    X = feat.loc[train, cols].to_numpy(np.float32)
    Y = np.stack([lab.loc[train, c].to_numpy() > 0.4 for c in vocab
                  if c in lab.columns], axis=1).astype(int)
    print(f'training on {len(X):,} TRAIN rows, {X.shape[1]} features, '
          f'{Y.shape[1]} classes')

    models = fit_models(X, Y, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(models['rf'], os.path.join(args.out_dir, 'rf.joblib'))
    joblib.dump(models['histgb'], os.path.join(args.out_dir, 'histgb.joblib'))
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({'vocab': [c for c in vocab if c in lab.columns],
                   'feature_cols': cols, 'seed': args.seed}, f, indent=2)
    print(f'wrote {args.out_dir}/{{rf,histgb}}.joblib + meta.json')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_fit_ml_baseline.py -v`
Expected: 3 passed

- [ ] **Step 5: Mutation-verify**

In `predict_proba_multilabel`, replace the one-vs-rest loop body with
`out[:, j] = (m.predict_proba(X)[:, 1] == m.predict_proba(X)[:, 1].max())`
(argmax-like behaviour).
Expected: `test_co_occurring_labels_are_both_predicted` FAILS. Revert via `cp`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fit_ml_baseline.py tests/test_fit_ml_baseline.py
git commit -m "Add RF and HistGB baselines on the 60 mrrsu parameters"
```

---

### Task 6: Tile scorer and the floor_test.sh hook

**Files:**
- Create: `scripts/classify_tile_baseline.py`
- Modify: `scripts/floor_test.sh` (add `CLASSIFY_CMD`)
- Test: `tests/test_classify_tile_baseline.py`

**Interfaces:**
- Consumes: `data.expert_rules.evaluate_rules`,
  `scripts.fit_ml_baseline.predict_proba_multilabel`,
  `scripts.classify_tile_supervised.load_tile` (for `valid_mask` identity)
- Produces: `<tid>_probs.npz` with keys `probs`, `valid_mask`, `transform`,
  `crs_wkt`, `class_names`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classify_tile_baseline.py
import glob, os
import numpy as np
import pytest
from scripts.classify_tile_baseline import assemble_npz_payload

def test_payload_is_structurally_identical_to_a_real_model_npz():
    """Compared against a REAL npz on disk, not a hand-written expectation --
    a hand-written one drifts from what the vectorizer actually reads."""
    real = sorted(glob.glob('/tmp/floor_test_*/*/*_probs.npz'))
    real += sorted(glob.glob('reports/floor_tests/*/*/*_probs.npz'))
    if not real:
        pytest.skip('no reference probs npz on this machine')
    ref = np.load(real[0], allow_pickle=True)

    H, W, C = 8, 9, 7
    payload = assemble_npz_payload(
        probs=np.zeros((H, W, C), np.float32),
        valid_mask=np.ones((H, W), bool),
        transform_arr=np.arange(6, dtype=np.float64),
        crs_wkt='PROJCS["x"]',
        class_names=['olivine','lcp','hcp','plagioclase','bland','alteration','junk'])

    assert set(payload) == set(ref.files), (
        f'keys differ: {sorted(payload)} vs {sorted(ref.files)}')
    assert payload['probs'].dtype == ref['probs'].dtype
    assert payload['valid_mask'].dtype == ref['valid_mask'].dtype
    assert payload['probs'].ndim == ref['probs'].ndim == 3

def test_rejects_a_vocabulary_the_vectorizer_would_not_accept():
    with pytest.raises(ValueError, match='class_names'):
        assemble_npz_payload(
            probs=np.zeros((2, 2, 2), np.float32),
            valid_mask=np.ones((2, 2), bool),
            transform_arr=np.arange(6, dtype=np.float64),
            crs_wkt='x', class_names=['mystery', 'vocab'])

def test_probs_channel_count_must_match_class_names():
    with pytest.raises(ValueError, match='channels'):
        assemble_npz_payload(
            probs=np.zeros((2, 2, 3), np.float32),
            valid_mask=np.ones((2, 2), bool),
            transform_arr=np.arange(6, dtype=np.float64),
            crs_wkt='x',
            class_names=['olivine','lcp','hcp','plagioclase','bland',
                         'alteration','junk'])
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_classify_tile_baseline.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/classify_tile_baseline.py
"""Score one tile with a baseline artifact and emit the standard probs npz.

The npz is the ONLY interface to the floor test, so a baseline written here runs
through floor_test.sh and the vectorizer completely unchanged -- same threshold
ladder, same smoothing, same summary tables. Any difference in polygon counts is
then attributable to the method rather than to the plumbing.

valid_mask is taken from classify_tile_supervised.load_tile (IMPORTED, not
reimplemented) so the baseline and the model mask identically. It is intersected
with mrrsu validity and BOTH counts are printed: a divergence must be visible
rather than absorbed into the comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.expert_rules import CLASSES_7, CLASSES_PYX  # noqa: E402

VALID_VOCABS = (CLASSES_7, CLASSES_PYX)


def assemble_npz_payload(probs, valid_mask, transform_arr, crs_wkt,
                         class_names) -> dict:
    names = [str(c) for c in class_names]
    if names not in [list(v) for v in VALID_VOCABS]:
        raise ValueError(
            f'class_names {names} is not a vocabulary the vectorizer accepts; '
            f'expected one of {[list(v) for v in VALID_VOCABS]}')
    if probs.shape[-1] != len(names):
        raise ValueError(
            f'probs has {probs.shape[-1]} channels but {len(names)} class_names')
    return {
        'probs': probs.astype(np.float32),
        'valid_mask': valid_mask.astype(bool),
        'transform': np.asarray(transform_arr, dtype=np.float64),
        'crs_wkt': crs_wkt,
        'class_names': np.array(names),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tile', required=True, help='mrral .img')
    ap.add_argument('--baseline', required=True,
                    help='expert rules .json, or an ML --out_dir')
    ap.add_argument('--model', choices=('rules', 'rf', 'histgb'), default='rules')
    ap.add_argument('--save_probs', required=True)
    ap.add_argument('--no_plot', action='store_true', help='accepted, ignored')
    args = ap.parse_args()

    import joblib
    from data.expert_rules import evaluate_rules
    from data.mrrsu_bands import read_mrrsu_cube
    from scripts.classify_tile_supervised import load_tile
    from scripts.fit_ml_baseline import predict_proba_multilabel

    mrrsu_img = args.tile.replace('_mrral_', '_mrrsu_')
    if not os.path.exists(mrrsu_img):
        raise SystemExit(f'no co-registered mrrsu tile at {mrrsu_img}')

    _data, valid_mask, transform, crs = load_tile(args.tile)
    cube, names = read_mrrsu_cube(mrrsu_img)
    if cube.shape[:2] != valid_mask.shape:
        raise SystemExit(
            f'mrrsu {cube.shape[:2]} is not co-registered with mrral '
            f'{valid_mask.shape}')
    mrrsu_valid = np.isfinite(cube).any(axis=-1)
    combined = valid_mask & mrrsu_valid
    print(f'valid pixels — mrral {valid_mask.sum():,}, '
          f'mrrsu {mrrsu_valid.sum():,}, both {combined.sum():,} '
          f'(mrral-only {int((valid_mask & ~mrrsu_valid).sum()):,})')

    if args.model == 'rules':
        cfg = json.load(open(args.baseline))
        scores = evaluate_rules(cube, names, cfg)
        vocab = cfg['vocab']
        probs = np.stack([scores[c] for c in vocab], axis=-1)
    else:
        meta = json.load(open(os.path.join(args.baseline, 'meta.json')))
        vocab = meta['vocab']
        model = joblib.load(os.path.join(args.baseline, f'{args.model}.joblib'))
        H, W, _ = cube.shape
        flat = cube.reshape(-1, cube.shape[-1])
        p = predict_proba_multilabel(model, flat, len(vocab))
        probs = p.reshape(H, W, len(vocab))

    probs[~combined] = 0.0
    payload = assemble_npz_payload(probs, combined,
                                   np.asarray(transform).flatten()[:6],
                                   crs.to_wkt() if crs else '', vocab)
    np.savez_compressed(args.save_probs, **payload)
    print(f'wrote {args.save_probs}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Add the floor_test.sh hook**

In `scripts/floor_test.sh`, replace the classify invocation inside `run_region`:

```bash
        # CLASSIFY_CMD lets a BASELINE produce the same probs npz and run
        # through this identical vectorization. Defaulting to the supervised
        # classifier keeps every existing caller unchanged. A forked copy of
        # this script would drift, and a drifted vectorization silently stops
        # being the same comparison.
        ${CLASSIFY_CMD:-$PYTHON scripts/classify_tile_supervised.py} \
            --tile "$img" --ckpt "$CKPT" \
            --save_probs "$npz" --no_plot ${CLASSIFY_EXTRA_ARGS:-}
```

Note `classify_tile_baseline.py` accepts `--ckpt` is NOT passed; instead invoke
it as:
`CLASSIFY_CMD="conda run -n crism python scripts/classify_tile_baseline.py --baseline config/expert_rules_7cls.json --model rules" CLASSIFY_EXTRA_ARGS="" bash scripts/floor_test.sh /dev/null rules_7cls`
so add `--ckpt` as an accepted-and-ignored argument to
`classify_tile_baseline.py`:

```python
    ap.add_argument('--ckpt', default=None,
                    help='accepted and ignored; floor_test.sh always passes it')
```

- [ ] **Step 5: Run to verify tests pass and floor_test.sh still parses**

```bash
conda run -n crism python -m pytest tests/test_classify_tile_baseline.py -v
bash -n scripts/floor_test.sh && echo "bash -n OK"
```
Expected: 3 passed; `bash -n OK`

- [ ] **Step 6: Mutation-verify**

Remove the `if names not in [...]` check in `assemble_npz_payload`.
Expected: `test_rejects_a_vocabulary_the_vectorizer_would_not_accept` FAILS.
Revert via `cp`.

- [ ] **Step 7: Commit**

```bash
git add scripts/classify_tile_baseline.py tests/test_classify_tile_baseline.py \
        scripts/floor_test.sh
git commit -m "Score tiles with baselines into the standard probs npz contract"
```

---

### Task 7: Atmospheric CO₂ diagnostic

**Files:**
- Create: `scripts/atmos_diagnostic.py`
- Test: `tests/test_atmos_diagnostic.py`

**Interfaces:**
- Consumes: a probs npz (any producer), the co-registered mrrde tile
- Produces: `detection_rate_by_decile(prob, valid, covariate, n=10) -> list[dict]`

**Reported, never gated.** Real HCP occurs at low elevation, so a hard veto
would suppress true detections. This makes the confound visible instead.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atmos_diagnostic.py
import numpy as np
from scripts.atmos_diagnostic import detection_rate_by_decile, air_mass

def test_detects_a_planted_elevation_dependence():
    """If detections concentrate at low elevation that is residual CO2, not
    clinopyroxene. The diagnostic must surface exactly that."""
    rng = np.random.default_rng(0)
    elev = rng.uniform(-4000, 2000, size=(100, 100)).astype(np.float32)
    prob = (elev < -2000).astype(np.float32) * 0.9      # planted: low only
    valid = np.ones_like(prob, bool)
    rows = detection_rate_by_decile(prob, valid, elev, threshold=0.5, n=10)
    assert len(rows) == 10
    assert rows[0]['rate'] > 0.5, 'lowest elevation decile should be hot'
    assert rows[-1]['rate'] == 0.0, 'highest elevation decile should be cold'

def test_flat_dependence_reports_flat():
    rng = np.random.default_rng(1)
    elev = rng.uniform(-4000, 2000, size=(100, 100)).astype(np.float32)
    prob = rng.random(elev.shape).astype(np.float32)
    rows = detection_rate_by_decile(prob, np.ones_like(prob, bool), elev,
                                    threshold=0.5, n=10)
    rates = [r['rate'] for r in rows]
    assert max(rates) - min(rates) < 0.15, f'spurious dependence: {rates}'

def test_air_mass_increases_with_incidence_angle():
    assert air_mass(np.float32(60.0), np.float32(0.0)) > \
           air_mass(np.float32(0.0), np.float32(0.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_atmos_diagnostic.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/atmos_diagnostic.py
"""Is an hcp detection clinopyroxene, or residual atmospheric CO2?

No CRISM summary parameter tracks GASEOUS atmospheric CO2 -- the CO2-named ones
(BD1435, BD3200, ICER1_2, ICER2_2) are all CO2 ICE. The physically correct proxy
is atmospheric path length, and mrrde carries it: elevation (band 15; pressure
falls ~exponentially with elevation, so CO2 column scales with it) and the
incidence/emission angles (bands 6/7) that give the air-mass factor.

HCPINDEX2 sits in the 2 um region where volcano-scan residual leaks, so this
matters for hcp specifically. REPORTED, NOT GATED: real HCP occurs at low
elevation too, and a hard veto would suppress true detections.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MRRDE_INA = 6
MRRDE_EMA = 7
MRRDE_ELEVATION = 15


def air_mass(ina_deg: np.ndarray, ema_deg: np.ndarray) -> np.ndarray:
    """1/cos(i) + 1/cos(e) — the standard two-way path-length factor."""
    i = np.clip(np.deg2rad(ina_deg), 0, np.deg2rad(89.0))
    e = np.clip(np.deg2rad(ema_deg), 0, np.deg2rad(89.0))
    return (1.0 / np.cos(i) + 1.0 / np.cos(e)).astype(np.float32)


def detection_rate_by_decile(prob: np.ndarray, valid: np.ndarray,
                             covariate: np.ndarray, threshold: float = 0.5,
                             n: int = 10) -> list[dict]:
    m = valid & np.isfinite(covariate)
    if not m.any():
        return []
    cov = covariate[m]
    det = (prob[m] >= threshold)
    edges = np.quantile(cov, np.linspace(0, 1, n + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rows = []
    for k in range(n):
        sel = (cov >= edges[k]) & (cov < edges[k + 1])
        rows.append({'decile': k + 1,
                     'lo': float(edges[k]), 'hi': float(edges[k + 1]),
                     'n': int(sel.sum()),
                     'rate': float(det[sel].mean()) if sel.any() else 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probs', required=True)
    ap.add_argument('--tile', required=True, help='mrral .img (mrrde inferred)')
    ap.add_argument('--klass', default='hcp')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    import rasterio

    d = np.load(args.probs, allow_pickle=True)
    names = [str(x) for x in d['class_names']]
    if args.klass not in names:
        raise SystemExit(f'{args.klass} not in {names}')
    prob = d['probs'][:, :, names.index(args.klass)]
    valid = d['valid_mask'].astype(bool)

    mrrde = args.tile.replace('_mrral_', '_mrrde_')
    if not os.path.exists(mrrde):
        raise SystemExit(f'no mrrde tile at {mrrde}')
    with rasterio.open(mrrde) as src:
        elev = src.read(MRRDE_ELEVATION + 1).astype(np.float32)
        ina = src.read(MRRDE_INA + 1).astype(np.float32)
        ema = src.read(MRRDE_EMA + 1).astype(np.float32)
    elev[elev == 65535.0] = np.nan
    am = air_mass(ina, ema)

    for label, cov in (('elevation (m)', elev), ('air mass', am)):
        print(f'\n{args.klass} detection rate by {label} decile '
              f'(threshold {args.threshold}):')
        print(f'  {"dec":>4}{"lo":>12}{"hi":>12}{"n":>10}{"rate":>8}')
        for r in detection_rate_by_decile(prob, valid, cov, args.threshold):
            print(f'  {r["decile"]:>4}{r["lo"]:>12.1f}{r["hi"]:>12.1f}'
                  f'{r["n"]:>10,}{r["rate"]:>8.3f}')
    print('\nDetections concentrated in the LOW-elevation / HIGH-air-mass '
          'deciles indicate residual atmospheric CO2 rather than clinopyroxene.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_atmos_diagnostic.py -v`
Expected: 3 passed

- [ ] **Step 5: Mutation-verify**

Change `edges = np.quantile(...)` to `edges = np.linspace(cov.min(), cov.max(), n+1)`
and re-run `test_flat_dependence_reports_flat`; it should still pass (equal-width
bins are also valid), so instead mutate `det = (prob[m] >= threshold)` to
`det = (prob[m] >= 0.0)`.
Expected: `test_detects_a_planted_elevation_dependence` FAILS on
`rows[-1]['rate'] == 0.0`. Revert via `cp`.

- [ ] **Step 6: Commit**

```bash
git add scripts/atmos_diagnostic.py tests/test_atmos_diagnostic.py
git commit -m "Add hcp-vs-elevation/air-mass diagnostic for residual CO2"
```

---

## End-to-end run (after all tasks)

```bash
D=/xdisk/sbyrne/phillipsm/CRISM_MRDR/crism_classification
P=$D/data/mrral_pixels_7cls_handcore.parquet

conda run -n crism python scripts/extract_mrrsu_features.py \
    --parquet $P --out $D/data/mrrsu_features_handcore.parquet --smooth

conda run -n crism python scripts/fit_expert_rules.py \
    --features $D/data/mrrsu_features_handcore.parquet --labels $P \
    --vocab 7cls --out config/expert_rules_7cls.json

conda run -n crism python scripts/fit_ml_baseline.py \
    --features $D/data/mrrsu_features_handcore.parquet --labels $P \
    --vocab 7cls --out_dir $D/baselines/ml_7cls

for b in "rules config/expert_rules_7cls.json" "rf $D/baselines/ml_7cls" \
         "histgb $D/baselines/ml_7cls"; do
  set -- $b
  CLASSIFY_CMD="conda run -n crism python scripts/classify_tile_baseline.py --model $1 --baseline $2" \
    bash scripts/floor_test.sh /dev/null "baseline_$1"
done
```

Then read `reports/floor_tests/baseline_*/summary.md` against the model's, using
the same acceptance table from the `floor-test` skill — the baselines are the
first **absolute** anchor for those thresholds, which until now were set only
relative to previous checkpoints.

## Self-review notes

- **Spec coverage:** npz contract → T6; valid_mask confound → T6 step 3; band
  registry → T1; multi-label principle → T3 (tests 1–2); RPEAK1 window and 7×7
  regional smoothing → T2 (`--smooth`) and T3; carbonate exception → T3 test 3;
  ice→junk → T3 test 4; veto retention → T4 test 2; precision monotonicity → T4
  test 1 plus a runtime warning; RF/HistGB and NaN → T5 test 3; atmospheric
  diagnostic → T7; floor_test hook → T6 step 4; pyx vocabulary → `--vocab pyx`
  throughout.
- **Gap found and closed in review:** an earlier draft had Task 2 emit
  positional `p0..p59` columns and left Task 4 resolving them back to parameter
  names through an unspecified `MRRSU_HDR_FOR_NAMES` hook — a placeholder. Task 2
  now emits real parameter names directly, which removes the hook, removes the
  rename from Tasks 4 and 5, and adds a cross-tile band-order check that raises
  if any tile disagrees.
- **Type consistency checked:** `read_mrrsu_cube -> (cube, names)` (T1) feeds T2
  and T6; `evaluate_rules(cube, names, config)` (T3) feeds T6;
  `predict_proba_multilabel(model, X, n_classes)` (T5) feeds T6;
  `assemble_npz_payload` (T6) is the single npz writer. No name drift.
