# Confidence-Weighted Polygon Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let reviewers grade each polygon decision Low/Moderate/High → per-polygon training weights 0.5/0.75/1.0, threaded to the next training cycle, and fix a latent bug where reassigned minerals are mistrained as bland.

**Architecture:** A `REVIEW_CONFIDENCE_WEIGHTS` map in the review persistence layer stamps each confirmed / reassigned-mineral row with a literal `confidence_weight` and a `Reviewed-<tier>` label. `data/dataset.py::_collapse_labels` already passes unknown tiers through to the stamped weight verbatim (no change needed), so review weights are honored without disturbing the base parquet's High/Moderate/Low rows. `build_7cls_dataset.py` stops flattening confirmed weights to 2.0 and routes reassigned minerals out of the bland pool into weighted positives.

**Tech Stack:** Python, pandas, numpy, pyarrow, Streamlit (review UI), pytest.

---

## File Structure

- `scripts/review/persistence.py` — add weight map, thread `confidence` through `_rows_for_polygon` and both writers, add `confidence` decision-log column. (Tasks 1–3)
- `scripts/build_7cls_dataset.py` — preserve per-polygon confirmed weights; route reassigned minerals out of bland. (Tasks 4–5)
- `scripts/review/app.py` — confidence radio + `_record` wiring. (Task 6)
- `tests/test_review_persistence.py` — writer/decision-log tests. (Tasks 1–3)
- `tests/test_build_7cls_confidence.py` (new) — build preservation + routing tests. (Tasks 4–5)
- `tests/test_collapse_reviewed_tier.py` (new) — `_collapse_labels` fallthrough regression. (Task 7)

All tests run with: `conda run -n crism python -m pytest <path> -v`

---

### Task 1: Confidence weight map + confirmed-writer confidence

**Files:**
- Modify: `scripts/review/persistence.py` (`_rows_for_polygon`, `ConfirmedPixelsWriter.append_polygon`, new constant)
- Test: `tests/test_review_persistence.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_review_persistence.py`:

```python
import pytest as _pytest
from scripts.review.persistence import REVIEW_CONFIDENCE_WEIGHTS


@_pytest.mark.parametrize('confidence,weight', [
    ('High', 1.0), ('Moderate', 0.75), ('Low', 0.5),
])
def test_confirmed_writer_stamps_confidence(tmp_path, confidence, weight):
    pq = tmp_path / 'confirmed'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::a::0',
        rows=np.array([0, 1]), cols=np.array([0, 1]),
        spectra=np.zeros((2, 59), dtype=np.float32),
        label_class='hcp', confidence=confidence,
    )
    df = pd.read_parquet(str(pq))
    assert (df['confidence_weight'] == weight).all()
    assert (df['confidence_tier'] == f'Reviewed-{confidence}').all()


def test_review_confidence_weights_values():
    assert REVIEW_CONFIDENCE_WEIGHTS == {'High': 1.0, 'Moderate': 0.75, 'Low': 0.5}
```

Also update the existing `test_confirmed_writer_schema_matches_mrral_pixels` assertions (it appends without `confidence`, so it must reflect the new default tier label):

```python
    # was: assert df['confidence_tier'].iloc[0] == 'High'
    assert df['confidence_weight'].iloc[0] == 1.0
    assert df['confidence_tier'].iloc[0] == 'Reviewed-High'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -v`
Expected: FAIL — `ImportError: cannot import name 'REVIEW_CONFIDENCE_WEIGHTS'` and `append_polygon() got an unexpected keyword argument 'confidence'`.

- [ ] **Step 3: Add the constant and thread weight/tier through `_rows_for_polygon`**

In `scripts/review/persistence.py`, add the constant near the top (after `_DECISION_COLS`):

```python
# Reviewer confidence → per-polygon training sample weight. Stamped together
# with a 'Reviewed-<tier>' label that is intentionally OUTSIDE
# data/dataset.py::_TIER_WEIGHTS so _collapse_labels passes the literal weight
# through verbatim (leaving base-parquet High/Moderate/Low weights untouched).
REVIEW_CONFIDENCE_WEIGHTS = {'High': 1.0, 'Moderate': 0.75, 'Low': 0.5}
```

Change `_rows_for_polygon` signature and the two stamped lines:

```python
def _rows_for_polygon(
    tile_id: str,
    polygon_uid: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spectra: np.ndarray,
    label_dict: dict[str, float],
    weight: float = 1.0,
    tier: str = 'High',
) -> pd.DataFrame:
    n = spectra.shape[0]
    polygon_id_int = _polygon_id_int(polygon_uid)
    data = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id_int, dtype=np.int64),
        'pixel_row': rows.astype(np.int64),
        'pixel_col': cols.astype(np.int64),
    }
    for i in range(59):
        data[f'm{i}'] = spectra[:, i].astype(np.float64)
    for c in _LABEL_COLS:
        data[c] = np.full(n, label_dict[c], dtype=np.float64)
    data['confidence_weight'] = np.full(n, weight, dtype=np.float64)
    data['confidence_tier'] = [tier] * n
    data['split'] = ['train'] * n
    return pd.DataFrame(data, columns=confirmed_schema_columns())
```

- [ ] **Step 4: Thread `confidence` through `ConfirmedPixelsWriter.append_polygon`**

```python
    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray, label_class: str,
                        extra_classes: Optional[list] = None,
                        confidence: str = 'High') -> None:
        """Write rows for ``polygon_uid`` with positive labels for
        ``label_class`` and every class in ``extra_classes`` (co-occurring
        minerals), stamped with the reviewer ``confidence`` weight/tier."""
        all_classes = [label_class] + list(extra_classes or [])
        weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra,
                                _label_dict_for_many(all_classes),
                                weight=weight, tier=f'Reviewed-{confidence}')
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        _atomic_write_parquet(df, path)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -v`
Expected: PASS (new parametrized test + constant test + updated schema test).

- [ ] **Step 6: Commit**

```bash
git add scripts/review/persistence.py tests/test_review_persistence.py
git commit -m "review: per-polygon confidence weight on confirmed writer"
```

---

### Task 2: HardNegativesWriter confidence (mineral reassignments only)

**Files:**
- Modify: `scripts/review/persistence.py` (`HardNegativesWriter.append_polygon`)
- Test: `tests/test_review_persistence.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_hard_negatives_mineral_reassignment_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::a::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='olivine',
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['confidence_weight'].iloc[0] == 0.5
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Low'
    assert df['negative_of'].iloc[0] == ''


def test_hard_negatives_tag_reject_keeps_fixed_weight(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::b::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='ambiguous',
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['confidence_weight'].iloc[0] == 1.0
    assert df['confidence_tier'].iloc[0] == 'High'
    assert df['negative_of'].iloc[0] == 'ambiguous'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -k hard_negatives -v`
Expected: FAIL — `append_polygon() got an unexpected keyword argument 'confidence'`.

- [ ] **Step 3: Thread `confidence` into the mineral-reassignment branch only**

Replace `HardNegativesWriter.append_polygon` body:

```python
    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray,
                        predicted_class: str,
                        corrected_class: Optional[str],
                        confidence: str = 'High') -> None:
        # Three cases for the reject:
        #  - no corrected_class:    "not {predicted_class}" with no positive label
        #  - mineral corrected:     positive label for the corrected class,
        #                           stamped with the reviewer confidence weight
        #  - non-mineral tag (e.g. 'ambiguous'): all-zero labels, negative_of=tag
        weight, tier = 1.0, 'High'
        if not corrected_class:
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = predicted_class
        elif _is_mineral_class(corrected_class):
            label = _label_dict_for(corrected_class)
            negative_of = ''
            weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]
            tier = f'Reviewed-{confidence}'
        else:
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = corrected_class
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra, label,
                                weight=weight, tier=tier)
        df['negative_of'] = negative_of
        df = df[hard_negatives_schema_columns()]
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        _atomic_write_parquet(df, path)
```

Note: `bland`/`dust` corrected classes are mineral-class per `_is_mineral_class`
(they map to the `other` label), so a bland reassignment is *also* weighted —
this is harmless because build routes `other=1.0` rows to the bland pool, which
keeps fixed weighting regardless of the stamped value.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -k hard_negatives -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/review/persistence.py tests/test_review_persistence.py
git commit -m "review: confidence weight on hard-neg mineral reassignments"
```

---

### Task 3: `confidence` column in the decision log

**Files:**
- Modify: `scripts/review/persistence.py` (`_DECISION_COLS`)
- Test: `tests/test_review_persistence.py`

- [ ] **Step 1: Write the failing test and update the header test**

Update `test_decision_log_creates_csv_and_appends_header` column list to include the new trailing column:

```python
    assert list(df.columns) == [
        'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
        'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
        'co_occurring_classes', 'confidence',
    ]
```

Add a new test:

```python
def test_decision_log_records_confidence(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    rec = _record()
    rec['confidence'] = 'Moderate'
    log.append(rec)
    df = pd.read_csv(csv)
    assert df.iloc[0]['confidence'] == 'Moderate'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -k decision_log -v`
Expected: FAIL — column list mismatch / `confidence` column absent.

- [ ] **Step 3: Add `confidence` to `_DECISION_COLS`**

```python
_DECISION_COLS = [
    'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
    'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
    # Semicolon-separated list of co-occurring mineral classes when a polygon
    # is confirmed as the predicted class but also shows signals from others
    # (e.g. olivine + hcp). Empty for single-class confirms and all rejects.
    'co_occurring_classes',
    # Reviewer confidence (High/Moderate/Low) logged for every decision. Only
    # applied to the parquet weight for confirms and mineral reassignments.
    'confidence',
]
```

The existing `_migrate_schema_if_needed` rewrites older `decisions.csv` files
with the new column automatically — no extra migration code required.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -v`
Expected: PASS (all persistence tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/review/persistence.py tests/test_review_persistence.py
git commit -m "review: log confidence column in decisions.csv"
```

---

### Task 4: Preserve per-polygon confirmed weights in build

**Files:**
- Modify: `scripts/build_7cls_dataset.py` (`load_confirmed_mineral_positives`)
- Test: `tests/test_build_7cls_confidence.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_7cls_confidence.py`:

```python
import os
import numpy as np
import pandas as pd

from scripts.build_7cls_dataset import load_confirmed_mineral_positives


def _confirmed_row(polygon_id, label_col, weight, tier, n=3):
    d = {
        'tile_id': ['t1250'] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.arange(n), 'pixel_col': np.arange(n),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    d['confidence_weight'] = np.full(n, weight)
    d['confidence_tier'] = [tier] * n
    d['split'] = ['train'] * n
    return pd.DataFrame(d)


def test_confirmed_preserves_per_polygon_weight(tmp_path):
    cdir = tmp_path / 'confirmed'
    cdir.mkdir()
    _confirmed_row(1, 'olivine_t1', 1.0, 'Reviewed-High').to_parquet(
        cdir / 'p_00000001.parquet', index=False)
    _confirmed_row(2, 'hcp', 0.5, 'Reviewed-Low').to_parquet(
        cdir / 'p_00000002.parquet', index=False)
    template = _confirmed_row(0, 'olivine_t1', 1.0, 'Reviewed-High').assign(
        bland=0.0, junk=0.0)
    out = load_confirmed_mineral_positives(str(cdir), template)
    # weights are NOT flattened to 2.0 — each polygon keeps its stamped value
    assert set(np.unique(out['confidence_weight'])) == {1.0, 0.5}
    assert set(out['confidence_tier']) == {'Reviewed-High', 'Reviewed-Low'}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py::test_confirmed_preserves_per_polygon_weight -v`
Expected: FAIL — `confidence_weight` is all `2.0` (current flat override).

- [ ] **Step 3: Stop overwriting confirmed weight/tier**

In `scripts/build_7cls_dataset.py::load_confirmed_mineral_positives`, replace:

```python
    # stamp 7cls cols (plag in confirmed pixels is always 0 — no real plag in MC13)
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=0.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    df = _assign_tile_splits(df, SEED + 300)
```

with (preserve the per-polygon weight/tier already in the parquet; default
absent weights to 1.0 for legacy files that predate confidence stamping):

```python
    # stamp 7cls cols (plag in confirmed pixels is always 0 — no real plag in MC13)
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=0.0)
    # Preserve the per-polygon reviewer confidence weight/tier instead of the
    # old flat REVIEW_WEIGHT override. Legacy confirmed files (pre-confidence)
    # carry weight=1.0 / tier='High', which collapse to weight 1.0 downstream.
    if 'confidence_weight' not in df.columns:
        df['confidence_weight'] = np.float32(1.0)
    if 'confidence_tier' not in df.columns:
        df['confidence_tier'] = 'High'
    df = _assign_tile_splits(df, SEED + 300)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py::test_confirmed_preserves_per_polygon_weight -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_7cls_confidence.py
git commit -m "build: preserve per-polygon confirmed confidence weights"
```

---

### Task 5: Route reassigned minerals out of the bland pool

**Files:**
- Modify: `scripts/build_7cls_dataset.py` (`load_bland_review`, new `load_reassigned_minerals`, `main` wiring)
- Test: `tests/test_build_7cls_confidence.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_7cls_confidence.py`:

```python
from scripts.build_7cls_dataset import load_bland_review, load_reassigned_minerals


def _hardneg_row(polygon_id, label_col, n=3):
    d = {
        'tile_id': ['t1250'] * n,  # MC13 region
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.arange(n), 'pixel_col': np.arange(n),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    d['confidence_weight'] = np.full(n, 0.5)
    d['confidence_tier'] = ['Reviewed-Low'] * n
    d['split'] = ['train'] * n
    d['negative_of'] = [''] * n
    return pd.DataFrame(d)


def _write_hardneg_dir(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _hardneg_row(1, 'olivine_t1').to_parquet(hdir / 'p_01.parquet', index=False)
    _hardneg_row(2, 'other').to_parquet(hdir / 'p_02.parquet', index=False)
    return str(hdir)


def test_reassigned_minerals_routed_to_positives(tmp_path):
    hdir = _write_hardneg_dir(tmp_path)
    out = load_reassigned_minerals(hdir)
    assert (out['olivine_t1'] > 0).all()
    assert (out['bland'] == 0).all()
    assert set(out['confidence_weight'].unique()) == {0.5}


def test_bland_review_excludes_mineral_reassignments(tmp_path):
    hdir = _write_hardneg_dir(tmp_path)
    out = load_bland_review(hdir, 'mc13_blands', mc13=True, seed_offset=10,
                            n_bland=1000)
    # only the other=1.0 polygon survives in the bland pool
    assert (out['bland'] > 0).all()
    assert (out['olivine_t1'] == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py -k "reassigned or bland_review" -v`
Expected: FAIL — `cannot import name 'load_reassigned_minerals'`, and `load_bland_review` currently stamps the olivine polygon as bland.

- [ ] **Step 3: Add the mineral-label mask helper and `load_reassigned_minerals`**

In `scripts/build_7cls_dataset.py`, add a module constant near the other
constants:

```python
# Mineral label columns used to distinguish mineral reassignments from bland
# reassignments inside the negative_of='' hard-negative pool.
_REASSIGN_MINERAL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase']
```

Add a new loader after `load_bland_review`:

```python
def load_reassigned_minerals(hn_dir: str) -> pd.DataFrame:
    """Reject→mineral reassignments live in hard_negatives with negative_of=''
    and a mineral label = 1.0. Ingest them as weighted mineral positives
    (preserving the reviewer confidence weight/tier), capped per polygon and
    tile-split — NOT as bland (the prior behaviour, which mistrained them)."""
    pool = _read_hn_tag(hn_dir, tag=None)
    if pool.empty:
        return pool
    mineral_mask = (pool[_REASSIGN_MINERAL_COLS] > 0).any(axis=1)
    df = pool[mineral_mask].copy()
    print(f'  reassigned minerals: {len(df):,} rows '
          f'({df["tile_id"].nunique()} tiles, '
          f'{df.groupby(["tile_id","polygon_id"]).ngroups} polygons)')
    if df.empty:
        return df
    df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 400)
    df = _assign_tile_splits(df, SEED + 400)
    splits = df['split'].value_counts().to_dict()
    print(f'  reassigned minerals: splits {splits}')
    # Preserve the parquet's confidence weight/tier (do not zero plagioclase —
    # a reject→plagioclase reassignment is a real plag positive).
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=0.0,
                          zero_plag=False)
    if 'confidence_weight' not in df.columns:
        df['confidence_weight'] = np.float32(1.0)
    if 'confidence_tier' not in df.columns:
        df['confidence_tier'] = 'High'
    return df
```

- [ ] **Step 4: Exclude mineral reassignments from `load_bland_review`**

In `load_bland_review`, immediately after the region-filter block that produces
`df` (right after the `print(f'  {source_label}: {len(df):,} raw rows ...')`
line), drop the mineral-reassignment rows so only true blands remain:

```python
    # Mineral reassignments (negative_of='' with a mineral label=1.0) share this
    # pool; they belong in load_reassigned_minerals, not the bland pool.
    df = df[~(df[_REASSIGN_MINERAL_COLS] > 0).any(axis=1)].copy()
```

- [ ] **Step 5: Wire `load_reassigned_minerals` into `main`**

In `main`, after the bland-review block, add the loader:

```python
    # ── 3b. Reassigned minerals (reject→mineral) ─────────────────────────────
    print('\nLoading reassigned mineral positives …')
    reassigned = load_reassigned_minerals(args.hn_dir)
```

Then add it to the `fragments` assembly list (the `for label, frag in [...]`
loop) as a new entry:

```python
    for label, frag in [('confirmed', confirmed),
                         ('reassigned', reassigned),
                         ('mc13_bland', mc13_bland),
                         ('mc11_bland', mc11_bland),
                         ('junk', junk_df),
                         ('alteration', alt_df)]:
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py -v`
Expected: PASS (all four build tests).

- [ ] **Step 7: Commit**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_7cls_confidence.py
git commit -m "build: route reassigned minerals to weighted positives, not bland"
```

---

### Task 6: Confidence radio in the review UI

**Files:**
- Modify: `scripts/review/app.py` (confidence radio + `_record` wiring)

This task has no unit test (Streamlit UI); verify by byte-compiling the module
and confirming the wiring by inspection.

- [ ] **Step 1: Add the confidence radio**

In `scripts/review/app.py`, immediately before the
`p1, b1, b2, b3, n1 = st.columns([1, 1, 1, 1, 1])` line, add:

```python
    confidence = st.radio(
        'confidence', ['High', 'Moderate', 'Low'], horizontal=True, index=0,
        help='Per-polygon training weight: High=1.0, Moderate=0.75, Low=0.5. '
             'Applied to confirms and reject→mineral reassignments.',
    )
```

- [ ] **Step 2: Thread `confidence` into `_record`**

In the `_record` function, update the three call sites:

`log.append(...)` — add the field:

```python
        log.append(dict(
            source_gpkg=item.source_gpkg, layer=item.layer,
            polygon_uid=item.polygon_uid, tile_id=item.tile_id,
            predicted_class=mineral, decision=decision,
            corrected_class=(corrected if decision == 'reject' else ''),
            n_pixels=n_px, area_m2=item.area_m2,
            co_occurring_classes=(';'.join(co_occurring)
                                   if decision == 'confirm' else ''),
            confidence=confidence,
        ))
```

`confirmed_writer.append_polygon(...)` — add `confidence=confidence`:

```python
            confirmed_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                label_class=mineral,
                extra_classes=co_occurring or None,
                confidence=confidence,
            )
```

`hardneg_writer.append_polygon(...)` — add `confidence=confidence`:

```python
            hardneg_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                predicted_class=mineral,
                corrected_class=(corrected or None),
                confidence=confidence,
            )
```

- [ ] **Step 3: Byte-compile to verify no syntax error**

Run: `conda run -n crism python -m py_compile scripts/review/app.py`
Expected: exit 0, no output.

- [ ] **Step 4: Confirm the wiring by grep**

Run: `grep -n "confidence" scripts/review/app.py`
Expected: shows the radio definition and the three `confidence=confidence`
call sites.

- [ ] **Step 5: Commit**

```bash
git add scripts/review/app.py
git commit -m "review UI: confidence radio wired to writers and decision log"
```

---

### Task 7: `_collapse_labels` fallthrough regression test

**Files:**
- Test: `tests/test_collapse_reviewed_tier.py` (new)

Guards the core invariant: `Reviewed-*` tiers must bypass the global
`_TIER_WEIGHTS` map and use the stamped weight verbatim. No production code
change — this locks in existing behaviour the feature depends on.

- [ ] **Step 1: Write the test**

Create `tests/test_collapse_reviewed_tier.py`:

```python
import numpy as np
import pandas as pd

from data.dataset import _collapse_labels


def _row(tier, weight):
    d = {'olivine_t1': [0.0], 'olivine_t2': [0.0], 'lcp': [0.0], 'hcp': [1.0],
         'plagioclase': [0.0], 'other': [0.0],
         'confidence_tier': [tier], 'confidence_weight': [weight]}
    return pd.DataFrame(d)


def test_reviewed_tiers_use_stamped_weight():
    for tier, w in [('Reviewed-High', 1.0), ('Reviewed-Moderate', 0.75),
                    ('Reviewed-Low', 0.5)]:
        out = _collapse_labels(_row(tier, w))
        assert float(out['confidence_weight'].iloc[0]) == w, tier


def test_base_high_moderate_low_unchanged():
    # Global tiers still map through _TIER_WEIGHTS, NOT the stamped weight.
    assert float(_collapse_labels(_row('Moderate', 0.50))['confidence_weight'].iloc[0]) == 0.85
    assert float(_collapse_labels(_row('Low', 0.25))['confidence_weight'].iloc[0]) == 0.70
```

- [ ] **Step 2: Run the test**

Run: `conda run -n crism python -m pytest tests/test_collapse_reviewed_tier.py -v`
Expected: PASS (no code change needed — confirms the fallthrough invariant).

- [ ] **Step 3: Commit**

```bash
git add tests/test_collapse_reviewed_tier.py
git commit -m "test: lock Reviewed-* tier fallthrough to stamped weight"
```

---

## Final verification

- [ ] **Run the full affected test set**

Run:
```bash
conda run -n crism python -m pytest tests/test_review_persistence.py \
    tests/test_build_7cls_confidence.py tests/test_collapse_reviewed_tier.py -v
```
Expected: all PASS.

- [ ] **Push to the HPC tracking branch**

```bash
git push origin master:feature/spatial-mae-pretraining
```

After the next review session collects confidence-graded polygons, rebuild the
7-class dataset on HPC (`hpc_build_7cls_data.slurm`) and retrain — the weights
flow through automatically.
