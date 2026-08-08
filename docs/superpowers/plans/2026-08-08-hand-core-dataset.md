# Hand-Labeled-Core Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the 7-class training parquet so hand-labeled data is the core and review data augments it, with bland/junk review-only and alteration review-defined.

**Architecture:** Extend `scripts/build_7cls_dataset.py` with a per-class `SourcePolicy` plus explicit review-session provenance, so the ungraded legacy session can be admitted for named classes only. Add a sweepable train-time `--weight_scheme` in `data/dataset.py`. No new scripts; the existing unit-balanced joint re-split is reused untouched.

**Tech Stack:** Python 3.11, pandas, pyarrow (predicate pushdown), pytest, conda env `crism`.

## Global Constraints

- All commands run in the `crism` conda env: `conda run -n crism …`.
- Run pytest from `crism_classification/`: `conda run -n crism python -m pytest tests/… -v`.
- Spec of record: `docs/superpowers/specs/2026-08-08-hand-core-dataset-design.md`.
- Do **not** change `_joint_resplit`, `assign_unit_balanced_splits`, or the MTRDR plag synth path — the split logic is the leakage fix and must stay byte-identical in behaviour.
- Review quality bar: v3 session only, `Reviewed-High` + `Reviewed-Moderate`.
- Legacy session admitted for `alteration`, `lcp`, `hcp` only.
- `MAX_PX_PER_POLYGON = 20_000` stays the default cap; legacy **confirms** get 5_000.
- `--weight_scheme level` must reproduce today's effective weights exactly, including the deliberate `Reviewed-*` stamped-weight pass-through asserted by `tests/test_collapse_reviewed_tier.py`. That test must keep passing unmodified.
- Output file: `data/mrral_pixels_7cls_handcore.parquet`. Never overwrite `data/mrral_pixels_7cls.parquet`.
- Existing default behaviour of `build_7cls_dataset.py` must be unchanged when the new flags are left at their defaults. **AMENDED 2026-08-08 by user ruling:** the plan originally specified *filtering* defaults (`--review_grades High Moderate`, `--legacy_classes alteration lcp hcp`, `--legacy_confirm_cap 5000`), which contradicted this very constraint — a bare rebuild would have silently produced a policy-filtered dataset. All policy defaults are now **permissive/inert**; the hand-core recipe is opt-in via explicit flags. See the spec's flag table.
- "Inert" means byte-identical output, not merely "drops no rows": `_apply_legacy_policy`'s confirm branch must preserve ROW ORDER when the cap binds nothing, because `_joint_resplit` is order-sensitive at ties (~250 rows moved train↔val otherwise).

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `data/dataset.py` | `WEIGHT_SCHEMES` table + `set_weight_scheme()`; `_collapse_labels` consults the active scheme | Modify |
| `scripts/train.py` | `--weight_scheme` flag, applied before any dataset code reads weights | Modify |
| `scripts/build_7cls_dataset.py` | `review_session` provenance, grade filter, `SourcePolicy`, bland/legacy policy, new CLI flags | Modify |
| `tests/test_weight_schemes.py` | Weight scheme resolution + `level` parity with today | Create |
| `tests/test_build_handcore_sources.py` | Provenance, grade filter, legacy policy, bland policy | Create |

---

### Task 1: Sweepable weight schemes

**Files:**
- Modify: `data/dataset.py:68` (`_TIER_WEIGHTS`) and `data/dataset.py:123-138` (`_collapse_labels` weight block)
- Modify: `scripts/train.py:252` (label-cols swap block — add the scheme call next to it)
- Test: `tests/test_weight_schemes.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `data.dataset.WEIGHT_SCHEMES: dict[str, dict[str, float]]`, `data.dataset.set_weight_scheme(name: str) -> None`, `data.dataset.active_weight_scheme() -> str`.

**Background you need.** `_collapse_labels` currently lowercases `confidence_tier`, maps it through `_TIER_WEIGHTS = {'high':1.0,'moderate':0.85,'low':0.70}`, and for any tier NOT in that table falls back to the `confidence_weight` already stamped in the parquet. The v3 review tiers (`Reviewed-High/-Moderate/-Low`) deliberately miss the table so their per-polygon reviewer weight passes through. **Preserve that.** The `level` scheme adds no `reviewed-*` keys at all, so behaviour is bit-identical to today.

- [ ] **Step 1: Write the failing test**

Create `tests/test_weight_schemes.py`:

```python
import numpy as np
import pandas as pd
import pytest

import data.dataset as ds
from data.dataset import _collapse_labels


@pytest.fixture(autouse=True)
def _reset_scheme():
    ds.set_weight_scheme('level')
    yield
    ds.set_weight_scheme('level')


def _row(tier, weight):
    return pd.DataFrame({
        'olivine_t1': [0.0], 'olivine_t2': [0.0], 'lcp': [0.0], 'hcp': [1.0],
        'plagioclase': [0.0], 'other': [0.0],
        'confidence_tier': [tier], 'confidence_weight': [weight],
    })


def _w(tier, weight):
    return float(_collapse_labels(_row(tier, weight))['confidence_weight'].iloc[0])


def test_level_matches_todays_behaviour():
    # Hand tiers resolve through the table.
    assert _w('High', 0.1) == pytest.approx(1.0)
    assert _w('Moderate', 0.1) == pytest.approx(0.85)
    assert _w('Low', 0.1) == pytest.approx(0.70)
    # Reviewed-* deliberately pass the stamped weight through untouched.
    assert _w('Reviewed-High', 1.0) == pytest.approx(1.0)
    assert _w('Reviewed-Moderate', 0.75) == pytest.approx(0.75)
    assert _w('Reviewed-Low', 0.5) == pytest.approx(0.5)


def test_review_up_scales_reviewed_tiers():
    ds.set_weight_scheme('review_up')
    assert _w('Reviewed-High', 1.0) == pytest.approx(2.0)
    assert _w('Reviewed-Moderate', 0.75) == pytest.approx(1.7)
    assert _w('High', 0.1) == pytest.approx(1.0)   # hand untouched


def test_hand_up_scales_hand_tiers():
    ds.set_weight_scheme('hand_up')
    assert _w('High', 0.1) == pytest.approx(1.5)
    assert _w('Moderate', 0.1) == pytest.approx(1.3)
    assert _w('Reviewed-High', 1.0) == pytest.approx(1.0)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match='nonesuch'):
        ds.set_weight_scheme('nonesuch')


def test_active_scheme_roundtrip():
    ds.set_weight_scheme('review_up')
    assert ds.active_weight_scheme() == 'review_up'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_weight_schemes.py -v`
Expected: FAIL — `AttributeError: module 'data.dataset' has no attribute 'set_weight_scheme'`

- [ ] **Step 3: Write minimal implementation**

In `data/dataset.py`, replace the `_TIER_WEIGHTS` definition (line 68) with:

```python
# Tier → per-pixel sample weight. A scheme omits the 'reviewed-*' keys to let
# the per-polygon reviewer weight stamped by scripts/review/persistence.py pass
# through _collapse_labels verbatim (see tests/test_collapse_reviewed_tier.py).
# 'level' therefore reproduces the pre-2026-08-08 behaviour exactly.
WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    'level':     {'high': 1.0, 'moderate': 0.85, 'low': 0.70},
    'review_up': {'high': 1.0, 'moderate': 0.85, 'low': 0.70,
                  'reviewed-high': 2.0, 'reviewed-moderate': 1.7,
                  'reviewed-low': 1.4},
    'hand_up':   {'high': 1.5, 'moderate': 1.3, 'low': 1.0,
                  'reviewed-high': 1.0, 'reviewed-moderate': 0.85,
                  'reviewed-low': 0.70},
}

_ACTIVE_SCHEME = 'level'
_TIER_WEIGHTS = WEIGHT_SCHEMES[_ACTIVE_SCHEME]


def set_weight_scheme(name: str) -> None:
    """Select the tier→weight table used by _collapse_labels.

    Must be called before any dataset is constructed. Schemes that omit the
    'reviewed-*' keys leave the stamped per-polygon reviewer weight intact.
    """
    global _ACTIVE_SCHEME, _TIER_WEIGHTS  # noqa: PLW0603
    if name not in WEIGHT_SCHEMES:
        raise ValueError(
            f'unknown weight scheme {name!r}; known: {sorted(WEIGHT_SCHEMES)}')
    _ACTIVE_SCHEME = name
    _TIER_WEIGHTS = WEIGHT_SCHEMES[name]


def active_weight_scheme() -> str:
    return _ACTIVE_SCHEME
```

Then in `_collapse_labels`, the three references to `_TIER_WEIGHTS` must read the
*current* table rather than a value bound at import. Replace the weight block
(lines 123-138) with:

```python
    tier_weights = WEIGHT_SCHEMES[_ACTIVE_SCHEME]
    if 'confidence_tier' in out.columns:
        mapped = (
            out['confidence_tier']
            .astype(str).str.lower()
            .map(tier_weights)
        )
        if 'confidence_weight' in out.columns:
            # Unknown tiers (Reviewed-*, Ambiguous) fall back to the stamped
            # weight; only rows lacking both get the Moderate default.
            mapped = mapped.fillna(
                pd.to_numeric(out['confidence_weight'], errors='coerce'))
        out['confidence_weight'] = (
            mapped.fillna(tier_weights['moderate']).astype(np.float32)
        )
    elif 'confidence_weight' not in out.columns:
        out['confidence_weight'] = np.float32(tier_weights['moderate'])
```

- [ ] **Step 4: Run the new test and the existing tier test**

Run: `conda run -n crism python -m pytest tests/test_weight_schemes.py tests/test_collapse_reviewed_tier.py -v`
Expected: PASS — all of `tests/test_weight_schemes.py` plus both tests in `tests/test_collapse_reviewed_tier.py` (unmodified).

- [ ] **Step 5: Add the train.py flag**

In `scripts/train.py`, add to the argument parser (near the other vocab flags around line 64):

```python
    parser.add_argument('--weight_scheme', default='level',
                        choices=sorted(__import__('data.dataset',
                                                  fromlist=['x']).WEIGHT_SCHEMES),
                        help='Tier->sample-weight table (default: level, which '
                             'reproduces pre-2026-08-08 weights). Sweepable.')
```

Prefer the explicit import already present in that module; if `data.dataset` is
imported as `data.dataset` at the top of `main()`, use
`choices=sorted(data.dataset.WEIGHT_SCHEMES)` instead of the `__import__` form.

Then, in the LABEL_COLS swap block at line 252 — **before** any dataset is
constructed — add:

```python
    data.dataset.set_weight_scheme(args.weight_scheme)
    logging.info('weight scheme: %s', data.dataset.active_weight_scheme())
```

- [ ] **Step 6: Verify train.py accepts and logs the flag**

Run: `conda run -n crism python scripts/train.py --help | grep -A2 weight_scheme`
Expected: the flag appears with choices `hand_up, level, review_up`.

- [ ] **Step 7: Commit**

```bash
git add data/dataset.py scripts/train.py tests/test_weight_schemes.py
git commit -m "feat: sweepable tier->weight schemes via --weight_scheme"
```

---

### Task 2: Review-session provenance

**Files:**
- Modify: `scripts/build_7cls_dataset.py:140-163` (`_read_hn_tag`), `:355-400` (`load_confirmed_mineral_positives`)
- Test: `tests/test_build_handcore_sources.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: every review fragment carries a `review_session` column with values `'legacy'` or `'v3'`; helper `_session_of(path: str) -> str`.

**Why this exists.** Legacy rows are stamped `confidence_tier='High'` — identical to a hand-labeled High row and indistinguishable from a graded v3 `Reviewed-High` by tier alone. Every later task keys off `review_session`, so this must land first.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_handcore_sources.py`:

```python
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.build_7cls_dataset import _read_hn_tag, _session_of

_LABEL = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other',
          'alteration']


def _hn_frame(n, tier, negative_of, tile='t1250', poly_start=0):
    d = {'tile_id': [tile] * n,
         'polygon_id': [poly_start + i // 10 for i in range(n)],
         'pixel_row': list(range(n)), 'pixel_col': list(range(n)),
         'negative_of': [negative_of] * n,
         'confidence_weight': [1.0] * n, 'confidence_tier': [tier] * n,
         'split': ['train'] * n}
    for c in _LABEL:
        d[c] = [0.0] * n
    for i in range(59):
        d[f'm{i}'] = [0.1] * n
    return pd.DataFrame(d)


def _write_session(root, name, frame):
    d = root / name / 'hard_negatives'
    d.mkdir(parents=True)
    frame.to_parquet(d / 'p_0001.parquet')
    return str(d)


def test_session_of_classifies_dirs():
    assert _session_of('/x/data/mc13_review/hard_negatives') == 'legacy'
    assert _session_of('/x/data/mc13_review_7cls_v3/hard_negatives') == 'v3'


def test_read_hn_tag_stamps_session(tmp_path):
    legacy = _write_session(tmp_path, 'mc13_review',
                            _hn_frame(20, 'High', 'ambiguous'))
    v3 = _write_session(tmp_path, 'mc13_review_7cls_v3',
                        _hn_frame(30, 'Reviewed-High', 'ambiguous',
                                  poly_start=100))
    out = _read_hn_tag([legacy, v3], 'ambiguous')
    assert 'review_session' in out.columns
    assert set(out['review_session']) == {'legacy', 'v3'}
    assert (out['review_session'] == 'legacy').sum() == 20
    assert (out['review_session'] == 'v3').sum() == 30


def test_confirmed_loader_preserves_session_through_template_align(tmp_path):
    """Regression: the loader returns df[template.columns], which would drop
    review_session and make every legacy confirm look like a graded v3 row."""
    from scripts.build_7cls_dataset import load_confirmed_mineral_positives

    for name, tier, n in [('mc13_review', 'High', 10),
                          ('mc13_review_7cls_v3', 'Reviewed-High', 10)]:
        d = tmp_path / name / 'confirmed_pixels'
        d.mkdir(parents=True)
        f = _hn_frame(n, tier, '', poly_start=0)
        f['lcp'] = 1.0
        f.drop(columns=['negative_of']).to_parquet(d / 'p_0001.parquet')

    template = _hn_frame(1, 'High', '').drop(columns=['negative_of'])
    template['bland'] = 0.0
    template['junk'] = 0.0
    out = load_confirmed_mineral_positives(
        [str(tmp_path / 'mc13_review' / 'confirmed_pixels'),
         str(tmp_path / 'mc13_review_7cls_v3' / 'confirmed_pixels')],
        template)
    assert 'review_session' in out.columns
    assert set(out['review_session']) == {'legacy', 'v3'}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -v`
Expected: FAIL — `ImportError: cannot import name '_session_of'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/build_7cls_dataset.py`, add after `_as_dirs` (line 137):

```python
def _session_of(path: str) -> str:
    """Classify a review dir as the ungraded legacy session or the graded v3.

    Legacy rows are stamped confidence_tier='High', identical to hand-labeled
    High rows, so tier cannot identify the session — the path must.
    """
    return 'v3' if '_7cls_v3' in os.path.normpath(path) else 'legacy'
```

In `_read_hn_tag`, stamp each dir's frame before concatenating. Replace the loop
body's `parts.append(table.to_pandas())` with:

```python
        frag = table.to_pandas()
        frag['review_session'] = _session_of(hn_dir)
        parts.append(frag)
```

In `load_confirmed_mineral_positives`, replace the `parts.extend(...)` generator
(lines 367-368) with a per-dir stamping loop:

```python
        session = _session_of(confirmed_dir)
        for f in files:
            frag = pd.read_parquet(os.path.join(confirmed_dir, f))
            frag['review_session'] = session
            parts.append(frag)
```

**Trap — the column would otherwise be silently dropped.** That function ends by
aligning to the caller's template and returning `df[template.columns.tolist()]`
(line 398). The base frame has no `review_session`, so the stamp would be thrown
away on return and every legacy confirm would look like a v3 row. Replace
line 398 with:

```python
    keep = template.columns.tolist()
    if 'review_session' in df.columns and 'review_session' not in keep:
        keep.append('review_session')
    return df[keep]
```

Note the existing `_per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 300)` at
line 380 already caps confirms at 20k before Task 4's legacy cap sees them.
That is fine — 5,000 < 20,000, so the tighter cap still binds.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Verify the existing builder tests still pass**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py tests/test_build_7cls_exclude.py tests/test_split_units.py tests/test_split_units_pixel_leak.py -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_handcore_sources.py
git commit -m "feat: stamp review_session provenance on review fragments"
```

---

### Task 3: v3 grade filter

**Files:**
- Modify: `scripts/build_7cls_dataset.py` (new helper + `main()` wiring)
- Test: `tests/test_build_handcore_sources.py`

**Interfaces:**
- Consumes: `review_session` column from Task 2.
- Produces: `_filter_review_grades(df: pd.DataFrame, grades: list[str]) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_handcore_sources.py`:

```python
from scripts.build_7cls_dataset import _filter_review_grades


def _graded(session, tier, n, poly_start):
    f = _hn_frame(n, tier, 'ambiguous', poly_start=poly_start)
    f['review_session'] = session
    return f


def test_grade_filter_keeps_only_named_v3_grades():
    df = pd.concat([
        _graded('v3', 'Reviewed-High', 10, 0),
        _graded('v3', 'Reviewed-Moderate', 10, 10),
        _graded('v3', 'Reviewed-Low', 10, 20),
    ], ignore_index=True)
    out = _filter_review_grades(df, ['High', 'Moderate'])
    assert set(out['confidence_tier']) == {'Reviewed-High', 'Reviewed-Moderate'}
    assert len(out) == 20


def test_grade_filter_leaves_legacy_rows_untouched():
    # Legacy is stamped tier='High'; the v3 grade filter must not judge it.
    df = pd.concat([
        _graded('legacy', 'High', 15, 0),
        _graded('v3', 'Reviewed-Low', 10, 100),
    ], ignore_index=True)
    out = _filter_review_grades(df, ['High', 'Moderate'])
    assert (out['review_session'] == 'legacy').sum() == 15
    assert (out['review_session'] == 'v3').sum() == 0


def test_grade_filter_is_noop_on_empty():
    assert _filter_review_grades(pd.DataFrame(), ['High']).empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -k grade -v`
Expected: FAIL — `ImportError: cannot import name '_filter_review_grades'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/build_7cls_dataset.py` after `_session_of`:

```python
def _filter_review_grades(df: pd.DataFrame, grades: list[str]) -> pd.DataFrame:
    """Keep only v3 rows whose reviewer grade is in `grades`.

    Legacy rows pass through untouched — they were never graded, and their
    stamped confidence_tier='High' would otherwise be misread as a reviewer
    grade. Legacy admission is decided per-class in _apply_legacy_policy.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    if 'review_session' not in df.columns:
        return df
    keep_tiers = {f'Reviewed-{g}' for g in grades}
    is_v3 = df['review_session'] == 'v3'
    keep = (~is_v3) | df['confidence_tier'].isin(keep_tiers)
    return df[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -k grade -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_handcore_sources.py
git commit -m "feat: v3 reviewer-grade filter for review fragments"
```

---

### Task 4: Legacy per-class policy and confirm cap

**Files:**
- Modify: `scripts/build_7cls_dataset.py` (new helper + `main()` wiring)
- Test: `tests/test_build_handcore_sources.py`

**Interfaces:**
- Consumes: `review_session` (Task 2).
- Produces: `_apply_legacy_policy(df, target_class, legacy_classes, confirm_cap, seed, is_confirm=False) -> pd.DataFrame`.

**Cap semantics (from the spec).** `confirm_cap` (5,000) applies **only** to legacy *confirms*, which concentrate into 10–18 polygons. Legacy alteration hard-negatives keep `MAX_PX_PER_POLYGON` (20,000) because they are already spread across 65 polygons at a median of 255 px each.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_handcore_sources.py`:

```python
from scripts.build_7cls_dataset import _apply_legacy_policy


def test_legacy_dropped_for_unlisted_class():
    df = pd.concat([
        _graded('legacy', 'High', 20, 0),
        _graded('v3', 'Reviewed-High', 10, 100),
    ], ignore_index=True)
    out = _apply_legacy_policy(df, 'bland', ['alteration', 'lcp', 'hcp'],
                               confirm_cap=5000, seed=42)
    assert (out['review_session'] == 'legacy').sum() == 0
    assert (out['review_session'] == 'v3').sum() == 10


def test_legacy_kept_for_listed_class():
    df = _graded('legacy', 'High', 20, 0)
    out = _apply_legacy_policy(df, 'alteration', ['alteration', 'lcp', 'hcp'],
                               confirm_cap=5000, seed=42)
    assert len(out) == 20


def test_legacy_confirm_cap_applies_per_polygon():
    # 3 polygons x 40 rows, cap 10 -> 30 rows kept, legacy confirms only.
    f = _hn_frame(120, 'High', '', poly_start=0)
    f['polygon_id'] = [i // 40 for i in range(120)]
    f['review_session'] = 'legacy'
    out = _apply_legacy_policy(f, 'lcp', ['lcp'], confirm_cap=10, seed=42,
                               is_confirm=True)
    assert len(out) == 30
    assert out.groupby('polygon_id').size().max() == 10


def test_legacy_hard_negatives_ignore_confirm_cap():
    f = _hn_frame(120, 'High', 'alteration', poly_start=0)
    f['polygon_id'] = [i // 40 for i in range(120)]
    f['review_session'] = 'legacy'
    out = _apply_legacy_policy(f, 'alteration', ['alteration'], confirm_cap=10,
                               seed=42, is_confirm=False)
    assert len(out) == 120
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -k legacy -v`
Expected: FAIL — `ImportError: cannot import name '_apply_legacy_policy'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/build_7cls_dataset.py` after `_filter_review_grades`:

```python
def _apply_legacy_policy(df: pd.DataFrame, target_class: str,
                         legacy_classes: list[str], confirm_cap: int,
                         seed: int, is_confirm: bool = False) -> pd.DataFrame:
    """Drop legacy-session rows unless target_class is explicitly admitted.

    The legacy MC13 session was never graded, so it is excluded by default.
    The spec admits it only for alteration/lcp/hcp. Legacy CONFIRMS additionally
    get a tighter per-polygon cap (they concentrate into 10-18 polygons);
    legacy hard-negatives keep MAX_PX_PER_POLYGON.
    """
    if df is None or df.empty or 'review_session' not in df.columns:
        return df if df is not None else pd.DataFrame()
    is_legacy = df['review_session'] == 'legacy'
    if target_class not in legacy_classes:
        return df[~is_legacy].reset_index(drop=True)
    if not is_confirm:
        return df.reset_index(drop=True)
    legacy = _per_polygon_cap(df[is_legacy], confirm_cap, seed)
    return pd.concat([df[~is_legacy], legacy], ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -k legacy -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_handcore_sources.py
git commit -m "feat: per-class legacy admission policy with confirm cap"
```

---

### Task 5: Bland review-only

**Files:**
- Modify: `scripts/build_7cls_dataset.py:311-352` (`_build_base`)
- Test: `tests/test_build_handcore_sources.py`

**Interfaces:**
- Consumes: nothing from Tasks 2-4.
- Produces: `_build_base(path, n_bland_target, hand_minerals='all', bland_sources='all')`; when `bland_sources='review'`, the base parquet's `other > 0` rows are dropped entirely.

**Why dropped, not kept as negatives.** Retaining them with `bland=0` would assert that bland terrain is not bland — false negatives that suppress the bland head. The loss has no per-class row masking.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_handcore_sources.py`:

```python
from scripts.build_7cls_dataset import _build_base


def _base_frame(tmp_path):
    n = 40
    d = {'tile_id': ['t1250'] * n,
         'polygon_id': [i // 10 for i in range(n)],
         'pixel_row': list(range(n)), 'pixel_col': list(range(n)),
         'confidence_weight': [1.0] * n, 'confidence_tier': ['High'] * n,
         'split': ['train'] * n}
    for c in _LABEL:
        d[c] = [0.0] * n
    # First 20 rows are minerals, last 20 are bland ('other').
    d['lcp'] = [1.0] * 20 + [0.0] * 20
    d['other'] = [0.0] * 20 + [1.0] * 20
    for i in range(59):
        d[f'm{i}'] = [0.1] * n
    p = tmp_path / 'base.parquet'
    pd.DataFrame(d).to_parquet(p)
    return str(p)


def test_bland_sources_review_drops_base_other_rows(tmp_path):
    out = _build_base(_base_frame(tmp_path), 300_000, bland_sources='review')
    assert len(out) == 20
    assert (out['bland'] > 0).sum() == 0


def test_bland_sources_all_keeps_base_other_rows(tmp_path):
    out = _build_base(_base_frame(tmp_path), 300_000, bland_sources='all')
    assert len(out) == 40
    assert (out['bland'] > 0).sum() == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -k bland -v`
Expected: FAIL — `TypeError: _build_base() got an unexpected keyword argument 'bland_sources'`

- [ ] **Step 3: Write minimal implementation**

Change the `_build_base` signature (line 311) to:

```python
def _build_base(path: str, n_bland_target: int,
                hand_minerals: str = 'all',
                bland_sources: str = 'all') -> pd.DataFrame:
```

Then replace the bland-tile block (lines 337-351) with:

```python
    # ── bland tile rows: subsample to n_bland_target, or drop entirely ──
    if bland_sources == 'review':
        # Spec 2026-08-08: bland is review-only. These rows are DROPPED, not
        # retained as all-negative background — keeping them with bland=0
        # would assert that bland terrain is not bland, and the loss has no
        # per-class row masking to prevent that false negative.
        print(f'  bland_sources=review: dropping all '
              f'{int(bland_mask.sum()):,} base bland rows')
        out = non_bland.reset_index(drop=True)
        print(f'  base after modification: {len(out):,} rows')
        return out

    bland_df = df[bland_mask].copy()
    bland_df = _subsample(bland_df, n_bland_target, SEED)
    bland_df = _stamp_7cls_cols(bland_df, bland=1.0, junk=0.0, alteration=0.0)
    if len(bland_df):
        bland_df['split'] = assign_unit_balanced_splits(bland_df, ['other'], SEED + 1)
    print(f'  bland tiles after subsample: {len(bland_df):,} '
          f'(target {n_bland_target:,})')
    if len(bland_df):
        print('  bland tiles: unit-balanced achieved fractions:')
        print(achieved_fractions(bland_df, bland_df['split'], ['other'])
              .to_string())

    out = pd.concat([non_bland, bland_df], ignore_index=True)
    print(f'  base after modification: {len(out):,} rows')
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_build_handcore_sources.py -k bland -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_handcore_sources.py
git commit -m "feat: bland_sources=review drops base parquet bland rows"
```

---

### Task 6: CLI wiring and end-to-end dry run

**Files:**
- Modify: `scripts/build_7cls_dataset.py:632-720` (`main()`)
- Test: manual dry run (the unit behaviour is covered by Tasks 1-5)

**Interfaces:**
- Consumes: `_filter_review_grades` (Task 3), `_apply_legacy_policy` (Task 4), `_build_base(bland_sources=…)` (Task 5).
- Produces: the CLI surface documented in the spec.

- [ ] **Step 1: Add the new flags**

In `main()`, after the `--max_bland_raw` argument (line 658), add:

```python
    ap.add_argument('--review_grades', nargs='+', default=['High', 'Moderate'],
                    help='v3 reviewer grades to admit (default: High Moderate). '
                         'Legacy rows are ungraded and unaffected.')
    ap.add_argument('--legacy_classes', nargs='*',
                    default=['alteration', 'lcp', 'hcp'],
                    help='Classes for which the ungraded legacy session is '
                         'admitted. Empty list excludes legacy entirely.')
    ap.add_argument('--legacy_confirm_cap', type=int, default=5_000,
                    help='Per-polygon cap on legacy CONFIRMS only (they '
                         'concentrate into 10-18 polygons). Legacy hard '
                         f'negatives keep MAX_PX_PER_POLYGON '
                         f'({MAX_PX_PER_POLYGON:,}).')
    ap.add_argument('--bland_sources', choices=['all', 'review'], default='all',
                    help="'review': drop the base parquet's bland rows so bland "
                         'comes only from review rejects.')
```

- [ ] **Step 2: Thread the flags through main()**

Change the `_build_base` call (line 670) to:

```python
    base = _build_base(args.base_parquet, n_bland,
                       hand_minerals=args.hand_minerals,
                       bland_sources=args.bland_sources)
```

Then apply the two filters. **ORDERING IS LOAD-BEARING — read this before you
edit.** `main()` builds `all_cols = base.columns.tolist()` and then does
`fragments.append(frag[all_cols])`. The base frame has no `review_session`, so
that line **drops the provenance column** from every review fragment. Your
policy block must therefore run *before* it. Insert immediately before the
`# ── 6. Align schemas and concatenate ──` comment and its
`all_cols = base.columns.tolist()` line. If you insert it after, every filter
silently becomes a no-op and the legacy exclusion does nothing — with no error.

Insert here:

```python
    # ── 5b. Review policy: v3 grade bar + per-class legacy admission ─────────
    print('\nApplying review source policy '
          f'(grades={args.review_grades}, legacy_classes={args.legacy_classes}) …')
    _policy = [
        ('confirmed',   'lcp',        True),
        ('reassigned',  'lcp',        False),
        ('mc13_bland',  'bland',      False),
        ('mc11_bland',  'bland',      False),
        ('junk',        'junk',       False),
        ('alteration',  'alteration', False),
    ]
    _frames = {'confirmed': confirmed, 'reassigned': reassigned,
               'mc13_bland': mc13_bland, 'mc11_bland': mc11_bland,
               'junk': junk_df, 'alteration': alt_df}
    for _name, _cls, _is_confirm in _policy:
        _f = _frames[_name]
        if _f is None or len(_f) == 0:
            continue
        _before = len(_f)
        _f = _filter_review_grades(_f, args.review_grades)
        _f = _apply_legacy_policy(_f, _cls, args.legacy_classes,
                                  args.legacy_confirm_cap, SEED,
                                  is_confirm=_is_confirm)
        _frames[_name] = _f
        print(f'  {_name}: {_before:,} -> {len(_f):,} rows')
    confirmed, reassigned = _frames['confirmed'], _frames['reassigned']
    mc13_bland, mc11_bland = _frames['mc13_bland'], _frames['mc11_bland']
    junk_df, alt_df = _frames['junk'], _frames['alteration']
```

Note the `confirmed` fragment carries olivine/lcp/hcp together; it is keyed to
`'lcp'` because lcp/hcp are the classes the spec admits legacy for, and the
spec accepts that legacy olivine rides along as a dual label.

- [ ] **Step 3: Belt-and-braces — ensure the provenance column never ships**

`review_session` is a build-time artifact, not a training column. The
`frag[all_cols]` projection already strips it, so this is a guard against the
base frame ever carrying it. Immediately before the parquet write in `main()`,
add:

```python
    if 'review_session' in out.columns:
        out = out.drop(columns=['review_session'])
```

- [ ] **Step 3b: Prove the ordering is right**

A no-op filter is silent, so verify the policy block actually sees provenance.
Temporarily add a print inside the policy loop and run the Step 5 dry run:

```python
        print(f'    {_name}: sessions={sorted(set(_f["review_session"]))}'
              if 'review_session' in _f.columns else
              f'    {_name}: NO review_session COLUMN — ordering is wrong!')
```

Expected: fragments report `sessions=['legacy', 'v3']` or `['v3']` — never the
"NO review_session COLUMN" branch. Remove the print before committing.

- [ ] **Step 4: Verify default behaviour is unchanged**

Run: `conda run -n crism python scripts/build_7cls_dataset.py --dry_run --max_bland_raw 20000 2>&1 | tail -30`
Expected: completes; the printed per-source counts match a run of the same
command at the previous commit (defaults are `bland_sources=all`,
`legacy_classes=[alteration,lcp,hcp]`, `review_grades=[High,Moderate]`).
Record both outputs and diff them.

- [ ] **Step 5: Dry-run the hand-core recipe**

Run:

```bash
conda run -n crism python scripts/build_7cls_dataset.py \
    --dry_run --max_bland_raw 20000 \
    --bland_sources review \
    --review_grades High Moderate \
    --legacy_classes alteration lcp hcp \
    --legacy_confirm_cap 5000 \
    --ndviz_dir '' \
    --out data/mrral_pixels_7cls_handcore.parquet 2>&1 | tail -40
```

Expected: base drops its bland rows; per-source counts shrink at the policy
step; no traceback. `--max_bland_raw` caps raw reads, so absolute counts will
not match a real build — this checks wiring, not volumes.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_7cls_dataset.py
git commit -m "feat: wire hand-core source policy flags into the 7cls builder"
```

- [ ] **Step 7: Full build (HPC or long local run)**

Run without `--dry_run`/`--max_bland_raw`:

```bash
conda run -n crism python scripts/build_7cls_dataset.py \
    --bland_sources review --review_grades High Moderate \
    --legacy_classes alteration lcp hcp --legacy_confirm_cap 5000 \
    --ndviz_dir '' --out data/mrral_pixels_7cls_handcore.parquet
```

Then verify the composition against the spec's projected table:

```bash
conda run -n crism python -c "
import pandas as pd
df = pd.read_parquet('data/mrral_pixels_7cls_handcore.parquet',
                     columns=['olivine_t1','olivine_t2','lcp','hcp',
                              'plagioclase','bland','alteration','junk',
                              'confidence_tier','split'])
print(len(df), 'rows')
for c in ['olivine_t1','olivine_t2','lcp','hcp','plagioclase','bland',
          'alteration','junk']:
    print(f'  {c:12} {(df[c]>0).sum():>9,}')
print(df['confidence_tier'].value_counts().to_string())
print(df['split'].value_counts().to_string())
"
```

Expected: `bland` ≈ 743.8k and sourced only from review; `junk` ≈ 165.4k;
`alteration` ≈ 259k (111.8k hand + 147.5k review); no `Reviewed-Low` tier
present.

---

## Self-Review

**Spec coverage.** Decision 1 (grade bar) → Task 3. Decision 2 (bland review-only, rows dropped) → Task 5. Decision 3 (alteration uncapped on the hand side) → no code change needed; it is the *absence* of a cap, verified by the Task 6 Step 7 composition check. Decision 4 (junk review-only) → falls out of Task 4, since junk has no hand source and legacy is excluded for it. Decision 5 (legacy per-class) → Task 4. Decision 6 (5k confirm cap, 20k for legacy alteration HN) → Task 4, `is_confirm` branch. Decision 7 (ndviz disabled) → Task 6, `--ndviz_dir ''`. Decision 8 (sweepable weights) → Task 1. Output naming → Task 6.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code. Task 6 Step 2 names the loop's frame variables explicitly rather than saying "similar to above".

**Type consistency.** `_session_of(str) -> str`, `_filter_review_grades(DataFrame, list[str]) -> DataFrame`, `_apply_legacy_policy(DataFrame, str, list[str], int, int, bool) -> DataFrame`, `_build_base(..., bland_sources: str)`, `set_weight_scheme(str) -> None`, `active_weight_scheme() -> str`. Names match between definition and call sites in Task 6.

**Resolved during review.** An earlier draft left the `load_confirmed_mineral_positives` stamp vague. Reading `scripts/build_7cls_dataset.py:355-398` showed the function returns `df[template.columns.tolist()]`, which would have silently discarded `review_session` and made every legacy confirm indistinguishable from a graded v3 row — defeating Task 4 entirely. Task 2 now carries the exact replacement for both the read loop and the return, plus a regression test that catches the drop.
