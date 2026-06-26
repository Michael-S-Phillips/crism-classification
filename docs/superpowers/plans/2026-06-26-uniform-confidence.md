# Uniform Confidence on All Active Assignments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make every actively-assigned review class (confirm OR any reassignment) carry the reviewer's confidence weight; plain rejects stay discarded.

**Architecture:** Persistence applies confidence to all assignment branches and reverts alteration to a `negative_of='alteration'` tag; the build's three review loaders preserve the per-polygon weight instead of clobbering it to a flat 2.0.

**Tech Stack:** Python, pandas, pytest.

Spec addendum: `docs/superpowers/specs/2026-06-24-review-confidence-weighting-design.md` (Addendum 2026-06-26).

---

### Task A: Persistence — alteration as tag + confidence on all assignment branches

**Files:** Modify `scripts/review/persistence.py`; Test `tests/test_review_persistence.py`.

- [ ] **Step 1: Write failing tests.** Add to `tests/test_review_persistence.py`:

```python
def test_alteration_is_not_a_mineral_class():
    from scripts.review.persistence import _is_mineral_class
    assert _is_mineral_class('alteration') is False
    assert _is_mineral_class('olivine') is True


def test_hard_negatives_alteration_tag_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::alt::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='alteration',
        confidence='Moderate',
    )
    df = pd.read_parquet(str(pq))
    assert df['negative_of'].iloc[0] == 'alteration'
    assert df['alteration'].iloc[0] == 0.0          # tag, not a positive label
    assert df['confidence_weight'].iloc[0] == 0.75
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Moderate'


def test_hard_negatives_ambiguous_tag_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::amb::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='ambiguous',
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['negative_of'].iloc[0] == 'ambiguous'
    assert df['confidence_weight'].iloc[0] == 0.5
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Low'
```

Also UPDATE the existing `test_hard_negatives_tag_reject_keeps_fixed_weight` (added in Task 2): it asserted an ambiguous reject keeps weight 1.0 / tier 'High'. That behavior is intentionally changing — ambiguous tags now carry confidence. Replace its assertions with:

```python
    assert df['confidence_weight'].iloc[0] == 0.5
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Low'
    assert df['negative_of'].iloc[0] == 'ambiguous'
```

The `test_hard_negatives_blank_corrected_keeps_fixed_weight` test (pure reject) must remain unchanged and still pass (weight 1.0 / tier 'High').

- [ ] **Step 2: Run tests to verify they fail.**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -k "alteration or ambiguous or tag_reject" -v`
Expected: FAIL — `_is_mineral_class('alteration')` is currently True; tag branch currently writes fixed 1.0/High.

- [ ] **Step 3: Remove `'alteration'` from `_is_mineral_class`.** Current:

```python
def _is_mineral_class(label_class: str) -> bool:
    """True if ``label_class`` denotes a positive mineral assignment (vs. a
    non-mineral tag like 'ambiguous' that should be recorded as a negative)."""
    return label_class in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration') \
           or label_class in _BLAND_ALIASES
```

becomes:

```python
def _is_mineral_class(label_class: str) -> bool:
    """True if ``label_class`` denotes a positive label assignment (vs. a
    tag like 'alteration'/'ambiguous' recorded via negative_of with no positive
    label). Alteration is a tag: it has a label column but the 7-class build
    ingests it from negative_of='alteration' (load_alteration_mc11), matching
    the existing review data."""
    return label_class in ('olivine', 'lcp', 'hcp', 'plagioclase') \
           or label_class in _BLAND_ALIASES
```

- [ ] **Step 4: Apply confidence to the tag branch in `HardNegativesWriter.append_polygon`.** The current body initializes `weight, tier = 1.0, 'High'` and only the mineral branch overrides them. Change the `else` (non-mineral tag) branch to also set the confidence weight/tier. The three branches become:

```python
        weight, tier = 1.0, 'High'
        if not corrected_class:
            # pure reject: "not {predicted_class}", no positive label, fixed
            # weight (discarded by the build — no loader reads negative_of=predicted)
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = predicted_class
        elif _is_mineral_class(corrected_class):
            # active mineral/bland assignment → positive label, confidence-weighted
            label = _label_dict_for(corrected_class)
            negative_of = ''
            weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]
            tier = f'Reviewed-{confidence}'
        else:
            # active tag assignment (alteration / ambiguous) → recorded via
            # negative_of, no positive label, confidence-weighted
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = corrected_class
            weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]
            tier = f'Reviewed-{confidence}'
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra, label,
                                weight=weight, tier=tier)
        df['negative_of'] = negative_of
        df = df[hard_negatives_schema_columns()]
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        _atomic_write_parquet(df, path)
```

Also update the method docstring (added in Task 2 polish) so it no longer says tag rejects keep fixed weight — they now carry confidence; only pure rejects stay fixed.

- [ ] **Step 5: Run tests.**

Run: `conda run -n crism python -m pytest tests/test_review_persistence.py -v`
Expected: ALL PASS (including the updated ambiguous test and the unchanged pure-reject test).

- [ ] **Step 6: Commit.**

```bash
git add scripts/review/persistence.py tests/test_review_persistence.py
git commit -m "review: alteration as tag; confidence on all active-assignment branches"
```

---

### Task B: Build — review loaders preserve per-polygon confidence weight

**Files:** Modify `scripts/build_7cls_dataset.py`; Test `tests/test_build_7cls_confidence.py`.

- [ ] **Step 1: Write failing tests.** Append to `tests/test_build_7cls_confidence.py`:

```python
from scripts.build_7cls_dataset import load_junk_ambiguous, load_alteration_mc11


def _tagged_hardneg_row(polygon_id, tag, weight, tier, n=3, mc13=True):
    tid = 't1250' if mc13 else 't1100'  # both are MC13-range here; tag-based loaders ignore region
    d = {
        'tile_id': [tid] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.arange(n), 'pixel_col': np.arange(n),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d['confidence_weight'] = np.full(n, weight)
    d['confidence_tier'] = [tier] * n
    d['split'] = ['train'] * n
    d['negative_of'] = [tag] * n
    return pd.DataFrame(d)


def test_junk_preserves_per_polygon_weight(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _tagged_hardneg_row(1, 'ambiguous', 0.5, 'Reviewed-Low').to_parquet(
        hdir / 'p_amb.parquet', index=False)
    out = load_junk_ambiguous(str(hdir))
    assert (out['junk'] > 0).all()
    assert (out['confidence_weight'] == 0.5).all()
    assert (out['confidence_tier'] == 'Reviewed-Low').all()


def test_alteration_preserves_per_polygon_weight(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _tagged_hardneg_row(2, 'alteration', 0.75, 'Reviewed-Moderate').to_parquet(
        hdir / 'p_alt.parquet', index=False)
    out = load_alteration_mc11(str(hdir))
    assert (out['alteration'] > 0).all()
    assert (out['confidence_weight'] == 0.75).all()
    assert (out['confidence_tier'] == 'Reviewed-Moderate').all()


def test_bland_review_preserves_per_polygon_weight(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    # bland reassignment: negative_of='', other=1.0, confidence-weighted
    row = _hardneg_row(3, 'other')  # _hardneg_row sets confidence_weight=0.5, tier Reviewed-Low, negative_of=''
    row.to_parquet(hdir / 'p_bland.parquet', index=False)
    out = load_bland_review(str(hdir), 'mc13_blands', mc13=True, seed_offset=10,
                            n_bland=1000)
    assert (out['bland'] > 0).all()
    assert (out['confidence_weight'] == 0.5).all()
    assert (out['confidence_tier'] == 'Reviewed-Low').all()
```

(`_hardneg_row` already exists in this file from the earlier Task 5 tests and sets `confidence_weight=0.5`, `confidence_tier='Reviewed-Low'`, `negative_of=''`.)

- [ ] **Step 2: Run tests to verify they fail.**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py -k "preserves_per_polygon_weight" -v`
Expected: FAIL — the three loaders currently overwrite to `REVIEW_WEIGHT=2.0` / `'Reviewed'`.

- [ ] **Step 3: `load_bland_review` — preserve weight.** Replace its final two stamping lines:

```python
    df = _stamp_7cls_cols(df, bland=1.0, junk=0.0, alteration=0.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    return df
```

with:

```python
    df = _stamp_7cls_cols(df, bland=1.0, junk=0.0, alteration=0.0)
    df = _fill_confidence_defaults(df)
    return df
```

- [ ] **Step 4: `load_junk_ambiguous` — preserve weight.** Replace:

```python
    df = _stamp_7cls_cols(df, bland=0.0, junk=1.0, alteration=0.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    return df
```

with:

```python
    df = _stamp_7cls_cols(df, bland=0.0, junk=1.0, alteration=0.0)
    df = _fill_confidence_defaults(df)
    return df
```

- [ ] **Step 5: `load_alteration_mc11` — preserve weight.** Replace:

```python
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=1.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    return df
```

with:

```python
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=1.0)
    df = _fill_confidence_defaults(df)
    return df
```

- [ ] **Step 6: Remove the now-unused `REVIEW_WEIGHT` constant.** Delete the line `REVIEW_WEIGHT = 2.0` near the top of the file. Then verify it's truly unused: `grep -n REVIEW_WEIGHT scripts/build_7cls_dataset.py` must return nothing.

- [ ] **Step 7: Run tests + full build/persistence/collapse suite.**

Run: `conda run -n crism python -m pytest tests/test_build_7cls_confidence.py -v`
Then: `conda run -n crism python -m pytest tests/ -k "build or persistence or collapse" -q`
Expected: ALL PASS.

- [ ] **Step 8: Commit.**

```bash
git add scripts/build_7cls_dataset.py tests/test_build_7cls_confidence.py
git commit -m "build: review loaders preserve per-polygon confidence weight; drop flat REVIEW_WEIGHT"
```

---

## Final verification

- [ ] Run `conda run -n crism python -m pytest tests/ -k "review or build or collapse" -q` — all pass.
- [ ] `git push origin master:feature/spatial-mae-pretraining`
