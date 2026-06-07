# Ambiguous-Row Loss Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Use the 34 k `negative_of='ambiguous'` rows in `data/mc13_review/hard_negatives.parquet` as strong all-class negatives during fine-tuning, so the classifier learns to push every logit DOWN on pixels that are spectrally unlike any of the four primary minerals.

**Architecture:** Extend `scripts/build_review_augmented_train.py` to optionally include ambiguous rows in the training parquet, tagged with a new `confidence_tier='Ambiguous'` and a higher `confidence_weight` (default 3.0). No `train.py` or loss-function changes required for the first iteration — the existing BCE/ASL loss already pushes logits toward 0 when label==0; the higher weight just amplifies that pressure for these specifically curated negatives.

**Tech Stack:** Same as the rest of the pipeline — Python 3.11, pandas, pyarrow. Builds on `scripts/build_review_augmented_train.py` from 2026-06-07.

**Spec note:** there is no separate spec doc; this plan IS the spec because the design is small and self-contained.

---

## Background

The MC13 polygon review produced three categories of pixels:

| category | rows | confidence_tier | how the loss should see them |
|---|---|---|---|
| confirmed mineral | 1,182,088 | Reviewed | strong positives (label=1, weight=2.0) |
| reject + corrected (mineral) | 68,562 | Reviewed | strong positives (label=1 for corrected class, weight=2.0) |
| reject + `ambiguous` tag | 34,126 | **NEW: Ambiguous** | strong all-class negatives (label=0 for every class, weight=3.0) |

The third bucket is currently dropped by `build_review_augmented_train.py` because the user flagged them as "not any of our minerals". They have spectra we definitely *don't* want the classifier to call hcp/lcp/olivine/plag/bland. The existing BCE/ASL loss applied to label-all-zeros already pushes logits down — boosting the row weight is enough to make this a meaningful signal without touching the loss function.

If the simple-weight approach plateaus, a follow-up could add a dedicated "no-class" loss term (penalize `max(softmax(logits))`). Not in scope for this plan.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/build_review_augmented_train.py` | Modify: add `--include_ambiguous` flag + tier/weight controls. |
| `tests/test_build_review_augmented_train.py` | Create: unit tests for inclusion logic + tier/weight wiring. |
| `scripts/hpc_finetune_with_review.slurm` | Modify: pass the include-ambiguous flag in the build step. |

The build script already exists; we only extend it. No train.py changes.

---

### Task 1: Add `--include_ambiguous` to the parquet builder

**Files:**
- Modify: `scripts/build_review_augmented_train.py`
- Create: `tests/test_build_review_augmented_train.py`

- [ ] **Step 1: Write the failing test**

Path: `tests/test_build_review_augmented_train.py`

```python
"""Tests for the review-augmented training parquet builder."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# We'll invoke the builder as a subprocess so the CLI surface is exercised.
SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'scripts', 'build_review_augmented_train.py',
)


def _confirmed_schema():
    return (
        ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
        + [f'm{i}' for i in range(59)]
        + ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
        + ['confidence_weight', 'confidence_tier', 'split']
    )


def _make_row(uid_int=1, label_col='hcp', tile='t0001', pr=0, pc=0):
    cols = _confirmed_schema()
    data = {c: [0.0] * 1 for c in cols if c.startswith('m')}
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[c] = [0.0]
    data[label_col] = [1.0]
    data['tile_id'] = [tile]
    data['polygon_id'] = [uid_int]
    data['pixel_row'] = [pr]
    data['pixel_col'] = [pc]
    data['confidence_weight'] = [1.0]
    data['confidence_tier'] = ['High']
    data['split'] = ['train']
    return pd.DataFrame(data, columns=cols)


def _make_hard_neg_row(uid_int, negative_of='', label_col=None, tile='t0001', pr=10):
    cols = _confirmed_schema() + ['negative_of']
    data = {c: [0.0] for c in cols if c.startswith('m')}
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[c] = [0.0]
    if label_col:
        data[label_col] = [1.0]
    data['tile_id'] = [tile]
    data['polygon_id'] = [uid_int]
    data['pixel_row'] = [pr]
    data['pixel_col'] = [0]
    data['confidence_weight'] = [1.0]
    data['confidence_tier'] = ['High']
    data['split'] = ['train']
    data['negative_of'] = [negative_of]
    return pd.DataFrame(data, columns=cols)


def _write_inputs(tmp_path, n_existing=10, n_confirmed=5, n_ambiguous=3):
    """Synth existing baseline + confirmed + hard_negatives parquets."""
    existing = pd.concat([_make_row(uid_int=i, tile=f't{i:04d}', pr=i)
                          for i in range(n_existing)], ignore_index=True)
    existing.to_parquet(tmp_path / 'mrral_pixels.parquet', index=False)

    confirmed = pd.concat([_make_row(uid_int=100 + i, tile='t1000', pr=100 + i)
                            for i in range(n_confirmed)], ignore_index=True)
    confirmed.to_parquet(tmp_path / 'confirmed_pixels.parquet', index=False)

    rows = []
    # corrected-mineral reject (negative_of blank, positive label set)
    rows.append(_make_hard_neg_row(200, negative_of='', label_col='olivine_t1', pr=200))
    # ambiguous rejects (all-zero labels, negative_of='ambiguous')
    for i in range(n_ambiguous):
        rows.append(_make_hard_neg_row(300 + i, negative_of='ambiguous', pr=300 + i))
    hn = pd.concat(rows, ignore_index=True)
    hn.to_parquet(tmp_path / 'hard_negatives.parquet', index=False)


def _run_builder(tmp_path, *extra):
    out = tmp_path / 'merged.parquet'
    cmd = [
        sys.executable, SCRIPT,
        '--existing', str(tmp_path / 'mrral_pixels.parquet'),
        '--confirmed', str(tmp_path / 'confirmed_pixels.parquet'),
        '--hard_negatives', str(tmp_path / 'hard_negatives.parquet'),
        '--out', str(out),
    ] + list(extra)
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return pd.read_parquet(out)


def test_default_excludes_ambiguous(tmp_path):
    """Without --include_ambiguous, only confirmed + corrected reject rows
    are folded into the training parquet."""
    _write_inputs(tmp_path)
    df = _run_builder(tmp_path)
    assert (df['confidence_tier'] == 'Ambiguous').sum() == 0
    # 10 existing + 5 confirmed + 1 corrected-mineral reject = 16
    assert len(df) == 16


def test_include_ambiguous_adds_rows_with_correct_tier(tmp_path):
    """--include_ambiguous folds the negative_of='ambiguous' rows in with
    tier='Ambiguous' and the user-specified weight."""
    _write_inputs(tmp_path, n_ambiguous=3)
    df = _run_builder(tmp_path, '--include_ambiguous',
                       '--ambiguous_weight', '3.0')
    ambig = df[df['confidence_tier'] == 'Ambiguous']
    assert len(ambig) == 3
    assert (ambig['confidence_weight'] == 3.0).all()
    # All label columns must be zero on ambiguous rows
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        assert (ambig[c] == 0.0).all()
    # split is train (ambiguous rows are training-only)
    assert (ambig['split'] == 'train').all()


def test_ambiguous_weight_independent_of_review_weight(tmp_path):
    """--ambiguous_weight defaults to 3.0 and is independent of
    --review_weight (which controls confirmed/corrected-mineral rows)."""
    _write_inputs(tmp_path)
    df = _run_builder(tmp_path, '--include_ambiguous',
                       '--review_weight', '2.0',
                       '--ambiguous_weight', '5.0')
    reviewed = df[df['confidence_tier'] == 'Reviewed']
    ambig = df[df['confidence_tier'] == 'Ambiguous']
    assert (reviewed['confidence_weight'] == 2.0).all()
    assert (ambig['confidence_weight'] == 5.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism pytest tests/test_build_review_augmented_train.py -v`
Expected: failures on `--include_ambiguous` not being recognized (or rows not being included).

- [ ] **Step 3: Implement the changes in `scripts/build_review_augmented_train.py`**

Three changes:

**(a)** Add the new CLI args next to the existing ones (around the existing `--review_weight` arg):

```python
    ap.add_argument('--include_ambiguous', action='store_true',
                    help='Also include negative_of=\'ambiguous\' hard-neg rows '
                         'as all-class negatives (label=0 for every class).')
    ap.add_argument('--ambiguous_weight', type=float, default=3.0,
                    help='confidence_weight for ambiguous rows. Higher than '
                         '--review_weight by default so the model sees them '
                         'as the strongest negatives.')
```

Plus a constant near the top:

```python
AMBIGUOUS_TIER = 'Ambiguous'
```

**(b)** Add a new loader for ambiguous rows. Insert next to `_load_corrected_hard_neg`:

```python
def _load_ambiguous_hard_neg(path: str) -> pd.DataFrame:
    """From hard_negatives.parquet, take rows tagged ambiguous (all label
    columns already 0; negative_of='ambiguous'). Returned in the confirmed
    schema (drops the negative_of column) for concat compatibility."""
    df = pd.read_parquet(path)
    is_ambig = df['negative_of'].astype(str) == 'ambiguous'
    df = df[is_ambig]
    return df[confirmed_schema_columns()]
```

**(c)** In `main()`, after the existing `review_parts.append(hn)` block, add:

```python
        if args.include_ambiguous:
            ambig_df = _load_ambiguous_hard_neg(args.hard_negatives)
            if len(ambig_df) > 0:
                ambig_df = ambig_df.copy()
                ambig_df['confidence_weight'] = args.ambiguous_weight
                ambig_df['confidence_tier'] = AMBIGUOUS_TIER
                ambig_df['split'] = 'train'
                review_parts.append(ambig_df)
                print(f'  + {len(ambig_df):,} ambiguous-tagged rows '
                      f'(weight={args.ambiguous_weight}, tier={AMBIGUOUS_TIER})')
```

Also update the deferred-print near the existing ambiguous-row count message:

```python
        # Remove the old "deferred" message and replace with:
        if not args.include_ambiguous:
            ambig = _ambiguous_row_count(args.hard_negatives)
            if ambig:
                print(f'  (deferred: {ambig:,} ambiguous-tagged rows — '
                      f'pass --include_ambiguous to fold them in)')
```

**Important:** the existing code sets `review['confidence_weight'] = args.review_weight` and `review['confidence_tier'] = REVIEW_TIER` AFTER the concat, which would overwrite the ambiguous-row tier/weight. Move the per-part weight/tier assignment INTO the loaders or into a per-part loop so each bucket keeps its own settings.

Concretely, change the existing block:

```python
    review = pd.concat(review_parts, ignore_index=True)
    review['confidence_weight'] = args.review_weight
    review['confidence_tier'] = REVIEW_TIER
    review['split'] = 'train'
```

to assign tier/weight inside the loaders instead. Update both `_load_confirmed` and `_load_corrected_hard_neg` to take a `weight` and `tier` arg, and set them on the returned df. The ambiguous loader already does this in the new `(c)` block above. After this refactor, `review = pd.concat(review_parts, ignore_index=True)` is the only line left in the concat block (tier/weight are already set per-part).

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism pytest tests/test_build_review_augmented_train.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Manual smoke run on real data**

```bash
conda run -n crism python -u scripts/build_review_augmented_train.py \
    --include_ambiguous --dry_run 2>&1 | tail -20
```

Confirm the "Reviewed" + "Ambiguous" tier counts in the train-split summary look right: Reviewed should be ~1.25 M and Ambiguous ~34 k.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_review_augmented_train.py tests/test_build_review_augmented_train.py
git commit -m "feat(review): --include_ambiguous folds 34k all-class negatives into train"
```

---

### Task 2: Wire ambiguous-loss into the HPC slurm script

**Files:**
- Modify: `scripts/hpc_finetune_with_review.slurm` (optional second variant)
- Create: `scripts/hpc_finetune_with_review_ambiguous.slurm`

- [ ] **Step 1: Create the new slurm**

Path: `scripts/hpc_finetune_with_review_ambiguous.slurm`

Same as `hpc_finetune_with_review.slurm`, but:
- Job name: `crism_ft_with_review_ambig`
- Run name: `ft_with_review_ambig`
- Add `--include_ambiguous` to the parquet build step
- Use the same patch cache (the ambiguous rows just add ~34 k entries, rebuilt cache will reflect them)
- Output: `logs/ft_with_review_ambig_%j.log`

Note: the patch cache MUST be rebuilt to include the ambiguous rows. The slurm header should mention this.

- [ ] **Step 2: Commit**

```bash
git add scripts/hpc_finetune_with_review_ambiguous.slurm
git commit -m "feat(hpc): slurm variant that includes ambiguous-row negatives"
```

---

## Acceptance

- `scripts/build_review_augmented_train.py --include_ambiguous` produces a parquet whose train split contains a new tier `Ambiguous` with ~34 k rows, all label cols = 0, weight = 3.0.
- Three unit tests cover: default-exclusion, inclusion-with-correct-tier, weight-independence-from-review-weight.
- The new slurm script is ready to submit on HPC after the cache rebuild.

## Future work (out of scope)

- Dedicated "no-class" auxiliary loss: penalize `max(softmax(logits))` on ambiguous rows. Would push the model toward genuinely-uncertain predictions on these pixels rather than just "everything is unlikely". Implement only if the weighted-BCE approach plateaus.
- Hard-negative-mining variant: on rows tagged `negative_of='hcp'` (etc., not ambiguous), use a class-specific stronger weight only for the predicted-class output channel, not all five.
