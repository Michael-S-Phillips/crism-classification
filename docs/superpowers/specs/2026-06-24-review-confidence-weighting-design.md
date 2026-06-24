# Confidence-Weighted Polygon Review — Design

**Date:** 2026-06-24
**Status:** Approved (design); pending implementation plan

## Goal

Let a reviewer assign **Low / Moderate / High** confidence to each polygon
decision in the MC review app, mapping to per-polygon training sample weights
**0.5 / 0.75 / 1.0**, and thread that weight through to the next training cycle
so high-confidence polygons dominate the loss and low-confidence ones
contribute proportionally less.

## Scope

Confidence applies to the **positive-label** decisions the reviewer makes:

- **Confirm** — polygon confirmed as its predicted (and any co-occurring) mineral.
- **Reject → mineral reassignment** — polygon rejected but corrected to a real
  mineral (olivine / lcp / hcp / plagioclase).

Confidence does **not** apply to:

- Pure rejects (no corrected class) — recorded as `negative_of=predicted_class`.
- Non-mineral tag rejects (bland / alteration / ambiguous).

These keep their existing fixed weighting and tier (`weight=1.0`, `tier='High'`).

## Confidence scale and weight semantics

```
REVIEW_CONFIDENCE_WEIGHTS = {'High': 1.0, 'Moderate': 0.75, 'Low': 0.5}
```

Review rows are stamped with **two** fields:

- `confidence_weight` — the literal float (0.5 / 0.75 / 1.0).
- `confidence_tier` — a descriptive label `Reviewed-Low` / `Reviewed-Moderate`
  / `Reviewed-High`.

**Why a `Reviewed-*` tier label rather than `High`/`Moderate`/`Low`:**
`data/dataset.py::_collapse_labels` lowercases `confidence_tier` and maps it
through the global `_TIER_WEIGHTS = {high:1.0, moderate:0.85, low:0.70}`. Rows
whose tier is **not** in that map fall through to the stamped
`confidence_weight` verbatim. The base parquet already uses High/Moderate/Low
for ~1.08M rows (mapped to 1.0/0.85/0.70). Using `Reviewed-*` labels means the
review weights (0.5/0.75/1.0) are honored exactly **without** re-weighting any
existing base data. `_collapse_labels` needs **no change** — the fallthrough
already exists (it was the audit-bug-#1 fix).

## Components

### 1. UI — `scripts/review/app.py`

Add a confidence selector near the decision buttons:

```python
confidence = st.radio('confidence', ['High', 'Moderate', 'Low'],
                      horizontal=True, index=0)  # default High
```

- Default **High** — matches today's implicit weight-1.0 behavior; the reviewer
  deliberately downgrades when unsure.
- `confidence` is passed into `_record(decision)`.
- It is logged to `decisions.csv` for every decision (audit trail), but only
  *applied* to the parquet weight for confirms and mineral reassignments.

`_record` wiring:

- `log.append(... confidence=confidence ...)`.
- `confirmed_writer.append_polygon(..., confidence=confidence)`.
- `hardneg_writer.append_polygon(..., confidence=confidence)` — the writer
  itself decides whether to apply it (mineral reassignment) or ignore it
  (pure/tag reject).

### 2. Persistence — `scripts/review/persistence.py`

- `_DECISION_COLS` gains a trailing `confidence` column. The existing
  `_migrate_schema_if_needed` rewrites older `decisions.csv` files with the new
  column (empty for historical rows) automatically.
- `confirmed_schema_columns()` is unchanged — it already includes
  `confidence_weight` and `confidence_tier`.
- New module constant:
  ```python
  REVIEW_CONFIDENCE_WEIGHTS = {'High': 1.0, 'Moderate': 0.75, 'Low': 0.5}
  ```
- `_rows_for_polygon(...)` gains `weight: float = 1.0` and
  `tier: str = 'High'` params, replacing the hardcoded
  `confidence_weight = 1.0` / `confidence_tier = 'High'`.
- `ConfirmedPixelsWriter.append_polygon(...)` gains `confidence: str = 'High'`;
  derives `weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]` and
  `tier = f'Reviewed-{confidence}'`, passes both to `_rows_for_polygon`.
- `HardNegativesWriter.append_polygon(...)` gains `confidence: str = 'High'`.
  Only when `corrected_class` is a mineral (the reassignment branch) does it
  stamp `weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]` /
  `tier = f'Reviewed-{confidence}'`. The pure-negative and non-mineral-tag
  branches keep `weight=1.0` / `tier='High'`.

### 3. Training ingestion — `scripts/build_7cls_dataset.py`

**3a. Preserve per-polygon confirmed weights.**
`load_confirmed_mineral_positives` currently overwrites every row with
`confidence_weight = REVIEW_WEIGHT (2.0)` and `confidence_tier = 'Reviewed'`.
Change it to **keep** the `confidence_weight` / `confidence_tier` already in the
confirmed parquet. The per-polygon cap (`_per_polygon_cap`, 20k) and tile-level
split assignment are retained.

- *Migration effect:* existing MC13 confirmed pixels carry `tier='High'`,
  `weight=1.0` (written by the old persistence code), so they collapse to
  weight **1.0** — down from the flat **2.0** the current build applies. This is
  intentional and consistent with "High = 1.0".
- `REVIEW_WEIGHT` constant is removed (no longer used for confirmed positives).

**3b. Route reassigned minerals out of the bland pool (latent-bug fix).**
Reassigned minerals (reject→olivine/lcp/hcp/plag) live in
`hard_negatives` with `negative_of=''` and the corrected mineral column = 1.0.
`load_bland_review` reads `_read_hn_tag(tag=None)` (negative_of null/empty) and
stamps **every** such row `bland=1.0` — so the 27 reassigned-olivine polygons in
MC13 are currently mistrained as bland.

Fix: after reading the `negative_of=''` pool, split rows by their label column
**before** stamping:

- rows with `other=1.0` (and all mineral cols 0) → bland pool (unchanged path).
- rows with a mineral col = 1.0 → a new **mineral-positive** fragment, ingested
  like confirmed positives: keep their stamped `confidence_weight` /
  `confidence_tier`, apply `_per_polygon_cap(20k)`, tile-level split.

A small helper isolates the routing so `load_bland_review` stays focused on true
blands.

### 4. `data/dataset.py`

No change. The unknown-tier fallthrough in `_collapse_labels` already maps
`Reviewed-*` rows to their stamped weight.

## Data flow

```
reviewer picks confidence (High/Moderate/Low)
        │
app.py _record(decision, confidence)
        ├─ decisions.csv               (confidence column, all decisions)
        ├─ ConfirmedPixelsWriter        (confirm → weight+Reviewed-tier)
        └─ HardNegativesWriter          (reassign-mineral → weight+Reviewed-tier;
                                         tag/pure reject → fixed 1.0/High)
        │
build_7cls_dataset.py
        ├─ load_confirmed_mineral_positives → keep per-polygon weight, cap, split
        └─ hard_negatives negative_of='' → route: other→bland | mineral→positives
        │
data/dataset.py _collapse_labels → Reviewed-* falls through to stamped weight
        │
training loss: per-pixel sample weight = stamped weight
```

## Testing

- **persistence round-trip** (`scripts/review/`): confirm a polygon with each of
  High/Moderate/Low → the per-polygon parquet has `confidence_weight` in
  {1.0, 0.75, 0.5} and `confidence_tier` in {`Reviewed-High`, `Reviewed-Moderate`,
  `Reviewed-Low`}. A reject→olivine reassignment with confidence Low → hard-neg
  parquet has `olivine_t1=1.0`, `confidence_weight=0.5`,
  `confidence_tier='Reviewed-Low'`. A bland / ambiguous reject → unchanged
  `confidence_weight=1.0`, `confidence_tier='High'`.
- **build routing**: a synthetic confirmed parquet with mixed per-polygon weights
  → `load_confirmed_mineral_positives` preserves them (no flat 2.0). A synthetic
  hard-neg parquet with a reassigned-olivine polygon (`negative_of=''`,
  `olivine_t1=1.0`) → routed into the mineral-positive fragment with
  `olivine_t1=1.0` and `bland=0`, not into the bland pool.
- **collapse fallthrough**: a `Reviewed-Moderate` row with stamped
  `confidence_weight=0.75` → `_collapse_labels` yields `0.75` (asserts the global
  map is bypassed).

## Out of scope

- Re-grading already-collected MC13/MC11 decisions (they default to High/1.0).
- Any change to the global `_TIER_WEIGHTS` or base-parquet weighting.
- Confidence on non-mineral tag rejects.
