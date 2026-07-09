# Unit-Aware Pixel-Balanced Splits + Metric Fixes — Design

**Date:** 2026-07-08
**Status:** Approved (user: "implement all fixes")

## Problems (measured)

1. **Adjacent-tile unit leakage inflates val AP.** Base-parquet splits interleave
   adjacent tiles over the same geologic units (train t0359/t0362/t0364 vs val
   t0360/t0361/t0363/t0365 …). The same mapped unit's pixels land in train and
   val; `val_AP_plagioclase` saturates to 1.000 in plastic arms (historic
   encoder-limited plag ≈ 0.14). Olivine 0.94 likely inflated too.
2. **Pixel-count imbalance.** Splits assign tiles/polygons blind to pixel count:
   plag val = 4.0% (14k px), confirmed-minerals val = 6.9k px, junk val =
   **265 px** (AP = noise, bounces 0.01–0.25 within runs).
3. **Junk drags the stop metric.** `val_mAP` (early stop + best-checkpoint)
   averages in junk's noisy near-zero AP, deflating and destabilizing model
   selection. Junk is a catch-all (reviewer "ambiguous") — its AP is not a KPI.
4. **Junk is uncapped.** `load_junk_ambiguous` is the only review loader without
   `_per_polygon_cap`; top-5 polygons hold 56% of the class.

## Design

### A. Unit-balanced splitter (`scripts/split_units.py`)

New module, pure functions, no rasterio dependency:

- **Tile centers** parse from tile files present in the repo's known dirs is NOT
  required — centers come from a lat/lon lookup built once from filenames
  (`t{n}_mrral_{lat}{ns}{lon}` e.g. `30n328`) passed in as a dict, or derived
  from the global tile-number grid (`row = n // 72, col = n % 72`,
  lat = 87.5 − row·5, lon = col·5 + 2.5) — the grid formula is used (no file
  access), validated against a filename sample in tests.
- **Polygon centroids** (approx): tile center + 5°·((mean_col/W − ½),
  −(mean_row/H − ½)) with nominal W = H = 1500 px. Positional error ≤ ~8 km.
- **Units**: single-linkage connected components of polygons with centroid
  distance ≤ `LINK_DEG = 0.25°` (~15 km; absorbs centroid error, merges
  cross-tile continuations of the same unit, and exceeds the 7×7-patch overlap
  scale by ~10×).
- **Greedy multi-class pixel-balanced assignment**: units sorted by total
  pixels descending; each unit goes to the split with the largest weighted
  deficit across the classes it contains (targets 70/15/15 of each class's
  pixels). Deterministic (seeded tie-break).
- **Min-val guard**: after assignment, any class with val < 5% of its pixels
  forces the smallest donor unit containing that class from train → val
  (repeat until ≥5% or no donor). Same for test.
- **Report**: returns per-class achieved fractions; build prints them.

API: `assign_unit_balanced_splits(df, label_cols, seed, link_deg=0.25) ->
pd.Series` (split per row; df needs tile_id, polygon_id, pixel_row, pixel_col).

### B. Build integration (`scripts/build_7cls_dataset.py`)

One splitter for every labeled source (replaces `_assign_tile_splits` /
`_assign_polygon_splits` usages AND the base parquet's inherited splits):

- base gpkg mineral rows (non-bland): **splits overridden** by the unit splitter
  (this is where the interleave leakage lives).
- confirmed positives, reassigned minerals, junk, alteration: unit splitter
  (after their existing caps).
- bland review sources + bland tiles: unit splitter for uniformity (they're
  volume classes; balance is trivially achievable).
- MTRDR synth plag parquet: untouched (separate point observations,
  tile-level splits already sane).
- Label cols for balancing: olivine(t1|t2 combined), lcp, hcp, plagioclase,
  other/bland, alteration, junk.
- Build prints a per-class × split pixel table + achieved fractions.

Expected consequence (state it in the run notes): val_mAP will DROP vs current
runs — the metric becomes honest; plag val especially.

### C. Junk per-polygon cap

`load_junk_ambiguous`: `_per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 100)`
before split assignment, like every other review loader.

### D. Core stop metric (exclude junk)

- `evaluation/metrics.py`: `compute_map(y_true, y_score, exclude=())` — skips
  named classes (resolved via `_class_names`).
- `training/train_torch.py`: monitored metric becomes `val_mAP_core` =
  mAP excluding `junk` (7-class mode only; 5/6-class unaffected). Both
  `val_mAP` (all classes) and `val_mAP_core` are logged to wandb;
  early stopping + best-checkpoint selection use `val_mAP_core`.
- Checkpoint metadata `stop_metric` records `val_mAP_core`.

## Testing

- splitter: unit formation merges cross-tile neighbors / separates distant
  polygons; greedy hits 70/15/15 ±5% on a synthetic many-unit dataset; min-val
  guard fires when one class is concentrated; determinism (same seed → same
  assignment); no polygon spans splits; **no val polygon within LINK_DEG of a
  same-class train polygon** (leakage regression test).
- build: base gpkg rows get overridden splits; per-class val fraction ≥5% on
  synthetic data; junk cap applied.
- metrics: exclude arg drops the class from the mean; train_torch monitors core
  (unit-testable via the metrics helper; train loop wiring verified by grep +
  py_compile + a 7-class smoke of the metric block if feasible).

## Out of scope

- Re-extracting t0360's base-parquet rows (tracked separately).
- MTRDR synth split changes.
- Any change to the currently running arms.
