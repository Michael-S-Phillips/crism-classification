# Unit-Balanced Splits + Metric Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Honest, pixel-balanced train/val/test splits (kills adjacent-tile unit leakage), junk out of the stop metric, junk per-polygon cap.

**Architecture:** New pure-function splitter module (`scripts/split_units.py`) clusters polygons into geographic units and greedy-assigns whole units to splits balancing per-class pixel fractions; `build_7cls_dataset.py` applies it to every labeled source including the base parquet's gpkg rows; `train_torch.py` monitors `val_mAP_core` (junk excluded).

**Tech Stack:** Python, pandas, numpy, pytest.

Spec: `docs/superpowers/specs/2026-07-08-unit-balanced-splits-design.md`

---

### Task A: `scripts/split_units.py` — splitter module (TDD)

**Files:** Create `scripts/split_units.py`; Test `tests/test_split_units.py` (new).

API (from spec):
```python
LINK_DEG = 0.25
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
MIN_HOLDOUT_FRAC = 0.05

def tile_center_deg(tile_id: str) -> tuple[float, float]:
    """lat, lon from data/tile_centers.csv — a committed lookup generated once
    from the local tile FILENAMES (t{n}_mrral_{lat}{n|s}{lon}, e.g.
    t1444_mrral_30n328 -> (30, 328)). Filename coords are the tile's
    lower-left/reference corner; +2.5 each gives the center. Generate the csv
    as part of this task: glob /mnt/mrdr/mc*/t*_mrral_*.img, parse, write
    data/tile_centers.csv (tile_id,lat,lon; ~1,764 rows), commit it. The
    module loads it once (module-level cache). Raise KeyError with a clear
    message for unknown tiles. NOTE: do NOT attempt a closed-form n->latlon
    grid formula — tile numbering is not a uniform global grid across quads."""

def polygon_units(df, link_deg=LINK_DEG) -> pd.Series:
    """Unit id per row. Polygon centroid = tile center + 5*((mean_col/1500)-.5)
    lon, -(5*((mean_row/1500)-.5)) lat; single-linkage components at link_deg
    (simple pairwise union-find over polygon centroids; lon distance uses
    cos(lat) scaling and 360-wraparound)."""

def assign_unit_balanced_splits(df, label_cols, seed, link_deg=LINK_DEG) -> pd.Series:
    """Greedy: units by total px desc; assign to split with largest weighted
    per-class deficit vs SPLIT_FRACS targets; seeded tie-break; then min-val/
    min-test guard (force smallest donor unit from train while a class's
    val/test fraction < MIN_HOLDOUT_FRAC and a donor exists). Returns split
    per row. Also expose achieved_fractions(df, splits, label_cols) -> DataFrame."""
```

Tests must cover (write FIRST, watch fail, then implement):
1. `tile_center_deg` returns filename-derived centers: `t1444 → (32.5, 330.5)`,
   `t1249 → (22.5, 75.5)`, `t0434 → (-37.5, 320.5)` (lower-left + 2.5). Test
   reads the committed `data/tile_centers.csv`; also assert the csv covers all
   tiles present in `data/mrral_pixels.parquet`.
2. Cross-tile merge: two polygons on adjacent tiles with centroids <0.25° apart share a unit; polygons 2° apart don't.
3. Balance: synthetic data, 40 units of varied sizes, 3 classes → achieved fractions within ±5% of 70/15/15 per class.
4. Min-holdout guard: one class concentrated in 2 units → val and test each get ≥5% of it.
5. Determinism: same seed → identical assignment; polygons never span splits.
6. **Leakage regression:** no val polygon centroid within link_deg of any same-class train polygon centroid (they'd share a unit by construction — assert it).

Steps: write tests → `conda run -n crism python -m pytest tests/test_split_units.py -v` (FAIL) → implement → PASS → commit `"splits: unit-aware pixel-balanced splitter module"`.

**NOTE on step 1:** t1444 = 30N 328E and t1249 = 20N 73E ⇒ row counts from the north pole in 5° bands and tile numbering must be derived: fit `n → (lat, lon)` on the three samples; also glob `/mnt/mrdr/mc*/t*_mrral_*.img` filenames to build a full ground-truth mapping and assert the derived formula matches ALL of them (fallback: ship a `tile_latlon_from_name()` parser and a build-time lookup from the base parquet's tile list if no closed form fits).

---

### Task B: build integration

**Files:** Modify `scripts/build_7cls_dataset.py`; Test `tests/test_build_7cls_confidence.py`.

- Import `assign_unit_balanced_splits`.
- `_build_base`: after stamping, **override** `split` for non-bland (gpkg) rows:
  `non_bland['split'] = assign_unit_balanced_splits(non_bland, BALANCE_COLS, SEED)`.
  `BALANCE_COLS = ['olivine_t1','olivine_t2','lcp','hcp','plagioclase','alteration']`.
  Bland-tile rows: keep existing subsample, then same splitter with `['other']`.
- Replace `_assign_tile_splits`/`_assign_polygon_splits` calls in
  `load_confirmed_mineral_positives`, `load_reassigned_minerals`,
  `load_bland_review`, `load_junk_ambiguous`, `load_alteration_mc11` with the
  unit splitter (same seed offsets as today). Keep `_assign_tile_splits`/
  `_assign_polygon_splits` defined but unused only if other scripts import them
  — grep first; if nothing imports them, delete.
- After the final concat in `main`, print per-class × split pixel counts AND
  achieved fractions (use `achieved_fractions`).
- Tests: extend `tests/test_build_7cls_confidence.py` — synthetic multi-unit
  confirmed dir → per-class val fraction ≥5%; base-override test: frame with
  interleaved-adjacent-tile plag polygons pre-split train/val → after build's
  splitter, no val polygon within 0.25° of same-class train polygon.
- Run FULL suite `-k "build or persistence or collapse"` → PASS → commit
  `"build: unit-balanced splits for all labeled sources"`.

---

### Task C: junk per-polygon cap

**Files:** Modify `scripts/build_7cls_dataset.py::load_junk_ambiguous`; Test `tests/test_build_7cls_confidence.py`.

- Test first: junk dir with one 50k-px polygon + one 1k-px polygon → loader
  output has ≤ `MAX_PX_PER_POLYGON` rows for the big one.
- Implement: `df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 100)` after
  the raw read/print, before split assignment (mirror `load_bland_review`).
- Commit `"build: per-polygon cap on junk (ambiguous) source"`.

---

### Task D: core stop metric (exclude junk)

**Files:** Modify `evaluation/metrics.py`, `training/train_torch.py`; Test `tests/test_metrics_core.py` (new).

- `compute_map(y_true, y_score, exclude: tuple = ())`: resolve class names via
  `_class_names(y_score.shape[1])`; skip excluded names. Default unchanged.
- Test first (`tests/test_metrics_core.py`): 7-col synthetic scores where the
  junk column is garbage → `compute_map(..., exclude=('junk',))` >
  `compute_map(...)`; excluding a name absent from the label set is a no-op.
- `train_torch.py`: locate the val-metric block (computes `val_mAP`, logs to
  wandb, feeds early stopping / best checkpoint). In 7-class mode
  (`len(LABEL_COLS) == 7` at runtime / n_classes == 7):
  `val_mAP_core = compute_map(y_true, y_score, exclude=('junk',))`; log both;
  monitored/best metric + checkpoint `stop_metric` field become
  `val_mAP_core`. 5/6-class paths untouched (`val_mAP_core == val_mAP` there
  is acceptable if simpler — but then name stays `val_mAP_core` only when
  logged in 7-class mode to avoid dashboard confusion).
- Verify: pytest new file; `py_compile` train_torch; grep shows monitored
  metric switched. Commit `"train: early-stop on val_mAP_core (junk excluded)"`.

---

### Task E: docs + real-data verification

**Files:** Modify `scripts/hpc_build_7cls_data.slurm` (data-design comment: unit-balanced splits, junk capped, core stop metric); run real-data verification.

- Update slurm comments (build + finetune: note the monitored metric change and
  that val_mAP will drop vs prior runs — metric honesty, not regression).
- Real-data verification (local): run `load_confirmed_mineral_positives` +
  `_build_base`-equivalent with the new splitter on the actual parquet/dirs;
  print per-class fractions; assert plag val ≥5% and no same-class val/train
  polygon pair within 0.25°. Save output to `reports/split_rebalance_check.md`.
- Commit `"docs+verify: unit-balanced split rollout"`; push
  `git push origin master:feature/spatial-mae-pretraining`.

---

## Final verification
- [ ] `conda run -n crism python -m pytest tests/ -k "split_units or build or persistence or collapse or metrics_core" -q` — all pass.
- [ ] `reports/split_rebalance_check.md` shows every class ≥5% val and ≥5% test by pixels.
- [ ] Push. HPC rebuild + retrain happen AFTER current arms finish.
