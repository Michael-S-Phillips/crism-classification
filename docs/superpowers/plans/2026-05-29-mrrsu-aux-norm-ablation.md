# mrrsu-aux Normalization Ablation Plan (Task A)

**Goal:** Add a NODATA-aware, physical-range-aware normalization choice to the `spatial_vit_aux` pipeline. Run 3 HPC fine-tune jobs (zscore / minmax / per-tile zscore), each with the same plag-aware-relabeled-mtrdr recipe but differing only in normalization.

**Status of current pipeline (from audit):**
- `scripts/build_mrrsu_aux.py` extracts RPEAK1 (band 8) and BD1300 (band 17) via 7×7 mean (NODATA-excluded), writes `mrrsu_aux_{split}.npy` and `mrrsu_aux_stats.json` with `{mean, std}`.
- `data/dataset.py::MrrsuAuxPatchDataset` z-scores in-batch using stats JSON.
- `scripts/classify_tile_supervised.py::load_mrrsu_aux_rasters` mirrors training normalization at inference.
- Current NODATA filter only excludes sentinel `65535`. **Does not exclude** RPEAK1 outliers (values < 0.5 µm or > 1.0 µm — known CRISM failure modes) or BD1300 outliers.
- No `--norm_mode` CLI option exists.
- Existing tests: `tests/test_mrrsu_aux.py`, `tests/test_mrrsu_aux_dataset.py`, `tests/test_spatial_spectral_classifier_aux.py`.

---

## Normalization Variants

| Variant | Stats computed | Per-pixel transform | Failure mode handling |
|---|---|---|---|
| `zscore` (current behavior, but with stricter NODATA) | global mean, std on physically-valid train pixels | `(x - mean) / std` | invalid → 0 (= train mean after z) |
| `minmax` | global min, max on physically-valid train pixels | `(x - min) / (max - min)`, clipped to [0, 1] | invalid → 0 |
| `pertile_zscore` | per-tile mean, std at runtime, computed on in-tile valid pixels; **fallback to global stats** when in-tile valid pixel count < `MIN_VALID_PER_TILE` (default 1000) | `(x - tile_mean) / tile_std` | invalid → 0 |

**Physical-range NODATA rules:**
- RPEAK1: keep if `0.5 ≤ x ≤ 1.0` AND `x != 65535`
- BD1300: keep if `0.0 ≤ x ≤ 1.0` AND `x != 65535` (sanity bound; verify against train data before locking the upper bound)

These rules live in `data/mrrsu_aux.py` as a shared helper `physically_valid_mask(raw_array, band_name)`.

---

## Task 1 — NODATA helper with physical-range rules

**File:** `data/mrrsu_aux.py` (modify)

- Add: `BAND_VALID_RANGES = {"RPEAK1": (0.5, 1.0), "BD1300": (0.0, 1.0)}` at module level.
- Add: `physically_valid_mask(arr: np.ndarray, band: str) -> np.ndarray` returns a boolean mask of pixels where value is finite, not 65535, and within `BAND_VALID_RANGES[band]`.
- Add: `apply_invalid_to_nan(arr: np.ndarray, band: str) -> np.ndarray` returns a copy with invalid entries set to NaN (downstream code already handles NaN).
- Existing `mean_pool_nodata` stays — but a 7×7 window with all-invalid neighbors must still propagate NaN at the center (already does via `uniform_filter` on a masked array).

**Test:** Add to `tests/test_mrrsu_aux.py`:
- `physically_valid_mask` rejects 65535, NaN, inf, below-range, above-range values.
- `apply_invalid_to_nan` replaces only invalid entries.

---

## Task 2 — Aux cache builder with `--norm_mode`

**File:** `scripts/build_mrrsu_aux.py` (modify)

Add CLI args:
- `--norm_mode {zscore, minmax, pertile_zscore}` (default `zscore` to preserve current behavior)
- `--min_valid_per_tile` (default 1000; only relevant for `pertile_zscore`)

Behavior:
- Compute the 7×7-mean aux arrays as before, then immediately apply `apply_invalid_to_nan` per-band so downstream stat computation sees NaN, not raw 65535 / out-of-range.
- For `zscore`: compute `mean, std` over physically-valid train rows; write `{"mode": "zscore", "mean": [r,b], "std": [r,b]}` to `mrrsu_aux_stats.json`.
- For `minmax`: compute `min, max` over physically-valid train rows; write `{"mode": "minmax", "min": [r,b], "max": [r,b]}`.
- For `pertile_zscore`: **don't compute global stats here**; write `{"mode": "pertile_zscore", "fallback_mean": [r,b], "fallback_std": [r,b], "min_valid_per_tile": N}` where the fallback values are global zscore stats (used when a tile lacks enough valid pixels).

Augment the stats JSON to also carry: `{"physical_ranges": {"RPEAK1": [0.5, 1.0], "BD1300": [0.0, 1.0]}, "version": 2}` — so loaders can detect old vs new format and fail loudly on mismatch.

**Test:** small fixture in `tests/test_build_mrrsu_aux.py` (create if missing) — synth a 4×4×2 raw aux array with known NaN + outliers, run each mode end-to-end, assert stats and output ranges.

---

## Task 3 — Dataset honours `norm_mode` field

**File:** `data/dataset.py` — `MrrsuAuxPatchDataset` (modify)

- On `__init__`, parse the `mode` field from stats JSON. Fail loudly if `version != 2` (forces regeneration).
- Implement three private methods: `_apply_zscore`, `_apply_minmax`, `_apply_pertile_zscore`. The last one needs the tile_id of each pixel — use `df['tile_id']` already in the dataset.
- After normalization, set non-finite entries to `0.0` (== sample mean post-transform, same as current behavior).
- For `pertile_zscore`: precompute `{tile_id → (mean, std) or fallback}` once at init by scanning the parquet's tile_id grouping over physically-valid aux values.

**Test:** Augment `tests/test_mrrsu_aux_dataset.py` with one test per mode. Stub a stats JSON for each mode; check that the transform produces the expected numeric result on a 3-pixel fixture.

---

## Task 4 — Inference path matches

**File:** `scripts/classify_tile_supervised.py::load_mrrsu_aux_rasters` (modify)

- Read the stats JSON, branch on `mode`.
- For `pertile_zscore` at inference: need a tile-level mean/std. Compute it on-the-fly from the physically-valid raster pixels of the tile being classified; fall back to `fallback_mean/std` if the tile has < `min_valid_per_tile` valid pixels (use the threshold from stats JSON).
- For `zscore` and `minmax`: straightforward apply.

**Test:** Add `tests/test_inference_aux_norm.py` — fixture-driven, asserts the loaded raster matches expected after each mode.

---

## Task 5 — Slurm sweep

**Files:** Make `scripts/hpc_finetune_mrrsu_aux.slurm` accept `--norm_mode` via env var, and create `scripts/hpc_finetune_mrrsu_aux_sweep.slurm` that launches 3 array jobs.

Approach (simplest): keep the existing slurm script but parameterise via env var `MRRSU_NORM_MODE`, and write a small wrapper script `scripts/launch_mrrsu_aux_sweep.sh` that sbatches the slurm script 3 times with different env vars. Each run uses a distinct cache dir suffixed by the mode (`patch_cache_zscore/`, etc.) so they don't clobber each other.

Run names on wandb:
- `ft_mrrsu_aux_zscore`
- `ft_mrrsu_aux_minmax`
- `ft_mrrsu_aux_pertile_zscore`

All use:
- Same encoder warm-start: `plag_aware_mae_128d_6l_best.pt`
- Same hparams as current `hpc_finetune_mrrsu_aux.slurm` (encoder_lr_scale 0.001, epochs 100, patience 25, asl_loss, plag-aware-relabeled-mtrdr settings)
- `--apply_relabels data/olivine_relabels.csv` (already in mtrdr variant)
- `--synth_train_cache` and `--synth_train_parquet` pointing at MTRDR plag cache

**Out of scope for this plan:** actually submitting the slurm jobs. Plan delivers the launcher; user submits.

---

## Task 6 — Scoring & report

**File:** `scripts/eval_mrrsu_aux_sweep.py` (new)

Once all 3 runs finish on HPC and checkpoints are pulled locally to `checkpoints/ft_mrrsu_aux_{mode}_best.pt`, this script:
- Runs `eval_on_corrected_val.py` against each checkpoint
- Tabulates mAP + per-class AP in a single markdown report at `reports/mrrsu_aux_norm_sweep_results.md`

---

## Definition of Done (local deliverables)

- All 5 new/modified files exist, parse cleanly with `python -m py_compile`.
- All unit tests pass: `python -m pytest tests/test_mrrsu_aux.py tests/test_mrrsu_aux_dataset.py tests/test_build_mrrsu_aux.py tests/test_inference_aux_norm.py -v`
- Local CPU smoke test: build a tiny 1-tile cache in each mode (`--norm_mode zscore/minmax/pertile_zscore`) and verify the output `mrrsu_aux_stats.json` shape and the `mrrsu_aux_train.npy` distribution is sensible (no NaNs, range plausible).
- Slurm launcher script exists and `sbatch --test-only` parses (don't actually submit).
- Eval-collation script exists and runs against existing `ft_plag_aware_relabeled_best.pt` as a smoke test (just to verify the markdown is generated).

## Out of Scope

- Adding more aux channels beyond RPEAK1 / BD1300.
- Modifying the spectral patch clip [0, 0.5].
- Submitting HPC jobs.
- Pulling and scoring checkpoints (user runs `eval_mrrsu_aux_sweep.py` after.)
