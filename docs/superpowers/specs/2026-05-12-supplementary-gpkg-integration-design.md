# Supplementary GPKG Integration Design

**Date:** 2026-05-12
**Status:** Approved
**Scope:** Bring 10 newly-annotated GeoPackage files from `/mnt/mrdr/categorized_mineral_units/sup/` into the labeled training dataset by giving them a `Category` column consistent with the existing 40 files. Classifier retraining only — MAE pre-training is unaffected.

---

## Goal

Add ~150 additional annotated polygons (across 10 CRISM observations) to the classifier training set without changing the label schema, the parser, or the downstream training pipeline. The supplementary files lack the `Category` column the parser expects; this work runs the existing categorization procedure (from `/mnt/mrdr/categorize_gpkg_ratio_files.ipynb`) on them so they become indistinguishable from the main set.

---

## Inputs and Outputs

**Input directory:** `/mnt/mrdr/categorized_mineral_units/sup/`

10 GeoPackage files: `T0573, T0608, T0644, T0645, T0682, T0685, T0818, T0822, T0886, T1020`. Each has the standard columns (`Polygon Number`, `Color`, `Mineral ID 1`–`Mineral ID 4`, `Spectrum Mean`, `Ratio Spectrum`, `geometry`, etc.) but **no `Category` column**.

All 10 matching `mrral*.img` tiles exist locally under `/mnt/mrdr/mc##/`.

**Output directory:** `/mnt/mrdr/categorized_mineral_units/`

10 new GeoPackage files, same filenames, each with the original columns **plus a synthesised `Category` column** matching the format the existing parser consumes (`"hcp + olivine (Moderate)"`, etc.).

**Originals in `sup/` are left in place** as a backup.

---

## Categorization Rules

Port `categorize_minerals` from `/mnt/mrdr/categorize_gpkg_ratio_files.ipynb` verbatim, then prepend one extra skip rule for contaminated denominator polygons.

### Skip rule (added)

A row is **dropped entirely** (no `Category` assigned, polygon excluded from output) when:

- `Mineral ID 1 == "denom"` (case-insensitive, whitespace-stripped) **AND**
- Any of `Mineral ID 2`, `Mineral ID 3`, `Mineral ID 4` is non-blank

This catches 5 known contaminated denoms in the current batch:

| File | Polygon | Contamination marker |
|---|---|---|
| T0644 | 7 | `bad probably has olivine in it` (ID 2) |
| T0644 | 13 | `not bland probably has olivine and hcp` (ID 2) |
| T0644 | 15 | `probably has hcp and olivine` (ID 2) |
| T0645 | 13 | `±pyroxene` (ID 2) |
| T0818 | 18 | `probably has some pyroxene` (ID 2) |

The remaining 67 (of 72) "clean" denom rows pass through and become `"Other (High)"` — preserving the existing spectrally-bland training pathway.

### Notebook rules (verbatim port)

For each surviving row, build a `Category` string by iterating `Mineral ID 1`–`Mineral ID 4`:

- **Start tier = High**
- **`±` qualifier:** strip the `±`; if in `Mineral ID 1` → tier = Low; else → tier = Moderate (unless already Low)
- **`uncertain` substring:** if in `Mineral ID 1` → tier = Low; else → tier = Moderate (unless already Low)
- **`felsic` substring:** tier = Low
- **`alteration` substring** (in non-ID1 cells): tier = Low
- **`slope` substring:** tier = Low
- **Token recognition:** if the (de-`±`'d) cell value equals one of `{olivine, plagioclase, lcp, hcp, red slope, felsic, alteration, spinel}` → append to category list
- After all four cells: sort tokens alphabetically, join with `" + "`, append `(<tier>)` → `Category`
- **No recognized tokens →** `"Other (<tier>)"`

This is the same logic that produced the existing 40 categorized files, so the resulting `Category` strings are by construction within the `parse_category()` vocabulary.

---

## Conflict Handling

Filenames in `sup/` do not collide with the existing 40 files in this batch. The script must still guard against future collisions:

- Before writing each output, check if the target path already exists.
- If it does, **raise an error and abort the entire run** (don't process subsequent files). Error message names the conflicting file and instructs the user to resolve manually — auto-merging polygon data is not safe (different files mean different annotation sessions, possibly different CRS or schema).
- If no conflicts, write all 10 outputs.

---

## Components

### 1. `scripts/categorize_sup_gpkg.py` (new)

Standalone Python script. Single entry point, runs to completion.

**CLI:**
```bash
python scripts/categorize_sup_gpkg.py \
    --input_dir /mnt/mrdr/categorized_mineral_units/sup \
    --output_dir /mnt/mrdr/categorized_mineral_units
```

Both flags have these as defaults so the typical invocation is just `python scripts/categorize_sup_gpkg.py`.

**Behaviour:**
1. Pre-flight: list all `*.gpkg` in `--input_dir`. For each, check whether `<output_dir>/<filename>` exists — if any conflict found, abort with a clear error before doing any work.
2. For each input file, load with `geopandas.read_file`, apply the contamination skip, apply `categorize_minerals` row-wise to produce the `Category` column, write the result to `<output_dir>/<filename>` with `driver='GPKG'`.
3. Per-file logging: `T0573.gpkg: 30 rows → 30 categorized, 0 contaminated denoms skipped`.
4. Final assertion (belt-and-braces): after writing each file, re-open it and confirm every `Category` value parses cleanly through `data.label_parser.parse_category` (i.e., produces a non-empty mineral parse OR is `"Other (...)"`). Fail loudly if any row produces an unrecognised category token — this would mean a token like `felsic` slipped through and the existing parser doesn't know it.

### 2. Downstream pipeline (no code changes)

After running the script, the standard data refresh applies:

1. `python data/extract_pixels.py` — regenerates `data/mrral_pixels.parquet` and `data/pixels.parquet`. The script already iterates `*.gpkg` in `categorized_mineral_units/`, so the 10 new tiles are picked up automatically.
2. Regenerate `data/patch_cache/mrral_{train,val,test}_patches_p7.npy` if patch-cached training is used.
3. Re-run any/all classifier training workflows the user wants to compare (e.g., `training/train_torch.py` for the SpatialSpectralClassifier; the various `config/sweep_*.yaml` baselines). Each picks up the new parquet automatically. Reuse the MAE checkpoint (`spatial_mae_128d_6l_best.pt`) unchanged.

---

## What Is Not in Scope

- **No changes to `data/label_parser.py`.** The notebook's `categorize_minerals` only emits tokens already in the parser's `_TOKEN_MAP`. Verified by the post-write assertion.
- **No changes to `data/extract_pixels.py`** for this batch. (If future sup batches arrive, the conversion script handles them the same way.)
- **No MAE retraining.** The MAE was pre-trained on the global unlabeled MRDR dataset; adding labeled polygons does not affect it.
- **No re-curation of the existing 40 files.** Any latent issues in how denom-only or `"Other (...)"` rows are treated by the training pipeline are pre-existing and out of scope here.
- **No new evaluation splits or metrics.** The new pixels flow into the existing train/val/test split logic in `data/extract_pixels.py`.

---

## Testing

- **Unit test:** add a test file (e.g., `tests/test_categorize_sup_gpkg.py`) that exercises `categorize_minerals` against representative row inputs — clean primary, `±` qualifier in each position, `uncertain` in each position, multi-mineral rows, denom + blank secondaries, denom + contaminated secondaries, all-empty row.
- **Smoke run on a single file** (e.g., `T0822.gpkg`) to a temporary output dir before doing the full 10. Inspect the resulting `Category` values manually.
- **Post-conversion parity check:** verify each output file is loadable with `geopandas` and that every `Category` value parses without warning through `parse_category`.
- **Pipeline run:** after the parquet regenerates, sanity-check that the new tile IDs (`t0573` … `t1020`) appear in `mrral_pixels.parquet` with non-zero pixel counts and that the new pixels distribute across confidence tiers as expected (most denoms → High `Other`, ± rows → Moderate/Low).

---

## Acceptance Criteria

1. `scripts/categorize_sup_gpkg.py` runs to completion on the 10 `sup/` files without errors.
2. 10 new files appear in `/mnt/mrdr/categorized_mineral_units/`, each with a populated `Category` column.
3. 5 contaminated denom rows are dropped (verified by row-count diff per file).
4. Every emitted `Category` parses cleanly through `parse_category`.
5. `data/extract_pixels.py` re-run produces a parquet that includes the 10 new tiles, with new pixels labelled across all confidence tiers.
6. At least one classifier training workflow (e.g., the SpatialSpectralClassifier fine-tune) re-runs against the updated parquet and reports its primary metric. (Whether val_mAP improves over the current 0.7175 best is a research question, not an acceptance condition.)
