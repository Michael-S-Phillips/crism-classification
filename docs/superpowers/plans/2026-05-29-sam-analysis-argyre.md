# SAM Analysis on Argyre Test Tiles (Task B)

**Goal:** Three SAM (Spectral Angle Mapper) analyses + classifier-plag-in-SAM diagnostic on the Argyre val/test tiles (`t0434`, `t0435`). Output per-class angle rasters, histograms, and candidate-pixel parquets that downstream tasks (C: test-set expansion, D: contrastive learning, E: linear unmixing) will consume.

**Status of current pipeline (from audit):**
- `sam_analysis/p.py` **does not exist**. Build new module from scratch.
- Endmember spectra (mean/median per class, 59 MRDR bands) already on disk at `/mnt/mrdr/endmember_extraction/crism_endmembers/crism_endmember_spectra.xlsx` — but **plagioclase not included** (current extractor covers olivine_t1, olivine_t2, LCP, HCP only).
- Argyre tiles: `/mnt/mrdr/mc26/t0434_mrral_*.img`, `/mnt/mrdr/mc26/t0435_mrral_*.img`.
- MTRDR scenes for the FeldsReview plag work live under `/mnt/mrdr/categorized_mineral_units/FeldsReview/`. Argyre-overlapping CRISM obsids need to be identified by spatial intersection.
- Wavelength resampling: `data/synthetic_plag.py::interp_to_mrral_wavelengths(lib_wl, lib_refl, target_wl)` — linear interp, NODATA-aware.
- Champion encoder: `checkpoints/ft_plag_aware_relabeled_best.pt`. Embedding extraction pattern: `model.encoder(patches)` → take center token at idx 25 for 7×7 patches → (B, 128).
- Labeled pixels: `data/mrral_pixels.parquet` with columns `olivine_t1, olivine_t2, lcp, hcp, plagioclase, other, split, tile_id, confidence_tier`.

---

## Module layout

```
sam_analysis/
  __init__.py                  (new)
  endmembers.py                (new — endmember library loader)
  sam.py                       (new — SAM angle core)
  embedding_sam.py             (new — embedding-space cosine distance)
  diagnostic.py                (new — classifier-plag-in-SAM analysis)
  run_argyre_sam.py            (new — CLI driver)
tests/
  test_sam_core.py             (new)
  test_endmember_loader.py     (new)
```

---

## Task 1 — Endmember library loader

**File:** `sam_analysis/endmembers.py` (new)

```python
def load_endmember_library(
    xlsx_path: str = "/mnt/mrdr/endmember_extraction/crism_endmembers/crism_endmember_spectra.xlsx",
    parquet_path: str = "/mnt/mrdr/crism_classification/data/mrral_pixels.parquet",
) -> dict[str, np.ndarray]:
    """Return dict mapping class name to mean reflectance spectrum (59,).

    Classes:
      - 'olivine' (union mean of olivine_t1 + olivine_t2 from xlsx if present, else from parquet)
      - 'lcp', 'hcp' (from xlsx)
      - 'plagioclase' (computed from parquet — high-confidence train pixels; xlsx does not have it)

    All spectra are length 59 (MRDR band count).
    """
```

Plagioclase endmember computation (since xlsx lacks it):
- Read `mrral_pixels.parquet`, filter to `split == 'train'` AND `plagioclase >= 0.7` AND `confidence_tier == 'High'`.
- Mean across `m0..m58` columns.

**Test:** `tests/test_endmember_loader.py` — assert all 4 classes returned, each length 59, no NaN, values within [0, 0.5] (clipped reflectance range).

---

## Task 2 — SAM angle core

**File:** `sam_analysis/sam.py` (new)

```python
def spectral_angle(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute SAM angle in radians between target spectrum/spectra and a reference.

    target: (N, B) or (B,)  — pixel spectra
    ref:    (B,)             — reference endmember
    returns: (N,) or scalar angle in radians

    angle = arccos( dot(t,r) / (||t|| * ||r||) )
    Robust to: zero-magnitude vectors (returns pi/2 ≡ "no information"); NaN bands (ignored pairwise).
    """

def sam_raster(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Apply spectral_angle to every pixel in a (H, W, B) cube.

    Returns: (H, W) float32 angle raster. NaN where pixel is all-invalid.
    """
```

Implementation notes:
- Use `np.einsum` or simple `np.tensordot` for batched dot products.
- Mask NaN bands consistently between target and ref (pairwise valid mask).
- For embedding-space (Task 4): same algorithm, just different `ref` dimensionality. Re-use `spectral_angle`; don't duplicate.

**Test:** `tests/test_sam_core.py`
- Identical-vector → angle 0.
- Orthogonal → angle π/2.
- Anti-parallel → π (won't occur with reflectance ≥ 0, but verify math).
- All-NaN target → NaN angle.
- 5-pixel batched fixture matches per-pixel scalar calls.

---

## Task 3 — Mode 1: spectral SAM on MRDR Argyre tiles

**File:** `sam_analysis/run_argyre_sam.py` (new, callable as script and importable)

For each Argyre tile (`t0434`, `t0435`):
- Open the mrral .img via `rasterio` (preserve transform + CRS).
- Mask CRISM NoData (65535).
- For each class endmember from Task 1: call `sam_raster`, write output `.npz`:
  - Path: `sam_analysis/outputs/argyre/{tile}_{class}_sam_mrdr.npz`
  - Contents: `angles=(H,W,) float32`, `transform=(6,)`, `crs_wkt=str`, `mode='spectral_mrdr'`

Also write per-tile-per-class histograms as PNG to `reports/sam_argyre/{tile}_{class}_sam_mrdr_hist.png` for quick visual review.

**Smoke check:** mean SAM angle across pixels should be < 0.5 rad (i.e. ~30°) for at least one class endmember on each tile — sanity check that the dot-product orientation is correct.

---

## Task 4 — Mode 2: SAM on MTRDR-subsetted spectra (Argyre)

**File:** `sam_analysis/run_argyre_sam.py` (extend)

This mode is conditional: only run if an Argyre-overlapping MTRDR scene exists in `/mnt/mrdr/categorized_mineral_units/FeldsReview/`.

Steps:
1. Build a script `sam_analysis/find_argyre_mtrdr.py` that:
   - For each Argyre tile, reads its CRS-projected bounding polygon from the mrral header.
   - Walks the FeldsReview directory tree, opens each `*_07_if*j_mtr3.img` header, reads its corner coordinates, checks intersection.
   - Outputs `sam_analysis/outputs/argyre/mtrdr_pairings.json` mapping `tile_id → [matching_mtrdr_paths]`. May be empty.

2. For each pairing in the JSON:
   - Open the MTRDR cube, get its 489-band wavelength array from `.hdr`.
   - Per-pixel resample to 59 MRDR bands using `interp_to_mrral_wavelengths` from `data/synthetic_plag.py`. **Batch this** — write a small helper `resample_cube(cube_487, src_wl, dst_wl)` that vectorises the existing per-spectrum function (loop over pixels is acceptable for first pass; optimise only if too slow).
   - Run `sam_raster` per endmember.
   - Write `.npz` and histogram with mode tag `'spectral_mtrdr'` and the MTRDR obsid in the filename.

**Important:** MTRDR cubes may not exactly cover the Argyre tile footprint. The output rasters here are in the MTRDR scene's own coordinate grid, **not** the MRDR tile grid. That's fine for histogram/diagnostic purposes — we are comparing distributions, not co-registered rasters.

---

## Task 5 — Mode 3: embedding-space SAM-analog

**File:** `sam_analysis/embedding_sam.py` (new)

```python
def load_encoder(ckpt_path: str = "checkpoints/ft_plag_aware_relabeled_best.pt"):
    """Load the champion classifier and return just the encoder + the head's center-token index."""

def extract_embeddings(cube: np.ndarray, encoder, patch_size: int = 7, batch_size: int = 4096) -> np.ndarray:
    """Run the encoder on every 7x7 patch centered on every valid pixel.

    cube: (H, W, 59) raw mrral reflectance, NaN where NoData.
    Returns: (H, W, 128) float32 embedding raster; NaN at pixels where center or any neighbor is invalid.
    """

def class_centroids(parquet_path, encoder, splits=('train',), conf_tier='High') -> dict[str, np.ndarray]:
    """Compute mean 128-d embedding per class on high-confidence pixels."""
```

Mode-3 pipeline:
- For each Argyre tile, extract per-pixel embeddings (one forward pass batch-wise).
- Compute class centroids (cached after first computation).
- Per pixel: cosine distance to each class centroid → angle raster per class (`arccos(cos_sim)` for symmetry with spectral SAM).
- Same .npz + histogram outputs as Mode 1/2, with mode tag `'embedding'`.

**Important:** Encoder is small (4.85 MB ckpt); embedding extraction at 7×7 stride-1 will be RAM-bound. Use float32 streaming + write embeddings to .npz per-tile.

---

## Task 6 — Classifier-plag-in-SAM diagnostic

**File:** `sam_analysis/diagnostic.py` (new)

For each Argyre tile, for each of the three modes:
- Load the existing classifier probability raster for the tile (from previous Argyre run — look under `data/vector_argyre_v3_bland_cont1/` for outputs, or re-classify if missing).
- For all pixels where `plag_prob >= 0.5`, get their SAM angle to the plag endmember (from this mode's output).
- Plot:
  - Histogram of plag-angle for (a) classifier-plag pixels, (b) labeled-plag pixels in the tile, (c) labeled-olivine pixels in the tile.
  - 2D scatter: classifier plag probability (x) vs SAM angle to plag endmember (y) — should be inversely correlated if classifier finds real plag.
- Output: `reports/sam_argyre/{tile}_classifier_plag_diagnostic_{mode}.png` + a stats CSV with per-bucket counts.

Identify and write the **misclassified-plag-actually-olivine** candidate set: pixels where `plag_prob >= 0.5` AND `sam_angle_to_plag > threshold_θ_n`. Save as parquet for downstream Task D consumption:
- Path: `sam_analysis/outputs/argyre/{tile}_hard_negatives_{mode}.parquet`
- Schema: `{row, col, tile_id, plag_prob, sam_angle_plag, sam_angle_olivine, mode}`

θ_n default: median + 1σ of labeled-olivine SAM-angle-to-plag distribution (so we're capturing things that are spectrally as far from plag as olivine typically is).

---

## Task 7 — Top-level summary figure

**File:** `sam_analysis/run_argyre_sam.py` (extend `main`)

After all three modes run, generate a single side-by-side figure (`reports/sam_argyre_summary.png`) with rows = Argyre tiles, columns = (mrdr-spectral, mtrdr-spectral, embedding) showing the classifier-plag SAM-angle distributions for each mode + labeled-plag and labeled-olivine reference overlays.

This is the figure that answers "which SAM mode best separates classifier-plag-that-is-olivine from classifier-plag-that-is-real-plag."

---

## Definition of Done (local deliverables)

- New `sam_analysis/` module with all files above.
- All unit tests pass: `pytest tests/test_sam_core.py tests/test_endmember_loader.py -v`
- `python -m sam_analysis.run_argyre_sam --tiles t0434 t0435 --modes mrdr embedding` runs end-to-end and produces:
  - Per-tile .npz angle rasters for olivine, LCP, HCP, plag in both modes
  - Histogram PNGs
  - Classifier-plag diagnostic PNGs and stats CSV
  - Hard-negative candidate parquet(s)
- `python -m sam_analysis.find_argyre_mtrdr` produces a `mtrdr_pairings.json` (may be empty if no Argyre MTRDR found — that's acceptable; document in output).
- If pairings non-empty: mtrdr mode runs and produces equivalent outputs.
- Summary figure `reports/sam_argyre_summary.png` exists.
- A short written report `reports/sam_argyre_summary.md` (3–5 paragraphs) interprets the histograms and recommends the best mode for downstream use.

## Out of Scope

- Running on MC13 (Task C consumes Argyre results before generalising).
- Linear unmixing (Task E).
- Contrastive learning (Task D — consumes hard-negative parquets emitted here).
- Adding plagioclase to the extracted-endmember xlsx (we compute it from parquet on the fly).
