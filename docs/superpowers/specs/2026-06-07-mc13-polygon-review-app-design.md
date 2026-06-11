# MC13 Polygon Review App — Design

> **Status:** Approved in conversation 2026-06-07. Author: phillipsm + Claude.

## Goal

Build a Streamlit app that walks the user through MC13 model-predicted polygons one at a time, lets them confirm or reject each polygon's predicted label, and harvests interior pixels from the confirmed polygons into an alternative training set. Rejections are kept as hard-negative candidates rather than thrown away.

The driving insight: the relabeled (May 29) MC13 product is the larger pool to mine (200k olivine, 64k LCP, 132k HCP polygons), and Phase 0 calibration analysis shows it is overconfident — so a human review pass should produce both a cleaner positive set and a useful hard-negative set in the same workflow. The contrastive product is much smaller (1.4k LCP, 385 HCP) and is deferred to a later pass.

## Scope

**In scope:**
- Olivine / LCP / HCP only.
- Source: `data/vector_mc13_relabeled/{olivine,lcp,hcp}.gpkg`. Each gpkg has five threshold layers (`thresh_0.85`, `0.90`, `0.93`, `0.95`, `0.97`).
- Per-class pixel target: **30,000 confirmed pixels**. App displays running progress; soft gate (banner) when reached, but user can continue.
- Output: ledger CSV (decisions, resumable), confirmed-pixels parquet (training set), hard-negatives parquet.

**Deferred (out of scope for this build):**
- Plagioclase. MC13 has no true plag regions; targeted ENVI ROI flow handles plag separately.
- "Other" class. Existing dust-patch harvest stands; the polygon-review tool does not touch it.
- Contrastive gpkg review (`data/vector_mc13_contrastive/`). Same code will handle it later by pointing `--gpkg_dir` at a different path.
- Cross-source spatial dedup.
- 7×7 patch extraction. Downstream concern; the existing patch-cache builder runs over the new parquet once review is done.

## Data flow

```
data/vector_mc13_relabeled/{class}.gpkg  (5 threshold layers)
            │
            ▼  queue iterator
       polygon record  ──────▶  mrral tile  ──────▶  interior pixel spectra
  (tile_id, layer, polygon_id, geom, pred_prob)        (n_pixels × 59 bands)
            │
            ▼  Streamlit card (mean spectrum + ±1σ envelope)
       [Confirm] / [Reject (+ optional corrected_class)] / [Skip]
            │
            ├──▶  data/mc13_review/decisions.csv      (append-only ledger)
            │
            ├──▶  data/mc13_review/confirmed_pixels.parquet
            │       schema: tile_id, polygon_id, pixel_row, pixel_col,
            │               m0..m58, olivine/lcp/hcp/plagioclase/other,
            │               confidence_weight=1.0, confidence_tier='High',
            │               split='train'   (matches mrral_pixels.parquet)
            │
            └──▶  data/mc13_review/hard_negatives.parquet
                    schema: same + negative_of column
                    (corrected_class blank → negative_of=predicted_class;
                     corrected_class set   → positive labels for corrected_class)
```

## Queue order

Per class:
1. Walk threshold layers high → low: `thresh_0.97`, `0.95`, `0.93`, `0.90`, `0.85`.
2. Within a layer, sort polygons by area descending (biggest first → maximum pixel yield per decision).
3. Skip polygons already in `decisions.csv` (resumable).

## UI layout

Top bar (always visible):
- Mineral selector: `[olivine] [lcp] [hcp]` (segmented control). Switching class swaps queue + counters; does not lose progress.
- Progress: `<confirmed_pixels_for_class> / 30,000`.
- Counters: `reviewed = N • confirm K1 / reject K2 / skip K3`.

Polygon card:
- Header line: `tile_id | layer | polygon_id | n_pixels | pred_prob`.
- Plot (plotly): mean spectrum (solid) + ±1σ envelope (shaded). X-axis: wavelength (nm, ~410 → 2457). Y-axis: reflectance. No band markers.
- Buttons (in this order):
  - `Confirm` — adds polygon's interior pixels to `confirmed_pixels.parquet`.
  - `Reject` — adds to `hard_negatives.parquet`; uses dropdown value if set (corrected class), else marks `negative_of={predicted}`.
  - `Skip` — logs decision='skip', no pixel rows written; will not reappear (avoid re-encountering ambiguous ones each session).
- Dropdown (below buttons): "If rejected, actually: [—, olivine, lcp, hcp, other]". Default `—` (blank). Only consulted on Reject.

Stop gate:
- When `confirmed_pixels_for_class >= 30000`, display banner: "30k reached for {class}. Switch mineral above or keep reviewing for more headroom."

## Persistence

### `data/mc13_review/decisions.csv` — append-only ledger
Columns:
- `ts` (iso8601 utc)
- `source_gpkg` (str — basename only, e.g. `vector_mc13_relabeled/hcp.gpkg`)
- `layer` (str — e.g. `thresh_0.95`)
- `polygon_uid` (str — `f"{tile_id}::{layer}::{polygon_index_in_layer}"` — stable across sessions)
- `tile_id` (str)
- `predicted_class` (one of olivine / lcp / hcp)
- `decision` (one of `confirm` / `reject` / `skip`)
- `corrected_class` (one of olivine / lcp / hcp / other / `<blank>`)
- `n_pixels` (int — interior pixel count for the polygon)
- `area_m2` (float)

Append-on-decision. On app startup, read full CSV → build set of decided `polygon_uid`s → skip those in queue.

### `data/mc13_review/confirmed_pixels.parquet`
- Built **incrementally** in the app: on `Confirm`, the polygon's interior pixel spectra are extracted from the mrral tile and the rows are appended to an in-memory list. A "Save parquet" button writes them to disk; this also runs automatically every 50 confirms and on app shutdown (best effort, via streamlit's session-state pattern).
- Schema **exactly** matches `data/mrral_pixels.parquet` so downstream pipelines work unchanged.
- All confirmed rows: `confidence_weight=1.0`, `confidence_tier='High'`, `split='train'`.

### `data/mc13_review/hard_negatives.parquet`
- Same row schema + extra `negative_of` column.
- If `corrected_class` is blank: write row with all label columns 0, `negative_of={predicted_class}`. Interpretation: "this pixel is *not* {predicted_class}; we don't claim what it *is*."
- If `corrected_class` is set: write row with positive label for `corrected_class`, `negative_of=NULL`. Interpretation: equivalent to a confirmed pixel of the corrected class — could be merged into `confirmed_pixels.parquet` downstream, but we keep it separate for provenance.

## Files

```
scripts/review/
    __init__.py
    queue.py          # PolygonQueue iterator (per-class, threshold + area sorted)
    loader.py         # polygon_interior_pixels(gdf_row, mrral_tile_path) -> ndarray
    persistence.py    # DecisionLog (csv ledger) + parquet writers
    app.py            # Streamlit app
tests/
    test_review_queue.py
    test_review_loader.py
    test_review_persistence.py
```

## Open question (answered, recorded here)

- **Interior pixel definition**: "all pixels inside polygon" vs "all pixels whose 7×7 neighborhood is inside polygon".
  - Decision: harvest **all** interior pixels in this app. Downstream patch-cache builder filters to pixels with valid 7×7 neighborhoods. Keeps the review tool simple + reversible.

## Risks / non-obvious behavior

- **Polygon counts are huge for relabeled** (132k HCP, 200k olivine). User won't review all. The 30k pixel target may be reached after a few hundred confirms if polygons are big.
- **Polygon-uid stability**: relies on `polygon_index_in_layer` being deterministic across geopandas reads. This is true for fiona-backed gpkg reads with no filter pushed down; we assert it in `queue.py` by reading via `gpd.read_file(..., layer=L).reset_index(drop=True)`.
- **Resumability**: `decisions.csv` is the source of truth (polygon-level decisions, append-on-each-click). Both parquet files are *derived* — on app startup, the app reconciles parquets against `decisions.csv` and re-extracts pixels for any confirmed/rejected polygons missing from the parquets. This makes a crash mid-parquet-write recoverable on next launch.

## Acceptance

- App launches via `conda run -n crism streamlit run scripts/review/app.py`.
- Mineral selector switches between olivine / lcp / hcp queues.
- A first confirm produces a 1-row+ parquet update.
- App restart skips already-decided polygons.
- 30k banner appears when threshold reached.
- Unit tests cover queue ordering, polygon-uid stability, decision-log append/read, parquet schema match against `mrral_pixels.parquet`.
