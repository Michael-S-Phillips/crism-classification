# Label Quantification: SAM Endmembers + N-D Visualizer — Design

**Date:** 2026-07-09
**Status:** Approved (user-directed; runs while HPC rebuild trains)

## Goal

Quantify the mineral label corpus: combine every label source (original hand
labels + review-app relabels), find ideal per-class endmembers via spectral
angle analysis, flag suspect polygons, and provide an interactive **N-D
Visualizer** (ENVI-style) to inspect the labeled spectra as vectors in
band-space. Analysis restricted to **450–2500 nm** → 57 mrral bands
(**m2..m58**, 534–2457 nm; m0/m1 fall below 450).

## Data sources (label corpus)

| source tag | where | labels |
|---|---|---|
| `hand` | `data/mrral_pixels.parquet`, rows with any mineral label (other<=0.5) | olivine_t1/t2, lcp, hcp, plagioclase |
| `confirmed` | `data/mc13_review/confirmed_pixels` + `data/mc13_review_7cls_v3/confirmed_pixels` | olivine_t1/t2, lcp, hcp, alteration (+co-occurring) |
| `reassigned` | both `hard_negatives` dirs, `negative_of=''` rows with a mineral label | olivine_t1, lcp, hcp, plagioclase (+co-occurring alteration) |

Classes analyzed: **olivine** (t1|t2 collapsed), **lcp**, **hcp**,
**plagioclase**, **alteration**. Bland excluded from endmember analysis but
loadable in the visualizer as a reference cloud (toggle).

Multi-label rows (>1 positive class) are EXCLUDED from endmember math
(impure by definition) but kept in the visualizer flagged `multi=True`.
`confidence_weight` and source are carried through everywhere.

## Component 1 — corpus assembler (`scripts/label_quant/assemble_labeled_spectra.py`)

Output `data/labeled_spectra.parquet`: one row per (pixel, class) with columns
`class, source, tile_id, polygon_id, confidence_weight, multi, m2..m58`.
Dedupe identical (tile, pixel_row, pixel_col, class) across sources with
precedence reassigned > confirmed > hand (a relabel supersedes the hand label).
Also writes `data/labeled_spectra_viz.parquet`: per-class subsample (≤5,000
px/class, seeded, per-polygon-cap 200 so big polygons don't dominate the cloud)
for interactive use; bland gets one 5,000-px reference sample.

## Component 2 — SAM endmember analysis (`scripts/label_quant/sam_endmembers.py`)

Spectral angle θ(a,b) = arccos(a·b / |a||b|) over the 57-band window (raw
reflectance v1; continuum removal noted as future work). Pixel-count-robust:
all class-level math runs on **polygon mean spectra** (single-label rows only).

Per class:
- **Medoid endmember**: polygon mean minimizing mean angle to all same-class
  polygon means (the most representative spectrum). Report top-5 candidates.
- **Discriminative endmember**: polygon mean maximizing the margin
  (min angle to other-class medoids − angle to own medoid).
- **Intra-class spread**: distribution of angles to own medoid (mean/p50/p90).

Corpus level:
- **Inter-class angle matrix** (medoid vs medoid) — the separability table.
- **Polygon purity report**: for every polygon, angle to own-class medoid,
  angle to nearest other class medoid, margin; polygons with negative margin
  (closer to another class) flagged as suspect, with source/tile/weight so the
  user can re-review them.

Outputs → `reports/label_quantification/`: `endmembers.csv` (class, kind,
tile, polygon, 57 band values), `class_angle_matrix.csv`,
`polygon_purity.csv`, `summary.md` (tables + suspect-polygon list + per-source
breakdown of purity).

## Component 3 — N-D Visualizer (`scripts/label_quant/nd_visualizer_app.py`)

Streamlit + plotly (matches review-app stack), loads the viz parquet:
- **3-D scatter** of spectra as points in band-space via selectable projection:
  PCA (default, fit on L2-normalized window spectra), any 3 raw bands, or
  **ENVI-style random orthonormal projection** with a "shuffle" button
  (re-randomize) — the modern equivalent of the n-D Visualizer's rotation.
- Color by: class (default, review-app palette) / source / confidence /
  **angle to a chosen endmember** (continuous scale + threshold slider).
- Filters: class toggles, source toggles, bland-reference toggle, multi-label
  toggle, confidence floor.
- **Spectra panel**: mean ±1σ of the current filter selection, with chosen
  endmember spectra overlaid; x-axis 450–2500 nm.
- Endmembers loaded from `reports/label_quantification/endmembers.csv` when
  present.
- Launcher: `scripts/label_quant/run_nd_visualizer.sh` (detached, port 8502 —
  review app owns 8501).

## Testing

- assembler: schema/window columns; provenance precedence on a planted
  duplicate; multi-label exclusion flag; per-polygon viz cap.
- SAM: angle correctness vs hand-computed values; medoid recovery on synthetic
  clusters; planted mislabeled polygon gets negative margin + flagged.
- app: py_compile + Streamlit AppTest smoke with a tiny synthetic parquet
  (loads, projection modes switch, no exceptions).

## Out of scope (v1)

- Continuum removal; MNF projections; PPI. Noted as future options.
- Any change to training pipelines — this is analysis/tooling only.
