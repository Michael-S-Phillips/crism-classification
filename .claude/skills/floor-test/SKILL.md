---
name: floor-test
description: Use when a new 7-class CRISM checkpoint needs acceptance testing — e.g. checkpoints just pulled from HPC, after retraining, or when asked to "run the floor test" / "test the checkpoint" on Nili/Argyre tiles.
---

# Checkpoint floor test (Nili + Argyre)

## Overview

The floor test is the model acceptance check: classify the 6 standard tiles,
vectorize per-mineral threshold polygons, and judge the counts against known
good/bad signatures. Core principle: **if a model doesn't look clean on
terrain it trained on (t1250/t1322 are train tiles — deliberate), nothing
downstream can be trusted.**

## Run

```bash
bash scripts/floor_test.sh <checkpoint.pt> [tag]
# e.g. bash scripts/floor_test.sh checkpoints/ft_7cls_v3b_lrscale001_best.pt
```

- ~35 min cold (8 tiles × ~4 min); per-tile skip-if-exists → resumable.
- Output: `reports/floor_tests/<tag>/summary.md` (+ styled gpkgs per region for QGIS).
- Tiles: Nili t1249 t1250 t1321 t1322 (`/mnt/mrdr/mc13`); Argyre t0434 t0435
  (`/mnt/mrdr/mc26`); MC11 t1086 t1087 (`/mnt/mrdr/mc11`). Do not vectorize a region
  with the wrong tile_dir — the script passes each region's own tile_dir; it was a
  real failure once.
- MC11 is the altered/dusty **OOD probe** — no established good/bad count signature
  yet (unlike the Nili/Argyre table below). Read it qualitatively: false minerals
  flooding altered ground is the classic MC11 failure; alteration collapsing to bland
  is the conservative failure. Neither candidate model has solved MC11 alteration.

## Judge the result

Read `summary.md`, then the previous floor test's summary (path is printed at
the bottom). Render a verdict per criterion:

| Criterion | Pass | Known-bad signature |
|---|---|---|
| HCP contained | @0.50 < ~800/region AND hcp.gpkg < ~2 MB | v2 flood: Nili 2,772 @0.50 / 10.5 MB; Argyre 1,733 / 3.5 MB |
| LCP alive on Nili | @0.50 ≥ ~150 | v2 collapse: 62 @0.50 (LCP↔HCP swap) |
| Olivine confident on Nili | ≥ ~300 polygons @0.99 | trained-terrain memorization shows as flood at 0.50 with weak 0.99 tail |
| Argyre olivine peaks high | count peaks at 0.85–0.90 | peaking at 0.50 = diffuse/unconfident |
| Plag near-zero on Argyre | expected ≈ 0 | a surge is a distribution change — investigate, not auto-fail |

Reference-good (v3b_lrscale001): Nili @0.50 oliv 1,136 / lcp 325 / hcp 371;
Argyre @0.50 oliv 82 (peak 1,153 @0.90) / hcp 148.

**Any metric moving >2× in the wrong direction vs the previous floor test is a
flag even if the absolute threshold passes.** Report the per-mineral tables,
the verdicts, and the comparison — then let the user eyeball the gpkgs in QGIS.

## Common mistakes

- Judging only @0.50: memorization looks "strong" at low thresholds; the 0.99
  tail and peak-threshold position carry the signal.
- Comparing against a different arm (lrscale0001 vs 001) — compare same-arm
  runs, or say explicitly that arms differ.
- Deleting `/tmp/floor_test_<tag>` mid-run: that's the resume cache.
