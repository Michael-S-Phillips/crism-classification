# Dust hard negatives for the dual-CR 7-class model — design

**Date:** 2026-08-17
**Status:** awaiting review
**Creates:** `scripts/mine_dust_hard_negatives.py`,
`scripts/merge_hard_negatives.py`, `scripts/hpc_finetune_handcore_dualcr_hardneg.slurm`,
`tests/test_mine_dust_hard_negatives.py`, `tests/test_merge_hard_negatives.py`
**Touches:** nothing existing — the merge writes a NEW parquet rather than mutating
`mrral_pixels_7cls_handcore.parquet`

---

## Problem, measured

On Nili t1321 the dual-CR 7-class model at e87 fires `lcp >= 0.99` on 125,757 px
(7.0% of the tile). **35% of those have LCPINDEX2 ≈ 0** — no pyroxene absorption
at all. The population is bimodal, not noisy: one mode at LCPINDEX2 0.000–0.010,
another at 0.030–0.070, with the middle nearly empty, and the two modes are
spatially disjoint (rows 64–593 vs 1265–1614).

Scanning all 60 `mrrsu` parameters, the false mode is **bright red dust**:

| | false LCP | true LCP | Cohen d |
|---|---|---|---|
| RBR | 6.01 | 3.75 | −4.8 |
| R770 | 0.264 | 0.155 | −5.4 |
| RPEAK1 | 0.869 | 0.742 | −7.6 |

Three findings constrain the fix:

1. **It is not a representation failure.** Raw absolute band depth over
   0.85–1.15 µm is nearly identical in the two modes (0.0118 dusty vs 0.0098
   real), but *after* hull-CR they sit at 0.059 and 0.150. CR separates them 2.5×.
   The information is in the model's input and the classifier discards it.
2. **The model already half-knows.** `p_bland` separates the two modes at
   AUC 0.93 — but at 0.087 vs 0.0067, far below any threshold, because
   multi-label sigmoids never force `p_lcp` down when `p_bland` rises.
3. **Confidence is non-monotonic against truth.** Mean LCPINDEX2 by `p_lcp` bin:
   0.0163 at 0.5–0.7 → **0.0076 at 0.95–0.99** → 0.0457 at ≥0.999. The
   0.85–0.99 band is worse than 0.5–0.7. Post-hoc calibration is monotonic and
   therefore cannot fix this.

The `--bland_gate` shipped in `683d833` is a partial mitigation: at 0.03 it
removes 51% of Nili lcp pixels and still leaves false detections, because it
catches only 84% of the *provably* false block. It stays available; it is not the
fix.

## What this does not attempt

The 0.85–0.99 miscalibration is plausibly driven by ASL `γ⁻=4.0`, which
down-weights easy negatives and so never sharpens that boundary. **Deliberately
out of scope** — one variable at a time. If this retrain improves the floor test,
we know hard negatives did it. ASL is the next experiment, not part of this one.

## Design

### Stage 1 — mine, locally (`mine_dust_hard_negatives.py`)

Local, because the `mrral`/`mrrsu` tiles and the 183-tile deployment probs are
here and the training parquet is not.

A pixel is a dust hard negative when all five hold:

1. **No mafic signature**, tile-relative: `OLINDEX3`, `LCPINDEX2` and
   `HCPINDEX2` all below that tile's 40th percentile. Tile-relative because
   absolute cuts provably do not transfer — t1249's whole-tile LCPINDEX2 median
   (0.0299) exceeds t1321's 90th percentile (0.0314) region.
2. **No alteration signature**: `BD1900_2`, `D2300`, `BD2210_2` below tile p60.
   Without this we would mine real alteration and teach the model to miss it.
3. **Dusty**: `RBR` and `R770` both above tile p60. Two parameters, not one, so a
   merely dark-and-featureless pixel is not mined — those are already `bland` in
   the label set and add nothing.
4. **Hard**: some existing model fires ≥ 0.90 for a mineral there. Easy negatives
   are already learned; only the ones that fool a model carry gradient. Source:
   `data/mc_deploy_pyx_physmax/probs` (183 tiles, mc11/13/26, post-`PHYS_MAX`).
   The pyx model is 6-class, but it shares the backbone and data build, so its
   dust confusion is the same phenomenon.
5. **Physically valid**: passes the `PHYS_MAX`/nodata test, and the whole 7×7
   patch is valid — the classifier reads a patch, so a mined centre with a
   nodata neighbour teaches the padding, not the dust.

**Spatial thinning.** Cap per tile and enforce a minimum pixel separation, so one
large dust mantle cannot supply the whole negative set. Without this the model
learns one location rather than one spectral class.

**Exclusion — labels.** Drop any `(tile_id, pixel_row, pixel_col)` already present
in the labeled parquet. Never contradict a hand label.

**Exclusion — the floor-test tiles.** Mine nothing from t1249, t1250, t1321,
t1322, t0434, t0435, t1086, t1087. All eight sit in mc11/mc13/mc26 and would
otherwise be mined, and training on them would turn the floor test into a partial
train-on-test — destroying the one property `MODELS.md` relies on it for ("the
only leakage-immune cross-era comparator"). t1321 is the sharpest case: it is
simultaneously the tile the diagnosis came from, prime dust-mining territory, and
the tile carrying the primary success metric.

This costs mining yield, and that is the correct trade. The three MC quadrants
hold 183 tiles; losing eight leaves 175.

Output: `data/hard_negatives_dust.parquet` — `tile_id`, `pixel_row`,
`pixel_col`, `band_00..band_58`, and the `mrrsu` values used, so the selection is
auditable after the fact rather than only reproducible.

### Stage 2 — merge, on HPC (`merge_hard_negatives.py`)

Runs where `mrral_pixels_7cls_handcore.parquet` lives, and reads that file's
schema rather than assuming it: the local proxy parquets call the bland class
`other`, the 7-class build calls it `bland`. Hard-coding either is a silent
mislabel.

- Assign each mined pixel a synthetic `polygon_id` (`dustneg_<n>`), one per
  thinned cluster.
- Set the bland column to 1 and every mineral column to 0.
- `confidence_weight` / `confidence_tier`: match whatever the existing bland rows
  use, read from the target parquet. Do not invent a tier.
- **Splits: reuse `split_units.assign_unit_balanced_splits` on the concatenated
  frame. Do not assign splits directly.** `polygon_units` links polygon centroids
  at 0.25° and unions anything sharing a literal pixel, so a mined negative near
  a val unit is absorbed *into* that val unit and follows it. That is exactly the
  leakage guard we want, and it already exists and is tested. Writing
  `split='train'` by hand would put dust pixels from val terrain into train.
- Write a NEW parquet. The input is an input.

Then the existing cache chain, unchanged: `cache_mrral_patches.py` → raw cache →
`build_cr_labeled_cache.py --dual` → dual cache → fine-tune.

### Stage 3 — retrain

`hpc_finetune_handcore_dualcr_hardneg.slurm`, a copy of the e87 job with the new
parquet and cache paths and nothing else changed: same backbone
(`spatial_mae_dualcr_denoising_256d_6l_best.pt`), same ASL, same
`weight scheme: level`, same `stop_metric val_mAP_core`, patience 40. Carries the
checkpoint-on-improvement and resume machinery, so a walltime kill does not lose
the run.

## How we will know it worked

`val_mAP_core` is not the test — it is measured on the same label distribution
that produced the problem, and adding negatives can raise it while changing
nothing on a map. The tests are:

1. **Primary.** Re-measure the t1321 false share: of pixels firing `lcp ≥ 0.99`,
   the fraction with LCPINDEX2 below tile p40. **Now 35%; target < 10%.**
2. **Over-correction guard, equally important.** t1249 is genuinely LCP-rich
   (whole-tile LCPINDEX2 median 0.0299, 24.9% of the tile at `lcp ≥ 0.99`, false
   share 0.1%). Its confident-LCP pixel count must not fall by more than ~15%. A
   model that has merely learned "bright ⇒ never mafic" will pass test 1 and fail
   this one.
3. **Floor test** at the same tag conventions, compared in **pixels retained**,
   not polygon counts — a subtractive change fragments regions and *raises*
   polygon counts (measured: Nili lcp @0.50 went 1,675 → 3,622 under the bland
   gate while losing 51% of pixels).
4. `dualcr_level_e87` and `dualcr_level_e87_gated` are the two baselines.

## Risks

- **Over-correction into brightness-phobia** — the model learns albedo, not
  mineralogy, and loses real LCP on bright terrain. Guard: test 2 above.
- **Mining real minerals.** Criteria 1–2 are tile-relative percentiles, not
  physics; a tile that is *entirely* mafic has a 40th percentile that is still
  mafic. Mitigation: report per-tile mined counts and the mrrsu distributions of
  what was mined, and inspect the worst tiles before merging.
- **The pyx model as the hardness oracle** biases mining toward *its* errors.
  Accepted: it shares backbone and data build with the target, and the
  alternative — running e87 inference over 183 tiles — costs ~20 h for a
  second-order gain.
- **Dust is genuinely spectrally featureless**, so `bland` may be the wrong
  concept if dust and featureless bedrock need separating later. The 8-class
  `dust` option was considered and declined to preserve floor-test comparability;
  revisit if this retrain underperforms.
