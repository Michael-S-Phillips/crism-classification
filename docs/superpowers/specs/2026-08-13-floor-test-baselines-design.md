# Floor-test baselines: expert band-parameter rules and classical ML — design

**Date:** 2026-08-13
**Status:** approved, pending implementation plan
**Creates:** `scripts/extract_mrrsu_features.py`, `scripts/fit_expert_rules.py`,
`scripts/fit_ml_baseline.py`, `scripts/classify_tile_baseline.py`,
`config/expert_rules_7cls.json`, `config/expert_rules_pyx.json`
**Modifies:** `scripts/floor_test.sh` (one env-var hook)

---

## Problem

The deep model has no comparator. Every floor test to date compares a checkpoint
against an earlier checkpoint, so the whole family could be worse than a
band-parameter map and nothing in the pipeline would say so. A paper needs a
floor: what does standard practice give you on the same tiles?

Two baselines, spanning the methods a reviewer will ask about:

1. **Expert band-parameter rules** — an explicit, auditable ruleset over the
   CRISM summary parameters, with vetoes and dominance logic. This is
   domain-expert practice made reproducible, not naive univariate thresholding.
2. **Classical ML** — RandomForest and HistGradientBoosting on the 60 summary
   parameters. This is "multivariate classical", the obvious middle rung.

Together with the deep model that gives three rungs: expert rules → classical ML
on expert-designed features → learned spatial-spectral representation.

## Architecture — one contract, everything else reused

The floor test's real interface is the probs npz written by
`classify_tile_supervised.py:456`:

| key | shape / type |
|---|---|
| `probs` | (H, W, C) float32 |
| `valid_mask` | (H, W) bool |
| `transform` | affine, 6 floats |
| `crs_wkt` | str |
| `class_names` | (C,) str |

Any producer of that file plugs into `floor_test.sh` and
`vectorize_per_mineral_thresholds_nili_6cls.py` unchanged — same
`[0.50, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99]` ladder, same 3×3 median
smoothing (`MEDIAN_SIZE=3, MEDIAN_ITERS=1`), same `MIN_PIXELS=9`, same summary
tables. **That identity is what makes this a comparison rather than two
pipelines read side by side:** any difference in polygon counts is attributable
to the method, because nothing downstream differs.

    mrrsu tile (60 params, co-registered) ─┐
    mrrde tile (geometry, elevation) ──────┼─→ scorer ─→ <tid>_probs.npz ─→ [unchanged floor test]
    labeled parquet, TRAIN split only ─────┘

`class_names` must be exactly one of the vocabularies the vectorizer accepts
(`vectorize_per_mineral_thresholds_nili_6cls.py:67-70`); anything else raises,
which is the desired behaviour:

    7-class  ['olivine','lcp','hcp','plagioclase','bland','alteration','junk']
    pyx      ['olivine','pyx','plagioclase','bland','alteration','junk']

Co-registration is verified: for `t1250`, mrral / mrrsu / mrrde are all
1538 × 1636. The rules operate on the same grid as the model output, pixel for
pixel, with no resampling.

### The valid_mask confound

The deep model derives `valid_mask` from mrral nodata handling. The baselines
read mrrsu. If those footprints differ, polygon counts differ for reasons
unrelated to method quality — a silent confound that would be read as a result.

**Decision:** the baseline derives `valid_mask` from **mrral, exactly as
`classify_tile_supervised.py` does**, then intersects with mrrsu validity, and
**prints both pixel counts plus their difference**. A divergence is then visible
in the log rather than absorbed into the comparison.

## Verified band indices

Read from `mc13/t1250_mrrsu_20n078_0327_4.hdr` (60 bands) and
`..._mrrde_...hdr` (19 bands). The four mineral indices match the values
already documented in `CLAUDE.md`, which is an independent confirmation.

| mrrsu | param | role |
|---:|---|---|
| 0 | `R770` | brightness (alternative) |
| 8 | `RPEAK1` | reflectance-peak **wavelength**, plag discriminant |
| 15 | `OLINDEX3` | olivine |
| 17 | `BD1300` | plagioclase |
| 18 | `LCPINDEX2` | low-Ca pyroxene |
| 19 | `HCPINDEX2` | high-Ca pyroxene |
| 20 | `VAR` | spectral variance (junk) |
| 23 | `BD1435` | CO₂ **ice** |
| 25 | `ICER1_2` | ice ratio |
| 28 | `BD1900R2` | bound H₂O |
| 33 | `MIN2200` | Al-OH / hydrated silica |
| 34 | `BD2210_2` | Al-OH |
| 40 | `BD2290` | Fe,Mg-OH |
| 41 | `D2300` | Mg,Fe-OH / carbonate |
| 43 | `SINDEX2` | polyhydrated sulfate |
| 44 | `ICER2_2` | ice ratio |
| 45 | `MIN2295_2480` | Mg-carbonate doublet |
| 47 | `BD2500_2` | Mg-carbonate |
| 50 | `BD3200` | CO₂ **ice** |

| mrrde | band | role |
|---:|---|---|
| 6 / 7 | INA / EMA at areoid | air mass |
| 11 / 12 | INA / EMA at surface from MOLA | air mass |
| 15 | Elevation (m rel. MOLA) | CO₂ column proxy |
| 17 | Bolometric albedo | context |

## The expert ruleset

### Design principle: the vocabulary is multi-label

`LABEL_COLS` are independent columns with a sigmoid each. A pixel can be olivine
**and** hcp — olivine-bearing basalt is ordinary. Mutually exclusive gates
therefore fight the label structure and would systematically suppress real
co-occurrence. Gates are split in two:

- **Veto (hard)** — only for genuinely incompatible or artifactual conditions:
  ice, saturation, non-physical values, hydration-excludes-dust for plagioclase.
- **Dominance (soft, a tier modifier)** — for cross-responding index pairs. Both
  labels can fire; the dominant one fires at a higher tier. The expert supplies
  "cross-response is real and dominance is informative"; the calibration
  **measures** how much dominance is worth rather than asserting it.

### Rules

| class | rule |
|---|---|
| **olivine** | `OLINDEX3 ≥ t` **AND NOT** junk. No mafic vetoes — olivine+hcp and olivine+lcp are real assemblages. |
| **lcp** | `LCPINDEX2 ≥ t` **AND NOT** junk. Dominance `LCPINDEX2 > HCPINDEX2` raises the tier. |
| **hcp** | `HCPINDEX2 ≥ t` **AND NOT** junk. Dominance `HCPINDEX2 > LCPINDEX2` raises the tier. |
| **plagioclase** | `RPEAK1` **in window** **AND** `BD1300 ≥ t` **AND** `BD1900R2` low (veto: dust) **AND NOT** junk. Mafic-low is a tier modifier, not a veto — plag+pyroxene basalt is real. |
| **alteration** | disjunction below, **AND** ice veto, **AND NOT** junk. |
| **bland** | every mineral and alteration rule fails, and not junk. |
| **junk** | `ICER1_2` high **OR** `ICER2_2` high **OR** `BD1435`/`BD3200` high (CO₂ ice) **OR** `R770` non-physical **OR** `VAR` extreme. |

**Alteration is a disjunction of mineral groups**, each requiring a *specific*
diagnostic rather than generic hydration — this is what stops dust qualifying:

| group | condition |
|---|---|
| Fe/Mg-phyllosilicate | `D2300` high AND `BD2290` high AND `BD1900R2` high |
| Al-phyllosilicate | `BD2210_2` high AND `BD1900R2` high |
| hydrated silica | `MIN2200` high AND `BD1900R2` high |
| sulfate | `SINDEX2` high AND `BD1900R2` high |
| **carbonate** | `BD2500_2` high AND `D2300` high — **no hydration required** |

The carbonate exception is load-bearing: carbonates are anhydrous, so a blanket
hydration requirement would silently reject them, and Nili Fossae
olivine-carbonate is exactly the terrain this project cares about.

**Global ice veto** on alteration: `ICER1_2` and `ICER2_2` low, so seasonal
frost cannot register as alteration.

### RPEAK1 is a wavelength, not an amplitude

`data/mrrsu_aux.py` documents two things this rule depends on:

> The plag-vs-olivine discriminant (RPEAK1) is regional, not per-pixel, so we
> feed the classifier a 7x7-mean of the mrrsu parameter rasters.
> …real plagioclase RPEAK1 sits ~0.7-0.8 um

So the plagioclase term is a **two-sided window**, calibrated as the 5th–95th
percentile of `RPEAK1` among plag-positive training pixels — not a one-sided
"high" threshold, which would admit everything above 0.8 as well. And `RPEAK1`
enters as a **7×7 mean**, matching both the existing aux precedent and the deep
model's receptive field. `BAND_VALID_RANGES` from `mrrsu_aux.py`
(`RPEAK1 (0.5, 1.0)`, `BD1300 (-0.5, 0.5)`) is reused rather than reinvented,
and the same pattern extended to the other indices.

`R770` is available as an OR'd brightness alternative behind a config flag. It
is scene-dependent (dust cover shifts absolute albedo), so if enabled it is
calibrated per-tile as a percentile of valid pixels, never as a global constant.

## Calibration: expert structure, data-fitted cut points

The split that keeps this defensible:

- **The logical form is fixed by mineralogy and never fitted.** No search over
  rule structures, so it cannot overfit into an uninterpretable rule.
- **Each veto threshold retains a specified fraction of that class's own
  training positives** (default 90%, configurable). Olivine's junk veto sits
  where it rejects pixels more artifact-like than 90% of real olivine. Self
  calibrating, interpretable, and structurally unable to silently annihilate a
  class — if a veto would drop below the retention floor, the fit logs it.
- **The primary index threshold is swept** to generate the ladder, vetoes held
  fixed. Each rung carries its **empirical precision on the training split**,
  and that precision is the probability written to the npz.

So a ladder position means "this rule at this strictness is right *p*% of the
time on training data" — the same axis as model activation, which is what makes
the two directly comparable at 0.5 / 0.9 / 0.99.

The entire ruleset — index names, directions, thresholds, retained fractions,
and precision at every rung — serialises to one human-readable JSON. A domain
reader can audit it and disagree with a specific number without touching code.

**Calibration uses the TRAIN split only.** Some floor tiles are also training
tiles — `t1250`/`t1322` deliberately so, per the `floor-test` skill: *"if a model
doesn't look clean on terrain it trained on, nothing downstream can be
trusted."* The supervised baselines therefore have the same partial scene
overlap as the deep model, by design rather than by accident, and it is stated
in the report rather than hidden.

## The baselines anchor the acceptance criteria

The `floor-test` skill's judgement table is entirely **relative to previous
checkpoints** — "v2 flood: Nili 2,772 @0.50", "v2 collapse: 62 @0.50",
"HCP contained: @0.50 < ~800/region". Those thresholds were set by comparing
models to each other, so nothing says whether ~800 is good in absolute terms.

Running the same table on the baselines gives the first **absolute anchor**:

- if the deep model's hcp count sits between the expert rules and the ML
  baseline, "contained" means something;
- if a baseline beats the model on a criterion, that criterion's threshold was
  measuring the wrong thing.

The baselines are therefore judged with the *same* criteria table and reported
in the same `summary.md` format, so the rows are directly stackable. This is
additional value from the shared npz contract and costs nothing extra.

## Classical ML baselines

- **Features:** the 60 mrrsu summary parameters per labeled pixel. This is what
  a domain scientist would actually do with classical ML and needs no patch
  cache. The comparison then reads as *learned spatial-spectral features vs
  expert-designed indices*, which is the honest framing.
- **Models:** `RandomForestClassifier` (native multi-output) and
  `HistGradientBoostingClassifier` (one-vs-rest per class). HistGB is chosen
  specifically because it **handles NaN natively** — mrrsu carries 65535 nodata,
  and imputation would bias the comparison. No new dependency; both are sklearn.
- **Output:** per-class probability → the same probs npz.

## Atmospheric CO₂ — a diagnostic, not a correction

No summary parameter tracks **gaseous** atmospheric CO₂. MRDR is already
volcano-scan corrected and nothing measures the residual. The CO₂-named
parameters (`BD1435`, `BD3200`, `ICER1_2`, `ICER2_2`) track CO₂ **ice** and are
used for the junk/ice veto only.

The physically correct proxy for residual CO₂ is **atmospheric path length**,
available from mrrde: elevation (pressure falls ~exponentially with elevation,
so CO₂ column scales with it) and the air-mass factor
`1/cos(INA) + 1/cos(EMA)`. `HCPINDEX2` sits in the 2 µm region where residual
CO₂ leaks, so this matters for hcp specifically.

**Reported, not gated.** Real HCP occurs at low elevation too, so a hard veto
would suppress true detections. The baseline reports **hcp detection rate by
elevation decile and by air-mass decile**. If hcp detections concentrate at low
elevation and high air mass, that is residual CO₂ rather than clinopyroxene, and
it is visible rather than silently corrected away. The same diagnostic runs on
the deep model's hcp channel for free, since it reads the probs npz.

`mrrwv` (water vapour) ships as `.tab`/`.lbl` with no ENVI raster and is not used.

## Components

| script | responsibility | output |
|---|---|---|
| `extract_mrrsu_features.py` | 60 params at each labeled pixel, 65535→NaN, optional 7×7 mean | parquet sidecar, row-aligned |
| `fit_expert_rules.py` | calibrate thresholds + precision ladder from TRAIN | `config/expert_rules_7cls.json`, `config/expert_rules_pyx.json` |
| `fit_ml_baseline.py` | train RF + HistGB | joblib + metadata JSON |
| `classify_tile_baseline.py` | score one tile with any artifact | `<tid>_probs.npz` |
| `floor_test.sh` | `CLASSIFY_CMD` env override | *(unchanged default)* |

`floor_test.sh` gains an env hook rather than a fork: a forked copy drifts, and
a drifted vectorization would silently stop being the same comparison.

**Row alignment is the one place the extraction can go silently wrong** — the
same failure the MTRDR plag caches had. If the sidecar is misaligned with the
parquet, every label attaches to the wrong pixel's parameters and produces a
plausible but meaningless baseline. It gets a per-row recoverable fingerprint
asserted in a test, not a shape check.

## Validation

- **npz structural identity** — a baseline npz and a real `classify_tile_supervised`
  npz must have identical keys, dtypes, shapes and `class_names`. Compared
  against a real one on disk, not against a hand-written expectation.
- **Row alignment** — unique recoverable per-row value survives extraction.
- **Rule semantics** — synthetic pixels constructed to satisfy exactly one rule
  each produce exactly that label; a pixel satisfying olivine and hcp produces
  **both** (the multi-label guarantee, which is the regression this design
  exists to prevent).
- **Veto retention floor** — a veto that would drop a class below its retention
  fraction is logged, not silently applied.
- **Carbonate without hydration is still alteration** — the anhydrous exception.
- **Ice is junk, not alteration.**
- **Precision monotonicity** — precision should be non-decreasing along the
  ladder; a violation is reported rather than smoothed over, since it indicates
  a badly behaved index.

Every test must be seen failing under a mutation of the code it covers. Eleven
tests in this project have shipped unable to fail for their stated reason.

## Scope

**In:** the eight floor tiles (4 Nili, 2 Argyre, 2 MC11), both vocabularies —
each fitted separately to its own `config/expert_rules_<vocab>.json`, since the
pyx merge removes the lcp/hcp dominance term entirely rather than just renaming a
channel — the three baselines, and the atmospheric diagnostic.

**Also out:** a fully unsupervised variant using published Viviano-Beck detection
thresholds instead of calibrated ones. It would need no labels and so would have
zero scene overlap, which is attractive for the paper, but it puts the results on
a different x-axis from the model's probability ladder. Recorded as a known
follow-on, not built here.

**Out, deliberately:** test-split metrics for the paper. The npz contract
already carries everything needed, so that is an additive pass with no rework —
and the numbers are only meaningful once the dual-CR fine-tunes finish.

## Risks and open questions

- **`BD1900R2` may win alteration index selection for the wrong reason.** Dust
  is hydrated, so a generic hydration index can score well while being
  mineralogically uninformative. The disjunction structure already requires a
  specific OH/carbonate/sulfate feature alongside hydration, which mitigates
  this; the fit additionally prints the full per-group AUC ranking so a
  degenerate winner is visible.
- **lcp/hcp separation is expected to be poor.** Both indices respond to
  pyroxene and discrimination rests on 2 µm band-centre position. This is not a
  defect to engineer around — it is independent evidence for the pyx merge, and
  the pyx-vocabulary run is expected to be the stronger baseline.
- **Plagioclase is expected to be weak.** `BD1300` is a shallow feature and plag
  competes with bland for featureless bright ground. Consistent with hand plag's
  SAM recall of 0.29. Predicted in advance so a poor result is not
  rationalised afterwards.
- **Multi-label rules inflate polygon counts** relative to a mutually exclusive
  classical map, because a pixel can now carry several labels. This is correct
  for this vocabulary and matches the deep model's behaviour, but it must not be
  read as over-firing when comparing against published single-label maps.
- **Argyre denominator artifacts** (`data/argyre_plag_suspect_denominators.gpkg`)
  are a known problem for plag in this region. The baseline will inherit them;
  the elevation/air-mass diagnostic may make them visible.
