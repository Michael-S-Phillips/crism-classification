# Hand-labeled-core training dataset — design

**Date:** 2026-08-08
**Status:** approved, pending implementation plan
**Touches:** `scripts/build_7cls_dataset.py`, `data/dataset.py`, `scripts/train.py`

---

## Goal

Rebuild the 7-class training parquet so that **hand-labeled data is the core** of
the dataset and review data augments it, rather than the current build where 9.5M
ungraded review rows sit against 2.6M hand-labeled rows and review effectively
defines the dataset.

Three classes are deliberate exceptions, because hand labels cannot carry them:

| Class | Why it is an exception |
|---|---|
| bland / dust | Not labeled as dust in the hand set — the `other` catch-all is not the same concept |
| alteration | Review coverage is better and more consistently judged |
| junk | Does not exist in the hand set at all |

## Source inventory (measured 2026-08-08)

Hand-labeled base — `data/mrral_pixels.parquet`, 2,619,784 rows:

| tier | rows |
|---|---|
| High | 1,535,293 |
| Moderate | 912,884 |
| Low | 171,607 |

Positives: olivine 1.59M · lcp 545,963 · hcp 434,305 · plagioclase 353,023 ·
`other` 876,748 · alteration 111,825.

Review v3 (`data/mc13_review_7cls_v3/`) — confidence-graded, 1,347 decisions.
Grades are stamped into `confidence_tier` as `Reviewed-High/-Moderate/-Low`.

| target | High | Moderate | Low |
|---|---|---|---|
| bland (rejects) | 741,971 | 1,837 | 8,414 |
| junk (ambiguous) | 164,859 | 572 | 880 |
| alteration (HN) | 1,853 | 719 | 5,790 |
| confirms (all) | 300,274 | 140,587 | 23,459 |

Confirm positives by grade: olivine 240,027 H / 134,555 M · lcp 15,227 H /
6,238 M · hcp 57,541 H / 28,935 M · alteration 40,580 H / 462 M ·
**plagioclase 0** · **`other` 0**.

Legacy MC13 review (`data/mc13_review/`) — **ungraded**, 334 decisions,
8,262,052 hard-negative rows (8,123,976 rejects · 103,895 alteration ·
34,181 ambiguous) and 1,244,075 confirms across only **49 polygons**
(median 9,513 px/poly, max 190,404). Its alteration hard-negatives are by
contrast well spread: 103,895 px across **65 polygons**, median 255 px/poly,
max 10,638.

ndviz relabels — 81 rows, one tile, all `ambiguous`.

## Decisions

1. **Review quality bar: v3 session, `Reviewed-High` + `Reviewed-Moderate`.**
   Low-grade review is excluded. The bar is only meaningful for the v3 session;
   the legacy session was never graded.

2. **Bland is review-only.** The base parquet's 876,748 `other > 0` rows — today
   the "bland tiles" source, subsampled to 300,000 — are **dropped entirely**,
   not retained as all-negative background. Retaining them with `bland = 0`
   would assert that bland terrain is not bland, producing false negatives that
   suppress the bland head; the loss has no per-class row masking to avoid this.

3. **Alteration: review is the source of truth, hand alteration is uncapped.**
   Hand-labeled alteration rides along wherever it co-occurs, because alteration
   in the base parquet is overwhelmingly a *dual label* on rows that also carry
   mineral labels. Capping it would delete hand-labeled mineral pixels as a side
   effect. Resulting mix ≈ 111,825 hand vs 147,509 review (~57% review).

4. **Junk is review-only** — automatic, it does not exist in the hand set.
   165,431 px from v3 at High+Moderate.

5. **Legacy session is excluded except for `alteration`, `lcp`, `hcp`.**
   Alteration because v3 supplies only 2,572 HN px against legacy's 103,895.
   lcp/hcp because v3 confirms supply only 21,465 / 86,476 px.

6. **Legacy per-polygon cap: 5,000 px on legacy *confirms* (lcp/hcp) only.**
   At the standard 20k the legacy confirm contribution concentrates into 10
   polygons for lcp and 18 for hcp — 131k lcp pixels from ten places is a
   memorization risk, not a generalization fix. 5k/poly spreads the same budget
   across all 49 confirm polygons (145,858 px).

   **Legacy alteration hard-negatives keep the standard 20k/poly cap.** They do
   not have the concentration problem: 103,895 px spread across 65 polygons,
   median 255 px/poly, max 10,638. A 5k cap would trim them to 81,889 px (−21%)
   for no diversity gain. The tight cap exists to fight concentration, so it is
   scoped to the source that is actually concentrated.

7. **ndviz pixel supersede is disabled.**

8. **Weight scheme moves to train time and becomes sweepable.** See below.

## Weight scheme

The build stamps `confidence_tier` only. `--weight_scheme` is a **train-time**
flag selecting the tier→weight table, so one parquet supports a full sweep
without a ~1 GB rebuild per point.

```python
WEIGHT_SCHEMES = {
    'level': {'high': 1.0, 'moderate': 0.85, 'low': 0.70,
              'reviewed-high': 1.0, 'reviewed-moderate': 0.85,
              'reviewed-low': 0.70},
    'review_up': {'high': 1.0, 'moderate': 0.85, 'low': 0.70,
                  'reviewed-high': 2.0, 'reviewed-moderate': 1.7,
                  'reviewed-low': 1.4},
    'hand_up': {'high': 1.5, 'moderate': 1.3, 'low': 1.0,
                'reviewed-high': 1.0, 'reviewed-moderate': 0.85,
                'reviewed-low': 0.70},
}
```

Default `level`: hand-labeled is the core by volume, and review dominates
bland/junk/alteration only because it is the sole source there.

**The `Reviewed-*` pass-through is deliberate, not a bug.** `_collapse_labels`
lowercases `confidence_tier` and looks it up in a three-key table
(`high`/`moderate`/`low`); the v3 tiers `Reviewed-High/-Moderate/-Low` miss that
lookup **by design**. `scripts/review/persistence.py` stamps them outside
`_TIER_WEIGHTS` precisely so the per-polygon reviewer weight
(`REVIEW_CONFIDENCE_WEIGHTS` = High 1.0 / Moderate 0.75 / Low 0.5) passes through
verbatim, and `tests/test_collapse_reviewed_tier.py` locks that behaviour.

`--weight_scheme level` must therefore **preserve** the pass-through, not
override it. Measured current state:

| source | `confidence_tier` | stamped weight |
|---|---|---|
| hand base | High / Moderate / Low | via `_TIER_WEIGHTS` → 1.0 / 0.85 / 0.70 |
| legacy review (9.5M rows) | `High` | 1.0 |
| v3 review | `Reviewed-High/-Moderate/-Low` | 1.0 / 0.75 / 0.5 |

So the effective weighting today is already close to `level`. The flag exists to
make the scheme *sweepable*, not to repair a defect. (The 2.0 / 3.0 review
weights live in `build_review_augmented_train.py`, which builds the superseded
`mrral_pixels_with_review*.parquet` and is not part of this pipeline.)

**Provenance trap.** Legacy rows are stamped `confidence_tier='High'`. A grade
filter keyed on `confidence_tier` alone cannot distinguish an ungraded legacy
row from a reviewer-graded v3 `Reviewed-High` row, so the legacy exclusion would
silently fail. Session provenance must be tracked explicitly at read time — see
the `review_session` column in the plan.

## Source policy

A `SourcePolicy` dataclass in `build_7cls_dataset.py`, one entry per class:

| Class | Hand | v3 (High+Mod) | Legacy | Cap |
|---|---|---|---|---|
| olivine | core | confirms | — | 20k/poly |
| plagioclase | core (+ MTRDR synth at train time) | none exist | — | 20k/poly |
| lcp, hcp | core | confirms | **confirms** | 20k v3 / **5k legacy confirms** |
| alteration | rides along, uncapped | confirms + HN | **HN** | 20k/poly (both) |
| bland | **dropped** | rejects | — | 20k/poly |
| junk | absent | ambiguous | — | 20k/poly |

New flags. **Every default is permissive — the hand-core recipe is opt-in.**
A bare `python scripts/build_7cls_dataset.py` must reproduce the pre-hand-core
dataset exactly (same counts, same splits, same tier histogram), so the
champion's data lineage stays reproducible. The values below are what you pass
to *get* the hand-core recipe; the defaults are in the right-hand column.

| flag | hand-core value | default (inert) |
|---|---|---|
| `--review_grades` | `High Moderate` | `High Moderate Low` |
| `--legacy_classes` | `alteration lcp hcp` | `_ALL_POLICY_CLASSES` (all) |
| `--legacy_confirm_cap` | `5000` | `MAX_PX_PER_POLYGON` (20,000) |
| `--bland_sources` | `review` | `all` |
| `--ndviz_dir` | `''` (disabled) | the ndviz dir |
| `--out` | `data/mrral_pixels_7cls_handcore.parquet` | `data/mrral_pixels_7cls.parquet` |

Full hand-core invocation:

```
python scripts/build_7cls_dataset.py \
  --bland_sources review --review_grades High Moderate \
  --legacy_classes alteration lcp hcp --legacy_confirm_cap 5000 \
  --ndviz_dir '' --out data/mrral_pixels_7cls_handcore.parquet
```

**Inertness is stricter than "drops no rows".** `_apply_legacy_policy`'s confirm
branch rebuilds the frame with `pd.concat` even when the cap removes nothing,
which changes row ORDER — and `_joint_resplit` is order-sensitive at ties, so
that alone moved ~250 rows between train and val. The confirm branch therefore
returns the original frame unchanged when the cap binds nothing. Verified: the
bare run is identical to the pre-hand-core build on every count, split and tier
histogram.

Unchanged, deliberately: the joint unit-balanced re-split over the combined
frame (`_joint_resplit` — the adjacent-tile leakage fix), the MTRDR plag synth
injection, and the excluded-polygon filter.

## Known consequences

**Admitting legacy for pyroxene also admits a large block of legacy
olivine — and it is NOT dual labels.** An earlier draft of this spec claimed
these were olivine labels riding along on pyroxene rows, by analogy with
alteration. That was wrong. Measured over the 1,244,075 legacy confirms:

| | rows |
|---|---:|
| olivine-positive | 444,254 |
| — of which **olivine-ONLY** (no lcp/hcp) | **440,793** (99.2%) |
| — of which genuine oliv+pyx dual labels | 3,461 (0.8%) |
| pyx-positive (lcp\|hcp) | 803,282 |
| — of which pyx-ONLY | 799,821 |

So the `confirmed` fragment being keyed to `'lcp'` in the policy table does not
merely let a few olivine co-labels ride along — it admits ~440k **independent,
ungraded legacy olivine-only rows** through a door opened for pyroxene. That is
a different decision from the alteration dual-label case and must be made
deliberately, not inherited by analogy.

Options considered: (a) accept it, (b) zero the olivine columns on legacy rows
admitted for pyx, or (c) admit only legacy confirm rows that are actually
pyx-positive.

**RULING 2026-08-08: (a) accept as-is**, after the correction above was put to
the user explicitly. Rationale: olivine is the strongest class in every floor
test, the 5k/polygon confirm cap keeps realised volume far below the raw 440k,
and additional olivine coverage is unlikely to hurt. Option (b) was rejected
because zeroing would create false negatives on the 3,461 genuine dual-label
rows — teaching "this olivine pixel is not olivine", the same failure mode that
made bland review-only rather than zeroed.

**Review augmentation is a small effect for the mineral classes.** Even with the
legacy supplement, lcp and hcp remain overwhelmingly hand-labeled. This build
will not on its own address the LCP OOD collapse that the floor tests keep
surfacing — that is a separate two-population problem.

**Dataset shrinks.** Dropping 876,748 hand `other` rows and excluding ungraded
legacy bland removes several million rows relative to the current 7cls build.
Expected, and the point.

## Output

`data/mrral_pixels_7cls_handcore.parquet` — a new file. The existing
`mrral_pixels_7cls.parquet` is not clobbered, so the current champion's data
lineage stays reproducible.

Vocabulary stays 7-class in the parquet (`olivine | lcp | hcp | plagioclase |
bland | alteration | junk`). The pyx merge is a train-time collapse
(`_collapse_labels` derives `pyx` from lcp/hcp), so `--pyx` / `--pyx_alt` runs
consume this same parquet without a rebuild.

## Testing

- Grade filter admits only `Reviewed-High` / `Reviewed-Moderate` from v3.
- Bland contains zero rows originating from the base parquet.
- Junk is review-only and non-empty.
- Legacy rows appear only for `alteration`, `lcp`, `hcp`.
- `--legacy_confirm_cap` is enforced per polygon on legacy confirms, and does
  **not** apply to legacy alteration hard-negatives (which use
  `MAX_PX_PER_POLYGON`).
- Every weight scheme resolves every tier present in the parquet, and
  `level` reproduces today's effective weights exactly — including the
  deliberate `Reviewed-*` stamped-weight pass-through that
  `tests/test_collapse_reviewed_tier.py` asserts.
- Legacy rows (stamped `confidence_tier='High'`) are not admitted by the v3
  grade filter; provenance, not tier, decides session membership.
- Splits stay unit-balanced; no polygon straddles train/val.
- Alteration hand rows are not dropped when they co-occur with mineral labels.
