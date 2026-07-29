# Pyroxene (pyx) Merge — Design (Spec A)

**Date:** 2026-07-29
**Status:** Approved (design) — pending spec review, then implementation plan
**Scope:** Spec A of 2. This covers the deep-model label merge (7→6 class). The
post-hoc ortho/clino Ca-typing overlay is **Spec B (deferred)**.

## Goal

Merge LCP + HCP into a single **`pyx`** (pyroxene) class for the deep classifier —
a 6-class target `olivine · pyx · plagioclase · bland · alteration · junk` — as a
**single-variable change vs the champion** `ft_7cls_v3b_lrscale001`. The model
detects pyroxene robustly; ortho-vs-clino (LCP/HCP) is deferred to a post-hoc
band-parameter overlay (Spec B).

## Motivation

LCP vs HCP is **marginally separable on MRDR mrral**, by every measure we have:
- Deep-model failure modes: LCP OOD collapse (champion floor test: Nili LCP → 0)
  and LCP↔HCP swaps (a named floor-test known-bad).
- Even Viviano's newest purpose-built 2µm band parameters separate confirmed LCP vs
  HCP at only **~0.76 AUC** when evaluated leakage-immune (polygon-grouped); the
  0.99 from random CV was pixel/polygon leakage.
- Physical root cause: mrral samples the 2µm region at **~40 nm**, ~6× coarser than
  full CRISM (~6.55 nm) — too coarse to resolve the ortho/clino Band II center shift
  per-pixel.

Forcing LCP/HCP as separate deep classes therefore chases a distinction the data
barely supports. Merging removes the brittle boundary; pyroxene becomes olivine's
robust neighbor. The Ca distinction moves to physically-grounded band parameters
applied post-hoc, as a confidence-graded continuum (Spec B).

## Relationship to the CR representation work (decision: raw first)

pyx is a label change, orthogonal to the input representation. The CR floor test
(2026-07-28) is a **partial** win: Nili LCP dramatically restored, but Argyre
olivine went diffuse (peaks @0.50, not @0.90 — a known-bad signature) and HCP was
over-predicted. Decision: **build pyx on the RAW representation first**, as a clean
single-variable change vs the champion. Rationale: (1) CR isn't a clean win yet —
folding it in carries the Argyre olivine regression and confounds attribution;
(2) the merge may itself *subsume* CR's main benefit (no lone LCP to collapse), so
pyx-on-raw tests whether merging alone fixes pyroxene without CR's cost/side-effect;
(3) clean attribution. CR stays a parallel track; combine later only if pyx-on-raw
still shows pyroxene weakness CR would fix.

## Design

### Merge mechanics (at the training target only)
- In `build_7cls_dataset.py`, collapse `lcp` + `hcp` into `pyx = max(lcp, hcp)`
  (element-wise, multi-label-safe), producing a 6-class target
  `olivine · pyx · plagioclase · bland · alteration · junk`.
- **Preserve the original lcp/hcp labels** everywhere else — `labeled_spectra.parquet`,
  review outputs, ndviz. Only the *deep-model training target* merges. Spec B's Ca
  classifier needs the lcp/hcp labels to calibrate; the merge must not destroy them.

### Model & training (mirror the champion, change only the labels)
- Warm-start the same v3-bland encoder `ft_bland_v3_lrscale0001_cont1_best.pt`;
  `--model spatial_vit`; raw reflectance.
- Same honest unit-balanced splits; same MTRDR synth-plag injection the champion
  used (kept here — this is the single-variable control vs the champion).
- 3-arm sweep `encoder_lr_scale {0.001, 0.01, 0.1}`; stop metric `val_mAP_core`
  (junk excluded — mean AP over `olivine, pyx, plagioclase, bland, alteration`;
  excludes only junk, matching the champion's 7-class core for single-variable parity).
- Produces `ft_6cls_pyx_lrscale{0001,001,01}_best.pt`.

### 6-class pyx vocabulary wiring
The new vocab threads through: build LABEL_COLS, `train.py`/dataset `n_classes` and
class list, `metrics.py` per-class AP keys + `val_mAP_core` core-set, and the
floor-test vectorizer's class list (emit a `pyx` gpkg). The repo has done 6-class
before; this is a new *vocabulary*, not new machinery.

### Floor-test criteria (simplified)
The pyroxene-splitting criteria collapse into one: **pyx detected robustly on Nili
(pyroxene-rich) without flooding; olivine still confident (≥~300 @0.99); plag ≈ 0 on
Argyre.** The "LCP alive," "HCP contained," and "LCP↔HCP swap" criteria are replaced
by the single pyx check. Baseline for comparison = champion's lcp+hcp combined.

## Feeds Spec B (deferred)
- Global deployment of the 6-class model → per-tile **pyx probability maps** (the
  overlay's input: which pixels are pyroxene).
- Preserved confirmed **lcp/hcp labels** → the overlay's calibration set.
- Spec B computes Viviano band params (validated Python port — corr 0.94 BI / 0.75–0.85
  BII vs IDL — or read from crevo where available) on pyx pixels and places them on a
  confidence-graded ortho↔clino continuum. Out of scope here.

## Non-goals
- The Spec B post-hoc overlay.
- CR integration (raw first; separate track).
- Any ortho/clino output from the deep model (that is the whole point of deferring).

## Risks & mitigations
- **pyx over-prediction / flood** — watch the floor-test pyx count + gpkg size; the
  combined lcp+hcp on the CR run ran high, so this is a real watch item.
- **6-class wiring touches several files** — cover with the build/metrics/vectorizer
  unit tests below; keep bland/junk handling byte-identical.
- **Loss of a genuinely-separable subset** — if some regions *do* cleanly split
  LCP/HCP, the merge defers that to Spec B rather than losing it (labels preserved).

## Testing
- **build merge (unit):** `pyx = max(lcp, hcp)` per row; 6-class LABEL_COLS emitted;
  `lcp`/`hcp` columns preserved in `labeled_spectra` (not dropped).
- **metrics (unit):** 6-class `val_mAP_core` excludes only junk = mean AP over olivine/pyx/plagioclase/bland/alteration.
- **train (smoke):** a 6-class step runs on synthetic data without shape errors.
- **vectorizer (smoke):** emits a `pyx` gpkg for the 6-class model.
- **acceptance:** the simplified floor-test gate above, vs the champion's lcp+hcp.
