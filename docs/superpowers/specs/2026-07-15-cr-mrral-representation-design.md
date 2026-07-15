# Continuum-Removed mrral Representation + CR-Native Encoder — Design

**Date:** 2026-07-15
**Status:** Approved (design) — pending spec review, then implementation plan
**Scope:** Spec 1 of 2 (see Decomposition). This spec covers the representation +
encoder core. The publication-defensibility evaluation layer is deferred to Spec 2.

## Goal

Replace the raw-reflectance input representation with **continuum-removed (CR)
mrral spectra**, and pretrain a **CR-native encoder**, so the learned
representation keys on mineral absorption bands rather than albedo/continuum.
Prove the change on the existing honest splits via the floor test before building
any downstream evaluation.

## Motivation

The current pipeline feeds **raw mrral reflectance** to the encoder (`data/dataset.py`
returns raw patches; `models/spatial_spectral_transformer.py` first op is
`band_embed = nn.Linear(59, 128)`), and the denoising MAE was pretrained to
reconstruct raw reflectance. Raw-reflectance angle/variance is dominated by
**brightness/albedo/continuum**, not mineralogy — every inter-class medoid angle is
<3° while the LCP hand-vs-confirmed split is 6.5°.

Consequence: the same mineral at different brightness lands in different regions of
the representation, so classes do not transfer across terrains. Demonstrated by the
honest-splits recovery run (`ft_7cls_v3b_lrscale001`): LCP val AP 0.838 but the
floor test collapsed Nili LCP to **0 polygons** at every threshold. Continuum
removal collapses the LCP two-population spectral-angle gap from **6.50° → 2.03°**
(2026-07-15 analysis, `reports/lcp_two_populations*.png`), confirming most of the
separation is continuum/albedo, not mineralogy.

This is a planet-wide generalization problem: a representation that entangles
nuisance variance (albedo, continuum slope, photometric/atmospheric residual) with
mineral signal cannot be defensible across the range of Martian terrains. The fix
is a representation invariant to those nuisances while preserving diagnostic
absorptions. Auxiliary CR features at the classifier head were rejected as a
band-aid: they leave the encoder brightness-entangled.

## Robustness bars (chosen by the user; earned in Spec 2)

1. **Held-out-region generalization** — leave-whole-regions-out transfer.
2. **Match published CRISM maps** — validation against known-mineralogy sites
   (qualitative, since we stay on MRDR mrral rather than reproducing the ratioed
   MTRDR summary products pixel-for-pixel).
3. **Physically-interpretable features** — decisions tied to absorption bands, not
   albedo/artifacts.

These bars drove two decisions: (a) a **physically-grounded** representation (CR)
over black-box learned invariance, because CR is interpretable and connects to
published spectroscopy; (b) staying on **MRDR mrral** (user choice) rather than
moving the data foundation to ratioed MTRDR/TRDR.

## Decomposition

- **Spec 1 (this doc):** CR representation + patch-cache builder change + CR-native
  denoising-MAE pretrain (with an encoder-size probe) + fine-tune on honest splits +
  floor-test go/no-go gate.
- **Spec 2 (deferred):** leave-one-region-out evaluation framework, known-site /
  literature validation, and interpretability/attribution. Only pursued if Spec 1
  passes its gate.

## Architecture & data flow

The model architecture is unchanged. The only structural change is *what the encoder
sees*. Pipeline:

```
mrral tile (raw reflectance, 59 bands)
   │  continuum removal (upper convex hull, good-band window, 1µm overlap excluded)
   ▼
CR patch  +  brightness scalar (mean good-band reflectance, pre-CR)
   │                                   │
   ▼ (CR spectrum only)                │
CR denoising-MAE encoder (pretrained)  │
   ▼                                   ▼
        classifier head  ◄── brightness scalar (late-fusion aux, 1-D)
   ▼
7-class logits
```

1. **CR at cache-build time.** A new continuum-removal step in the patch-cache
   builder converts each pixel's raw spectrum to CR before the patch is written.
   Applied to both the global pretrain cache and the labeled fine-tune caches.
2. **CR-native denoising-MAE pretrain** on the CR global cache → CR encoder.
3. **Fine-tune** the classifier on CR patches (+ brightness scalar) from the CR
   encoder, on the current honest unit-balanced splits.
4. **Inference** continuum-removes each tile identically before classifying.

### Continuum removal (the representation)

- **Method:** upper convex hull continuum, divide spectrum by the hull → CR
  reflectance ≤ 1, absorptions dip below 1. Parameter-free (no polynomial degree or
  tie-points to defend); the standard spectroscopic choice.
- **Window:** good bands only — m2..m58 (534–2457 nm) with the 1 µm detector-overlap
  region (1000–1065 nm) excluded, matching the existing band-exclusion convention
  (`scripts/label_quant/sam_endmembers.py`).
- **Degenerate/low-SNR pixels:** NODATA (65535) and non-finite pixels are handled as
  today (→ 0, masked). Pixels whose hull is degenerate (flat/near-zero) get CR set to
  1.0 (no bands) rather than dividing by ~0. CR must never emit NaN/Inf.
- **Brightness scalar:** mean good-band reflectance *before* CR, retained per pixel
  as one explicit input. CR discards absolute albedo, which is a real cue for
  bland/dust; the scalar preserves it. Injected at the **classifier head** via the
  existing late-fusion aux path (`models/spatial_spectral_classifier_aux.py`), as a
  single deliberate 1-D feature — not a substitute for fixing the representation.

### Pretrain (CR denoising MAE + encoder-size probe)

- **Corpus:** rebuild the global patch cache with CR applied, all mc## tiles, 7×7
  patches. The current global cache is truncated from the tile-download corruption
  and must be rebuilt regardless, so CR piggybacks on required work.
- **Model:** same `DenoisingSpatialSpectralMAE`; input = CR spectrum; corrupt →
  reconstruct **in CR space** (CR precedes the denoising corruption, so the model
  learns to denoise where we classify). Brightness scalar is NOT part of the MAE.
- **Encoder-size probe (disciplined, not a grid):** two pretrains —
  **128-dim/6-layer** (proven config) as primary, and **256-dim/6-layer** as a single
  larger comparison arm.
- **Selection metric:** a **frozen linear probe on the honest val** (`val_mAP_core`),
  NOT MAE reconstruction loss — recon loss rewards encoding brightness/texture, the
  nuisance we are removing. Keep the larger encoder only if it wins the linear probe;
  otherwise ship 128/6 (lighter planet-wide inference, less overfitting risk on the
  small labeled set).

### Fine-tune

- Classifier on CR patches (+ brightness scalar) from the selected CR encoder.
- **Current honest unit-balanced splits**, `val_mAP_core` stop metric — so the run is
  a **single-variable** change (representation only) vs today's champion
  `ft_7cls_v3b_lrscale001`, directly comparable through the floor test.
- Labels unchanged for this spec (the weak-hand-LCP purity issue is a parallel track;
  keeping labels fixed isolates CR as the variable).

## Evaluation — the go/no-go gate

Spec 1 is judged by the **leakage-immune** comparators, not val_mAP:

- **Floor test** (Nili t1249/t1250/t1321/t1322 + Argyre t0434/t0435) vs the current
  champion's floor test (`reports/floor_tests/v4honest_lrscale001/`).
- **MC11 alteration visual** on the altered tiles (e.g. t1450), vs the review-only and
  6cls-denoise baselines already captured.

**Pass criteria (concrete):**
- **LCP survives OOD** — Nili LCP produces a non-trivial, threshold-graded polygon
  population (contrast: honest recovery = 0 at all thresholds).
- **Mafic/alteration do not collapse** — no return of the mafic→olivine flood or the
  alteration→bland collapse seen in the two MC11 baselines.

If the gate fails, **stop**: CR-on-mrral is insufficient and we reconsider (e.g. the
Spec-2 data-foundation fork) rather than building the eval layer on a dead core.

## Components (each independently testable)

| component | responsibility | file(s) |
|---|---|---|
| CR transform | raw spectrum → CR (+ brightness scalar); NaN/Inf-safe | new `data/continuum_removal.py` |
| patch-cache CR hook | apply CR when writing patches | `scripts/build_global_patch_cache.py`, `data/dataset.py` cache path |
| CR denoising-MAE pretrain | pretrain CR encoder; 2-arm size probe | new `scripts/hpc_pretrain_cr_denoising.slurm` (from `pretrain_spatial_mae_denoising.py`; `--embed_dim {128,256}`) |
| linear-probe selector | frozen-encoder val_mAP_core to pick encoder size | new `scripts/linear_probe_encoder.py` |
| CR fine-tune | classifier on CR + brightness, honest splits | new `scripts/hpc_finetune_cr.slurm` |
| inference CR | CR tiles before classify | `scripts/classify_tile_supervised.py` |

## Non-goals / out of scope

- The Spec 2 evaluation layer (region-holdout, literature validation, interpretability).
- Label cleanup / LCP purity remediation (parallel track).
- Any move to MTRDR/TRDR or canonical ratioing (user chose to stay on mrral).
- Contrastive/learned-invariance representations (rejected: weakest on interpretability).
- Architecture changes beyond the encoder-size probe and the brightness aux input.

## Risks & mitigations

- **CR amplifies noise on low-albedo/low-SNR pixels.** Mitigate with the degenerate-
  hull guard (CR→1.0) and the retained denoising objective; watch the floor test for
  noise-driven false positives.
- **Brightness scalar leaks the albedo nuisance back in.** It's a single 1-D feature
  at the head, not in the encoder; if it dominates, drop it (ablation in fine-tune).
- **Bigger encoder overfits the small labeled set.** Guarded by linear-probe
  selection and keeping 128/6 as default.
- **Compute:** two pretrains (~days each on HPC). Bounded by the 2-arm probe, not a
  grid; the global-cache rebuild is owed regardless.
- **Denoise/CR ordering wrong.** Fixed: CR before corruption in pretraining.

## Testing

- **CR transform (unit):** hull correctness on synthetic spectra; CR ≤ 1 everywhere;
  band depth recovered on a planted absorption; NaN/Inf-safe on flat/zero/NODATA
  input; brightness scalar = mean good-band reflectance.
- **Cache builder (unit):** a built patch equals the CR of the raw patch; shape/NODATA
  handling unchanged.
- **Pretrain/fine-tune (smoke):** CPU smoke that the CR encoder pretrains a step and
  the classifier consumes CR + brightness without shape errors.
- **Parity:** inference CR path matches the cache-build CR path on the same pixels.
