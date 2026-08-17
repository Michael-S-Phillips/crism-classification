# Contrastive refinement of the dual-CR encoder — design

**Date:** 2026-08-17
**Status:** approved (design), awaiting spec review
**Creates:** `scripts/build_dust_contrastive_pools.py`,
`scripts/train_contrastive_dualcr.py`, `scripts/eval_embedding_separation.py`,
`tests/test_build_dust_contrastive_pools.py`,
`tests/test_eval_embedding_separation.py`
**Reuses unchanged:** `models/contrastive_encoder.py`, `training/contrastive_train.py`
**Depends on:** the mining + merge stages of
`2026-08-17-dust-hard-negatives-design.md`

---

## Why contrastive, and why this shape

The dual-CR MAE objective is **reconstruction only** — MSE of a denoised
reconstruction against the clean patch, over all 49 positions
(`models/denoising_spatial_mae.py:86-139`). There is no contrastive term, and
nothing in pretraining ever asks the encoder to place two spectra *apart*.

This project has run contrastive refinement once before, and it worked. Per
`wiki/Plagioclase Detection.md`: a 30-epoch InfoNCE refinement of the plag-aware
MAE encoder, using SAM-mined hard negatives against vetted ROI positives, gave
the **best plagioclase AP on record (0.152)** and a **frozen-encoder linear probe
that beat every prior full fine-tune**, with a side effect of HCP 0.70 → 0.80 —
attributed to the plag/olivine task forcing the encoder to disentangle the 1 µm
absorption shape. A targeted contrastive task produced a general gain.

That is the shape reused here, against the failure we can currently measure.

## The three pools

Mapping the plag recipe onto the dust problem:

| plag run | here |
|---|---|
| positives: hand-vetted plag ROIs + labeled gpkg plag | **mafic confirmed twice**: inside a hand-labeled lcp/hcp/olivine polygon AND with the matching index (LCPINDEX2 / HCPINDEX2 / OLINDEX3) above the 90th percentile of that index over all valid pixels of that tile |
| hard negatives: classifier-plag that is spectrally olivine | **mined dust the model calls mafic** — the pool from the hard-negatives spec |
| soft negatives: labeled olivine (the confusable class) | **ordinary bland**: index-free but *dark* — fails the dusty test (RBR or R770 below tile p60) |

Splitting bland into dark-featureless (soft) and bright-dusty (hard) is the
substance of the design. Both belong away from mafic, but only the dusty half is
being confused with it, and the 2.0 / 1.0 hard/soft weighting is where that
asymmetry is expressed. Collapsing them into one negative pool would spend the
objective's capacity on a boundary that is not broken.

Positives require agreement between a hand label and an independent index
because the hand labels are known-noisy — the same audit that puts hand
plagioclase at SAM recall 0.29 is why an index-only or label-only positive pool
would teach the encoder the label noise.

## Representation

Pools are **118-channel dual-CR 7×7 patches, per-block standardised** (hull
÷0.0705, linear ÷0.1726) — identical to what the encoder was pretrained on and
what the classifier consumes. The mining stage emits raw 59-band rows, so the
pools go through the existing dual conversion rather than being built ad hoc; a
pool built at a different scaling would train the encoder on a distribution
nothing else ever sees.

## Split discipline — the load-bearing constraint

**Every pool is restricted to train-split units.** The contrastive stage trains
the *shared encoder*; if its pools contain val or test pixels, every downstream
number is contaminated — including the linear probe that is supposed to be the
cheap honest read, and the fine-tune val that decides the checkpoint.

Concretely this creates an ordering dependency, and "in parallel" does not mean
"simultaneously":

```
mine dust hard negatives            (shared, spec A stage 1)
        v
merge + assign_unit_balanced_splits (shared, spec A stage 2 — splits exist here)
        v
   +----+---------------------------+
   |                                |
build contrastive pools        cache + fine-tune arm A
(filter split == 'train')      (MAE backbone)
   |                                |
contrastive refinement              |
   |                                |
fine-tune arm B                     |
(contrastive backbone)              |
   +----------------+---------------+
                    v
        compare — encoder is the only difference
```

Splits come from spec A's merge, which runs `assign_unit_balanced_splits` over
the concatenated frame. Pools filter that column. Nothing assigns splits twice.

**Note on the precedent:** the prior plag contrastive run drew its hard negatives
from Argyre SAM pools and I have **not** verified that it respected the split.
Its 0.152 may therefore be optimistic. That does not change this design — it is
the reason this design states the constraint explicitly rather than inheriting
the earlier practice.

## Training

`ContrastiveEncoder(n_bands=118, patch_size=7, embed_dim=256, n_heads=4,
n_layers=6, proj_dim=64)` — the existing class takes all of these as constructor
arguments, so **no model code changes**. Warm-start with
`load_encoder_state_dict` from `spatial_mae_dualcr_denoising_256d_6l_best.pt`.

InfoNCE on L2-normalised projections at the settings that worked: `tau=0.07`,
`hard_weight=2.0`, `soft_weight=1.0`, 30 epochs, batch 64, lr 1e-4,
`encoder_lr_scale 0.01`. These are inherited deliberately rather than re-tuned;
tuning them is a later experiment, not part of establishing whether the idea
works at all.

The projection head is discarded. The saved encoder state dict is structurally
identical to the MAE's, so it warm-starts a classifier unchanged.

## Evaluation, in cost order

**1. Embedding separation — minutes, before any fine-tune, and pre-registered.**
Take t1321's `lcp >= 0.99` pixels, split them into the false group
(LCPINDEX2 < tile p40) and the true group (> 0.03), embed both with the encoder,
and measure how separable they are — AUC of a held-out logistic probe on the
embedding. **Measure the MAE encoder first and record the number before training
the contrastive one**, so the comparison cannot be chosen after the fact.

Reference points already measured on this population: hull-CR band depth
separates them 2.5× (0.059 vs 0.150); the classifier's own `p_bland` separates
them at AUC 0.93.

**2. Linear probe** — frozen encoder, single Linear head, val_mAP and per-class
AP. This is how the plag result was established.

**3. Two fine-tune arms** on identical hard-negative data, differing only in
backbone, then the floor test. Success criteria are inherited from spec A so the
two experiments are judged on one yardstick: t1321 false share **35% → target
< 10%**, and the t1249 over-correction guard (confident-LCP count must not fall
more than ~15%). Compared in **pixels retained**, not polygon counts — a
subtractive change fragments regions and inflates polygon counts (measured: Nili
lcp @0.50 went 1,675 → 3,622 under the bland gate while losing 51% of pixels).

## Risks

- **This may not be the bottleneck.** The dual-CR representation already
  separates the two populations 2.5× in CR band depth and the classifier discards
  it, which points at the head and the ASL setting rather than the encoder. If
  so, contrastive refinement reshapes something that was not broken. Evaluation
  step 1 costs minutes and tests exactly this before either fine-tune is
  committed — if embedding separation is already high on the MAE encoder, stop
  and go after the head instead.
- **Representation collapse.** InfoNCE can drive all embeddings together.
  Monitor embedding variance and the positive/negative similarity gap, not loss
  alone — a falling loss is consistent with collapse.
- **Overfitting to the mined tiles.** All eight floor-test tiles (t1249, t1250,
  t1321, t1322, t0434, t0435, t1086, t1087) are excluded from every pool, by the
  same exclusion the mining stage applies. t1321 is the sharpest case: diagnostic
  tile, prime dust-mining territory, and holder of the primary success metric.
- **A shared encoder is a shared risk.** Refinement aimed at dust could degrade
  an unrelated class. The floor test across all five minerals is the guard, which
  is why arm B is floor-tested rather than judged on the dust metric alone.

## Out of scope

Tuning `tau` / `hard_weight` / `proj_dim`; self-supervised augmentation pairs;
joint MAE+contrastive pretraining; changing ASL. Each is a separate variable and
this experiment already carries one.
