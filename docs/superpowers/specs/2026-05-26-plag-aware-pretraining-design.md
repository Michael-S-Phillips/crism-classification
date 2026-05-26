# Plag-Aware Multi-Task Pretraining — Design

> Status: approved design, pending implementation plan
> Date: 2026-05-26

## Problem & Motivation

Plagioclase classification is stuck at val_AP ≈ 0.13, and three independent diagnostics
have established this is an **encoder-representation** ceiling, not a classifier/loss problem:

| diagnostic | plag val_AP |
|---|---|
| Linear probe on frozen cont1 encoder (fresh head) | 0.131 |
| Focal-loss fine-tune sweep (best of γ∈{1.5,2.0,3.0}) | 0.138 |
| Binary plag-vs-rest, encoder unfrozen, posw∈{3,5,10} | 0.134 |

All cluster at ~0.13. Loss-side and head-side interventions are exhausted. The encoder
(`spatial_mae_denoising_128d_6l_best.pt`) was pretrained with a pure denoising-MAE
reconstruction objective, which rewards predicting strong/smooth spectral features and
discards the subtle plagioclase 1.25 µm absorption (diagnostic SNR ≈ 2). The fix must act
on the **representation**.

## Goal

Lift plagioclase val_AP above the ~0.13 ceiling by co-training the denoising-MAE encoder
with a supervised auxiliary objective, warm-started from the existing encoder so the classes
that already work (olivine 0.87, LCP 0.93, HCP 0.69) do not regress. Secondarily, fold the
available plagioclase spectral library into the fine-tuning data.

**Success bar:**
- **Publishable target:** plag val_AP ≈ **0.6** (on real val pixels). This is the goal that
  makes the result defensible in peer review.
- **Minimum signal gate:** plag val_AP > ~0.20. Below this, the encoder approach is not
  working and we escalate to input-level changes (inject mrrsu PLG band parameter as an
  auxiliary channel) rather than iterating further on this design.
- **Guardrail:** olivine/LCP/HCP APs must not regress materially.

## Decisions (locked during brainstorming)

1. **New plag data** (`/mnt/mrdr/plagioclase-targeted/`, 30 mean spectra) enters via
   **tile + augment → fine-tuning set only**. Not used in pretraining.
2. **Auxiliary loss** = multi-task: MAE reconstruction + supervised auxiliary head.
3. **Init** = warm-start from `spatial_mae_denoising_128d_6l_best.pt` (~30–50 epochs),
   NOT from scratch.
4. **Aux head** = full **5-class multi-label** (independent sigmoid logits), trained with the
   **same ASL loss** as fine-tuning (`asl_gamma_neg=4.0`) plus `class_weights` upweighting
   plagioclase. Rationale: keeping HCP/LCP separable in the embedding is strictly more
   informative; the HCP/LCP collapse is a downstream label-space decision, not an
   encoder-capacity one.
5. **Data flow** = dual-stream (Approach 1): unlabeled global patch cache (recon) +
   labeled mrral patches (recon + aux).

## Architecture

### Model — `MultiTaskDenoisingMAE`

A thin wrapper around the existing `DenoisingSpatialSpectralMAE` (encoder + decoder + recon
path unchanged). Adds exactly one module:

- `aux_head = nn.Linear(embed_dim=128, 5)` — reads the **full-visibility center-token**
  embedding (the same token `SpatialSpectralClassifier` uses downstream).

Methods:
- `forward_recon(x)` → existing masked denoising path → recon loss. (Encoder sees only the
  visible 25% of tokens; center token usually masked.)
- `forward_aux(x_labeled)` → **full** encoder pass (all 49 tokens visible) → center token →
  5 logits.

**Why two passes for labeled batches:** the recon objective needs masking (75%), but the
downstream classifier consumes the encoder at full visibility on the center token. The aux
head must match downstream usage, so labeled batches run the encoder twice — masked (for
recon) and full (for aux). This is the correct alignment, not a workaround.

### Data flow — dual stream

- **Stream U (unlabeled):** `CRISMCachedPatchDataset` over `global_patch_cache_dir`
  (sharded `global_patches_*.npy`). Recon only. Preserves global spatial diversity and
  guards the warm-started encoder against forgetting general features.
- **Stream L (labeled):** mrral patches — `mrral_train_patches_p7.npy` memmap + labels from
  `mrral_pixels.parquet` (train split). Recon + aux.
- Each training step draws one U batch and one L batch.

### Loss

```
total = recon_mse(U ∪ L) + λ · ASL(aux_logits_L, labels_L)
```

- ASL: reuse the existing fine-tuning loss machinery (`asl_gamma_neg=4.0`, `asl_gamma_pos=0.0`,
  `asl_clip=0.05`) with `class_weights` upweighting plagioclase.
- **λ schedule:** ramp 0 → λ_target over a warmup (≈5 epochs) so the cold aux head does not
  yank the warm-started encoder. λ_target is a tunable hyperparameter (start ≈1.0).
- Log `recon` and `asl` separately. If `recon` climbs as `asl` falls, λ is too high.

### Training

- Warm-start encoder + decoder from `spatial_mae_denoising_128d_6l_best.pt`; aux head random.
- ~30–50 epochs, HPC `gpu_standard`, ~1 day wall.
- Checkpoint selection: best by a combined criterion — low recon AND high plag AP on a small
  monitoring slice **held out from the train split** (NOT the official val split, which must
  stay clean for the 3-way fine-tuning comparison below). Save `encoder_state` in the existing
  MAE checkpoint format so `SpatialSpectralClassifier.load_encoder_state_dict` consumes it
  unchanged.

## Synthetic Plagioclase Patches (fine-tuning data)

New `scripts/build_synthetic_plag_patches.py`:

1. Parse both ENVI spectral libraries in `/mnt/mrdr/plagioclase-targeted/`
   (`unratioed_plag_highconfidence`, `unratioed_plag_moderateconfidence_w_2micron`;
   30 spectra total, 545 bands, 364–3937 nm).
2. Subset 545 → 59 mrral wavelengths (`m0..m58`, 410–2457 nm) by nearest-wavelength match.
3. Clip to [0, 0.5] to match `CRISMSpectralPatchDataset` normalization.
4. Per spectrum, generate N augmented variants. Each variant: tile the 59-band spectrum to
   7×7×59, then add **per-pixel** Gaussian noise (σ≈0.005) + small per-band jitter + slight
   continuum scaling. Per-pixel (not uniform-tile) noise is the mitigation against a
   flat-patch shortcut — neighbors differ so the encoder cannot learn "49 identical pixels ⇒
   plag." N is a tunable hyperparameter (start ≈ a few hundred per spectrum, target a plag
   row count comparable to a real labeled tile's contribution).
5. Emit:
   - `synth_plag_patches_p7.npy` — (N_total, 7, 7, 59) float32, parquet-row order.
   - A parquet fragment with matching rows: `tile_id='SYNTH_PLAG_<libname>_<i>'`,
     `plagioclase=1`, all other labels 0, `confidence_tier` from the source library
     (high/moderate), `confidence_weight` per existing convention, **`split='train'` only**.

**Eval integrity:** synthetic rows are TRAIN-only and never enter val/test. All reported plag
AP is measured on real labeled pixels.

Dataset loading: a small extension to `CRISMSpectralPatchDataset` (or a thin concat dataset)
joins the real train cache + the synthetic cache when the train split is requested.

## Evaluation — 3-way comparison

After plag-aware pretraining, fine-tune the 5-class classifier with the **identical cont1
config** (`encoder_lr_scale=0.001`, ASL, lr=5e-4, batch=256, 6L/128d/4H, patch 7), varying only
the lever under test:

| run | encoder | train data | isolates |
|---|---|---|---|
| baseline (cont1) | old denoising MAE | real only | — |
| A | plag-aware MAE | real only | encoder effect |
| B | plag-aware MAE | real + synthetic | encoder + data |

- **Primary metric:** plag val_AP on real val pixels vs the 0.13 baseline.
- **Guardrail:** olivine/LCP/HCP val_AP must not regress materially (watch for the
  focal-sweep-style HCP dip).
- **Decision:** publishable target plag AP ≈ 0.6. Minimum signal gate ~0.20 — clearing it
  means the approach works and is worth tuning (λ, N, epochs) toward 0.6; falling below it
  means escalate to input-level change (mrrsu PLG channel injection), out of scope here.

## Files

New:
- `models/multitask_denoising_mae.py` — wrapper + `aux_head`.
- `scripts/pretrain_plag_aware_mae.py` — dual-stream training loop (adapted from
  `scripts/pretrain_spatial_mae_denoising.py`).
- `scripts/build_synthetic_plag_patches.py` — ENVI → tiled/augmented cache + parquet fragment.
- `scripts/hpc_pretrain_plag_aware.slurm` — HPC submission.

Modify:
- Dataset loader (`data/dataset.py`) — concat synthetic patch cache for the train split.
- Config — add path for the synthetic cache; confirm `global_patch_cache_dir` is set on HPC.

Checkpoint format: emit `encoder_state` exactly as the existing MAE checkpoints so the
downstream classifier loads it with no changes.

## Testing

- **Unit:** ENVI library parse + nearest-match band subset (assert 59 bands, monotonic
  nearest indices); augmentation produces N variants with shape (7,7,59), `plagioclase=1`,
  `split='train'`; aux-head forward shape (B,5); multi-task loss assembles non-NaN with a
  mixed labeled/unlabeled step.
- **Smoke:** 1-epoch pretraining on a tiny subset — both `recon` and `asl` move; checkpoint
  saves and reloads into `SpatialSpectralClassifier.load_encoder_state_dict` with empty
  missing/unexpected encoder keys.
- **Integration:** fine-tune smoke from the new encoder loads cleanly and trains one epoch.

## Out of Scope

- Collapsing HCP/LCP into a pyroxene class + mrrsu disambiguation (separate intervention).
- Injecting the mrrsu PLG band parameter as an auxiliary input channel (the escalation path
  if this design does not clear ~0.20).
- Re-fetching the 18 source CRISM scenes from PDS to reconstruct real plag ROIs (blocked on
  ROI masks we do not have; the colleague's deliverable we chose not to wait for).
