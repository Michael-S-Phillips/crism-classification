# CRISM Primary-Mineral Map: Improvement Roadmap

> Drafted 2026-06-06 from the 4-agent project audit + literature research synthesis. Reframed 2026-06-06 to put the global map first.

**Primary deliverable: a global per-pixel map of primary minerals (olivine, LCP, HCP, plagioclase) across the MRDR data.** A publishable per-class AP — particularly for plag — is a derived secondary win, but the model exists to produce the *map*, not the metric.

**Implication for ranking interventions:**

- A plag gain that costs >1 AP point on HCP, LCP, or olivine is a **net negative** for the map. Map quality is jointly determined by all four mafic-vs-feldspar classes.
- **Polygon-level accuracy + spatial coherence are the map-facing metrics.** Phase 0 (a polygon eval harness) is therefore not just nice-to-have — it's the only honest measurement of progress.
- Domain generalization across MC quadrants matters as a first-class concern. Hellas weakness is a map-quality bug, not a footnote.

**Two empirical anchors that constrain every choice:**

1. **Plag is encoder-limited.** Five separate head/loss interventions plateaued at ~0.14 AP. The 2026-05-25 linear probe showed fresh heads gained only +0.006 over baseline. Anything that *only* touches the classifier head is unlikely to break the ceiling.
2. **Plag/HCP trade-off has been reversed exactly once.** Contrastive learning is the only intervention where pushing plag (+0.024) also lifted HCP (+0.099). Every other pro-plag move costs HCP. Proposals must not undo that.

**Near-term deployable win** (before any new intervention): re-vectorize MC13 with the contrastive linear-probe checkpoint. The current MC13 product (`data/vector_mc13_relabeled/`, made with `ft_plag_aware_relabeled_best.pt`) has the *worse* HCP (0.669 uncorrected val). Contrastive lifts it to 0.797. That's a +0.13 AP HCP improvement available right now without any further training. The HCP layer in MC13 has been your biggest visual complaint; redeploying alone should clean it up.

Out: re-pretraining from scratch, foundation-model swap-in, full FT from the contrastive encoder (already shown to destabilize). In: small additive losses, augmented contrastive, decoupled head retraining, attention pooling, multi-task regression on the *physical* discriminative axes.

---

## Phase 0 — Eval infrastructure (mandatory; do before any claim)

The 2026-05-22 polygon-mean eval showed **val_mAP 0.72 corresponds to only 46% polygon-level accuracy** (and 8% in Hellas). Without a polygon-level harness, AP gains aren't meaningful. We've been operating blind on the metric that actually matters.

### Task 0.1 — Polygon-level eval harness
- **File:** `scripts/eval_polygon_accuracy.py` (new — model the existing 2026-05-22 polygon-mean eval)
- **What it does:** Score any checkpoint at the *polygon* level: classify every interior pixel, aggregate by mean probability per polygon, output (a) per-polygon predicted class vs labeled class confusion matrix, (b) tile-region-stratified accuracy (cmu / argyre / Hellas), (c) spatial coherence metric (median connected-component size by class)
- **Output:** `reports/polygon_eval_<ckpt_stem>.md` + companion JSON
- **Why now:** locks in a "real" metric before any new intervention. Run it once against `ft_plag_aware_real_only_best.pt`, `ft_plag_aware_relabeled_best.pt`, `contrastive_plag_v1_best.pt` (linear probe applied) to set today's baseline numbers in polygon terms.
- **Cost:** ~6 hr implementation; ~10 min/checkpoint to score.
- **Definition of done:** the three baseline polygon-accuracy numbers are recorded in `Experiments & Results.md` so subsequent phases have a real target.

### Task 0.2 — Calibration curves + Brier/ECE
- **File:** add to `scripts/eval_on_corrected_val.py` (extend, don't replace) and/or `scripts/eval_polygon_accuracy.py`
- **What it does:** binned reliability diagram (predicted probability vs empirical positive rate) per class, Brier score, Expected Calibration Error.
- **Why:** the SAM diagnostic showed classifier-plag pixels have mean P(plag) ≈ 0.5 even when spectrally olivine. Confidence is uncalibrated. Without calibration we can't safely threshold for product deployment.
- **Cost:** ~3 hr; runs alongside the polygon eval.

---

## Phase 1 — Cheap, high-prior wins (parallel-able, <1 day each)

These all compose with the existing contrastive encoder. Run each independently first to measure isolated impact, then combine the winners.

### Task 1.1 — BD1300 / RPEAK1 as auxiliary regression targets [**single highest expected gain**]
- **Source:** [arxiv 2407.16384](https://arxiv.org/html/2407.16384v1) (multitask HSI cls+reg); local 2026-05-21 RPEAK1 discovery memo.
- **Why this lever:** we have direct empirical evidence (RPEAK1 + BD1300 separate plag from olivine in mrrsu space) and we currently use them only as *input* features in the aux-norm sweep — which plateaued at plag AP 0.148 because head-level inputs can't fix encoder geometry. Treating them as *regression targets* forces the encoder to preserve those discriminative axes.
- **Files:** `models/spatial_spectral_classifier_aux.py` already has the dual-input plumbing; extend it. New head: `nn.Linear(embed_dim, 2)` predicting `(BD1300, RPEAK1)` per pixel. Loss: `cls_loss + 0.3 * MSE(reg_head_output, [BD1300, RPEAK1])`. Training data already has the values per row in `mrrsu_aux_train.npy` (the cached values we built for the failed aux ablation).
- **Implementation sketch:**
  ```python
  class SpatialSpectralClassifierAuxReg(SpatialSpectralClassifier):
      def __init__(self, ..., n_aux_targets=2):
          super().__init__(...)
          self.aux_head = nn.Linear(self.encoder.embed_dim, n_aux_targets)
      def forward(self, x):
          h = self.encoder(x)[:, self._center_idx]
          return self.head(h), self.aux_head(h)
  ```
  In `train_torch.py` extend the loss to `cls_loss + λ * MSE(aux_pred, aux_target)` with `λ=0.3`. Use the existing `mrrsu_aux_<split>.npy` files as the regression target source.
- **HPC slurm:** new `scripts/hpc_finetune_aux_regression.slurm` modeled on `hpc_finetune_plag_aware_relabeled.slurm`.
- **Cost:** ~6 hr implementation + 1 HPC run.
- **Risk:** Low. Sibling head; can ablate cleanly with `λ=0`. Worst case: same as ft_plag_aware_real_only (since `λ=0` recovers it exactly).
- **Definition of done:** model trained, scored on polygon eval + uncorrected val. If plag AP > 0.20, this is the new champion.

### Task 1.2 — SimCLR-style augmented positives in InfoNCE
- **Source:** [arxiv 2505.12482](https://arxiv.org/pdf/2505.12482) (HSI SSL few-shot), [arxiv 2408.08447](https://arxiv.org/pdf/2408.08447) (SpectralEarth).
- **Why:** the project's existing CRISM-physics noise augmentation (gaussian + 1µm spike + column bias, σ values from real label data) is already implemented in `models/noise_augmentation.py`. It's used during MAE pretraining but **not** in the contrastive loop's positive-pair construction. Literature shows aug positives are the standard +3-5% AP win in HSI contrastive.
- **Files:** `training/contrastive_train.py`. After `pos = next(positive_pool)`, add `pos_aug = noise_augmentation(pos)` and use as a second positive (two-positive InfoNCE). Alternatively, simpler: every anchor gets paired with `anchor_aug = noise_aug(anchor)` as well as a cross-pixel positive — both positives, equal weight in the InfoNCE numerator. The contrastive sweep already has a `--noise_aug` flag; reuse it but apply to anchor too, not just to all batch tensors.
- **Sketch:**
  ```python
  # Currently: pos comes from a different polygon's plag pixel
  # Add: anchor_self_pos = noise_aug(anchor)  (corrupted view of same pixel)
  # InfoNCE numerator: sum over (z_pos, z_anchor_self_pos)
  ```
- **Cost:** ~3 hr (it's mostly aug-pipeline plumbing).
- **Risk:** Low — additive on top of current contrastive. Run as variant 5 of the existing sweep.

### Task 1.3 — Copy-paste plag positives (data-side)
- **Source:** [CVPR 2021 Ghiasi et al.](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf), [X-Paste ICML 2023](https://proceedings.mlr.press/v202/zhao23f.html).
- **Why:** we have only 5,157 plag positives, of which 1,817 are the high-quality ROI patches. Center-pixel-only classification means spatial paste artifacts don't propagate to the output. Effectively 5–10× the plag positive count.
- **Files:** `data/dataset.py::CRISMSpectralPatchDataset`. Add a `copy_paste_pool` arg that takes pre-built plag patches (from `data/contrastive/extra_plag_roi/patches.npy` plus the gpkg-positive subset). With probability `p_copy_paste=0.3`, replace the center pixel of a sampled non-plag patch with a center pixel from the plag pool, and update the label.
- **Sketch:**
  ```python
  def __getitem__(self, idx):
      patch, label = ...                            # original
      if self.copy_paste_pool is not None and rng.random() < self.p_copy_paste:
          plag_idx = rng.integers(0, len(self.copy_paste_pool))
          plag_patch = self.copy_paste_pool[plag_idx]
          patch[3, 3, :] = plag_patch[3, 3, :]      # center pixel
          label = self.PLAG_LABEL                   # one-hot or soft
      return patch, label
  ```
- **Cost:** ~6 hr.
- **Risk:** Low. If Phase 4 attention-pooling head (P1 in research) is later added, copy-paste of *only* the center pixel becomes less effective — but neighborhood paste is straightforward to extend.

### Task 1.4 — Decoupled classifier retraining (cRT)
- **Source:** [arxiv 1910.09217](https://arxiv.org/abs/1910.09217) Kang et al. (the original), [CIT 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12374) long-tail decoupled training.
- **Why:** stage-1 instance-balanced training learns a good encoder; stage-2 freezes the encoder and retrains the head with class-balanced sampling. Documented +2–5% tail AP on chest-X-ray long-tail. Most importantly: **diagnostic value.** If cRT doesn't move plag past 0.18, we've reconfirmed the encoder ceiling beyond any doubt and we should escalate to Phase 3 (architectural).
- **Files:** new `scripts/cRT_head_only.py` — load any FT or contrastive checkpoint, freeze encoder, reset head, retrain head with `WeightedRandomSampler` (per-class plag weight = 5–10x).
- **Cost:** ~3 hr. Reuses existing checkpoints + dataset; head training is fast (~10 min on GPU).
- **Risk:** Very low — read-only modification of the head; encoder untouched.

---

## Phase 2 — Compose the winners + multi-class contrastive (~1 week)

After Phase 1, we know which of {1.1, 1.2, 1.3, 1.4} contributed real AP. Compose them and run the multi-class contrastive refinement on top.

### Task 2.1 — Multi-class supervised contrastive (SupCon) with prototypes
- **Source:** [Pattern Recognition 2025](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000141) (rebalanced SupCon with prototypes), [arxiv 2503.17024](https://arxiv.org/abs/2503.17024) ("Tale of Two Classes").
- **Why:** the current InfoNCE is 2-class (plag vs olivine). HCP and LCP are bystanders. Multi-class SupCon with all 5 classes + learnable per-class prototypes preserves the +0.10 HCP gain and prevents the plag/HCP trade-off from reasserting itself once we layer other improvements.
- **Files:** replace `info_nce_loss` in `training/contrastive_train.py` with a SupCon-with-prototypes loss. Keep the existing hard-negative weighting mechanism (the SAM-mined olivine pixels still get weight 2.0 in the denominator).
- **Sketch:**
  ```python
  # Per-batch: collect z_anchor, z_pos, all hard_negs, all soft_negs
  # Add per-class prototype p_c (learnable on the unit sphere)
  # Loss: SupCon over (z_anchor, z_pos_same_class) + α·MSE(z_anchor, p_class)
  # Prototype init: uniformly distributed on hypersphere; reduces collapse
  ```
- **Cost:** ~12 hr.
- **Risk:** Medium — if the SupCon temperature is wrong, can re-introduce instability. Keep encoder_lr_scale at 0.01 (same as current). Frozen-encoder linear probe protocol unchanged.
- **Sequencing:** depends on Phase 1 — initialize SupCon training from the best-of-Phase-1 contrastive checkpoint.

### Task 2.2 — Combined production run
- After SupCon converges, run linear probe + cRT (Task 1.4) on top.
- Compare polygon-level accuracy + uncorrected val + Hellas region (domain test).
- If we hit plag AP ≥ 0.25 *and* HCP ≥ 0.80, this is the new MC13 deployment champion. Re-vectorize MC13.

---

## Phase 3 — Architectural refits (only if Phase 2 plateaus)

### Task 3.1 — Attention pooling head
- **Source:** [arxiv 2309.06891](https://arxiv.org/pdf/2309.06891) (SimPool), [arxiv 2112.13692](https://arxiv.org/pdf/2112.13692) (attention aggregation).
- **Why:** the classifier currently reads only the *center pixel* token of the 7×7 patch. We're discarding 48 of 49 spatial tokens at every classification step. A learned-query attention pooler over all tokens should weakly improve on this without breaking the contrastive encoder.
- **Files:** `models/spatial_spectral_transformer.py::SpatialSpectralClassifier`. Replace center-token slicing with a single-query attention layer.
- **Caveat:** if Task 1.3 (copy-paste, center-pixel only) is the load-bearing Phase 1 win, attention pooling may hurt it. Run them as alternatives, not in combination, in this phase.
- **Cost:** ~4 hr + 1 sweep.

### Task 3.2 — Spectral derivative as auxiliary input branch
- **Source:** [arxiv 2407.18593](https://arxiv.org/pdf/2407.18593) (magnitude-derivative complementary learning for HSI).
- **Why:** first derivative of the spectrum suppresses albedo / illumination and amplifies absorption-band shape — exactly the plag/olivine 1 µm discriminator.
- **Files:** new `data/dataset.py::SpectralDerivativeDataset` wrapper; `models/spatial_spectral_transformer.py` extended to a 2-channel input (`[B, 2, 7, 7, 59]`) where channel 1 is raw, channel 0 is derivative.
- **Cost:** ~8 hr + 1 sweep.

---

## Explicitly NOT doing (yet)

| | Why deferred |
|---|---|
| **SpectralGPT / Prithvi / SatMAE foundation model swap** | Mars CRISM (atmosphere + mineral assemblages) is far OOD from Earth Sentinel-2. Resampling 13-band → 59-band is invasive; expected gain unclear; cost ~40 hr GPU. Only revisit if Phase 3 plateaus. |
| **Full MAE re-pretraining with symmetric decoder + dual-mask** | Invalidates the current contrastive checkpoint; ~80 hr GPU; high blast radius. Defer until other levers are exhausted. |
| **PU learning** | Plag positives are well-curated; the "noisy negatives" problem is real but partially addressed by SAM mining. PU adds class-prior estimation complexity that may not pay back. |
| **Mixture of experts** | Adds inference-time complexity; the encoder-bottleneck finding suggests an ensemble of heads on the same encoder won't help much. |
| **Domain-shift training on Hellas explicitly** | Out of scope for this roadmap; tracked separately. |

---

## Recommended kickoff order (next 2 weeks)

| day | task | who | gate |
|---|---|---|---|
| 1 | 0.1 (polygon eval harness) | me | baselines recorded |
| 2 | 0.2 (calibration metrics) — in parallel: kick off 1.4 (cRT) on existing contrastive ckpt | me | calibration baseline + cRT delta |
| 3 | 1.2 (augmented positives in InfoNCE) — launch on HPC | me | sweep result |
| 3-4 | 1.1 (BD1300/RPEAK1 regression head) | me | new candidate ckpt |
| 5 | 1.3 (copy-paste plag) | me | new train run |
| 6-7 | Phase 1 polygon-eval round-up + pick combination | user + me | direction for Phase 2 |
| 8-12 | 2.1 (SupCon w/ prototypes) on the best Phase 1 base | me | new champion candidate |
| 13 | 2.2 (production run + MC13 redeploy if champion) | user | MC13 product refresh |
| 14 | Decision point: Phase 3 or write up | user | — |

The Phase 1 tasks are independent and can be kicked off in parallel HPC jobs. Phase 2 depends on Phase 1's outputs.

---

## Decision checklist for the user

Before launching any of this:

1. **Confirm priorities** — is the goal still "publishable plag AP" or has it shifted to "reliable HCP product for the upcoming MC13 deployment"? They imply different ranking.
2. **Confirm Phase 0 mandatory** — do you want polygon-level eval before any AP-chasing intervention?
3. **Confirm the explicit-not-doing list** — anything on it that you'd want to challenge?
4. **Compute budget** — Phase 1 is ~40 hr HPC; Phase 2 is ~60 hr; Phase 3 ~30 hr. Acceptable?
