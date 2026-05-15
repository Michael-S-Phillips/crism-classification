# Adversarial Signal/Noise Decomposition Design (DecompSpVitAdv, v2)

**Date:** 2026-05-15
**Status:** Approved (direction approved by user; implementation delegated)
**Scope:** Replace v1's multiplicative-atmosphere decomposition with an additive signal+noise decomposition driven by gradient-reversal adversarial decorrelation. Drop the atmosphere head. Reuse the MAE pre-training checkpoint unchanged.

---

## Motivation — what v1 told us, and why v2 is different

v1 (`DecompSpVit`) factored I/F as `T(λ)·s + b(λ) + ε` with per-patch atmospheric `T` and `b`. The four-config v1 sweep produced two findings:

1. **The decomposition collapsed to a no-op.** `val_T_mean ≈ 1.0` and `val_b_mean ≈ 0` across all configs — the model sat at the atmospheric priors and let the signal decoder reproduce the input directly (`s_hat ≈ x`). Reconstruction loss dropped to 5e-5 immediately.
2. **Classification performance was unchanged** — within 0.01-0.02 mAP of v4_fixed across every configuration.

Two root causes:

- **Physical misfit.** CRISM MRDR data has already been atmospherically corrected by the PDS pipeline. The multiplicative `T_atm` term we tried to estimate doesn't exist in the data we're training on; what remains is mostly additive (detector boundary artifacts at ~1 µm, column-correlated dark current residuals, stochastic detector noise) and small in magnitude.
- **No real disentanglement pressure.** The regularizers (`||ε||²`, `||T−1||²`, `||b||²`, TV smoothness) only prevent gross failure modes. They don't actively force the signal embedding to be class-informative and the noise embedding to be class-uninformative — so the model is free to let both branches encode whatever helps reconstruction, which collapses to the identity-like solution above.

v2 rebuilds the decomposition around the actual physics of processed CRISM data, and adds a positive learning signal that *forces* the signal/noise split to mean something.

## Decomposition (additive, two-stream)

```
x  ≈  s + n
```

where `s` is the per-pixel surface reflectance estimate (the signal) and `n` is the per-pixel residual that absorbs everything else — detector seam artifacts, column noise, stochastic residual. We do **not** parametrize `n` further into named physical sources (e.g., `n_col`, `n_det`). Instead the adversarial objective lets the model discover whatever structure `n` actually has, subject only to the constraint that it be class-uninformative.

This matches the user's framing: "these are processed data, so noise has been reduced and atmosphere is ~mostly all corrected out … there still is a good amount of 'noise' (actual noise and instrument-specific oddities)."

## Architecture (`DecompSpVitAdv`)

```
                                    ┌──────────────────────┐
   x  (B, 7, 7, 59)  ───────────────►  Shared encoder      │
                                    │  (SpatialSpectralTransformer,
                                    │   6L · 4H · 128-d,
                                    │   MAE-pretrained)    │
                                    └────────┬─────────────┘
                                             │  z: (B, 50, 128)
                                             │  (CLS + 49 spatial tokens)
                                             ▼
                  ┌──────────────────────────┴──────────────────────────┐
                  │                                                     │
        ┌─────────▼─────────┐                                ┌──────────▼──────────┐
        │ Signal projection │                                │ Noise projection    │
        │ Linear 128 → 128  │                                │ Linear 128 → 128    │
        │ (per token)       │                                │ (per token)         │
        └─────────┬─────────┘                                └──────────┬──────────┘
                  │ s_emb: (B, 49, 128)                                 │ n_emb: (B, 49, 128)
                  │                                                     │
        ┌─────────┴───────────────────┐                       ┌─────────┴───────────────────┐
        ▼                             ▼                       ▼                             ▼
┌──────────────────┐         ┌──────────────────┐    ┌──────────────────┐         ┌──────────────────┐
│ Signal decoder   │         │ Classifier head  │    │ Noise decoder    │         │ Gradient-reversal│
│ MLP per-token    │         │ Linear 128 → 5   │    │ MLP per-token    │         │ + Discriminator  │
│ 128 → 256 → 59   │         │ (from center     │    │ 128 → 256 → 59   │         │ MLP 128 → 64 → 5 │
│ → s_hat          │         │ pixel s_emb)     │    │ → n_hat          │         │ (from center     │
│ (B, 49, 59)      │         │ → logits (B, 5)  │    │ (B, 49, 59)      │         │ pixel n_emb)     │
└──────────────────┘         └──────────────────┘    └──────────────────┘         └──────────────────┘
                                                                                            │
                                                                                            ▼
                                                                                    disc_logits (B, 5)

Reconstruction:    x̂ = s_hat + n_hat
```

### Module breakdown

- **`SpatialSpectralTransformer` (unchanged)** — existing encoder, identical to v1. MAE checkpoint loads unchanged via `load_encoder_state_dict`.

- **Signal projection** (new, lightweight) — `Linear(128, 128)` applied per-token. Produces `s_emb ∈ ℝ^(B, 49, 128)`, the signal-only encoder representation.

- **Noise projection** (new, lightweight) — `Linear(128, 128)` applied per-token. Produces `n_emb ∈ ℝ^(B, 49, 128)`, the noise-only encoder representation.

- **Signal decoder** (new) — MLP per-token, `128 → 256 → 59`. Maps `s_emb` to per-pixel reflectance prediction `s_hat`.

- **Noise decoder** (new) — MLP per-token, `128 → 256 → 59`. Maps `n_emb` to per-pixel residual `n_hat`. Note: the noise decoder needs enough capacity to encode the column/detector-boundary patterns the user described.

- **Classifier head** (same as v1) — `Linear(128, n_classes)` reading the **center-pixel `s_emb`**, NOT the raw encoder embedding. This is the critical change: the classifier no longer has free access to noise-correlated features.

- **Discriminator + Gradient Reversal Layer (GRL)** — A small MLP (`128 → 64 → n_classes`) attached to the center-pixel `n_emb` *through* a gradient-reversal layer. Forward: identity. Backward: gradients flipped and scaled by `λ_adv`.
  - The discriminator's parameters receive **positive** gradient → it learns to predict class from noise embedding.
  - The encoder + noise projection receive **reversed** gradient → they are pressured to make `n_emb` *uninformative* about the class.
  - This is the standard DANN (Domain-Adversarial Neural Network, Ganin & Lempitsky 2015) pattern. The "domain" here is the class label, and we want noise to be domain-invariant.

### Forward pass (pseudocode)

```python
def forward(self, x):                                # x: (B, 7, 7, 59)
    z = self.encoder(x)                              # (B, 50, 128)
    tokens = z[:, 1:]                                # (B, 49, 128) — skip CLS

    s_emb = self.signal_projection(tokens)           # (B, 49, 128)
    n_emb = self.noise_projection(tokens)            # (B, 49, 128)

    s_hat = self.signal_decoder(s_emb)               # (B, 49, 59)
    n_hat = self.noise_decoder(n_emb)                # (B, 49, 59)
    x_hat = s_hat + n_hat                            # (B, 49, 59) — additive recon

    center_s_emb = s_emb[:, self._center_idx]        # (B, 128) — center pixel signal
    center_n_emb = n_emb[:, self._center_idx]        # (B, 128) — center pixel noise

    logits = self.classifier(center_s_emb)           # (B, 5)
    n_emb_grl = GradientReversal.apply(center_n_emb, self.lambda_adv)
    disc_logits = self.discriminator(n_emb_grl)      # (B, 5)

    return logits, s_hat, n_hat, x_hat, disc_logits, center_s_emb, center_n_emb
```

### Center-pixel-only classifier and discriminator?

We use the **center-pixel** tokens for both classifier and discriminator (matching v1's classifier). The reconstruction operates on all 49 spatial tokens (we want `s_hat + n_hat ≈ x` everywhere in the patch). This keeps the adversarial game between encoder and discriminator focused on the single token whose mineralogy we actually care about.

## Composite loss

```
L_total = L_cls
        + λ_recon · L_recon
        + λ_adv   · L_adv        (gradient-reversed for the encoder/projections)
        + λ_smooth · L_smooth
```

| Term | Definition | Default λ | Purpose |
|---|---|---|---|
| `L_cls` | ASL on `(logits, labels, sample_weights)` with class_weights `(1, 1, 1.5, 3, 1)` | 1.0 | drives classification, same as v5 |
| `L_recon` | MSE on `(s_hat + n_hat)` vs valid pixels of `x` | **10.0** | enforces decomposition closure — *much* higher than v1's 1.0 because the recon target is the entire input and we need real pressure |
| `L_adv` | ASL on `(disc_logits, labels, sample_weights)` — same loss family as classifier for consistency | 1.0 | discriminator learns to predict class from `n_emb`. Encoder receives this gradient *reversed* via GRL with multiplier `λ_adv_schedule` (see below). |
| `L_smooth` | total-variation penalty on `s_hat` across spatial dims | 0.001 | mineralogy is spatially smooth at the patch scale |

### GRL λ_adv schedule (DANN-style warmup)

Constant `λ_adv` from epoch 1 can be unstable. We use the standard DANN schedule:

```
p = current_epoch / total_epochs   (∈ [0, 1])
λ_adv(p) = (2 / (1 + exp(-10·p))) - 1     ∈ [0, 1]
```

This grows smoothly from 0 at epoch 1 to ~1 at the end. Multiplied by a target maximum `λ_adv_max` (default `1.0`). Implemented as a hook called once per epoch that updates the GRL layer's stored `lambda_adv`.

### Regularizers we are NOT including (and why)

- `||n_hat||²` on raw magnitude — would prevent the trivial "all-signal" collapse but also fight the legitimate "noise is real" objective. The adversarial pressure is a more honest disentanglement signal.
- `||s_hat||²` — same reasoning; we don't want to suppress signal magnitude.
- Atmosphere-style priors — not applicable (no multiplicative term).
- A per-class explicit weighting on `L_adv` — keep symmetric with classifier loss for first iteration.

## Why this avoids v1's collapse

In v1, `s_hat ≈ x, n_hat ≈ 0` was a valid trivial solution because nothing prevented the model from putting all reconstruction effort into the signal branch. In v2:

- If the encoder puts class-discriminating information into `n_emb`, the discriminator wins, `L_adv` is small, and **the encoder gets a strong negative gradient via GRL pushing it to remove class info from `n_emb`**.
- Conversely, anything class-discriminating that ends up in `s_emb` is rewarded by `L_cls`.
- The reconstruction loss (now 10× stronger) forces `s_hat + n_hat ≈ x` — so if class-relevant features were excluded from `n_emb`, the only place they can go and still satisfy reconstruction is `s_emb`.

The fixed-point of the adversarial game is: `n_emb` carries only class-orthogonal information (instrument artifacts, column patterns, stochastic noise — the "noise" the user described); `s_emb` carries class-relevant surface signal.

The discriminator's val accuracy `val_disc_acc` is the diagnostic for whether the game is working. We expect:
- Early epochs: discriminator accuracy moderate-to-high (encoder hasn't separated yet)
- Mid-to-late epochs: discriminator accuracy approaches the marginal class prior (encoder has won; n_emb is class-uninformative)

If `val_disc_acc` stays high throughout, the encoder isn't winning the game — either `λ_adv` is too small or the encoder lacks capacity to separate.

## Training procedure

**Stage 1 — Pre-training (unchanged).** MAE checkpoint `spatial_mae_128d_6l_best.pt` (epoch 194, MSE 0.016) loads unchanged.

**Stage 2 — Fine-tuning.** Same as v5/decomp_v1 except:
- New model: `DecompSpVitAdv`
- New loss: `AdversarialDecompositionLoss`
- New diagnostic metric: `val_disc_acc` (should decrease over training)
- Per user feedback, **drop the frozen-encoder condition**: sweep only `encoder_lr_scale ∈ {0.001, 0.01, 0.1}` (3-task array instead of 4)
- Single optimizer (AdamW), single learning rate schedule — the GRL handles the adversarial sign internally

## Files & components

| Path | Action | Responsibility |
|---|---|---|
| `models/decomp_spatial_vit_adv.py` | Create | `DecompSpVitAdv` module + `GradientReversalLayer` autograd function |
| `training/adv_decomp_losses.py` | Create | `AdversarialDecompositionLoss` — composite loss returning total + named components |
| `training/train_torch.py` | Modify | Add a `DecompSpVitAdv` branch parallel to the existing `DecompSpVit` branch; expose `lambda_adv` schedule; log `val_disc_acc` |
| `scripts/train.py` | Modify | Add `decomp_spatial_vit_adv` to `TORCH_MODELS`; one new CLI flag `--lambda_adv_max` (default 1.0). Reuse the existing `--decomp_lambda_recon` and `--decomp_lambda_smooth` flags (their semantics are the same for both v1 and v2; the slurm script just overrides the default `λ_recon=1.0` → `10.0` for v2). |
| `scripts/hpc_ablation_decomp_v2.slurm` | Create | 3-task sweep (lrscale {0.001, 0.01, 0.1}) — no frozen baseline |
| `tests/test_decomp_spatial_vit_adv.py` | Create | Shape contracts, GRL sign check, MAE checkpoint compatibility |
| `tests/test_adv_decomp_losses.py` | Create | Loss component tests including the adversarial gradient direction |
| `scripts/figures/fig_decomp_v2_architecture.py` | Create | Architecture diagram for v2 (deferred — generated after first sweep so we can include disc-accuracy plot) |

## Acceptance criteria

1. `DecompSpVitAdv` forward returns the 7-tuple `(logits, s_hat, n_hat, x_hat, disc_logits, s_emb_center, n_emb_center)` with documented shapes.
2. Backward pass shows opposite signs on the encoder vs the discriminator for the adversarial loss component (verified by unit test with controlled inputs).
3. MAE encoder checkpoint loads cleanly: no `unexpected` keys, no `missing` keys for encoder weights.
4. On at least one HPC training run, `val_disc_acc` decreases from epoch 1 → end of training. If `val_disc_acc` *increases* or stays flat at chance + 5%, the adversarial game converged immediately and we need to investigate (likely `λ_adv` is too weak or discriminator too strong).
5. `val_eps_norm_mean` of `n_hat` is non-trivially nonzero (say, > 0.005). If `n_hat ≈ 0` everywhere, the model is again ignoring the noise branch and we need to revisit.
6. Per-class APs are at minimum on par with v5 best (val_mAP ≥ 0.55). The headline win we're hoping for: HCP↔LCP center-pixel cosine similarity (from the same diagnostic in `fig_v5_embedding.png`) drops below 0.84.

## Out of scope for v2

- **Separate noise-source heads** (column, detector boundary, stochastic ε). If `n_hat` shows column structure post-training, a v3 could split it explicitly — but the user wanted to let the model discover the structure rather than pre-specify.
- **Larger ViT encoder** (256-d / 8-layer / 6-head). Deferred until we know whether v2's classification ceiling is set by capacity or by representation quality.
- **Variational / Bayesian formulation** of the signal/noise factors. Clean theory, additional tuning surface — defer.
- **Adversarial training with a separate optimizer for the discriminator** (alternating updates, different LRs). Standard DANN with a single optimizer + GRL is simpler and usually adequate as a first try.

## Open hyperparameters

- `λ_adv_max` — default 1.0. May need to be 0.1 if the adversarial signal destabilizes training, or 5.0 if disc accuracy doesn't drop.
- `λ_recon` — default 10.0 (vs v1's 1.0). The recon term needs to actually push the model; v1 showed 1.0 was too small.
- `lambda_adv_schedule_speed` — the `10` constant in the DANN schedule. Default keeps the standard DANN behavior; could slow to `5` if instability shows.
- Discriminator capacity — default `128 → 64 → 5`. May need to shrink to `128 → 5` (single linear) if discriminator wins too easily, or widen if it can't predict at all.
