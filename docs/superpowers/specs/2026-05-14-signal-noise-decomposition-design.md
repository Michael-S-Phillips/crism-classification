# Signal / Noise Decomposition Encoder Design (DecompSpVit)

**Date:** 2026-05-14
**Status:** Approved (direction approved by user; implementation delegated)
**Scope:** Replace the monolithic SpatialSpectralClassifier encoder + head with a four-component decomposition that explicitly separates surface signal from atmospheric, scene-bias, and stochastic noise terms. Drop-in replacement for the existing fine-tuning pipeline; pre-training MAE checkpoint is reused.

---

## Motivation

The current SpatialSpectralClassifier encodes raw I/F reflectance directly into a 128-d embedding, then a linear head produces per-class logits. Three failure modes have emerged:

1. **HCP↔LCP confusion** (cosine similarity 0.84 in the v4_fixed embedding diagnostic) — the encoder hasn't fully separated spectrally adjacent pyroxenes.
2. **Plagioclase struggles** (val_AP ~0.08 in v4_fixed) — the diagnostic 1.3 µm absorption is subtle and easily confounded with continuum slope or atmospheric residuals.
3. **No physical interpretability** — when the model fails, we can't say whether it's confused by surface mineralogy or by instrumental/atmospheric artefacts.

CRISM observations are not pure surface reflectance. The canonical Sun→surface→sensor equation is:

```
I/F_observed(λ, r, c) ≈ T_atm(λ) · R_surface(λ, r, c) + b_path(λ) + n_column(c, λ) + ε(r, c, λ)
```

where `T_atm` is wavelength-dependent atmospheric transmission (multiplicative), `b_path` is atmospheric path radiance (additive, spatially uniform at a 7×7 patch scale ~1 km), `n_column` is detector column-correlated artefact (additive), and `ε` is stochastic per-pixel noise.

A purely additive `signal + noise ≈ input` decomposition cannot represent the multiplicative `T_atm` term cleanly. We want a decomposition that respects the physical structure of CRISM measurements.

The hypothesis: forcing the encoder to factor I/F into physical components — surface signal, multiplicative correction, additive bias, and stochastic residual — gives the classifier a cleaner, less-confounded input than raw I/F, and produces a more interpretable embedding.

## Decomposition (B′)

We adopt the **B′ variant**: per-pixel surface signal, patch-level atmospheric terms, per-pixel stochastic residual. Column-correlated artefacts are lumped into the residual ε for this first iteration (can be split out in a follow-on if the residual shows obvious column structure).

```
x[r, c, λ]  ≈  T(λ) · s[r, c, λ]  +  b(λ)  +  ε[r, c, λ]
```

| Symbol | Shape per patch | Per pixel or per patch | Physical meaning |
|---|---|---|---|
| `x` | (7, 7, 59) | input | Observed I/F |
| `s` | (7, 7, 59) | per-pixel | Surface reflectance — **the signal** |
| `T` | (59,) | **per-patch** | Multiplicative atmospheric transmission |
| `b` | (59,) | **per-patch** | Additive path radiance |
| `ε` | (7, 7, 59) | per-pixel | Stochastic + column residual |

The patch-level assumption for `T` and `b` is justified because (a) atmospheric scale heights are kilometers, larger than a 7×7 patch (~1 km), (b) CRISM image footprints don't change atmospheric path enough at patch scale to matter, and (c) it dramatically reduces the parameter budget and trivial-solution risk.

## Architecture (DecompSpVit)

```
                              ┌──────────────────────┐
   patch x (7×7×59) ────────► │ Shared encoder       │
                              │ (existing 6L ViT,    │
                              │  128-d, 4-head)      │
                              └────────┬─────────────┘
                                       │  (B, 49+1, 128)
                                       ▼
                              ┌──────────────────────┐
                              │ CLS token (B, 128)   │───────────────────┐
                              │ + 49 spatial tokens  │                   │
                              └────────┬─────────────┘                   │
                                       │                                 │
              ┌────────────────────────┼─────────────────────┐           │
              │                        │                     │           │
              ▼                        ▼                     ▼           ▼
     ┌────────────────┐      ┌────────────────┐    ┌────────────────┐  ┌──────────────────────┐
     │ Signal decoder │      │ Atmosphere head│    │ Residual head  │  │ Classification head  │
     │ MLP per token: │      │ MLP from CLS:  │    │ MLP per token: │  │ Linear from          │
     │   128 → 59     │      │   128 → 2·59   │    │   128 → 59     │  │   s_center → 5       │
     │ → s_hat        │      │   split into   │    │ → ε_hat        │  │ + sigmoid            │
     │   (B,49,59)    │      │   T_hat, b_hat │    │   (B,49,59)    │  └──────────────────────┘
     └────────────────┘      │   (B,59)×2     │    └────────────────┘
                             └────────────────┘
              │                        │                     │
              └─────────► reconstruct: x_hat = T_hat · s_hat + b_hat + ε_hat
```

### Module breakdown

- **`SpatialSpectralTransformer` (unchanged)**. The existing encoder produces (B, 50, 128) — 49 spatial tokens + 1 CLS. Pre-trained MAE checkpoint loads in unmodified.

- **Signal decoder** (new) — small MLP applied per-token, mapping 128-d encoder output → 59-d spectrum estimate at that spatial position. Output: `s_hat` ∈ ℝ^(B,49,59), reshapable to (B, 7, 7, 59).

- **Atmosphere head** (new) — reads only the CLS token (which has full spatial receptive field), MLP 128 → 2·59. Output split into:
  - `T_hat` ∈ ℝ^(B, 59): pass through a sigmoid then scale to a sensible range (e.g., `0.3 + 0.7·σ(·)` keeps it in [0.3, 1.0])
  - `b_hat` ∈ ℝ^(B, 59): unconstrained additive offset (or pass through `tanh · max_offset` to bound)

- **Residual head** (new) — MLP per-token mapping 128 → 59. Output: `ε_hat` ∈ ℝ^(B, 49, 59).

- **Classification head** (new) — single linear layer 128 → 5 reading the **center-pixel token output by the encoder**.

  The encoder is shared by both the classifier and the signal decoder. The reconstruction objective (via the signal decoder) pressures the encoder to represent surface mineralogy faithfully; the classifier consumes that same shared embedding. The classifier does **not** read `s_hat` (the per-pixel reflectance reconstruction) — that would force the classifier to operate in a 59-band space where it loses the encoder's rich representation. Reading the encoder embedding directly is both more parameter-efficient and more representationally expressive.

  This means the joint optimization works as follows:
  - The reconstruction loss says "encoder embedding should be sufficient to recover the surface reflectance via the signal decoder."
  - The classification loss says "encoder embedding should be sufficient to predict the mineral class."
  - Together: the embedding represents whatever is necessary to recover *both* clean reflectance and mineral class — which by construction is "the surface signal, separated from noise."

### Forward pass

```python
def forward(x):                              # x: (B, 7, 7, 59)
    z = encoder(x)                           # (B, 50, 128)   — CLS + 49 spatial
    cls = z[:, 0]                            # (B, 128)
    tokens = z[:, 1:]                        # (B, 49, 128)

    s_hat = signal_decoder(tokens)           # (B, 49, 59)
    epsilon_hat = residual_decoder(tokens)   # (B, 49, 59)
    Tb = atmosphere_head(cls)                # (B, 2*59)
    T_hat = sigmoid_scale(Tb[:, :59])        # (B, 59), bounded in [T_min, T_max]
    b_hat = Tb[:, 59:]                       # (B, 59), unconstrained

    # Per-pixel reconstruction (broadcast T_hat and b_hat across the 49 tokens)
    x_hat = (
        T_hat[:, None, :] * s_hat
        + b_hat[:, None, :]
        + epsilon_hat
    )                                        # (B, 49, 59) — reshape to (B,7,7,59)

    # Classifier reads the center-pixel SIGNAL embedding
    center_token = tokens[:, 24]             # (B, 128)   token index 24 = (3,3)
    logits = class_head(center_token)        # (B, 5)
    return logits, s_hat, T_hat, b_hat, epsilon_hat, x_hat
```

## Loss

```
L_total = L_cls
        + λ_recon · L_recon
        + λ_eps   · L_eps_reg
        + λ_T     · L_T_reg
        + λ_b     · L_b_reg
        + λ_smooth · L_smooth
```

| Term | Definition | Purpose | Default λ |
|---|---|---|---|
| `L_cls` | ASL on logits vs labels (existing) | drives classification | 1.0 |
| `L_recon` | MSE on `x_hat` vs valid pixels of `x` | enforces the decomposition is consistent with the data | 1.0 |
| `L_eps_reg` | `‖ε_hat‖²` averaged over tokens | residual should be small — prevents the model from dumping everything into ε | 0.1 |
| `L_T_reg` | `‖T_hat − 1‖²` averaged over bands | weakly prior T toward "no attenuation" | 0.01 |
| `L_b_reg` | `‖b_hat‖²` averaged over bands | weakly prior b toward zero | 0.01 |
| `L_smooth` | total-variation penalty on `s_hat` across spatial dimensions, mean over bands | encourages spatial smoothness on the surface signal (mineralogy varies slowly within 7×7 patches) | 0.001 |

### Why these regularizers prevent the trivial solution

The trivial solution is `s_hat = (x - b)/T`, `ε_hat = 0`, with `T, b` chosen freely. Without regularization, the model can satisfy reconstruction perfectly while leaving the signal embedding meaningless. The priors force structure:

- `L_T_reg` and `L_b_reg` push the atmospheric terms toward "no correction needed" by default — they only deviate when the data demands it.
- `L_eps_reg` keeps the residual small, so the model can't just dump leftover error into `ε`.
- `L_smooth` exploits the fact that mineralogy is spatially correlated within 7×7 patches — the signal should be smoother than the raw I/F, which contains spatially-uncorrelated stochastic noise.

The classifier loss `L_cls` is the **primary** signal that the encoder's center-pixel token must represent surface mineralogy. The reconstruction loss is consistency with physics. Together they pin the decomposition.

### NODATA handling

CRISM I/F has nodata pixels (encoded as 65535 in the raw mrral). The training-time dataset already nullifies these (sets to 0 per `extract_pixels.py`). For the reconstruction loss, we additionally weight by a valid mask: pixels with `|x| < 1.0` after clipping participate; saturated/zeroed pixels are excluded from `L_recon` only. The classifier loss is unchanged.

## Training procedure

**Stage 1 — Pre-training (unchanged).** SpatialSpectralMAE on the full unlabeled global MRDR dataset. Existing checkpoint `spatial_mae_128d_6l_best.pt` (epoch 194) is reused without retraining. The encoder weights load into `DecompSpVit.encoder` via the existing `load_encoder_state_dict` interface.

**Stage 2 — Fine-tuning on the labeled dataset.** Same setup as v5:
- Stratified parquet (`data/mrral_pixels.parquet`, 2.42M rows)
- Hard olivine labels + tier-based confidence weights (commit `9f0b3a3`)
- Class-balanced ASL loss with weights `(1, 1, 1.5, 3, 1)`
- Cosine LR schedule, AdamW, encoder_lr_scale sweep `{frozen, 0.001, 0.01, 0.1}`
- Epochs 100, patience 25, min_delta 0.001 (same as v5)
- New per-sweep additions: the reconstruction + regularizer losses with λ values above

**Stage 3 — Diagnostics (per checkpoint).** After each training run we save:
- Per-class val AP and test AP
- Mean `T_hat`, `b_hat`, `‖ε_hat‖` over the val set — physical sanity check (T should be ~0.7-0.95 on average, b should be small, ε should be smaller than s)
- A spectrum-decomposition figure for a handful of val pixels (input, signal, atmosphere, residual)
- Center-pixel embedding cosine-similarity matrix between classes (the same diagnostic from `fig_v5_embedding.png`)

## Files & components

| Path | Action | Responsibility |
|---|---|---|
| `models/decomp_spatial_vit.py` | Create | `DecompSpVit` module: shared encoder + 4 heads + forward returning all components |
| `training/decomp_losses.py` | Create | Composite loss with classifier + reconstruction + regularizers; configurable λ values |
| `training/train_torch.py` | Modify | Add a `model_type='decomp_spatial_vit'` path that uses the composite loss and logs the additional metrics (mean T, b, ε norms) |
| `scripts/train.py` | Modify | Add `decomp_spatial_vit` to `TORCH_MODELS` and wire CLI args for the λ values |
| `scripts/hpc_ablation_decomp_v1.slurm` | Create | First v1 sweep of the decomposition encoder (4-config ablation matching v5 conventions) |
| `tests/test_decomp_spatial_vit.py` | Create | Shape + invariant tests: forward pass returns all components with correct shapes; reconstruction `T·s + b + ε ≈ x` holds at init when regularizers are off |
| `scripts/figures/fig_decomposition.py` | Create | Spectrum-decomposition figure for a few val pixels |
| `scripts/figures/fig_decomp_architecture.py` | Create | New architecture diagram using the scientific-schematics skill |

## Implementation order

1. **`DecompSpVit` module** — clean class wrapping the existing encoder with the four new heads. Tests for shape correctness and load-from-MAE-checkpoint compatibility.
2. **`DecompositionLoss`** — composite loss with the six terms. Unit test that each term contributes the expected gradient direction.
3. **`train_torch.py` integration** — branch on model_type; log new metrics.
4. **`scripts/train.py` CLI** — `--model decomp_spatial_vit` plus `--lambda_recon`, `--lambda_eps`, etc.
5. **First sweep** — run `hpc_ablation_decomp_v1.slurm` against the v5 stratified parquet/cache. Same 4 lr_scale configs to compare directly to the v5 baseline. Different sweep name (`decomp_v1`) to avoid checkpoint collisions.
6. **Diagnostics & figures** — `fig_decomposition.py` (decomposed spectrum for example pixels), `fig_decomp_architecture.py` (new schematic), updated `fig_v5_embedding.png` re-run pointed at a decomp checkpoint.
7. **Wiki update** — new methodology log section on the decomposition approach, with figures embedded.

## Acceptance criteria

1. `DecompSpVit` forward produces the 5 outputs with the documented shapes. ✓ via test.
2. Reconstruction `‖T·s + b + ε − x‖` is small on validation pixels at the end of training (specific threshold determined empirically — expect ~10% of the raw spectrum norm).
3. Mean `T_hat` over val pixels is in physically plausible range (~0.7-0.95 — CRISM atmospheric transmission is typically in this band).
4. The decomposition runs successfully on HPC for at least one of the 4 lr_scale configurations.
5. A test-split eval of the new decomp checkpoint produces per-class AP; if HCP↔LCP cosine similarity in the center-pixel embedding drops below 0.84, that's the headline win we're after.

## What's NOT in scope (deferred)

- **Column-noise model** (the `n_column` term in the canonical equation). If `ε_hat` shows clear column structure after v1, split it out in a v2 with a per-column-of-patch (or per-band-and-column) head.
- **Per-pixel atmospheric terms**. We assume patch-level `T` and `b`. If the data shows per-pixel `T` variation matters (e.g., topographic shadowing varies by pixel), we can move `T_hat` to per-token in a v2.
- **Bayesian / variational version** with explicit priors on `T`, `b`, `ε`. A clean follow-on but adds tuning complexity. v1 is point-estimate (MAP) only.
- **Adversarial decorrelation** between signal embedding and residual embedding (option C from the brainstorm). Possible v2 add-on if the physics regularizers alone leave too much classifier-relevant information in the residual.

## Open hyperparameters (to tune empirically)

- The six λ values. Defaults above are reasonable first guesses but will need a small sweep — keeping them coupled (e.g., L_T_reg and L_b_reg always equal) reduces the search dimension.
- Bounds on `T_hat`: starting with [0.3, 1.0] but could narrow if data demands.
- Whether `T_hat` should be exposed as one global value per band per patch (current proposal) or per-row-of-patch (potential v1.5 if shadowing/illumination varies meaningfully across 7 rows of CRISM at 180 m/pixel).
