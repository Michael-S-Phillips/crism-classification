# Dual continuum representation (hull-CR ⊕ linear-CR) — design

**Date:** 2026-08-10
**Status:** approved, pending implementation plan
**Touches:** `data/continuum_removal.py`, `data/dataset.py`, the patch-cache
builders, `models/spatial_spectral_*`, the MAE pretrain, `scripts/train.py`,
`scripts/classify_tile_supervised.py`

---

## Problem

Upper-hull continuum removal destroys the diagnostic feature of the two classes
that have never worked.

Alteration's diagnostic is a broad **convex arch over 1–2 µm**. Upper-hull CR
divides by the convex hull — and a broad convex arch *is* approximately the hull,
so CR divides the feature out. Measured on the hand-core training data:

| class | RAW arch | hull-CR arch | retained |
|---|---:|---:|---:|
| **alteration (v3 review)** | **0.1678** | 0.0683 | **41%** |
| plagioclase | 0.0397 | 0.0226 | 57% |
| olivine | 0.0617 | 0.0416 | 67% |
| lcp | 0.0454 | 0.0373 | 82% |
| bland | 0.0266 | 0.0225 | 84% |

(arch = convexity at 1625 nm against the 984–2205 nm chord; positive = convex.
The RAW column is normalised by the reflectance level so it is comparable across
classes of different albedo, while the ratio-spectrum columns are absolute — so
"retained" is indicative of the magnitude lost, not an exact ratio. The
per-pixel AUC table below is the rigorous version and does not depend on this
normalisation.)

Alteration has **the largest raw arch of any class**, 6× bland's, and hull-CR
removes 59% of it. The consequence is that alteration's hull-CR endmember becomes
the **flattest class in the vocabulary** (mean CR depth 0.0381, flatter than
bland's 0.0412) and therefore acts as an attractor for featureless ground.

That is observable in the deployed model. `scripts/audit_confident_predictions.py`
on `ft_7cls_handcore_level`, Nili t1250, own-class spectral agreement from
≥0.50 → ≥0.99:

| model class | own agreement | drifts toward |
|---|---|---|
| olivine | 0.12 → 0.17 | alteration 0.30 → **0.54** |
| lcp | 0.32 → 0.31 | hcp 0.32 → 0.45 |
| plagioclase | **0.45 → 0.02** | alteration 0.24 → **0.97** |

Controls confirm the diagnostic works: model-bland improves 0.66 → 0.95 and
model-alteration 0.69 → 0.97 over the same bands. So when confident predictions
have spectral support, it shows.

**Why the network cannot simply learn CR itself.** Three reasons, and the third
is the operative one:

1. The hull is a combinatorial operation — which bands are hull vertices — so it
   is discontinuous in the input and a poor fit for gradient descent.
2. An MAE reconstructing *raw* spectra spends its capacity on level and slope,
   because that is where the variance is; band depths of a few percent barely
   register in the loss. **The pretraining objective decides what gets
   represented.**
3. The model has no incentive to be albedo-invariant, because albedo is
   predictive in-distribution. Median brightness spans **1.76×** across classes
   and orders them almost monotonically. On Mars albedo tracks dust, illumination
   and atmospheric path — all correlated with *location* — so a raw-fed model
   learns a shortcut that fails out-of-distribution.

So hull-CR does not add information; it **removes a shortcut and imposes an
invariance the model would never adopt voluntarily**. That is why it changes
results despite being a deterministic function of its input — and why it is a
*trade*, not an upgrade.

## The fix

**118 channels: hull-CR (59) ⊕ linear-CR (59).**

`linear_continuum_removed(spec)` divides each spectrum by a **per-spectrum
least-squares line** fitted over the 55 good bands (the 1021–1056 nm
detector-overlap window excluded, as elsewhere). This removes level and slope —
the albedo nuisance — but **cannot** remove curvature, because a line has none.

Measured, per pixel, using the 1–2 µm arch alone to separate alteration from
every other class:

| transform | AUC vs all others | worst per-class |
|---|---:|---|
| raw arch | 0.991 | 0.978 (junk) |
| **linear-CR arch** | **0.990** | **0.974 (junk)** |
| hull-CR arch | 0.856 | **0.719 (hcp)** |
| brightness alone (the shortcut) | 0.764 | 0.531 (plagioclase) |

Linear-CR matches raw (0.990 vs 0.991) while hull-CR loses a third of the
discriminating power, and sags worst exactly against hcp and lcp — the confusions
the model actually makes.

The mechanism is that linear-CR makes curvature a **signed** feature: alteration
comes out **+0.174** (convex, above the chord) while bland is **−0.118**
(concave, absorption-dominated). Opposite sides of zero, not merely far apart.
Most classes go negative; alteration and junk go positive.

**Least-squares, not endpoint-anchored.** Both were tested: identical
discriminating power (AUC 0.990 each), but lsq removes albedo slightly better
(residual spread 1.00× vs 1.05×) and cannot be tilted by a single artifact band —
which matters because band 0 (410 nm) carries the known blue-edge spike up to
~1180 I/F.

**Why keep hull-CR rather than replace it.** Hull-CR's stronger invariance is the
most plausible cause of the one unambiguous win this project has: Nili LCP
surviving out-of-distribution in `ft_7cls_cr_lrscale0001`, where every raw-mrral
model collapsed it to zero. Nothing proves that causally, so this design keeps it
instead of trading it away on an assumption.

## Scale handling

This is the part that would otherwise be discovered too late. The two channels
are not on the same scale:

| | range | std |
|---|---|---:|
| hull-CR | [0, 1] bounded by construction | 0.0705 |
| linear-CR | −8.65 … +10.20 (p1–p99: 0.24 … 1.25) | **0.1726** |

Linear-CR carries **2.45× the variance**. Under pooled reconstruction MSE the
pretrain would spend most of its capacity on the linear channel — the same
failure mode as a raw-space MAE, merely relocated. Therefore:

1. **Clip linear-CR to [0, 2]** before caching. p99.99 is 1.415, so this keeps
   all real signal with headroom and removes the tails that would dominate
   gradients.
2. **Standardize per channel block** (divide each 59-band block by its global
   std) before the encoder.
3. **Log the MAE denoising loss per channel block.** For monitoring only — see
   the correction below.

**CORRECTION 2026-08-10.** An earlier version of this section claimed that
computing the loss per channel and averaging was necessary because "pooling
silently reweights the objective by the variance ratio". **That is wrong.** For
equal-sized blocks, `mean([mean(A), mean(B)])` is algebraically identical to the
pooled `mean(A ∪ B)`:

    mean(A) = sum(A)/n,  mean(B) = sum(B)/n   (equal n)
    (mean(A) + mean(B))/2 = (sum(A) + sum(B))/(2n) = pooled mean

Verified numerically: the two differ by 1.9e-9, pure float rounding. Averaging
equal-sized block means *is* pooling.

**Step 2 above is the actual mechanism.** Dividing each block by its own std
makes the cached *targets* comparable — measured on real spectra, the
standardised blocks come out at std 0.9936 (hull) and 0.9655 (linear), a ratio of
1.029×. Once the targets are on the same scale, a pooled MSE already weights them
equally. Nothing further is required.

The per-block loss machinery is retained purely as a **diagnostic**: reporting
the two block losses separately is how you would notice a cache written
un-standardised, or CR_SCALES going stale relative to the transform. It does not
change the optimisation.

**CORRECTION 2026-08-12 — the sentence above ("a pooled MSE already weights them
equally") is wrong as stated, and the diagnostic is what caught it.** Equal
target *variance* does not produce equal *loss*. Measured on the completed
pretrain (`spatial_mae_dualcr_denoising_256d_6l`, job 23548835, 200 epochs):

| epoch | hull block | linear block | hull share of total |
|---:|---:|---:|---:|
| 1 | 136.09 | 23.77 | 85% |
| 20 | 0.1649 | 0.0454 | 78% |
| 100 | 0.0977 | 0.0172 | 85% |
| 200 | 0.0846 | 0.0138 | 86% |

The hull block carries **85–88% of the reconstruction loss throughout**, and the
ratio *widens* from 5.7× to ~7×. So the residual objective is dominated by the
59 channels that flatten alteration's arch — the opposite of the imbalance this
section was written to prevent, and invisible without the per-block split.

(The pooled-equals-average identity itself still holds exactly, as the same log
confirms: (0.0846 + 0.0138)/2 = 0.0492, the reported total to four decimals. The
2026-08-10 correction above stands.)

**This is probably not a pathology, and the distinction matters.** The linear
block reaches a *lower* absolute error, so it is fit well rather than neglected —
a well-fit block contributes little gradient precisely because little error
remains. Independent evidence that the model learns real linear-block structure:
against the trivial baseline "predict each masked position as the mean of the
visible positions in the same patch" (`scripts/plot_mae_reconstructions.py`), the
MAE wins by **1.33× on the linear block against 1.14× on hull**, and by **1.60×
on alteration** — its largest margin, in the block and class the design predicts.

What it does mean: the pretrain's spare capacity goes to hull-CR, so if Task 8
finds the dual representation underperforming, an obvious next lever is
reweighting the two blocks in the loss — which the existing `n_channel_blocks`
machinery already makes a one-line change, since the per-block losses are already
computed. Do not do this pre-emptively; it would add a second variable.

## Held constant — exactly one variable

| | value | why |
|---|---|---|
| vocab | **7-class** | the pyx merge is evidence-backed but gets its own spec |
| loss | `--asl_loss`, unchanged | calibration gets its own spec |
| encoder | 256d, 6 layers | matches `spatial_mae_cr_denoising_256d_6l` |
| data | `mrral_pixels_7cls_handcore.parquet` | unchanged |
| lr / schedule | unchanged | |

Comparison target: **`ft_7cls_handcore_level`** — same data, same vocab, same
loss, differing only in representation.

## Components

| component | change |
|---|---|
| `data/continuum_removal.py` | add `linear_continuum_removed(spec) -> (..., 59)`, lsq over good bands, clip [0, 2] |
| global pretrain cache | new 118-channel builder (or a `--dual` mode on `build_global_patch_cache.py`) |
| labeled cache | 118-channel mode on `build_cr_labeled_cache.py`, brightness sidecar retained |
| `data/dataset.py` | a `dual_cr` mode serving (7, 7, 118); the existing `cache_is_cr` fail-fast guard must cover it |
| MAE pretrain | `n_bands=118`, per-channel loss |
| model | `n_bands=118` on the encoder and classifier |
| `scripts/train.py` | flag to select the representation |
| `scripts/classify_tile_supervised.py` | build the 118-channel input at inference |

## Validation, defined before the run

The design makes a **falsifiable prediction**: if hull-CR's flattening of the
arch is what makes alteration an attractor, then restoring curvature should stop
other classes drifting toward alteration.

1. **`audit_confident_predictions.py` on Nili t1250** — olivine's drift to
   alteration should fall from 0.54, plagioclase's from 0.97, and plagioclase's
   own-agreement inversion (0.45 → 0.02) should at minimum flatten. This is the
   mechanism check, and it is the one that can falsify the design.
2. **Floor test** vs `ft_7cls_handcore_level` on the same 8 tiles.
3. **Nili LCP must survive.** If it collapses, hull-CR's invariance was doing
   more than the linear channel can replace — which would be a real finding, not
   a failure to hide.
4. `val_mAP_core` is reported but **not** the arbiter: it is not comparable
   across representation changes, and this project's record is that val
   repeatedly overruled reality.

## Cost

| step | cost |
|---|---|
| global 118-ch pretrain cache | ~1h20m (measured: 1h20m for 59-ch, 50 shards, ~5M patches) |
| MAE pretrain | ~1h10m (measured for 256d, 200 epochs) |
| labeled 118-ch cache | 4–8h (I/O bound) |
| fine-tune | ~1 day |
| **disk** | **~90 GB extra** — global 58 → 116 GB, labeled 32 → 64 GB |

Disk lands on xdisk. Note `/groups` filling up previously killed two CR-cache
builds with `Errno 28`, so the target must be xdisk explicitly.

## Risks and open questions

- **RESOLVED 2026-08-12 — the MAE does favour one channel, and it is hull.**
  This risk was written as "watch the per-channel losses; they should be
  comparable." They are not: hull holds 85–88% of the loss for the whole run (see
  the 2026-08-12 correction above). Not judged a pathology — the linear block
  simply reaches a lower error — but the premise that standardisation alone
  equalises the two blocks is retired. Block reweighting is the lever if Task 8
  disappoints.
- **The noise model had to be rescaled for the standardised representation, and
  the pretrain log does not show it.** `CrismNoiseAugmentation`'s sigmas are
  absolute and were estimated against hull-CR of std 0.0705; against standardised
  dual data (std ≈ 1) they would have been 14× too weak, giving a nominally
  denoising MAE with negligible corruption. Fixed in `a822a08` (scale by
  `1/hull_std`, auto-detected from `n_bands == 118`, seam spike mirrored into the
  linear block at band 74). **But `pretrain_spatial_mae_denoising.py` logs the
  pre-scale CLI values** — job 23548835 printed `σ_gauss=0.0087` when the
  effective value was ~0.1233 — so the log cannot be used to confirm the scaling
  was active. Fix the log line to print effective sigmas.
- **118 channels doubles patch memory.** At batch 256 that is ~11.6 MB/batch,
  which is not a constraint, but the labeled cache doubling to 64 GB is.
- **Band 0 is corrupt in both channels.** The reader already masks I/F > 1.0 to
  nodata, and lsq is robust to it, but the band contributes nothing either way.
- **The raw global pretrain cache (May 18) predates the July 8 tile refresh** and
  may carry zero-fill corruption. The new 118-ch cache must be built from tiles
  directly, post-refresh — do not derive it from that cache. Run
  `scripts/audit_spectra_quality.py` on the product.
- **Alteration's two sources disagree.** Hand alteration's raw arch is 0.0341
  against review's 0.1678 — 5× apart, so they are not the same material. This
  design preserves whatever alteration signal exists but does not settle which
  definition is correct; that is a labelling question.
