# LCP Edge-Pixel Over-Prediction — Investigation and Solutions

**Date:** 2026-05-21
**Author:** investigation prompted by observation that LCP appears uniquely sensitive to tile-edge pixels in the MC13 v3-denoising classifier output

---

## Priority-ordered fix list (read this first)

Empirically established on tile t1322; ordered by how completely each addresses the **core** LCP-edge-bleed mechanism.

| # | Fix | Addresses | Retrain? | Verified Δ on LCP@d≤3 |
|---|---|---|---|---|
| 1 | **Train-time edge-pad augmentation** + bland-relabel data | Both OOD-fallback *and* model's edge-region prior | YES | _(not yet — predict ≤ 1.1×)_ |
| 2 | **Run the bland-relabel sweep and re-measure** before doing anything else | The "Other = within-scene leftover" mechanism, which is plausibly the LARGEST single contributor | NO (sweep already in flight) | _(unknown — re-probe in AM)_ |
| 3 | **Erode `valid_mask` by `PAD=3` at inference** (solution G) | Eliminates the visible symptom regardless of cause; honest about boundary uncertainty | NO | 1.77 → **1.28×** |
| 4 | **Reflect-pad + masked-zero normalization combined** (solution F) | The OOD aspect of the patch-padding mechanism; helps olivine/hcp/plagioclase even more than LCP | NO | 1.77 → **1.53×** |
| 5 | **Masked-zero normalization alone** (solution B) | Half of the padding-mechanism issue | NO | 1.77 → 1.61× |
| 6 | **Reflect-pad alone** (solution C) | Half of the padding-mechanism issue (different half) | NO | 1.77 → 1.67× |
| 7 | **OOD output head** (calibration-class fix) | Residual interior LCP over-prediction (broader than edges) | YES | n/a |
| 8 | **Class-balanced or focal loss reweight** | Reduces overall LCP overprediction baseline | YES | n/a |
| 9 | **Temperature scaling on a held-out edge set** | Symptom only, not cause | NO (needs labeled edges) | n/a |

**The core lesson from the empirical work:** *no inference-time fix gets LCP edge enrichment below 1.28×.* The model has a baked-in LCP-leaning prior for edge-context inputs, and the only way to remove it is to expose the model to edge patches during training (#1) and to give "Other" a real identity so OOD inputs have a non-mineral destination (#2).

**Recommended sequencing:** (#2) first because it's already running and may largely solve the problem on its own; (#3) immediately as a safe interim measure; (#4) alongside (#3) for the surface area we keep; (#1) as the durable fix.

---

## Side-by-side comparison of inference-time fixes on tile t1322 (d ≤ 3 px ring)

| Fix | Padding | Normalization | LCP enrich | olivine | hcp | plag |
|---|---|---|---:|---:|---:|---:|
| **A — current** | zero (const) | include zeros in μ,σ | **1.77×** | 0.17× | 0.04× | 0.37× |
| B | zero (const) | mask zeros in μ,σ | 1.61× | 1.01× | 0.59× | 1.75× |
| C | reflect | include zeros in μ,σ | 1.67× | 0.42× | 0.28× | 0.90× |
| **F** — best inference-only | reflect | mask zeros in μ,σ | **1.53×** | 1.01× | 0.63× | 1.76× |
| **G** — erode valid_mask by PAD | const | include zeros (no change) | **1.28×** *(visible)* | 1.00× | 0.68× | 1.62× |
| **ideal (target)** | — | — | ~1.0× | ~1.0× | ~1.0× | ~1.0× |

**Key takeaway:** **no inference-time fix gets LCP edge enrichment below 1.28×.** Even with reflect-padding and masked normalization combined (F), LCP retains a 1.53× over-enrichment at edges. With boundary refusal (G — erode the valid_mask by PAD and discard those predictions), the visible enrichment drops to 1.28× — but that's measured on pixels 3-6 from the original boundary, whose patches still touched the (now-nodata) outer ring. The residual cannot be eliminated by inference-time tweaks alone; the model itself has a learned LCP-leaning prior in the edge-context region of latent space, and the only way to remove it is to **expose the model to edge patches during training**.

---

## TL;DR

The LCP edge-bleed is real and measurable. It is the result of **two compounding failures** that both stem from the same root: the model has never been trained to handle partial-context inputs, and at inference time it converts that ambiguity into a confident LCP prediction.

- **Root cause:** zero-padded edge patches are far out-of-distribution; the model's latent space happens to map "ambiguous/normalized-noise" inputs preferentially to LCP, so OOD inputs at tile boundaries get classified as high-confidence LCP.
- **Empirical evidence:**
  - LCP positives are **1.79× over-enriched** within 3 pixels of any tile boundary; olivine 0.30×, HCP 0.11×, plagioclase 0.33× — all three *suppressed* at edges. LCP is the only over-enriched class.
  - 10.09 % of all 54.2 M LCP positives across MC13 (~5.47 M predictions) sit in the 3-pixel boundary ring, which is only 5.63 % of valid area.
  - The training data contains only 0.52 % edge rows (within 3 px of any boundary). The model has effectively zero exposure to edge-padded patches at training time.
  - For a *random standard-normal patch* — which is what naive normalization produces from a half-zero edge patch — the model assigns probability ≥ 0.85 to LCP **37.5 %** of the time, vs olivine 16.0 %, HCP 5.3 %, plagioclase 15.0 %. **LCP is the OOD fallback class.**
  - LCP "edge" polygons (centroids within 6 px of a tile boundary) have **20 % lower mean reflectance magnitude** than LCP interior polygons. Consistent with zero-mixed normalization shrinking the magnitude.

There are several ways to attack this — ordered below by how well each addresses the *core* failure rather than the symptom.

---

## The mechanism, in one paragraph

At inference, `classify_tile_supervised.py` zero-pads the tile by `PAD = patch_size // 2 = 3` pixels before extracting 7×7 patches (line 73). Patches centred on the first or last 3 pixels of any row or column therefore have one or more entire rows/columns that are literal zero. The next step (`normalize_patches`, lines 86-92) computes per-patch `μ`, `σ` over **all** 2891 values including those zeros. The included zeros pull `μ` down and inflate `σ`, so the normalized "valid" portion of the patch looks like roughly-uniform noise centred near `μ = 0` with `σ = 1` — i.e. a sample from the same distribution as `torch.randn(...)`. Empirically we showed that the v3-denoising classifier maps that distribution preferentially to LCP. The training set contained essentially no patches with this signature (0.52 % edge rows, all of which likely had interior context anyway because polygons were drawn well inside the data area), so the model has no calibrated response for these inputs and falls back to the LCP-leaning direction of its latent space.

The CRISM atmospheric/photometric correction residuals at scene boundaries probably amplify this, but the dominant factor — by the size of the empirical enrichment — is patch-padding interacting with naive normalization.

---

## Detailed solution write-ups

The 8 sections below are *not* in priority order — they're a complete catalogue with rationale, cost, and where empirical evidence exists. **For the actual priority ranking, see the table at the top of this document.** Cross-references in the table point into these sections.

---

### 1. Retrain with explicit edge-padding augmentation (root fix)

**What:** During training, randomly mask a contiguous rectangular sub-region of each patch (matching the geometry of real edge padding — e.g. zero out the first K rows or columns, K ∈ {0, 1, 2, 3} with non-trivial probability), and keep the label of the centre pixel. This teaches the model that partial-context patches are valid input and that the label should still come from the visible (non-zero) part of the patch.

**Why it's the strongest fix:** It addresses the underlying mismatch between training and inference distributions. After this, edge patches are no longer OOD, and the model's OOD fallback to LCP is irrelevant at edges. It's also robust to future scenes with mid-tile nodata blocks (e.g. atmospheric correction failures), not just literal tile boundaries.

**Cost:** Requires retraining the classifier (~5 hrs on UArizona HPC). Best done in tandem with the in-flight `ft_bland_*` sweep (Phase 6). Easiest: add the augmentation to `training/train_torch.py`'s patch loader, ship a `ft_bland_v3_lrscale001_edgeaug` sweep variant.

**Evaluation:** Re-run the same edge-enrichment probe (`/tmp/probe_lcp_edge.py`) and look for LCP enrichment at d≤3 dropping from 1.79× to ≤ 1.1×. Confirm that olivine/HCP coverage at edges *increases* (because the OOD fallback no longer suppresses them either).

---

### 2. Erode the inference-time `valid_mask` by `PAD` pixels (boundary refusal)

**What:** Before passing the tile to the classifier, run `valid_mask = scipy.ndimage.binary_erosion(valid_mask, iterations=PAD)` so that pixels closer than `PAD = 3` to any tile boundary or nodata region are marked invalid up-front. The classifier still runs everywhere (so spatial context is preserved), but the output for those edge pixels is suppressed to nodata in the downstream `probs_hw` and `valid_mask` written to disk.

**Why it's strong:** By construction, the model never gets *asked* about ambiguous-context pixels, so there is no opportunity for the LCP-fallback to fire there. Honest about data quality — the 3-px ring is genuinely a region where we can't compute a spatial-spectral signal — and the user sees the missing area as a clear nodata band instead of a misleading LCP halo.

**Cost:** One-line change to `scripts/classify_tile_supervised.py:load_tile` (and the equivalent in `classify_targeted_observation.py:load_targeted` and `classify_tile_embeddings.py:load_tile`). No retraining.

**Edge case:** Within-tile nodata regions (caused by atmospheric correction artifacts or sensor data drops) also have edge-padded patches; binary_erosion will handle them too as long as `valid_mask` correctly reflects the nodata.

**Loss:** 5.6 % of valid area per tile (the 3-pixel ring) becomes nodata. For MC13 that's ~5.8 M pixels — but those are exactly the pixels where the model can't trust itself anyway.

**Evaluation:** Same edge-enrichment probe; LCP coverage in the d≤3 ring drops to 0 (by construction). For d in (3, 6], enrichment should drop toward 1.0× because the patches whose centres are 4-6 px from the edge no longer get edge-padded.

---

### 3. Switch `mode='constant'` to `mode='reflect'` in the inference-time padding

**What:** Replace `np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')` with `mode='reflect'` (or `mode='symmetric'`). At each edge, the padding becomes a mirror image of the adjacent interior pixels rather than zeros. Patches at the boundary then look like extensions of the interior — same spectral statistics, similar texture — instead of half-zero artifacts.

**Empirical result (tile t1322):**

| mineral | depth | A: zero-pad+incl | C: reflect-pad+incl |
|---|---|---|---|
| olivine | 3 | 0.17× | 0.42× |
| lcp | 3 | 1.77× | **1.67×** |
| hcp | 3 | 0.04× | 0.28× |
| plagioclase | 3 | 0.37× | 0.90× |

Reflect-pad reduces LCP edge enrichment from 1.77 to 1.67 — a smaller improvement than masked-zero normalization (1.61). And critically, **LCP remains 1.67× over-enriched** even with realistic reflected data in the patch. This says some of the LCP edge bleed is *not* about zero-padding at all — see "Residual LCP bias" below.

**Cost:** One-line change. No retraining.

---

### 4. Mask zeros out of per-patch normalization (port from targeted classifier)

**What:** In `classify_tile_supervised.py:normalize_patches`, replace the current implementation with the masked version already used in `classify_targeted_observation.py:192-202`. That version computes μ and σ over only the non-zero positions of each patch, then leaves zeros at zero and normalizes the rest.

**Why it's not enough by itself:** The model *still sees* literal zeros in the padded positions; only the magnitude of the surviving signal is now more realistic. So we expect a partial reduction in LCP edge bleed but not full elimination — the encoder's first attention layer still operates on a mix of valid tokens and "zero" tokens (which may project to a non-zero position in token space depending on the `band_embed` linear layer).

**Cost:** ~10 LOC. No retraining.

**Empirical result (tile t1322, 1.73 M valid pixels):**

| mineral | depth | A: zeros-in-norm | B: zeros-masked | effect |
|---|---|---|---|---|
| olivine | 3 | 0.17× | **1.01×** | edge suppression eliminated |
| lcp | 3 | 1.77× | 1.61× | small reduction |
| hcp | 3 | 0.04× | 0.59× | edge suppression eliminated |
| plagioclase | 3 | 0.37× | **1.75×** | from suppressed → over-represented |

**Read:** the masked-zero normalization is a **strong fix for olivine, HCP and plagioclase** (whose edge predictions were being *suppressed* by the include-zeros normalization — the model was misclassifying their would-be-edge-positives as LCP); it's only a partial fix for LCP itself (1.77 → 1.61). So solution 4 alone is necessary but not sufficient to eliminate the LCP-specific edge bleed — solution 1 (training-time augmentation) or solution 2 (boundary refusal) is needed on top of it.

Also note plagioclase becomes *over*-enriched at edges with masked normalization (1.75×). That's because plagioclase was a low-base-rate class whose few true predictions got swamped by the zero-normalization bias previously. Once the bias is removed, plagioclase fires at edges roughly as much as LCP does — both classes share an OOD-fallback issue at edges, just with very different absolute counts.

LCP polygon counts on t1322 also drop overall: 837,809 → 790,638 (-5.6%) — i.e. masked normalization reduces some interior LCP over-prediction too, not just edge bleed.

---

### 5. Zero-fraction-based patch rejection at inference

**What:** For every patch, count the fraction of pixels that are exactly zero. If above a threshold (e.g. 25 %), mark the output for that centre pixel as nodata.

**Why it's a useful safety net:** Handles within-tile nodata blocks (not just tile boundaries) and any future case where a partial patch sneaks past `valid_mask` checks.

**Cost:** ~10 LOC, no retraining.

**Caveat:** It's basically a generalisation of solution 2 (which uses geometric distance from the boundary). Solution 2 is preferable if the only source of zeros is boundary padding, because it's simpler and doesn't require setting a threshold. If within-tile nodata is also a problem, both are complementary.

---

### 6. Output calibration / temperature scaling targeted at edges

**What:** Fit a per-class temperature parameter on a held-out set of edge polygons such that the calibrated LCP probability matches the (much lower) actual base rate of LCP at edges. Apply the temperature only to predictions near tile boundaries.

**Why it's weak:** It treats the symptom (over-confident LCP at edges) rather than the cause (OOD-default-to-LCP). The model's underlying latent geometry still maps ambiguous inputs to LCP — calibration just turns those high probabilities into lower ones. Pixels that get rejected at the new threshold may include legitimate LCP detections near the edge too.

**Cost:** Moderate (need a held-out edge label set, which we don't currently have).

---

### 7. Replace sigmoid head with softmax + reserve "other" for OOD

**What:** Change the multi-label sigmoid output to a 5-class softmax, so the probabilities are mutually exclusive. Train so that "other" absorbs ambiguous-input mass instead of LCP.

**Why it's interesting but disruptive:** It architecturally pushes the OOD-fallback class to be "other" (which is what we *want* it to be) instead of LCP. But: it makes the model strictly single-label, losing the genuine multi-mineral pixels (e.g. olivine + LCP intergrowths). And it requires retraining the entire classifier.

**Cost:** Major architectural change + full retrain.

**Better alternative:** Stick with multi-label sigmoid, but train an explicit "is OOD" output head alongside the 5 mineral heads. At inference, suppress mineral predictions when the OOD head fires. This is a bigger lift but it directly attacks the OOD-fallback problem the model currently has.

---

### 8. The bland-relabel sweep may partially address this on its own

**Observation:** The current v3 classifier's "other" class was trained on within-scene leftover polygons in mineral-bearing scenes — i.e. spectra adjacent to mineral pixels. That means the model probably learned "other = lcp-context-but-not-lcp" rather than "other = bland surface". When confronted with an OOD edge patch, the model's option "this might be LCP context" beats "this might be other (=lcp-context-too)", and LCP wins.

The new bland-relabel parquet (Phase 6 of the methodology log) replaces those 677 K within-scene "other" pixels with 877 K bland-tile spectra. The new "other" class will have a genuine "bland / featureless" identity. The OOD direction in latent space *should* shift away from LCP and toward this new "other" — but we won't know for sure until the sweep completes.

**Action:** Re-run the same edge-enrichment probe (`/tmp/probe_lcp_edge.py`) against the ft_bland champion once it's pulled down tomorrow. If LCP edge enrichment drops substantially (say from 1.79× toward 1.0×), some of the work in solutions 1-3 may not be needed. If it stays high, solutions 1-3 are critical regardless.

---

## Residual LCP bias (after all inference fixes)

The fact that LCP enrichment stays ≥ 1.28× even with the most aggressive inference-time fixes is significant. Three plausible mechanisms for the residual:

1. **The model's "Other" class learned the wrong thing.** In the v3 parquet, "Other (High)" polygons were drawn *within mineral-bearing scenes* — often as a "this nearby pixel isn't the mineral but isn't a clear absence either" label. So "Other" became spectrally close to LCP in latent space; when an edge patch produces an ambiguous representation, the model picks "the most common-and-similar class," which is LCP. The **bland-relabel sweep (Phase 6)** directly attacks this by giving "Other" a real bland-surface identity. **Predict:** the ft_bland champion will show much lower LCP edge enrichment without any of the inference fixes.

2. **CRISM swath-edge artifacts.** Real edges of CRISM observations have lower SNR, residual atmospheric calibration errors, and detector-edge column biases. Even with reflect-pad, those few real edge pixels are inside the patch, and they carry spectral artifacts the model has interpreted as LCP-like. The only way to fix this is to either (a) teach the model that these artifacts are not signal — which requires training on edge patches — or (b) refuse to predict on them.

3. **Genuine geological signal at edges.** A CRISM target scene is often centred on a feature; the swath edges fall on adjacent terrain that *may* genuinely contain different mineralogy (often more pyroxene-rich Martian regolith versus the target's specific composition). Small effect, but it exists.

Of these, (1) is the dominant one and should mostly resolve with the bland-relabel sweep. (2) requires either retraining or refusal. (3) is unavoidable but small.

---

## Recommended action plan

The decision depends on how *cleanly* you want this fixed before scaling the figures up to MC13 with the new bland-relabel model:

**Step zero (must do):** Re-run the edge-enrichment probe (`/tmp/probe_lcp_edge.py`, pointed at fresh probs from the ft_bland champion) once the sweep produces a winning checkpoint. If the new model already shows LCP edge enrichment < 1.3× without any inference changes, that tells us mechanism (1) was indeed dominant and we may not need to do much more.

**Pragmatic / fast path (today/tomorrow, regardless of sweep outcome):**
1. Apply **F = reflect-pad + masked-zero normalization** in `classify_tile_supervised.py` (and the targeted variant, which already has the masked normalization). Brings LCP edge enrichment from 1.77 to 1.53 immediately. ~15 LOC change. No retraining.
2. Apply **solution 2 (erode `valid_mask` by PAD)** in the same scripts. Reduces *visible* LCP edge enrichment further to 1.28× by discarding the most-problematic ring. ~5 LOC change. Accept the 5.5% area loss as honest reporting.
3. Re-classify the Nili 4-tile set with the ft_bland champion + steps 1 and 2. Inspect.

**Principled / longer path (this week):**
4. Implement **solution 1 (edge-padding augmentation in the training pipeline)**. The right place is in the patch-cache build (`scripts/cache_mrral_patches.py`) — or, simpler, in the patch loader in `training/train_torch.py` (apply random rectangular cutouts to a random fraction of training patches). Suggested augmentation distribution:
   - 70 % of patches: no cutout
   - 20 % of patches: zero out 1-3 rows OR columns from one edge of the patch
   - 10 % of patches: zero out 1-3 rows AND 1-3 columns from one corner
   Match the geometry of real edge-padding cases.
5. Re-run the ft_bland sweep with edge augmentation enabled (`ft_bland_*_edgeaug`).
6. Compare on the same Nili tiles. Target: LCP edge enrichment at d≤3 drops to 1.0-1.1×, and all other classes' edge enrichment is similarly normalized.

**If even with all of the above LCP is still over-predicted in the interior** (i.e., the issue is broader than edges):
7. Add an explicit "OOD" output head trained to fire on synthetic/non-CRISM inputs. At inference, suppress all mineral predictions when the OOD head fires. This addresses the calibration/class-imbalance issue at the source. Requires retraining but is the most principled fix for the model's general OOD overconfidence.

---

## Empirical artifacts produced during this investigation

- `/tmp/probe_lcp_edge.py` — per-mineral edge enrichment across all 54 MC13 tiles
- `/tmp/probe_training_edge_coverage.py` — what fraction of training rows are at edges (0.52 %)
- `/tmp/probe_model_bias.py` — output bias, all-zero input behavior, random-noise OOD behavior
- `/tmp/probe_lcp_spectra_edge_vs_interior.py` — edge vs interior LCP polygon mean spectra
- `/tmp/probe_normalization_swap.py` — A/B between current and masked-zero normalization (tile t1322)
- `/tmp/probe_reflect_pad.py` — C/E: reflect-pad and symmetric-pad versions
- `/tmp/probe_combined_fixes.py` — F (reflect + masked) and G (boundary refusal) combinations
- `/tmp/render_edge_fix_comparison.py` — visual side-by-side of A vs F vs G
- `reports/per_mineral_mc13/edge_fix_compare_t1322.png` — 3-panel figure visualising A vs F vs G LCP coverage
