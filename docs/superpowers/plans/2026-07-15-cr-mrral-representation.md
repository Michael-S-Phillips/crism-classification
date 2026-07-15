# CR-mrral Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the raw-reflectance model input with continuum-removed (CR) mrral spectra + a retained brightness scalar, pretrain a CR-native denoising-MAE encoder, and fine-tune the classifier on CR, proven via the floor-test gate on honest splits.

**Architecture:** One new pure-function module does upper-hull CR over the 59-band good-band window (1 µm overlap excluded) and returns CR spectrum + brightness scalar. A hook applies it at patch-cache-build time and at inference. The MAE/classifier architectures are unchanged; the classifier uses the existing `SpatialSpectralClassifierAux` with `aux_dim=1` (brightness). Pretrain and fine-tune are new slurm scripts (run on HPC).

**Tech Stack:** Python, numpy, torch, rasterio; existing `models/`, `data/`, `scripts/` packages; conda env `crism`; HPC slurm.

## Global Constraints

- 59-band mrral input, band cols `m0..m58` (`data/dataset.py:MRRAL_BAND_COLS`). Patch layout `(P,P,59)`, patch_size 7.
- NODATA = 65535 → 0; CR must never emit NaN/Inf.
- 1 µm detector-overlap exclusion window: **1000–1065 nm** (matches `scripts/label_quant/sam_endmembers.py` `BAD_BAND_RANGES_NM`).
- Honest unit-balanced splits; stop metric `val_mAP_core`. Single-variable vs champion `ft_7cls_v3b_lrscale001` (representation is the only change).
- Encoder-size probe: `{128d/6L, 256d/6L}`; select by frozen linear probe `val_mAP_core`, not MAE recon loss.
- HPC runs deferred to the runbook (Task 9); everything else verified locally.

---

### Task 1: Continuum-removal module + 59-band good-band mask

**Files:**
- Create: `data/continuum_removal.py`
- Create: `data/mrral_wavelengths_59.json` (committed sidecar: m0..m58 nm)
- Test: `tests/test_continuum_removal.py`

**Interfaces:**
- Produces:
  - `good_band_mask_59() -> np.ndarray` (bool, len 59; False inside 1000–1065 nm)
  - `continuum_removed(spec: np.ndarray) -> np.ndarray` — spec `(..., 59)` raw reflectance → CR `(..., 59)`, upper-hull over good bands, CR=1.0 on excluded/degenerate bands, ≤1.0001 everywhere, NaN/Inf-safe.
  - `brightness_scalar(spec: np.ndarray) -> np.ndarray` — mean over good bands (pre-CR), shape `(...,)`.
  - `cr_patch(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — patch `(P,P,59)` → (CR patch `(P,P,59)`, brightness map `(P,P)`).

**CR algorithm (upper convex hull, per spectrum, over good bands):** monotone-chain upper hull on `(wl[good], spec[good])`; interpolate hull across good wl; `cr = spec/hull` on good bands, `1.0` elsewhere; if hull ≤ 1e-6 (flat/zero pixel) return all-ones.

- [ ] Step 1: Write failing tests — hull CR of a synthetic 1-absorption spectrum recovers band depth; CR≤1.0001 everywhere; flat spectrum → all ones; NODATA(0) pixel → all ones (no NaN); `good_band_mask_59()` has False only in 1000–1065 nm; `brightness_scalar` = mean of good bands.
- [ ] Step 2: Run `python -m pytest tests/test_continuum_removal.py -v` → FAIL (module missing).
- [ ] Step 3: Build `mrral_wavelengths_59.json` from a representative `.hdr` (`wavelength = {...}`), then implement the module.
- [ ] Step 4: Run tests → PASS.
- [ ] Step 5: Commit.

### Task 2: CR hook in the global patch-cache builder

**Files:**
- Modify: `scripts/build_global_patch_cache.py` (`extract_patches_from_tile`, after the clip on line ~54)
- Test: `tests/test_global_cache_cr.py`

**Interfaces:**
- Consumes: `data.continuum_removal.cr_patch`.
- Produces: `--continuum_removed` CLI flag; when set, each written patch is CR and a parallel `brightness` array `(n, P, P)` is saved alongside (`*_brightness.npy`).

- [ ] Step 1: Failing test — with CR on, returned patch equals `continuum_removed(raw_patch)` for a tiny synthetic tile; brightness array matches `brightness_scalar`; shapes/NODATA unchanged.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement: apply `cr_patch` per patch when flag set; accumulate brightness; write sidecar.
- [ ] Step 4: Tests PASS.
- [ ] Step 5: Commit.

### Task 3: CR + brightness in the labeled patch dataset

**Files:**
- Modify: `data/dataset.py` (`CRISMPatchDataset`: add `continuum_removed: bool`, `return_brightness: bool`; apply CR in both the cache and on-the-fly `__getitem__` paths; return brightness as the aux)
- Test: `tests/test_dataset_cr.py`

**Interfaces:**
- Consumes: `data.continuum_removal.cr_patch`.
- Produces: when `continuum_removed=True, return_brightness=True`, `__getitem__` returns `(cr_patch_tensor, brightness_tensor(1,), label, weight)`; cache path CR-transforms cached raw patches on read (or consumes a CR cache built by Task 2 — controlled by `cache_is_cr: bool`).

- [ ] Step 1: Failing test — on a synthetic df+cache, `__getitem__` returns CR patch matching `continuum_removed`, brightness shape `(1,)`; label/weight unchanged; raw mode still returns raw.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Tests PASS.
- [ ] Step 5: Commit.

### Task 4: Trainer wiring for CR + brightness aux

**Files:**
- Modify: `scripts/train.py` (add `--continuum_removed`, `--brightness_aux`; when set, build `SpatialSpectralClassifierAux(aux_dim=1)` and pass brightness as aux)
- Modify: `training/train_torch.py` (batch unpacking to include aux when present)
- Test: `tests/test_train_torch.py` (extend: a CR+aux smoke step runs without shape error)

**Interfaces:**
- Consumes: Task 3 dataset (CR patch + brightness aux); `models.spatial_spectral_classifier_aux.SpatialSpectralClassifierAux`.
- Produces: a runnable `--model spatial_vit_aux --continuum_removed --brightness_aux --seven_class` path.

- [ ] Step 1: Failing/extended test — one training step with CR dataset + `aux_dim=1` model produces a loss and backward without error on synthetic data.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement wiring.
- [ ] Step 4: Tests PASS.
- [ ] Step 5: Commit.

### Task 5: CR denoising-MAE pretrain (input CR) + slurm

**Files:**
- Modify: `scripts/pretrain_spatial_mae_denoising.py` (add `--continuum_removed`: CR the input patches before the denoising corruption; keep reconstruction target in CR space)
- Create: `scripts/hpc_pretrain_cr_denoising.slurm` (2-arm array: `--embed_dim 128` / `256`, both `--n_layers 6`, `--continuum_removed`, global CR cache)
- Test: `tests/test_pretrain_cr_smoke.py`

**Interfaces:**
- Consumes: CR global cache (Task 2) OR on-read CR of the raw global cache.
- Produces: `checkpoints/spatial_mae_cr_denoising_{128,256}d_6l_best.pt` (encoder_state).

- [ ] Step 1: CPU smoke test — one pretrain step with `--continuum_removed` on a synthetic cache runs and the reconstruction is computed in CR space.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement `--continuum_removed`; write the 2-arm slurm from `hpc_pretrain_denoising.slurm`.
- [ ] Step 4: Smoke PASSES; `bash -n` the slurm.
- [ ] Step 5: Commit.

### Task 6: Linear-probe encoder selector

**Files:**
- Create: `scripts/linear_probe_encoder.py`
- Test: `tests/test_linear_probe.py`

**Interfaces:**
- Consumes: a pretrained CR encoder ckpt + the labeled CR val cache.
- Produces: prints/returns frozen-encoder `val_mAP_core` for one encoder (compare 128 vs 256 by running twice).

- [ ] Step 1: Failing test — on synthetic data, freezing an encoder + fitting a linear head yields a finite `val_mAP_core` in `[0,1]`.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement (freeze encoder, extract center-token features, logistic/linear head, compute per-class AP excluding junk → mean).
- [ ] Step 4: Tests PASS.
- [ ] Step 5: Commit.

### Task 7: CR fine-tune slurm

**Files:**
- Create: `scripts/hpc_finetune_cr.slurm` (from `hpc_finetune_7cls_v3bland.slurm`; `--continuum_removed --brightness_aux --model spatial_vit_aux`, CR encoder from Task 5's winner, honest-splits parquet/cache, `val_mAP_core`)

**Interfaces:**
- Consumes: selected CR encoder; `data/mrral_pixels_7cls.parquet` + CR patch cache.
- Produces: `checkpoints/ft_7cls_cr_lrscale{0001,001,01}_best.pt`.

- [ ] Step 1: `bash -n scripts/hpc_finetune_cr.slurm` (syntax) + grep-confirm the CR/aux flags and encoder path are present.
- [ ] Step 2: Commit.

### Task 8: Inference CR path

**Files:**
- Modify: `scripts/classify_tile_supervised.py` (add `--continuum_removed --brightness_aux`; CR each pixel patch identically to training; pass brightness aux to the aux model)
- Test: `tests/test_classify_cr_parity.py`

**Interfaces:**
- Consumes: `data.continuum_removal`, a CR+aux checkpoint.
- Produces: `--continuum_removed` inference producing a `(H,W,7)` prob array; CR path matches the cache-build CR on the same pixels (parity).

- [ ] Step 1: Failing parity test — inference CR of a small synthetic tile equals `cr_patch` applied to the same extracted patches.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Tests PASS.
- [ ] Step 5: Commit.

### Task 9: HPC runbook

**Files:**
- Create: `docs/hpc_runbook_cr_mrral.md`

Ordered HPC commands: (1) git pull; (2) rebuild global CR patch cache; (3) `sbatch` 2-arm CR pretrain; (4) linear-probe both encoders → pick size; (5) build/confirm labeled CR cache; (6) `sbatch` CR fine-tune (3-arm lrscale); (7) floor test + MC11 visual on the winner; (8) the go/no-go pass criteria (LCP survives OOD; no mafic→olivine / alteration→bland collapse) and what to do on fail (stop; reconsider data foundation).

- [ ] Step 1: Write the runbook with exact commands + expected artifacts.
- [ ] Step 2: Commit.

---

## Self-review

- **Spec coverage:** CR representation (T1), cache CR (T2), dataset CR+brightness (T3), trainer wiring (T4), CR pretrain + size probe (T5), linear-probe selection (T6), fine-tune (T7), inference (T8), gate/runbook (T9). Brightness scalar via `aux_dim=1` (T3/T4/T8). All spec components covered.
- **Non-goals honored:** no Spec-2 eval framework, no label cleanup, no MTRDR/ratioing, no contrastive path.
- **Type consistency:** `cr_patch` returns (CR patch, brightness map) used identically in T2/T3/T8; `aux_dim=1` brightness consistent across T3/T4/T8.
- **Local vs HPC:** T1–T4, T6, T8 verified locally; T5/T7 are smoke+syntax locally, full runs in T9 runbook.
