# HPC Runbook — CR-mrral representation (Spec 1)

Ordered HPC steps to build the continuum-removed (CR) representation, pretrain a
CR-native encoder, fine-tune, and run the floor-test go/no-go gate.

- Spec: `docs/superpowers/specs/2026-07-15-cr-mrral-representation-design.md`
- Plan: `docs/superpowers/plans/2026-07-15-cr-mrral-representation.md`
- Repo on HPC: `/groups/sbyrne/phillipsm/crism_classification` (branch
  `feature/spatial-mae-pretraining`); python
  `/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python`.

All local code (CR module, dataset/trainer/inference wiring, converter, linear
probe) is committed and unit-tested. These steps run the GPU/large-cache work.

---

## 0. Pull the code

```bash
cd /groups/sbyrne/phillipsm/crism_classification
git pull
```

## 1. Rebuild the global patch cache as CR

The old global cache is truncated (tile-corruption era) and must be rebuilt anyway.
Build it CR (writes `global_patches_*.npy` + `_brightness.npy` sidecars +
`shard_index.json`):

```bash
python scripts/build_global_patch_cache.py \
    --continuum_removed \
    --output /xdisk/sbyrne/phillipsm/crism_patch_cache_cr
# uses cfg["data_root"] tiles; ~same size/time as the raw global cache build.
```
Verify: `ls /xdisk/sbyrne/phillipsm/crism_patch_cache_cr/global_patches_000.npy`.

## 2. Pretrain the CR encoder — 2-arm size probe

```bash
sbatch scripts/hpc_pretrain_cr_denoising.slurm      # array 0-1: embed_dim 128 & 256
```
Produces `checkpoints/spatial_mae_cr_denoising_{128,256}d_6l_best.pt`. Reads the CR
cache from step 1 (fed un-z-scored). ~32 h/arm.

## 3. Build the CR labeled cache (fine-tune input)

Convert the existing raw labeled cache to CR + brightness sidecar (one-time; on-read
CR every epoch is too slow):

```bash
python scripts/build_cr_labeled_cache.py \
    --raw_dir data/patch_cache_7cls \
    --out_dir data/patch_cache_7cls_cr \
    --splits train val test --patch_size 7
```
(If `data/patch_cache_7cls` is absent, build it first via
`hpc_build_7cls_data.slurm`, then convert.) Writes
`data/patch_cache_7cls_cr/mrral_{split}_patches_p7.npy` + `_brightness.npy`.

## 4. Pick the encoder size — linear probe

Run the frozen-encoder probe for BOTH encoders; keep the winner (256 only if it
beats 128 on `val_mAP_core`):

```bash
for D in 128 256; do
  echo "== embed_dim $D =="
  python scripts/linear_probe_encoder.py \
    --encoder_ckpt checkpoints/spatial_mae_cr_denoising_${D}d_6l_best.pt \
    --mrral_parquets data/mrral_pixels_7cls.parquet \
    --patch_cache_dir data/patch_cache_7cls_cr \
    --embed_dim $D --n_layers 6 --n_heads 4 --patch_size 7 \
    --seven_class --continuum_removed --cache_is_cr --brightness_aux
done
```
Selection metric is `val_mAP_core` (NOT MAE recon loss). If the winner is 256, edit
`scripts/hpc_finetune_cr.slurm`: set `ENCODER=...256d_6l_best.pt` and `EMBED_DIM=256`.

## 5. Fine-tune on CR — 3-arm lrscale sweep

```bash
sbatch scripts/hpc_finetune_cr.slurm     # array 0-2: lrscale 0.001/0.01/0.1
```
Produces `checkpoints/ft_7cls_cr_lrscale{0001,001,01}_best.pt`. Honest unit-balanced
splits, stop `val_mAP_core`. NOTE: MTRDR synth-plag injection is intentionally
dropped for this run (see slurm header) — revisit after the gate.

## 6. The go/no-go gate — floor test + MC11 visual

Pick the arm with the best `val_mAP_core`, then (on the workstation, where the tiles
and floor-test tooling live):

```bash
# floor test (Nili + Argyre), compare to the champion v4honest_lrscale001
bash scripts/floor_test.sh checkpoints/ft_7cls_cr_lrscale001_best.pt cr_lrscale001

# MC11 alteration visual on the altered tile, with the CR flags
conda run -n crism python scripts/classify_tile_supervised.py \
    --tile /mnt/mrdr/mc11/t1450_mrral_30n358_0327_4.img \
    --ckpt checkpoints/ft_7cls_cr_lrscale001_best.pt \
    --continuum_removed --brightness_aux \
    --save_probs /tmp/mc11_cr_t1450_probs.npz --no_plot
```
(Floor test / classify must use the CR flags so inference CR-matches training.)

### PASS criteria
- **LCP survives OOD:** Nili LCP produces a non-trivial, threshold-graded polygon
  population (champion honest recovery = 0 at all thresholds).
- **No mafic/alteration collapse:** no mafic→olivine flood, no alteration→bland
  collapse (the two MC11 baselines failed oppositely — see
  `[[reviewonly-and-mc11]]`).

### On PASS
Update `MODELS.md` (new `ft_7cls_cr_*` row + verdict) and proceed to Spec 2
(leave-one-region-out eval, known-site validation, interpretability). Re-add MTRDR
synth-plag as a follow-up variable.

### On FAIL
STOP. CR-on-mrral is insufficient — do NOT build the Spec 2 eval layer. Reconsider
the deferred data-foundation fork (canonical ratioed MTRDR) or the label-purity
track. Record the negative result in `MODELS.md` + memory.
