#!/bin/bash
#SBATCH --job-name=dualcr_ft
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/dualcr_ft_%j.log
#SBATCH --error=logs/dualcr_ft_%j.log

# JOB 4 of 4 — 7-class hand-core fine-tune on the 118-channel dual-CR
# representation, warm-started from the dual-CR MAE encoder.
#
# The comparator is ft_7cls_handcore_level. Every hyperparameter below is copied
# from scripts/hpc_finetune_handcore.slurm arm 0 so the arms differ in
# REPRESENTATION only:
#   spatial_vit_aux + brightness_aux, 7-class, weight_scheme level, ASL loss,
#   lr 5e-4, encoder_lr_scale 0.001, 150 epochs, patience 40, min_delta 0.0005,
#   batch 256, embed_dim 256, 4 heads, 6 layers, patch 7.
#
# ── ONE UNAVOIDABLE SECOND DIFFERENCE. READ BEFORE INTERPRETING THE RESULT ───
# ft_7cls_handcore_level injects MTRDR plagioclase via --synth_train_cache /
# --synth_val_cache. Those caches are 59-channel, so train.py REFUSES --dual_cr
# alongside any --synth_* rather than concatenating a 59-channel source into a
# 118-channel run. This arm therefore trains plagioclase on hand labels only.
#
# So a plagioclase difference between this run and ft_7cls_handcore_level is NOT
# evidence about hull-vs-dual CR. Task 8 must do one of:
#   (a) build a dual-CR MTRDR plag patch cache and re-run with --synth_*, or
#   (b) run a matched hull-CR control arm with --synth_* also omitted.
# Option (b) is cheaper and is the honest minimum. The ALTERATION and BLAND
# classes — the actual hypothesis under test — are unaffected either way.
#
# ── What is being tested ─────────────────────────────────────────────────────
# Upper-hull CR divides by the convex hull, and a broad convex arch IS
# approximately the hull, so hull CR destroys alteration's 1-2 µm arch (41%
# retained vs 84% for bland). Per-pixel the arch alone separates alteration from
# every other class at AUC 0.990 under linear CR against 0.856 under hull CR. If
# hull CR's flattening is what makes alteration a false attractor, the dual
# representation should reduce alteration over-prediction on the floor tiles.
#
# val_mAP is NOT comparable across data-build eras. The floor test (6 fixed
# Nili/Argyre tiles) is the leakage-immune comparator — see MODELS.md.
#
# Submit chained on the pretrain AND the labeled cache:
#   sbatch --dependency=afterok:<pretrain_job>:<labeled_cache_job> \
#          scripts/hpc_finetune_dualcr.slurm

set -uo pipefail

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification          # code + env only
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR                    # ALL bulk data
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
EMBED_DIM=256
ENCODER=${CKPT_DIR}/spatial_mae_dualcr_denoising_${EMBED_DIM}d_6l_best.pt

PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet
CACHE_DIR=${DATA_DIR}/patch_cache_handcore_dualcr
RUN_NAME=ft_7cls_handcore_dualcr_level

cd "$WORK_DIR" || { echo "ERROR: cannot cd $WORK_DIR"; exit 1; }
mkdir -p logs

# Checkpoints and all outputs on xdisk, never /groups.
if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi
mkdir -p "$CKPT_DIR"

# ── PREREQUISITES ────────────────────────────────────────────────────────────
for f in "$ENCODER" "$PARQUET"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
for split in train val; do
    for f in "${CACHE_DIR}/mrral_${split}_patches_p7.npy" \
             "${CACHE_DIR}/mrral_${split}_patches_p7_brightness.npy"; do
        [ -f "$f" ] || {
            echo "ERROR: dual-CR patch cache incomplete — missing $f"
            echo "Build it: sbatch scripts/hpc_build_dualcr_labeled_cache.slurm"
            exit 1; }
    done
done

# The dataset's byte-exact size guard would catch a 59-channel cache, and a
# 59-wide encoder warm-start would raise on the band_embed copy_ — but both only
# after SLURM has handed over a GPU. Check the two widths up front instead.
$PYTHON - "$ENCODER" "$CACHE_DIR" "$PARQUET" <<'EOF' || { echo "ERROR: encoder/cache width preflight failed"; exit 1; }
import os, sys, torch
import pandas as pd
enc_path, cache_dir, parquet = sys.argv[1], sys.argv[2], sys.argv[3]
ck = torch.load(enc_path, map_location='cpu', weights_only=False)
state = ck.get('encoder_state') or ck.get('mae_state') or ck
KEY = 'band_embed.weight'
hits = [k for k in state if k.endswith(KEY)]
assert hits, f'no *{KEY} in {enc_path}; keys: {sorted(state)[:6]}'
# band_embed is nn.Linear(n_bands, embed_dim) -> weight (embed_dim, n_bands), so
# shape[-1] IS the channel count. A 59-wide encoder would raise on the copy_ into
# a 118-wide param, but only after SLURM has handed over the GPU.
w = state[hits[0]]
assert w.shape[-1] == 118, f'encoder is {w.shape[-1]}-channel, not 118 (dual-CR)'
print(f'  encoder OK: {hits[0]} {tuple(w.shape)}')
# The LABELED cache is a raw HEADERLESS np.memmap (build_cr_labeled_cache.py:86),
# so np.load cannot report its shape and byte count is all we have — unlike the
# np.save'd GLOBAL shards in jobs 1/2, which are checked via np.load(...).shape.
# Different formats on purpose; do not unify the two checks.
# A bare bytes % (7*7*118*4) test proves nothing either, because
# 59 * (even n) == 118 * (n/2). The parquet row count is the real reference, and
# it is exactly what CRISMSpectralPatchDataset will demand at startup.
df = pd.read_parquet(parquet, columns=['split'])
for s in ('train', 'val'):
    n = os.path.getsize(os.path.join(cache_dir, f'mrral_{s}_patches_p7.npy'))
    rows = int((df['split'] == s).sum())
    want = rows * 7 * 7 * 118 * 4
    assert n == want, (f'{s} cache is {n:,} bytes but the parquet has {rows:,} '
                       f'rows, needing exactly {want:,} at 118 channels')
    print(f'  cache OK: {s} {rows:,} patches x 7x7x118')
EOF

# Refuse to train on spectra we have not checked. Cheap next to a wasted GPU day.
echo "=== spectra audit ==="
$PYTHON scripts/audit_spectra_quality.py "$PARQUET" --fail_tile_over 5 || {
    echo "ERROR: $PARQUET failed the spectra audit. Refusing to train."; exit 1; }

echo "=== $RUN_NAME start $(date) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --seven_class \
    --continuum_removed --cache_is_cr --dual_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --weight_scheme level \
    --lr 5e-4 --encoder_lr_scale 0.001 \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss || { echo "ERROR: fine-tune failed"; exit 1; }

echo "=== $RUN_NAME end $(date) ==="
echo "Floor-test with (--dual_cr is REQUIRED; classify_tile_supervised verifies the"
echo "checkpoint's own first-layer width against it and aborts on a mismatch):"
# No vocab flag: classify_tile_supervised._set_n_classes reads head.weight and a
# 7-wide head auto-selects the 7-class vocab. --seven_class does not exist there.
echo "  CLASSIFY_EXTRA_ARGS=\"--continuum_removed --dual_cr --brightness_aux --embed_dim ${EMBED_DIM}\" \\"
echo "    bash scripts/floor_test.sh ${CKPT_DIR}/${RUN_NAME}_best.pt ${RUN_NAME}"
