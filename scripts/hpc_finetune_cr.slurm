#!/bin/bash
#SBATCH --job-name=crism_7cls_cr
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/7cls_cr_%a_%j.log
#SBATCH --error=logs/7cls_cr_%a_%j.log

# 7-class fine-tune on the CR representation — single-variable vs the champion
# ft_7cls_v3b_lrscale001 (representation = the only change).
# Spec: docs/superpowers/specs/2026-07-15-cr-mrral-representation-design.md
# Plan: docs/superpowers/plans/2026-07-15-cr-mrral-representation.md (Task 7)
#
# Reads a PRECOMPUTED CR labeled cache (scripts/build_cr_labeled_cache.py) with the
# brightness sidecar; --model spatial_vit_aux consumes the CR patch + brightness
# scalar (aux_dim=1). Encoder = the CR-pretrain winner (128 or 256; pick via
# scripts/linear_probe_encoder.py). Honest unit-balanced splits; stop val_mAP_core.
#
# DEVIATION (documented): the MTRDR synth-plag injection used by the raw champion
# is DROPPED here — those patches are raw and would need CR + brightness plumbing.
# CR's gate is the LCP/mafic floor test (representation), which plag-synth does not
# affect. Re-add synth in a follow-up once CR is validated.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

# CR-pretrain winner: 256d beat 128d on the linear probe (val_mAP_core 0.596 vs
# 0.541, 2026-07-27) — a clear 5.6 pp margin, so use the 256d encoder.
ENCODER=${CKPT_DIR}/spatial_mae_cr_denoising_256d_6l_best.pt
EMBED_DIM=256
CACHE_DIR=${WORK_DIR}/data/patch_cache_7cls_cr          # built by build_cr_labeled_cache.py
PARQUET=${WORK_DIR}/data/mrral_pixels_7cls.parquet

RUN_NAMES=("ft_7cls_cr_lrscale0001" "ft_7cls_cr_lrscale001" "ft_7cls_cr_lrscale01")
LR_SCALES=("0.001" "0.01" "0.1")
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
LR_SCALE=${LR_SCALES[$SLURM_ARRAY_TASK_ID]}

cd "$WORK_DIR"
mkdir -p logs checkpoints

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

for f in "$ENCODER" "$PARQUET"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7_brightness.npy" ]; then
    echo "ERROR: CR labeled cache/brightness sidecar missing at $CACHE_DIR"
    echo "Build it: python scripts/build_cr_labeled_cache.py --raw_dir data/patch_cache_7cls --out_dir $CACHE_DIR"
    exit 1
fi

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (encoder_lr_scale=$LR_SCALE, embed_dim=$EMBED_DIM) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --seven_class \
    --continuum_removed --cache_is_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --lr 5e-4 --encoder_lr_scale "$LR_SCALE" \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
