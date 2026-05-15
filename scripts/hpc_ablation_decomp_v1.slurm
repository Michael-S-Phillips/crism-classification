#!/bin/bash
#SBATCH --job-name=spvit_decomp_v1
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/decomp_v1_%a_%j.out
#SBATCH --error=logs/decomp_v1_%a_%j.err

# v1: first run of the signal/noise decomposition encoder (DecompSpVit).
# Matches v5 hyperparams except --model decomp_spatial_vit and the new
# --decomp_lambda_* flags. Spec at:
#   docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache

RUN_NAMES=("spvit_decomp_v1_frozen" "spvit_decomp_v1_lrscale0001" "spvit_decomp_v1_lrscale001" "spvit_decomp_v1_lrscale01")
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")
CLASS_WEIGHTS="1.0,1.0,1.5,3.0,1.0"

PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7.npy" ]; then
    echo "ERROR: mrral_train_patches_p7.npy not found at ${CACHE_DIR}" >&2
    exit 1
fi

mkdir -p logs checkpoints

echo "=== decomp_v1 ablation: ${RUN_NAMES[$SLURM_ARRAY_TASK_ID]} (task $SLURM_ARRAY_TASK_ID) ==="
${PYTHON} -u scripts/train.py \
    --model decomp_spatial_vit \
    --run_name "${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --epochs 100 \
    --patience 25 \
    --min_delta 0.001 \
    --batch_size 256 \
    --lr 5e-4 \
    --lr_t_max 100 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --class_weights "${CLASS_WEIGHTS}" \
    --decomp_lambda_recon 1.0 \
    --decomp_lambda_eps 0.1 \
    --decomp_lambda_T 0.01 \
    --decomp_lambda_b 0.01 \
    --decomp_lambda_smooth 0.001 \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}
