#!/bin/bash
#SBATCH --job-name=spvit_lr_ablation
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lr_ablation_%a_%j.out
#SBATCH --error=logs/lr_ablation_%a_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache

# ── Condition arrays (task ID 0-3) ───────────────────────────────────────────
RUN_NAMES=("spvit_frozen" "spvit_lrscale0001" "spvit_lrscale001" "spvit_lrscale01")
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")

# ── micromamba environment ─────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORK_DIR"

# ── Write config.local.yaml ──────────────────────────────────────────────────
cat > config.local.yaml <<EOF
data_root: ${WORK_DIR}/data
checkpoint_dir: ${CKPT_DIR}
EOF

# ── Patch cache guard ────────────────────────────────────────────────────────
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7.npy" ]; then
    echo "ERROR: mrral_train_patches_p7.npy not found at ${CACHE_DIR}" >&2
    exit 1
fi

mkdir -p logs checkpoints

# ── Training ─────────────────────────────────────────────────────────────────
echo "=== LR scale ablation: ${RUN_NAMES[$SLURM_ARRAY_TASK_ID]} (task $SLURM_ARRAY_TASK_ID) ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name "${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --epochs 50 \
    --patience 10 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}
