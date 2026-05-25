#!/bin/bash
#SBATCH --job-name=crism_finetune_bland_binplag
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_bland_binplag_%j.log
#SBATCH --error=logs/finetune_bland_binplag_%j.log

# Binary plag-vs-rest classifier, warm-started from bland-v3 lr0001 cont1.
#
# Diagnostic / potential win:
# Train a 2-class (single-logit) classifier whose only task is "is this pixel
# plagioclase?". The encoder is reused (warm-start) but the multi-class head
# is dropped and replaced with a single-output head.
#
# If this dramatically out-performs the 5-class plag head (val_AP_plagioclase
# in cont1 ≈ 0.13), multi-class competition was the bottleneck. If it scores
# similarly, the encoder doesn't discriminate plagioclase well — we'd need a
# different intervention (data augmentation, focal loss + plag-only training,
# or a different pretraining).
#
# Three runs vary the positive-class weight (effective plagioclase up-weight
# inside the binary loss): {3, 5, 10}. Uses focal loss (gamma=2.0) by default
# so hard plag examples drive learning.

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
CKPT_INIT="${CKPT_DIR}/ft_bland_v3_lrscale0001_cont1_best.pt"

# ── micromamba ────────────────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
EOF
fi
mkdir -p logs checkpoints

if [ ! -f "$CKPT_INIT" ]; then
    echo "ERROR: warm-start checkpoint missing: $CKPT_INIT"; exit 1
fi

# Each run: binary mode (n_classes=1, target=plagioclase). The 5-class head is
# automatically dropped from the warm-start checkpoint at load time (encoder
# weights are reused). Different positive-class weights are passed via
# --class_weights — for a 1-class model this single value scales the positive
# logit's loss contribution.

run_binplag () {
    local POSW=$1
    local TAG="ft_bland_v3_cont1_binplag_posw${POSW//./}"
    echo "=== ${TAG} (binary plag, pos_weight=${POSW}) ==="
    python -u scripts/train.py \
        --model spatial_vit \
        --run_name "${TAG}" \
        --init_ckpt "${CKPT_INIT}" \
        --n_classes 1 \
        --binary_target_class plagioclase \
        --class_weights "${POSW}" \
        --focal_loss --focal_gamma 2.0 \
        --encoder_lr_scale 0.001 \
        --epochs 100 --patience 25 \
        --batch_size 256 --lr 5e-4 \
        --embed_dim 128 --n_heads 4 --n_layers 6 \
        --patch_size 7
}

run_binplag 3
run_binplag 5
run_binplag 10

echo "=== Binary-plag sweep done ==="
