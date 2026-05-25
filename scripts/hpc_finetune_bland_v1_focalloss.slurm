#!/bin/bash
#SBATCH --job-name=crism_finetune_bland_focal
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_bland_focal_%j.log
#SBATCH --error=logs/finetune_bland_focal_%j.log

# Focal loss sweep, warm-started from bland-v3 lr0001 cont1.
#
# Goal: lift plagioclase val_AP beyond what class-weighting alone achieved.
# Focal loss (gamma > 0) automatically down-weights easy examples and up-
# weights hard ones — exactly plagioclase's situation. Combined with the
# class-weight scheme that did best in the prior class-balance sweep
# (plag=3, other=0.7), this should push the plag head harder.
#
# Three runs vary focal_gamma in {1.5, 2.0, 3.0}.
# Class weights fixed at "1.0,1.0,1.5,3.0,0.7" (oli, lcp, hcp, plag, other).
# All warm-start from cont1; 100 epochs / patience 25; same lr recipe as cont1.

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

run_focal () {
    local GAMMA=$1
    local TAG="ft_bland_v3_cont1_focal_g${GAMMA//./}"   # focal_g15, focal_g20, focal_g30
    echo "=== ${TAG} (focal_gamma=${GAMMA}) ==="
    python -u scripts/train.py \
        --model spatial_vit \
        --run_name "${TAG}" \
        --init_ckpt "${CKPT_INIT}" \
        --class_weights "1.0,1.0,1.5,3.0,0.7" \
        --focal_loss --focal_gamma "${GAMMA}" \
        --encoder_lr_scale 0.001 \
        --epochs 100 --patience 25 \
        --batch_size 256 --lr 5e-4 \
        --embed_dim 128 --n_heads 4 --n_layers 6 \
        --patch_size 7
}

run_focal 1.5
run_focal 2.0
run_focal 3.0

echo "=== Focal-loss sweep done ==="
