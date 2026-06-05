#!/bin/bash
#SBATCH --job-name=crism_contrastive_ft
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=18:00:00
#SBATCH --output=logs/contrastive_ft_%j.log
#SBATCH --error=logs/contrastive_ft_%j.log

# Full fine-tune of the contrastive-refined encoder.
# - Warm-starts from checkpoints/contrastive_plag_v1_best.pt
# - Attaches a fresh SpatialSpectralClassifier head
# - Trains end-to-end with ASL loss + encoder_lr_scale=0.001 (matches the
#   project's other supervised FTs for apples-to-apples comparison).
# - Saves best by val_mAP.

set -euo pipefail

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
CONTRASTIVE_CKPT="${CKPT_DIR}/contrastive_plag_v1_best.pt"
RUN_NAME="contrastive_plag_v1"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f "$CONTRASTIVE_CKPT" ]; then
    echo "ERROR: contrastive checkpoint missing: $CONTRASTIVE_CKPT"
    exit 1
fi

# Auto-discover the ROI extra-positives pool so the FT head trains on the
# hand-vetted plag patches too (same hook as the linear probe).
PROBE_EXTRA_ARGS=()
EXTRA_POS_DIR="${WORKDIR}/data/contrastive/extra_plag_roi"
if [ -f "${EXTRA_POS_DIR}/patches.npy" ] && [ -f "${EXTRA_POS_DIR}/meta.parquet" ]; then
    PROBE_EXTRA_ARGS+=(--extra_positive_pool_dirs "${EXTRA_POS_DIR}")
    echo "  including extra positive pool dir: ${EXTRA_POS_DIR}"
fi

echo "=== full fine-tune (ASL, encoder_lr_scale=0.001) ==="
python -u scripts/eval_contrastive.py \
    --ckpt "${CONTRASTIVE_CKPT}" \
    --mode finetune \
    --run_name "${RUN_NAME}" \
    --output_dir "${CKPT_DIR}" \
    --epochs 100 --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --encoder_lr_scale 0.001 \
    --asl_gamma_neg 4.0 --asl_gamma_pos 0.0 --asl_clip 0.05 \
    "${PROBE_EXTRA_ARGS[@]}"

echo "=== contrastive finetune done ==="
