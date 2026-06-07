#!/bin/bash
#SBATCH --job-name=crism_ft_with_review
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=14:00:00
#SBATCH --output=logs/ft_with_review_%j.log
#SBATCH --error=logs/ft_with_review_%j.log

# Fine-tune the spatial_vit classifier on baseline + MC13-review-augmented
# training data. The review pool added ~1.25 M manually-verified pixels
# (16× more olivine, ~2× lcp/hcp, ~0.06× bland; no plag) that the loss
# sees at confidence_weight=2.0 vs 1.0 for the original soft labels.
#
# Direct partner: ft_plag_aware_relabeled (same encoder, same recipe, no
# review augmentation). val_mAP delta tells us whether the review effort
# moved the needle.
#
# Required setup on HPC (do BEFORE sbatch):
#   1. rsync data/mrral_pixels_with_review.parquet to
#      /xdisk/sbyrne/phillipsm/CRISM_MRDR/data/ via filexfer.hpc.arizona.edu
#   2. Build the matching patch cache (one-time, ~1 h on local SSD):
#        python -u scripts/cache_mrral_patches.py \
#          --parquets data/mrral_pixels.parquet data/mrral_pixels_with_review.parquet \
#          --cache_dir data/patch_cache_with_review

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
CACHE_DIR="${WORKDIR}/data/patch_cache_with_review"
PARQUET_BASE="${WORKDIR}/data/mrral_pixels.parquet"
PARQUET_REVIEW="${WORKDIR}/data/mrral_pixels_with_review.parquet"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
EOF
fi

if [ ! -f "$ENCODER" ]; then echo "ERROR: encoder missing: $ENCODER"; exit 1; fi
if [ ! -f "$PARQUET_BASE" ]; then echo "ERROR: base parquet missing: $PARQUET_BASE"; exit 1; fi
if [ ! -f "$PARQUET_REVIEW" ]; then echo "ERROR: review parquet missing: $PARQUET_REVIEW"; exit 1; fi
if [ ! -d "$CACHE_DIR" ]; then echo "ERROR: cache dir missing: $CACHE_DIR — build it first (see header)"; exit 1; fi

python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_with_review \
    --pretrain_ckpt "${ENCODER}" \
    --mrral_parquets "${PARQUET_BASE}" "${PARQUET_REVIEW}" \
    --patch_cache_dir "${CACHE_DIR}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== ft_with_review done ==="
