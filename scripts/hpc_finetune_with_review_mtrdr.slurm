#!/bin/bash
#SBATCH --job-name=crism_ft_review_mtrdr
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=14:00:00
#SBATCH --output=logs/ft_review_mtrdr_%j.log
#SBATCH --error=logs/ft_review_mtrdr_%j.log

# Fine-tune the spatial_vit classifier on:
#   (1) the baseline mrral_pixels.parquet
#   (2) the MC13 polygon-review augmentation (review-augmented parquet)
#   (3) the 1,817 MTRDR-derived plagioclase ROI patches via the synthetic-
#       patch ConcatDataset mechanism
#
# Direct partners (both finished earlier):
#   - ft_plag_aware_relabeled  : baseline only
#   - ft_with_review           : baseline + review augmentation (regressed)
#
# Question this run answers: does adding 1,817 plag-positive MTRDR pixels to
# the supervised loss move plagioclase AP off its 0.14-0.16 plateau? The
# contrastive encoder already saw these patches via InfoNCE; this is the
# first time the supervised classifier head will see them as labeled data.
#
# Required setup on HPC (do BEFORE sbatch):
#   1. git pull on /groups/sbyrne/${USER}/crism_classification (this script
#      + scripts/build_mtrdr_plag_rows.py)
#   2. python -u scripts/build_mtrdr_plag_rows.py   (~5 sec, produces
#      data/mtrdr_plag_rows.parquet)
#   3. Confirm data/contrastive/extra_plag_roi/patches.npy exists locally
#      (it should — it was rsync'd during the contrastive build).
#   4. Confirm the review-augmented patch cache exists at
#      data/patch_cache_with_review (built earlier via
#      scripts/hpc_build_cache_with_review.slurm).

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
CACHE_DIR="${WORKDIR}/data/patch_cache_with_review"
PARQUET_BASE="${WORKDIR}/data/mrral_pixels.parquet"
PARQUET_REVIEW="${WORKDIR}/data/mrral_pixels_with_review.parquet"
MTRDR_PATCHES="${WORKDIR}/data/contrastive/extra_plag_roi/patches.npy"
MTRDR_ROWS="${WORKDIR}/data/mtrdr_plag_rows.parquet"

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

# Sanity checks
for f in "$ENCODER" "$PARQUET_BASE" "$PARQUET_REVIEW" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then echo "ERROR: cache dir missing: $CACHE_DIR"; exit 1; fi

python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_review_mtrdr \
    --pretrain_ckpt "${ENCODER}" \
    --mrral_parquets "${PARQUET_BASE}" "${PARQUET_REVIEW}" \
    --patch_cache_dir "${CACHE_DIR}" \
    --synth_train_cache "${MTRDR_PATCHES}" \
    --synth_train_parquet "${MTRDR_ROWS}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== ft_review_mtrdr done ==="
