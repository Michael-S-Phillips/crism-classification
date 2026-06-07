#!/bin/bash
#SBATCH --job-name=crism_cache_with_review
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/cache_with_review_%j.log
#SBATCH --error=logs/cache_with_review_%j.log

# Build a row-aligned patch cache for [mrral_pixels, mrral_pixels_with_review].
# Output: data/patch_cache_with_review/mrral_{train,val,test}_patches_p7.npy
#
# Required: both parquets must be in data/ before this runs:
#   data/mrral_pixels.parquet                (baseline)
#   data/mrral_pixels_with_review.parquet    (rsync'd from local; cp from /xdisk if needed)

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs data/patch_cache_with_review

if [ ! -f data/mrral_pixels.parquet ]; then
    echo "ERROR: data/mrral_pixels.parquet missing"; exit 1
fi
if [ ! -f data/mrral_pixels_with_review.parquet ]; then
    echo "ERROR: data/mrral_pixels_with_review.parquet missing — cp from /xdisk first"; exit 1
fi

python -u scripts/cache_mrral_patches.py \
    --parquets data/mrral_pixels.parquet data/mrral_pixels_with_review.parquet \
    --cache_dir data/patch_cache_with_review

echo "=== cache build done ==="
ls -la data/patch_cache_with_review/
