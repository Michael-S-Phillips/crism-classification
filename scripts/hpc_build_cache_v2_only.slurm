#!/bin/bash
#SBATCH --job-name=crism_cache_v2only
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/cache_v2_only_%j.log
#SBATCH --error=logs/cache_v2_only_%j.log

# Build the patch cache from the v2 augmented parquet ALONE.
#
# The v2 parquet is a self-contained replacement (capped base + review rows);
# the earlier base+v2 concat double-counted every base row and defeated the
# per-polygon cap (audit 2026-06-11, bug #2). This cache pairs with FT runs
# that pass --mrral_parquets v2 only.
#
# Input:  data/mrral_pixels_with_review_v2.parquet
# Output: data/patch_cache_v2_only/mrral_{train,val,test}_patches_p7.npy

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs data/patch_cache_v2_only

if [ ! -f data/mrral_pixels_with_review_v2.parquet ]; then
    echo "ERROR: data/mrral_pixels_with_review_v2.parquet missing — run hpc_build_review_v2.slurm first"; exit 1
fi

python -u scripts/cache_mrral_patches.py \
    --parquets data/mrral_pixels_with_review_v2.parquet \
    --cache_dir data/patch_cache_v2_only

echo "=== v2-only cache build done ==="
ls -la data/patch_cache_v2_only/
