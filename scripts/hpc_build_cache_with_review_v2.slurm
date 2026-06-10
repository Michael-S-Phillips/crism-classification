#!/bin/bash
#SBATCH --job-name=crism_cache_v2
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/cache_with_review_v2_%j.log
#SBATCH --error=logs/cache_with_review_v2_%j.log

# Build the patch cache aligned to the v2 augmented parquet.
#
# Inputs:
#   data/mrral_pixels.parquet                  (patched with alteration column)
#   data/mrral_pixels_with_review_v2.parquet   (per-polygon cap + alteration + ambiguous)
#
# Output: data/patch_cache_with_review_v2/mrral_{train,val,test}_patches_p7.npy
#
# Kept parallel to the v1 cache dir (data/patch_cache_with_review/) so the
# v1 FT runs remain reproducible. The 6-class FT slurm
# (hpc_finetune_6class_alteration.slurm) targets the v2 cache dir.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs data/patch_cache_with_review_v2

if [ ! -f data/mrral_pixels.parquet ]; then
    echo "ERROR: data/mrral_pixels.parquet missing — run patch script first"; exit 1
fi
if [ ! -f data/mrral_pixels_with_review_v2.parquet ]; then
    echo "ERROR: data/mrral_pixels_with_review_v2.parquet missing — run hpc_build_review_v2.slurm first"; exit 1
fi

# Sanity-check the alteration column propagated into v2
python -c "
import pandas as pd
for p in ['data/mrral_pixels.parquet', 'data/mrral_pixels_with_review_v2.parquet']:
    df = pd.read_parquet(p, columns=['alteration'])
    n = int((df['alteration'] > 0.5).sum())
    print(f'  {p}: {n:,} alteration-positive rows')
" || exit 1

python -u scripts/cache_mrral_patches.py \
    --parquets data/mrral_pixels.parquet data/mrral_pixels_with_review_v2.parquet \
    --cache_dir data/patch_cache_with_review_v2

echo "=== cache build done ==="
ls -la data/patch_cache_with_review_v2/
