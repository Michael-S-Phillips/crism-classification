#!/bin/bash
#SBATCH --job-name=crism_build_v2
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_review_v2_%j.log
#SBATCH --error=logs/build_review_v2_%j.log

# Build the v2 augmented training parquet on a compute node — the legacy
# hard_negatives parquet (2.6 GB on disk → ~9 GB in pandas) busts the login
# node's per-user memory cap.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs

python -u scripts/build_review_augmented_train.py \
    --max_pixels_per_polygon 20000 \
    --include_ambiguous \
    --with_alteration_column \
    --out data/mrral_pixels_with_review_v2.parquet

echo "=== build_review_v2 done ==="
ls -la data/mrral_pixels_with_review_v2.parquet
