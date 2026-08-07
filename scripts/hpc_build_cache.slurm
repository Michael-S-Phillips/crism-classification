#!/bin/bash
#SBATCH --job-name=build_mrral_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=0-2:00:00
#SBATCH --output=logs/build_cache_%j.log
#SBATCH --error=logs/build_cache_%j.log

# Build the 7x7 mrral patch cache from data/mrral_pixels.parquet.
# CPU-only — patch extraction is I/O-bound (reading tile data).
# Run this before scripts/hpc_ablation_lr_scale_v4.slurm.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"
mkdir -p logs

${PYTHON} -u scripts/cache_mrral_patches.py --patch_size 7
