#!/bin/bash
#SBATCH --job-name=cr_mae_pretrain
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/cr_mae_pretrain_%a_%j.log
#SBATCH --error=logs/cr_mae_pretrain_%a_%j.log

# CR-native denoising spatial-spectral MAE pretrain — encoder-size probe.
# Spec: docs/superpowers/specs/2026-07-15-cr-mrral-representation-design.md
# Plan: docs/superpowers/plans/2026-07-15-cr-mrral-representation.md (Task 5)
#
# Reads a PRECOMPUTED CR global cache (build_global_patch_cache.py
# --continuum_removed) fed UN-z-scored (per-patch z-score would rescale away
# absolute band depth). Two arms select encoder size by a downstream linear probe
# (scripts/linear_probe_encoder.py), NOT by MAE recon loss:
#   arm 0: embed_dim 128 (proven)      arm 1: embed_dim 256 (larger comparison)
# Keep the 256 encoder only if it wins the linear probe.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python
CR_CACHE=/xdisk/sbyrne/phillipsm/crism_patch_cache_cr   # built by build_global_patch_cache.py --continuum_removed

EMBED_DIMS=(128 256)
EMBED_DIM=${EMBED_DIMS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME=spatial_mae_cr_denoising_${EMBED_DIM}d_6l

cd "$WORK_DIR"
mkdir -p logs

# Point the global cache at the CR cache for this run.
cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF

if [ ! -d "$CR_CACHE" ] || [ -z "$(ls -A "$CR_CACHE"/global_patches_*.npy 2>/dev/null)" ]; then
    echo "ERROR: CR global cache missing at $CR_CACHE"
    echo "Build it first: python scripts/build_global_patch_cache.py --continuum_removed --output $CR_CACHE"
    exit 1
fi

echo "=== CR MAE pretrain arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (embed_dim=$EMBED_DIM) start $(date) ==="

${PYTHON} -u scripts/pretrain_spatial_mae_denoising.py \
    --continuum_removed \
    --epochs 200 \
    --warmup 10 \
    --batch_size 1024 \
    --patches_per_epoch 200000 \
    --num_workers 6 \
    --embed_dim ${EMBED_DIM} \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --decoder_layers 2 \
    --mask_ratio 0.75 \
    --sigma_gauss 0.0087 \
    --sigma_spike 0.0058 \
    --sigma_column 0.0049 \
    --spike_center_band 15 \
    --spike_fwhm_bands 3.0 \
    --run_name ${RUN_NAME}

echo "=== $RUN_NAME end $(date) ==="
