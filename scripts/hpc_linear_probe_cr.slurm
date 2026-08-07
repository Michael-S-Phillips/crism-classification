#!/bin/bash
#SBATCH --job-name=cr_linear_probe
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/cr_linear_probe_%j.log
#SBATCH --error=logs/cr_linear_probe_%j.log

# Frozen-encoder linear probe to pick the CR encoder size (128 vs 256) by
# val_mAP_core. Runs on a GPU node with 64gb — the login node OOM-kills this
# (it loads the full 7cls parquet + extracts features for all train patches).
# Plan step 4: docs/hpc_runbook_cr_mrral.md
# Keep the 256d encoder only if it beats 128d on val_mAP_core.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"
mkdir -p logs

CACHE_DIR=${DATA_DIR}/patch_cache_7cls_cr
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7_brightness.npy" ]; then
    echo "ERROR: CR labeled cache/brightness sidecar missing at $CACHE_DIR"
    echo "Build it: sbatch scripts/hpc_build_cr_labeled_cache.slurm"
    exit 1
fi

for D in 128 256; do
    CKPT=${DATA_ROOT}/crism_classification/checkpoints/spatial_mae_cr_denoising_${D}d_6l_best.pt
    if [ ! -f "$CKPT" ]; then echo "ERROR: missing $CKPT"; exit 1; fi
    echo "==================== embed_dim $D ===================="
    ${PYTHON} -u scripts/linear_probe_encoder.py \
        --encoder_ckpt "$CKPT" \
        --mrral_parquets "${DATA_DIR}/mrral_pixels_7cls.parquet" \
        --patch_cache_dir "$CACHE_DIR" \
        --embed_dim $D --n_layers 6 --n_heads 4 --patch_size 7 \
        --seven_class --continuum_removed --cache_is_cr --brightness_aux
done

echo "=== linear probe done: $(date) — compare the two val_mAP_core above ==="
