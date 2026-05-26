#!/bin/bash
#SBATCH --job-name=crism_plag_aware_mae
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/plag_aware_mae_%j.log
#SBATCH --error=logs/plag_aware_mae_%j.log

# Plag-aware multi-task pretraining: warm-start the denoising MAE encoder with a
# 5-class ASL aux head (dual stream: unlabeled global cache for recon, labeled
# mrral patches for recon + aux). Produces an encoder checkpoint whose encoder_state
# loads straight into SpatialSpectralClassifier.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
CKPT_DIR="${WORKDIR}/checkpoints"
INIT="${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f "$INIT" ]; then
    echo "ERROR: warm-start checkpoint missing: $INIT"; exit 1
fi

python -u scripts/pretrain_plag_aware_mae.py \
    --init "$INIT" \
    --epochs 40 \
    --warmup 5 \
    --aux_warmup 5 \
    --lambda_target 1.0 \
    --plag_class_weight 5.0 \
    --batch_size 512 \
    --labeled_batch_size 256 \
    --steps_per_epoch 400 \
    --num_workers 6 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --mask_ratio 0.75 \
    --asl_gamma_neg 4.0 --asl_gamma_pos 0.0 --asl_clip 0.05 \
    --run_name plag_aware_mae_128d_6l

echo "=== plag-aware pretraining done ==="
