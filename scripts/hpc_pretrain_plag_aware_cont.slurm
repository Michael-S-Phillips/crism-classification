#!/bin/bash
#SBATCH --job-name=crism_plag_aware_cont
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/plag_aware_mae_cont_%j.log
#SBATCH --error=logs/plag_aware_mae_cont_%j.log

# Longer continuation of the plag-aware multi-task pretraining. Warm-starts from
# the 40-epoch plag_aware_mae_128d_6l_best.pt (encoder + decoder + aux head all
# resume) and trains 100 more epochs. Tests whether plag-aware representation keeps
# improving with more pretraining. The aux head is already trained, so the lambda
# ramp is short (aux_warmup=1).
#
# NOTE: the monitor_plag_AP logged here is inflated by spatial autocorrelation
# (per-pixel carve from train) — it is only a checkpoint-selection signal, NOT a
# generalization metric. The honest test is a downstream fine-tune on the official
# val split.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
CKPT_DIR="${WORKDIR}/checkpoints"
INIT="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f "$INIT" ]; then
    echo "ERROR: plag-aware checkpoint to continue from is missing: $INIT"; exit 1
fi

python -u scripts/pretrain_plag_aware_mae.py \
    --init "$INIT" \
    --epochs 100 \
    --warmup 5 \
    --aux_warmup 1 \
    --lambda_target 1.0 \
    --plag_class_weight 5.0 \
    --batch_size 512 \
    --labeled_batch_size 256 \
    --steps_per_epoch 400 \
    --num_workers 6 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --mask_ratio 0.75 \
    --asl_gamma_neg 4.0 --asl_gamma_pos 0.0 --asl_clip 0.05 \
    --run_name plag_aware_mae_128d_6l_cont

echo "=== plag-aware pretraining continuation done ==="
