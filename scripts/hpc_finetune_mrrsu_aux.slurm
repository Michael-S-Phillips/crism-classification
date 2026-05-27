#!/bin/bash
#SBATCH --job-name=crism_ft_mrrsu_aux
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_mrrsu_aux_%j.log
#SBATCH --error=logs/ft_mrrsu_aux_%j.log

# Fine-tune the aux classifier (encoder + smoothed RPEAK1/BD1300 head) from the best
# available encoder. Ablation partner: ft_plag_aware_real_only (same encoder, no aux).
# Requires the aux cache (build_mrrsu_aux.py) to exist in data/patch_cache/.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
AUX_DIR="${WORKDIR}/data/patch_cache"

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

# Build the aux cache if missing (cheap: reads mrrsu tiles, 7x7-mean, sample).
if [ ! -f "${AUX_DIR}/mrrsu_aux_train.npy" ] || [ ! -f "${AUX_DIR}/mrrsu_aux_stats.json" ]; then
    echo "=== building mrrsu aux cache ==="
    python -u scripts/build_mrrsu_aux.py
fi

if [ ! -f "$ENCODER" ]; then
    echo "ERROR: encoder missing: $ENCODER"; exit 1
fi

python -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name ft_mrrsu_aux \
    --pretrain_ckpt "${ENCODER}" \
    --mrrsu_aux_dir "${AUX_DIR}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== mrrsu-aux fine-tune done ==="
