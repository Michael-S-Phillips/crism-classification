#!/bin/bash
#SBATCH --job-name=crism_ft_relabeled
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_relabeled_%j.log
#SBATCH --error=logs/ft_relabeled_%j.log

# Champion recipe (plag-aware encoder, ASL) re-run with the manual olivine
# relabels applied (soft pyroxene targets on the 118 reviewed polygons).
# Ablation partner: ft_plag_aware_real_only (identical config, no relabels).
# Goal: recover HCP AP that the olivine-label contamination was suppressing.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
RELABELS="${WORKDIR}/data/olivine_relabels.csv"

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

if [ ! -f "$ENCODER" ]; then echo "ERROR: encoder missing: $ENCODER"; exit 1; fi
if [ ! -f "$RELABELS" ]; then echo "ERROR: relabels CSV missing: $RELABELS"; exit 1; fi

python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_plag_aware_relabeled \
    --pretrain_ckpt "${ENCODER}" \
    --apply_relabels "${RELABELS}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== relabeled fine-tune done ==="
