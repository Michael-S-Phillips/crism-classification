#!/bin/bash
#SBATCH --job-name=crism_ft_plag_aware
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ft_plag_aware_%j.log
#SBATCH --error=logs/ft_plag_aware_%j.log

# 3-way plag-aware evaluation (runs A + B). The baseline run
# (ft_bland_v3_lrscale0001_cont1) is already in wandb; we do not rerun it.
#
#   Run A — encoder-only: plag-aware encoder, real data only  (isolates encoder)
#   Run B — encoder + synthetic plag patches in the train split (encoder + data)
#
# Both use the IDENTICAL cont1 fine-tune recipe (encoder_lr_scale=0.001, ASL,
# lr=5e-4, batch=256, 6L/128d/4H, patch 7) so the only variables are the encoder
# and the synthetic data. --pretrain_ckpt loads encoder_state only (fresh head).

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
SYNTH_NPY="${WORKDIR}/data/patch_cache/synth_plag_patches_p7.npy"
SYNTH_PARQUET="${WORKDIR}/data/patch_cache/synth_plag_rows.parquet"

# ── micromamba ────────────────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
EOF
fi
mkdir -p logs checkpoints

if [ ! -f "$ENCODER" ]; then
    echo "ERROR: plag-aware encoder missing: $ENCODER"; exit 1
fi

# ── Run A: encoder-only (real data) ─────────────────────────────────────────
echo "=== Run A: ft_plag_aware_real_only ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_plag_aware_real_only \
    --pretrain_ckpt "${ENCODER}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run B: encoder + synthetic plag data ────────────────────────────────────
if [ ! -f "$SYNTH_NPY" ] || [ ! -f "$SYNTH_PARQUET" ]; then
    echo "WARNING: synthetic plag cache missing — skipping Run B."
    echo "  expected: $SYNTH_NPY"
    echo "           $SYNTH_PARQUET"
    echo "  (scp them from the local box's data/patch_cache/, then resubmit Run B.)"
else
    echo "=== Run B: ft_plag_aware_real_plus_synth ==="
    python -u scripts/train.py \
        --model spatial_vit \
        --run_name ft_plag_aware_real_plus_synth \
        --pretrain_ckpt "${ENCODER}" \
        --encoder_lr_scale 0.001 \
        --epochs 100 --patience 25 \
        --batch_size 256 --lr 5e-4 \
        --embed_dim 128 --n_heads 4 --n_layers 6 \
        --patch_size 7 \
        --asl_loss \
        --synth_train_cache "${SYNTH_NPY}" \
        --synth_train_parquet "${SYNTH_PARQUET}"
fi

echo "=== plag-aware fine-tune eval done ==="
