#!/bin/bash
#SBATCH --job-name=spatial_vit_finetune
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.log

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"

# ── micromamba environment ─────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"

# ── Ensure config.local.yaml points to data on xdisk ─────────────────────────
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
fi

mkdir -p logs checkpoints

# ── Build mrral patch cache if not already present ────────────────────────────
CACHE_DIR="${WORKDIR}/data/patch_cache"
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7.npy" ]; then
    echo "Building mrral patch cache..."
    python -u scripts/cache_mrral_patches.py
    echo "Cache built."
fi

CKPT_SPATIAL="${WORKDIR}/checkpoints/spatial_mae_128d_6l_best.pt"

# ── Run 1: spatial_vit with pretrained MAE encoder ───────────────────────────
echo "=== Run 1: spatial_vit with spatial MAE encoder ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name spvit_spatial_mae \
    --epochs 50 \
    --patience 10 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --pretrain_ckpt "${CKPT_SPATIAL}" \
    --encoder_lr_scale 0.1

# ── Run 2: spatial_vit baseline (random init) ─────────────────────────────────
echo "=== Run 2: spatial_vit baseline (random init) ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name spvit_baseline \
    --epochs 50 \
    --patience 10 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "Fine-tuning comparison complete."
