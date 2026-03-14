#!/bin/bash
#SBATCH --job-name=spatial_mae_pretrain
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10             # 8 dataloader workers + 1 main + 1 wandb
#SBATCH --mem=48gb                     # 8 workers × ~1.3 GB + model + OS headroom
#SBATCH --time=10-00:00:00             # 10 days (max allowed)
#SBATCH --output=logs/pretrain_%j.log
#SBATCH --error=logs/pretrain_%j.log

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"    # flat tile directory on xdisk

# ── micromamba environment ─────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

# ── Machine-local config (creates if missing) ─────────────────────────────────
cd "$WORKDIR"
if [ ! -f config.local.yaml ]; then
    echo "data_root: ${DATA_ROOT}" > config.local.yaml
    echo "Created config.local.yaml with data_root=${DATA_ROOT}"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
mkdir -p logs checkpoints

python -u scripts/pretrain_spatial_mae.py \
    --epochs 400 \
    --patches_per_epoch 50000 \
    --batch_size 512 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --mask_ratio 0.75 \
    --warmup 40 \
    --num_workers 8

echo "Pre-training finished. Best checkpoint: checkpoints/spatial_mae_128d_6l_best.pt"
