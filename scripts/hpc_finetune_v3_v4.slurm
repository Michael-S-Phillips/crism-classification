#!/bin/bash
#SBATCH --job-name=crism_finetune_v3_v4
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/finetune_v3_v4_%j.log
#SBATCH --error=logs/finetune_v3_v4_%j.log

# Fine-tune sweep: denoising MAE, SPEND MAE, and March plain MAE encoders.
# Goal: disentangle whether downstream classification deficits come from the
# pretraining objective or from the labeled dataset getting harder
# (sup-GPKG and Hellas additions changed the val split since the March
# champion ran).
#
# 9 runs: 3 encoders × 3 lr_scales {0.001, 0.01, 0.1}
# Per-run budget ~5 hrs at --epochs 100 --patience 25; 48-hr wall time
# accommodates the full 9-run sweep with margin.

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"

CKPT_V3="${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt"
CKPT_V4="${CKPT_DIR}/spatial_mae_spend_128d_6l_best.pt"
CKPT_MARCH="${CKPT_DIR}/spatial_mae_128d_6l_march_best.pt"

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
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
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

# ── Run 1/9: v3 denoising encoder, lr_scale 0.001 ────────────────────────────
echo "=== Run 1/9: ft_v3_denoising_lrscale0001 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_v3_denoising_lrscale0001 \
    --pretrain_ckpt "${CKPT_V3}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 2/9: v3 denoising encoder, lr_scale 0.01 ─────────────────────────────
echo "=== Run 2/9: ft_v3_denoising_lrscale001 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_v3_denoising_lrscale001 \
    --pretrain_ckpt "${CKPT_V3}" \
    --encoder_lr_scale 0.01 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 3/9: v3 denoising encoder, lr_scale 0.1 ──────────────────────────────
echo "=== Run 3/9: ft_v3_denoising_lrscale01 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_v3_denoising_lrscale01 \
    --pretrain_ckpt "${CKPT_V3}" \
    --encoder_lr_scale 0.1 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 4/9: v4 SPEND encoder, lr_scale 0.001 ────────────────────────────────
echo "=== Run 4/9: ft_v4_spend_lrscale0001 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_v4_spend_lrscale0001 \
    --pretrain_ckpt "${CKPT_V4}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 5/9: v4 SPEND encoder, lr_scale 0.01 ─────────────────────────────────
echo "=== Run 5/9: ft_v4_spend_lrscale001 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_v4_spend_lrscale001 \
    --pretrain_ckpt "${CKPT_V4}" \
    --encoder_lr_scale 0.01 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 6/9: v4 SPEND encoder, lr_scale 0.1 ──────────────────────────────────
echo "=== Run 6/9: ft_v4_spend_lrscale01 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_v4_spend_lrscale01 \
    --pretrain_ckpt "${CKPT_V4}" \
    --encoder_lr_scale 0.1 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 7/9: March plain MAE encoder, lr_scale 0.001 ─────────────────────────
echo "=== Run 7/9: ft_marchmae_lrscale0001 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_marchmae_lrscale0001 \
    --pretrain_ckpt "${CKPT_MARCH}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 8/9: March plain MAE encoder, lr_scale 0.01 ──────────────────────────
echo "=== Run 8/9: ft_marchmae_lrscale001 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_marchmae_lrscale001 \
    --pretrain_ckpt "${CKPT_MARCH}" \
    --encoder_lr_scale 0.01 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

# ── Run 9/9: March plain MAE encoder, lr_scale 0.1 ───────────────────────────
echo "=== Run 9/9: ft_marchmae_lrscale01 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_marchmae_lrscale01 \
    --pretrain_ckpt "${CKPT_MARCH}" \
    --encoder_lr_scale 0.1 \
    --epochs 100 \
    --patience 25 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== All 9 fine-tune runs complete ==="
