#!/bin/bash
#SBATCH --job-name=crism_finetune_bland_lr0001_cont1
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_bland_lr0001_cont1_%j.log
#SBATCH --error=logs/finetune_bland_lr0001_cont1_%j.log

# Continuation of the three ft_bland_*_lrscale0001 runs.
# Wandb traces showed the lr0001 variants still had downward loss slope at
# epoch 100. This script warm-starts each from its prior `_best.pt` and runs
# another 100 epochs with the same recipe + asl loss + encoder_lr_scale 0.001.
#
# Mechanics:
#   - --init_ckpt loads the FULL classifier state (encoder + head) from the
#     finetune _best.pt. (Distinct from --pretrain_ckpt, which loads only the
#     encoder from an MAE checkpoint.) See scripts/train.py.
#   - Optimizer + scheduler restart from scratch; we don't have the prior
#     optimizer state saved. lr_t_max=50 (cosine annealing) keeps the schedule
#     gentle so the new run doesn't blow away the learned weights.
#   - Same hyperparams as the original run otherwise. patience=25 so we
#     early-stop if improvement has truly stalled.
#
# Per-run budget ~5 hrs; 3 runs + buffer → 24-hr wall time.

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"

CKPT_V3_PRIOR="${CKPT_DIR}/ft_bland_v3_lrscale0001_best.pt"
CKPT_V4_PRIOR="${CKPT_DIR}/ft_bland_v4_lrscale0001_best.pt"
CKPT_MARCH_PRIOR="${CKPT_DIR}/ft_bland_marchmae_lrscale0001_best.pt"

# ── micromamba environment ─────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"

# ── Ensure config.local.yaml exists (created by hpc_setup.sh) ────────────────
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

# ── Verify prior checkpoints exist ───────────────────────────────────────────
for ckpt in "$CKPT_V3_PRIOR" "$CKPT_V4_PRIOR" "$CKPT_MARCH_PRIOR"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: prior checkpoint missing: $ckpt"
        echo "       Make sure the ft_bland_v1 sweep produced all three lr0001 best.pt files."
        exit 1
    fi
done

# ── Run 1/3: v3 denoising encoder, lr_scale 0.001 (continuation) ────────────
echo "=== Run 1/3: ft_bland_v3_lrscale0001_cont1 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_bland_v3_lrscale0001_cont1 \
    --init_ckpt "${CKPT_V3_PRIOR}" \
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

# ── Run 2/3: v4 SPEND encoder, lr_scale 0.001 (continuation) ─────────────────
echo "=== Run 2/3: ft_bland_v4_lrscale0001_cont1 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_bland_v4_lrscale0001_cont1 \
    --init_ckpt "${CKPT_V4_PRIOR}" \
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

# ── Run 3/3: March plain MAE encoder, lr_scale 0.001 (continuation) ──────────
echo "=== Run 3/3: ft_bland_marchmae_lrscale0001_cont1 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_bland_marchmae_lrscale0001_cont1 \
    --init_ckpt "${CKPT_MARCH_PRIOR}" \
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

echo "=== All 3 lr0001 continuation runs complete ==="
