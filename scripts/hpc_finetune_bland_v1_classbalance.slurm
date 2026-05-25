#!/bin/bash
#SBATCH --job-name=crism_finetune_bland_classbal
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_bland_classbal_%j.log
#SBATCH --error=logs/finetune_bland_classbal_%j.log

# Class-balanced sweep, warm-started from the bland-v3 lr0001 cont1 champion.
#
# Background:
#   - bland-v3 lr0001 cont1 reached val_mAP=0.7262 with strong olivine/lcp/hcp/other
#     performance but the plagioclase head is suppressed (val_AP~0.09; almost
#     no polygons predicted plagioclase on Argyre/Nili).
#   - The encoder *can* discriminate plagioclase (a few pixels score 0.96+);
#     the head has learned to be ultra-conservative because the optimizer
#     down-weighted plagioclase given its lowest training count + lowest val_AP.
#   - Class weights re-balance the BCE/ASL loss to push the head to use the
#     existing plagioclase-discriminative features more aggressively.
#
# Class-weight ordering (LABEL_COLS): olivine, lcp, hcp, plagioclase, other.
# Three configs vary only the plagioclase weight {2, 3, 5}; "other" is
# down-weighted to 0.7 to break the over-confident-other tendency we kept
# seeing in inference.
#
# Each run is a warm-start (--init_ckpt) from cont1 and trains for up to 100
# more epochs with the same recipe (lr 5e-4, encoder_lr_scale 0.001, ASL loss,
# patience 25). 3 runs × ~5 hrs ≈ 15 hrs; 24-hr wall-time has headroom.

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"

CKPT_INIT="${CKPT_DIR}/ft_bland_v3_lrscale0001_cont1_best.pt"

# ── micromamba environment ─────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"

# ── Ensure config.local.yaml exists ──────────────────────────────────────────
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

# ── Verify warm-start checkpoint exists ──────────────────────────────────────
if [ ! -f "$CKPT_INIT" ]; then
    echo "ERROR: warm-start checkpoint missing: $CKPT_INIT"
    echo "       Make sure ft_bland_v3_lrscale0001_cont1_best.pt is on HPC."
    exit 1
fi

# ── Run 1/3: plagioclase weight = 2 ──────────────────────────────────────────
echo "=== Run 1/3: ft_bland_v3_cont1_classbal_plag2_oth07 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_bland_v3_cont1_classbal_plag2_oth07 \
    --init_ckpt "${CKPT_INIT}" \
    --class_weights "1.0,1.0,1.5,2.0,0.7" \
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

# ── Run 2/3: plagioclase weight = 3 ──────────────────────────────────────────
echo "=== Run 2/3: ft_bland_v3_cont1_classbal_plag3_oth07 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_bland_v3_cont1_classbal_plag3_oth07 \
    --init_ckpt "${CKPT_INIT}" \
    --class_weights "1.0,1.0,1.5,3.0,0.7" \
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

# ── Run 3/3: plagioclase weight = 5 ──────────────────────────────────────────
echo "=== Run 3/3: ft_bland_v3_cont1_classbal_plag5_oth07 ==="
python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_bland_v3_cont1_classbal_plag5_oth07 \
    --init_ckpt "${CKPT_INIT}" \
    --class_weights "1.0,1.0,1.5,5.0,0.7" \
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

echo "=== All 3 class-balanced runs complete ==="
