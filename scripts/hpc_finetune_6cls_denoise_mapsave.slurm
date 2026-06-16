#!/bin/bash
#SBATCH --job-name=crism_6cls_mapsave
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/6cls_denoise_mapsave_%j.log
#SBATCH --error=logs/6cls_denoise_mapsave_%j.log

# Denoise-backbone 6-class FT, explicitly tracking val_mAP as the stop metric
# so the saved best.pt is unambiguously the best primary-mineral epoch.
#
# Background: the mc11val_denoise run (2026-06-15) happened to also use
# val_mAP because --stop_metric was not wired in the spatial_vit branch
# (now fixed). Its best.pt is therefore already the epoch-10 peak
# (val_mAP=0.8138, plag=0.2425, HCP=0.830). This slurm is an explicit
# re-run if that checkpoint is unavailable or you want a fresh seed.
#
# Settings identical to ft_6cls_mc11val_denoise except:
#   --stop_metric val_mAP  (explicit; now correctly wired)
#   --patience 30          (epoch-10 peak → stops ~epoch 40)
#
# Also benefits from dual-checkpointing (committed alongside this slurm):
#   {run_name}_best.pt      <- best val_mAP epoch (the primary objective)
#   {run_name}_best_map.pt  <- only written when stop_metric != val_mAP (N/A here)
#   {run_name}_last.pt      <- final epoch weights

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
ENCODER=${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache_mc11val
PARQUET=${WORK_DIR}/data/mrral_pixels_with_review_mc11val.parquet
MTRDR_PATCHES=${WORK_DIR}/data/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${WORK_DIR}/data/mtrdr_plag_rows.parquet

PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"
mkdir -p logs checkpoints

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

for f in "$ENCODER" "$PARQUET" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: cache dir missing: $CACHE_DIR"
    exit 1
fi

$PYTHON -c "
import pandas as pd
df = pd.read_parquet('${PARQUET}', columns=['split','alteration'])
v = df[df.split=='val']; n = int((v.alteration>0.5).sum())
print(f'  val alteration positives: {n:,} (prevalence {n/len(v):.3f})')
if n < 5000:
    print('ERROR: clean MC11 val alteration too small'); raise SystemExit(1)
" || exit 1

echo "=== ft_6cls_denoise_mapsave: stop=val_mAP, patience=30, lr=5e-4 cosine, enc_scale=0.01 ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_6cls_denoise_mapsave \
    --pretrain_ckpt "$ENCODER" \
    --with_alteration \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    --lr 5e-4 --encoder_lr_scale 0.01 \
    --stop_metric val_mAP \
    --epochs 300 --patience 30 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== ft_6cls_denoise_mapsave done ==="
