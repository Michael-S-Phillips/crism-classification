#!/bin/bash
#SBATCH --job-name=crism_pyx_cr
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/pyxcr_%a_%j.log
#SBATCH --error=logs/pyxcr_%a_%j.log

# pyx-merge ON the CR representation — stacks the project's TWO independent fixes
# for the LCP OOD collapse:
#   * CR (representation): continuum removal exposes weak LCP band depth
#     (ft_7cls_cr_lrscale0001 restored Nili LCP where every raw model had LCP=0).
#   * pyx (label): collapse LCP+HCP -> one clean 'pyx' class, sidestepping the
#     two-population LCP entirely (ft_6cls_pyx_lrscale001 gave a confident merged
#     mafic class on RAW, floor-test winner among the pyx arms).
# This run = ft_7cls_cr_lrscale* with --seven_class -> --pyx. Everything else
# (CR-native 256d encoder, CR labeled cache + brightness sidecar, honest splits,
# stop=val_mAP_core) is IDENTICAL, so it is a clean single-variable test of the
# label merge on top of CR. Ortho/clino (LCP<->HCP) split stays post-hoc (Spec B),
# so nothing is lost by merging at the training target.
#
# Reuses the CR run's artifacts as-is — NO cache/encoder rebuild:
#   * encoder  spatial_mae_cr_denoising_256d_6l_best.pt   (CR-pretrain winner)
#   * cache    data/patch_cache_7cls_cr  (labeled CR patches + brightness sidecar;
#              --pyx collapses lcp+hcp at label-read time, so the 7-class cache is
#              exactly what pyx needs — no rebuild).
#
# Submit:  sbatch scripts/hpc_finetune_pyx_cr.slurm
# Then floor-test each arm with: CLASSIFY_EXTRA_ARGS="--continuum_removed \
#   --brightness_aux --embed_dim 256 --pyx" bash scripts/floor_test.sh <ckpt> <tag>

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

# CR-pretrain winner: 256d beat 128d on the linear probe (val_mAP_core 0.596 vs
# 0.541, 2026-07-27) — same encoder the 7-class CR run used.
ENCODER=${CKPT_DIR}/spatial_mae_cr_denoising_256d_6l_best.pt
EMBED_DIM=256
CACHE_DIR=${WORK_DIR}/data/patch_cache_7cls_cr          # built by build_cr_labeled_cache.py
PARQUET=${WORK_DIR}/data/mrral_pixels_7cls.parquet

RUN_NAMES=("ft_6cls_pyxcr_lrscale0001" "ft_6cls_pyxcr_lrscale001" "ft_6cls_pyxcr_lrscale01")
LR_SCALES=("0.001" "0.01" "0.1")
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
LR_SCALE=${LR_SCALES[$SLURM_ARRAY_TASK_ID]}

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

for f in "$ENCODER" "$PARQUET"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7_brightness.npy" ]; then
    echo "ERROR: CR labeled cache/brightness sidecar missing at $CACHE_DIR"
    echo "Build it: python scripts/build_cr_labeled_cache.py --raw_dir data/patch_cache_7cls --out_dir $CACHE_DIR"
    exit 1
fi

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (encoder_lr_scale=$LR_SCALE, embed_dim=$EMBED_DIM, pyx+CR) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --pyx \
    --continuum_removed --cache_is_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --lr 5e-4 --encoder_lr_scale "$LR_SCALE" \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
