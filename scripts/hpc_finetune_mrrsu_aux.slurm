#!/bin/bash
#SBATCH --job-name=crism_ft_mrrsu_aux
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_mrrsu_aux_%j.log
#SBATCH --error=logs/ft_mrrsu_aux_%j.log

# Halt on any unhandled error (build failure, missing files, etc.) so we don't
# cascade a build issue into a misleading train-time error.
set -euo pipefail

# Fine-tune the aux classifier (encoder + smoothed RPEAK1/BD1300 head) from the best
# available encoder. Ablation partner: ft_plag_aware_real_only (same encoder, no aux).
#
# Normalization-mode ablation: this script honors the MRRSU_NORM_MODE env var
# (one of zscore / minmax / pertile_zscore; default zscore for backwards
# compatibility). Each mode uses a SEPARATE aux cache dir so the three jobs in
# the sweep do not clobber each other.
#
# To launch the full 3-mode sweep, use scripts/launch_mrrsu_aux_sweep.sh.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"

NORM_MODE="${MRRSU_NORM_MODE:-zscore}"
case "$NORM_MODE" in
    zscore|minmax|pertile_zscore) ;;
    *)
        echo "ERROR: MRRSU_NORM_MODE must be zscore|minmax|pertile_zscore (got '$NORM_MODE')"
        exit 1
        ;;
esac
# Per-mode cache dir so the three sweep jobs share neither .npy files nor stats.
AUX_DIR="${WORKDIR}/data/patch_cache_${NORM_MODE}"
RUN_NAME="ft_mrrsu_aux_${NORM_MODE}"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints "$AUX_DIR"

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
EOF
fi

# Build the aux cache for this norm mode if missing.
# NB: we write to a mode-specific dir (--output_dir) so different modes coexist.
if [ ! -f "${AUX_DIR}/mrrsu_aux_train.npy" ] || [ ! -f "${AUX_DIR}/mrrsu_aux_stats.json" ]; then
    echo "=== building mrrsu aux cache (mode=${NORM_MODE}) ==="
    python -u scripts/build_mrrsu_aux.py \
        --norm_mode "${NORM_MODE}" \
        --output_dir "${AUX_DIR}"
else
    echo "=== aux cache already present at ${AUX_DIR}, skipping build ==="
fi

# Verify the build produced what training needs (catches silent build failures).
for f in mrrsu_aux_train.npy mrrsu_aux_val.npy mrrsu_aux_stats.json; do
    if [ ! -f "${AUX_DIR}/${f}" ]; then
        echo "ERROR: build did not produce ${AUX_DIR}/${f}" >&2
        echo "       check stdout above for the build step's error" >&2
        exit 2
    fi
done
echo "=== aux cache verified at ${AUX_DIR} ==="
ls -la "${AUX_DIR}"

if [ ! -f "$ENCODER" ]; then
    echo "ERROR: encoder missing: $ENCODER"; exit 1
fi

echo "=== fine-tune: run_name=${RUN_NAME}  aux_dir=${AUX_DIR} ==="

python -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "${RUN_NAME}" \
    --pretrain_ckpt "${ENCODER}" \
    --mrrsu_aux_dir "${AUX_DIR}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== mrrsu-aux fine-tune done (mode=${NORM_MODE}) ==="
