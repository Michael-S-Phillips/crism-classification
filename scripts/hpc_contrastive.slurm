#!/bin/bash
#SBATCH --job-name=crism_contrastive
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=18:00:00
#SBATCH --output=logs/contrastive_%j.log
#SBATCH --error=logs/contrastive_%j.log

# Contrastive plag-vs-olivine encoder refinement (Task D).
# - Warm-starts from the plag-aware MAE encoder.
# - Pre-builds the three patch pools if missing.
# - Trains InfoNCE with weighted hard/soft negatives.
# - Runs a linear probe at the end and prints per-class AP.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
POOL_DIR="${WORKDIR}/data/contrastive"
RUN_NAME="contrastive_plag_v1"
LABELED_GPKG_DIR="${DATA_ROOT}/categorized_mineral_units"
MC13_PLAG_GPKG="${WORKDIR}/data/vector_mc13_relabeled/plagioclase.gpkg"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints "$POOL_DIR"

if [ ! -f "$ENCODER" ]; then
    echo "ERROR: pretrain encoder missing: $ENCODER"
    exit 1
fi
USE_MC13=1
if [ ! -f "$MC13_PLAG_GPKG" ]; then
    echo "WARNING: MC13 plag gpkg missing: $MC13_PLAG_GPKG"
    echo "         proceeding without MC13 polygon hard-negatives — SAM parquets"
    echo "         (if present) will be the sole hard-negative source"
    USE_MC13=0
fi

# ---------- pre-build patch pools if missing ----------
NEED_POOLS=0
for sub in hard_negatives positives soft_negatives; do
    if [ ! -f "${POOL_DIR}/${sub}/patches.npy" ]; then
        NEED_POOLS=1
    fi
done
if [ "$NEED_POOLS" -eq 1 ]; then
    echo "=== building contrastive patch pools ==="
    # SAM-flagged hard-negative parquets from the Argyre Task B run.
    # Skip silently if they are not present on HPC.
    SAM_PARQUETS=()
    for p in \
        "${WORKDIR}/sam_analysis/outputs/argyre/t0434_hard_negatives_mrdr.parquet" \
        "${WORKDIR}/sam_analysis/outputs/argyre/t0435_hard_negatives_mrdr.parquet"; do
        if [ -f "$p" ]; then SAM_PARQUETS+=("$p"); fi
    done
    # Optional extra positive pool: sensor-space ROI plag patches from
    # scripts/build_plag_roi_patches.py. Auto-detected if present.
    EXTRA_POS_DIRS=()
    for d in \
        "${WORKDIR}/data/contrastive/extra_plag_roi"; do
        if [ -f "${d}/patches.npy" ] && [ -f "${d}/meta.parquet" ]; then
            EXTRA_POS_DIRS+=("$d")
        fi
    done

    BUILD_ARGS=(
        --labeled_gpkg_dir "$LABELED_GPKG_DIR"
        --output_dir "$POOL_DIR"
        --max_per_polygon 200
    )
    if [ "$USE_MC13" -eq 1 ]; then
        BUILD_ARGS+=(--mc13_plag_gpkg "$MC13_PLAG_GPKG" --mc13_threshold_layer thresh_0.92)
    else
        # No MC13 gpkg: tell the build script to skip the MC13 polygon harvest.
        BUILD_ARGS+=(--skip_hard_mc13)
    fi
    if [ ${#SAM_PARQUETS[@]} -gt 0 ]; then
        echo "  including ${#SAM_PARQUETS[@]} SAM hard-negative parquet(s)"
        BUILD_ARGS+=(--sam_hard_negative_parquets "${SAM_PARQUETS[@]}")
    else
        echo "  no SAM parquets found, using MC13 polygons only for hard negatives"
    fi
    if [ ${#EXTRA_POS_DIRS[@]} -gt 0 ]; then
        echo "  including ${#EXTRA_POS_DIRS[@]} extra positive pool dir(s) (ROI patches)"
        BUILD_ARGS+=(--extra_positive_pool_dirs "${EXTRA_POS_DIRS[@]}")
    fi
    python -u scripts/build_contrastive_data.py "${BUILD_ARGS[@]}"
else
    echo "=== patch pools already present, skipping build ==="
fi

# ---------- train ----------
echo "=== contrastive training ==="
python -u scripts/train_contrastive.py \
    --pretrain_ckpt "$ENCODER" \
    --pool_dir "$POOL_DIR" \
    --output_dir "$CKPT_DIR" \
    --run_name "$RUN_NAME" \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --encoder_lr_scale 0.01 \
    --tau 0.07 \
    --hard_weight 2.0 \
    --soft_weight 1.0 \
    --n_hard_per_batch 8 \
    --n_soft_per_batch 8 \
    --proj_dim 64 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --num_workers 4

# ---------- linear probe (fast) ----------
echo "=== linear probe ==="
PROBE_EXTRA_ARGS=()
if [ ${#EXTRA_POS_DIRS[@]:-0} -gt 0 ]; then
    PROBE_EXTRA_ARGS+=(--extra_positive_pool_dirs "${EXTRA_POS_DIRS[@]}")
fi
# (Re-discover the pool dirs in case the build was skipped and the variable
# wasn't set above.)
if [ ${#PROBE_EXTRA_ARGS[@]} -eq 0 ]; then
    PROBE_EXTRA=()
    for d in "${WORKDIR}/data/contrastive/extra_plag_roi"; do
        if [ -f "${d}/patches.npy" ] && [ -f "${d}/meta.parquet" ]; then
            PROBE_EXTRA+=("$d")
        fi
    done
    if [ ${#PROBE_EXTRA[@]} -gt 0 ]; then
        PROBE_EXTRA_ARGS+=(--extra_positive_pool_dirs "${PROBE_EXTRA[@]}")
        echo "  probe: including ${#PROBE_EXTRA[@]} extra positive pool dir(s)"
    fi
fi
python -u scripts/eval_contrastive.py \
    --ckpt "${CKPT_DIR}/${RUN_NAME}_best.pt" \
    --mode linear \
    --probe_epochs 5 \
    --batch_size 256 \
    "${PROBE_EXTRA_ARGS[@]}"

echo "=== contrastive pipeline done ==="
