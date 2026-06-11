#!/bin/bash
#SBATCH --job-name=crism_ft_6cls_alt
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=18:00:00
#SBATCH --output=logs/ft_6cls_alt_%j.log
#SBATCH --error=logs/ft_6cls_alt_%j.log

# Six-class fine-tune (v2, corrected): the five primary mineral classes PLUS
# alteration as a new positive output. Goal: a classifier we can scale up
# for global Mars-wide deployment that explicitly distinguishes alteration
# spectra (clays, sulfates, opal, prehnite, chlorite, etc.) from primary
# mafic minerals it currently confuses them with — especially HCP.
#
# CORRECTED per the 2026-06-11 audit (wiki: Project Audit 2026-06-11):
#   - Trains on the v2 parquet ALONE. v2 is a self-contained replacement
#     (capped base + review rows); the earlier base+v2 concat double-counted
#     every base row and defeated the per-polygon cap (bug #2).
#   - data/dataset.py now preserves the stamped confidence weights for
#     Reviewed (2.0) and Ambiguous (3.0) tiers instead of clobbering them
#     to 0.85 (bug #1). This is the first run where the review weighting
#     actually takes effect.
#
# Prerequisites (in order):
#   1. data/mrral_pixels.parquet patched with the alteration column:
#          python scripts/patch_mrral_pixels_with_alteration.py
#   2. v2 parquet built (rebuild after pulling the NaN-alteration fix):
#          sbatch scripts/hpc_build_review_v2.slurm
#   3. v2-only patch cache:
#          sbatch scripts/hpc_build_cache_v2_only.slurm
#
# Encoder warm-start: ft_review_mtrdr_best.pt (the verified 0.7644 5-class
# champion). Passed via --init_ckpt (it's a fine-tune checkpoint with
# `model_state`, not an MAE ckpt with `encoder_state`); train.py drops the
# 5-class head on shape mismatch and reuses only the encoder, leaving
# Linear(128, 6) freshly initialized. lr_scale=0.001 preserves the encoder.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/ft_review_mtrdr_best.pt"
CACHE_DIR="${WORKDIR}/data/patch_cache_v2_only"
PARQUET_REVIEW="${WORKDIR}/data/mrral_pixels_with_review_v2.parquet"
MTRDR_PATCHES="${WORKDIR}/data/contrastive/extra_plag_roi/patches.npy"
MTRDR_ROWS="${WORKDIR}/data/mtrdr_plag_rows.parquet"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs checkpoints

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${WORKDIR}/data/patch_cache
EOF
fi

# Sanity checks
for f in "$ENCODER" "$PARQUET_REVIEW" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: cache dir missing: $CACHE_DIR  (build via hpc_build_cache_v2_only.slurm)"
    exit 1
fi

# Verify the v2 parquet has the alteration column populated, no NaNs, and
# that the review confidence weights survived the build.
python -c "
import pandas as pd
df = pd.read_parquet('${PARQUET_REVIEW}',
                     columns=['alteration', 'confidence_tier',
                              'confidence_weight'])
n_alt = int((df.alteration > 0.5).sum())
n_nan = int(df.alteration.isna().sum())
print(f'  alteration positives: {n_alt:,}   NaNs: {n_nan:,}')
if n_alt == 0:
    print('ERROR: no alteration positives — run the patch script + rebuild v2.')
    raise SystemExit(1)
if n_nan:
    print('ERROR: NaNs in alteration column — rebuild v2 with the fixed build script.')
    raise SystemExit(1)
rev = df[df.confidence_tier == 'Reviewed']
if len(rev) and abs(float(rev.confidence_weight.iloc[0]) - 2.0) > 1e-6:
    print('ERROR: Reviewed rows do not carry weight 2.0 — stale v2 parquet?')
    raise SystemExit(1)
print(f'  Reviewed rows: {len(rev):,} at weight 2.0 — OK')
" || exit 1

python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_6cls_alteration_v2 \
    --init_ckpt "${ENCODER}" \
    --with_alteration \
    --mrral_parquets "${PARQUET_REVIEW}" \
    --patch_cache_dir "${CACHE_DIR}" \
    --synth_train_cache "${MTRDR_PATCHES}" \
    --synth_train_parquet "${MTRDR_ROWS}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== ft_6cls_alteration_v2 done ==="
