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

# Six-class fine-tune: the existing five primary mineral classes PLUS
# alteration as a new positive output. Goal: a classifier we can scale up
# for global Mars-wide deployment that explicitly distinguishes alteration
# spectra (clays, sulfates, opal, prehnite, chlorite, etc.) from primary
# mafic minerals it currently confuses them with — especially HCP.
#
# Data sources folded in (all carry an `alteration` column post-build):
#
#   1. Patched mrral_pixels.parquet — 111,825 alteration-positive pixels
#      across 157 Argyre / Hellas / Nili gpkg polygons (90k train / 12k val
#      / 9k test). These are mineral+alteration multi-label rows (e.g.
#      "alteration + olivine" ⇒ both olivine and alteration positive in the
#      same row).
#      Run BEFORE this slurm:
#          conda run -n crism python scripts/patch_mrral_pixels_with_alteration.py
#
#   2. data/mrral_pixels_with_review.parquet — confirmed + corrected-mineral
#      review pixels. Built BEFORE this slurm via:
#          conda run -n crism python scripts/build_review_augmented_train.py \
#              --max_pixels_per_polygon 20000 \
#              --include_ambiguous \
#              --with_alteration_column \
#              --out data/mrral_pixels_with_review_v2.parquet
#      The build also folds in the 103,895 review-tagged "alteration"
#      rejects (negative_of='alteration' in hard_negatives) as positive
#      alteration examples, and the 34k ambiguous rejects as universal
#      hard negatives.
#
#   3. data/mtrdr_plag_rows.parquet — 1,817 collaborator-contributed
#      plagioclase ROI patches, fed via SyntheticPatchDataset.
#
# Encoder warm-start: ft_review_mtrdr_best.pt (yesterday's 5-class
# champion). We pass it via --init_ckpt (not --pretrain_ckpt) because
# the file is a fine-tune checkpoint with `model_state`, not an MAE
# pretrain with `encoder_state`. The shape-mismatch path in train.py
# automatically drops the 5-class head weights and reuses only the
# encoder, leaving Linear(128, 6) freshly initialized.
# lr_scale=0.001 preserves the encoder representation.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/xdisk/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/ft_review_mtrdr_best.pt"
CACHE_DIR="${WORKDIR}/data/patch_cache_with_review_v2"
PARQUET_BASE="${WORKDIR}/data/mrral_pixels.parquet"
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
for f in "$ENCODER" "$PARQUET_BASE" "$PARQUET_REVIEW" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: cache dir missing: $CACHE_DIR  (rebuild via hpc_build_cache_with_review_v2.slurm)"
    exit 1
fi

# Verify the patched + augmented parquets actually have the alteration column
python -c "
import pandas as pd
for p in ['${PARQUET_BASE}', '${PARQUET_REVIEW}']:
    df = pd.read_parquet(p)
    if 'alteration' not in df.columns:
        print(f'ERROR: {p} is missing the alteration column. '
              'Run scripts/patch_mrral_pixels_with_alteration.py and '
              'rebuild --with_alteration_column.')
        raise SystemExit(1)
    n = int((df.alteration > 0.5).sum())
    print(f'  {p}: {n:,} alteration-positive rows')
" || exit 1

python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_6cls_alteration \
    --init_ckpt "${ENCODER}" \
    --with_alteration \
    --mrral_parquets "${PARQUET_BASE}" "${PARQUET_REVIEW}" \
    --patch_cache_dir "${CACHE_DIR}" \
    --synth_train_cache "${MTRDR_PATCHES}" \
    --synth_train_parquet "${MTRDR_ROWS}" \
    --encoder_lr_scale 0.001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== ft_6cls_alteration done ==="
