#!/bin/bash
#SBATCH --job-name=crism_ft_relab_mtrdr
#SBATCH --partition=gpu_standard
#SBATCH --account=sbyrne
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_relab_mtrdr_%j.log
#SBATCH --error=logs/ft_relab_mtrdr_%j.log

# Champion recipe with the LOWER encoder_lr_scale (0.0001), olivine relabels
# applied as soft-pyroxene targets, AND new MTRDR-derived plagioclase training
# patches added as a synth-style cache.
#
# Ablation partners (all on the SAME val labels for honest comparison via
# scripts/eval_on_corrected_val.py):
#   ft_plag_aware_relabeled        relabels only         (encoder 0.001)
#   ft_plag_aware_real_only (lower) lower encoder lr     (no relabels)
#   ft_plag_aware_relabeled_mtrdr  this run             (lower + relabels + MTRDR plag)
#
# Goal: push plagioclase AP up via the +8,671 real-MTRDR plag pixels across 18
# scenes, without sacrificing HCP (relabels keep olivine-only contamination
# corrected).

WORKDIR="/groups/sbyrne/${USER}/crism_classification"
DATA_ROOT="/groups/sbyrne/${USER}/CRISM_MRDR"
CKPT_DIR="${WORKDIR}/checkpoints"
ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
RELABELS="${WORKDIR}/data/olivine_relabels.csv"
MTRDR_NPY="${WORKDIR}/data/patch_cache/mtrdr_plag_patches_p7.npy"
MTRDR_PARQUET="${WORKDIR}/data/patch_cache/mtrdr_plag_rows.parquet"

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

for f in "$ENCODER" "$RELABELS" "$MTRDR_NPY" "$MTRDR_PARQUET"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done

python -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_plag_aware_relabeled_mtrdr \
    --pretrain_ckpt "${ENCODER}" \
    --apply_relabels "${RELABELS}" \
    --synth_train_cache "${MTRDR_NPY}" \
    --synth_train_parquet "${MTRDR_PARQUET}" \
    --encoder_lr_scale 0.0001 \
    --epochs 100 --patience 25 \
    --batch_size 256 --lr 5e-4 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== relabeled + MTRDR fine-tune done ==="
