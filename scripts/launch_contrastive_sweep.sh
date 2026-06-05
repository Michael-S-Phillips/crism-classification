#!/usr/bin/env bash
# scripts/launch_contrastive_sweep.sh
#
# 4-condition sweep: {plag-aware MAE, denoising MAE} × {no noise, with noise}.
# Each variant runs the full contrastive pipeline (train + linear probe) with
# its own RUN_NAME / checkpoint, so they coexist on HPC without clobbering.
#
# Defaults assume the denoising-encoder source is the v3 fine-tune checkpoint
# (contains the same encoder weights under a `model_state` key — the
# ContrastiveEncoder loader extracts just the encoder portion). If a clean
# MAE-only denoising checkpoint exists at
# `$CKPT_DIR/spatial_mae_denoising_128d_6l_best.pt`, edit DENOISING_ENCODER
# below to use that path instead.
#
# Usage:
#   bash scripts/launch_contrastive_sweep.sh           # submit all 4
#   bash scripts/launch_contrastive_sweep.sh --dry_run # print commands only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/hpc_contrastive.slurm"
CKPT_DIR="/groups/sbyrne/${USER}/crism_classification/checkpoints"

PLAG_AWARE_ENCODER="${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
# Fall back to the FT checkpoint if a clean MAE-only denoising ckpt isn't there
if [ -f "${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt" ]; then
    DENOISING_ENCODER="${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt"
else
    DENOISING_ENCODER="${CKPT_DIR}/ft_v3_denoising_lrscale001_best.pt"
fi

# Sweep with longer epochs (60 vs original 30) since plag was still trending
# in the linear-probe trajectory at epoch 30.
COMMON_ENV="CONTRASTIVE_EPOCHS=60"

CONDITIONS=(
    # ENCODER_PATH       NOISE   SUFFIX
    "$PLAG_AWARE_ENCODER 0       _plag_clean"
    "$PLAG_AWARE_ENCODER 1       _plag_noise"
    "$DENOISING_ENCODER  0       _denoise_clean"
    "$DENOISING_ENCODER  1       _denoise_noise"
)

DRY_RUN=0
if [ "${1:-}" = "--dry_run" ] || [ "${1:-}" = "-n" ]; then
    DRY_RUN=1
fi

i=0
for cond in "${CONDITIONS[@]}"; do
    # parse the row — three whitespace-separated tokens
    read -r enc noise suffix <<<"$cond"
    cmd="CONTRASTIVE_ENCODER=${enc} \
CONTRASTIVE_NOISE_AUG=${noise} \
CONTRASTIVE_RUN_SUFFIX=${suffix} \
${COMMON_ENV} \
sbatch ${SLURM_SCRIPT}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$cmd"
    else
        echo "=== submitting variant: ${suffix} (noise=${noise}) ==="
        eval "$cmd"
        # stagger to avoid simultaneous IO ramp on the shared share
        if [ "$i" -lt $(( ${#CONDITIONS[@]} - 1 )) ]; then
            sleep 30
        fi
    fi
    i=$(( i + 1 ))
done

echo
echo "all 4 sweep jobs ${DRY_RUN:+would be }submitted."
echo "monitor: squeue -u \$USER"
echo "wandb runs will appear as contrastive_plag_v1{${CONDITIONS[*]##* }}_linear_probe"
