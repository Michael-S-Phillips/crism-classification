#!/bin/bash
# scripts/launch_mrrsu_aux_sweep.sh
#
# Sweep launcher for the mrrsu-aux normalization ablation. sbatches the
# fine-tune slurm script 3 times with different MRRSU_NORM_MODE env vars
# (zscore / minmax / pertile_zscore), each writing to its own cache dir and
# wandb run name (ft_mrrsu_aux_{mode}).
#
# Usage:
#   bash scripts/launch_mrrsu_aux_sweep.sh           # submit all three
#   bash scripts/launch_mrrsu_aux_sweep.sh --dry_run # print sbatch commands only
#
# After all three runs finish and checkpoints sync back locally, run
# scripts/eval_mrrsu_aux_sweep.py to score them on the corrected val split.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/hpc_finetune_mrrsu_aux.slurm"

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: slurm script not found: $SLURM_SCRIPT" >&2
    exit 1
fi

DRY_RUN=0
if [ "${1:-}" = "--dry_run" ] || [ "${1:-}" = "-n" ]; then
    DRY_RUN=1
fi

MODES=(zscore minmax pertile_zscore)

for mode in "${MODES[@]}"; do
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "MRRSU_NORM_MODE=${mode} sbatch ${SLURM_SCRIPT}"
    else
        echo "=== submitting ${mode} ==="
        MRRSU_NORM_MODE="${mode}" sbatch "${SLURM_SCRIPT}"
    fi
done

echo "all three sweep jobs ${DRY_RUN:+would be }submitted."
echo "monitor with: squeue -u \$USER"
echo "after completion, score with: python scripts/eval_mrrsu_aux_sweep.py"
