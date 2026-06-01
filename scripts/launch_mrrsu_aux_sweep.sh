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

# Stagger submissions by 60s so that if slurm places multiple jobs on the same
# node, they don't all hit the memory-intensive dataset-construction phase at
# the same instant (previous batch all OOM-killed at exactly the same second).
for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "MRRSU_NORM_MODE=${mode} sbatch ${SLURM_SCRIPT}"
        continue
    fi
    echo "=== submitting ${mode} ==="
    MRRSU_NORM_MODE="${mode}" sbatch "${SLURM_SCRIPT}"
    # delay between subsequent submissions, not after the last one
    if [ "$i" -lt $(( ${#MODES[@]} - 1 )) ]; then
        echo "  sleeping 60s before next submission to avoid simultaneous placement..."
        sleep 60
    fi
done

echo "all three sweep jobs ${DRY_RUN:+would be }submitted."
echo "monitor with: squeue -u \$USER"
echo "after completion, score with: python scripts/eval_mrrsu_aux_sweep.py"
