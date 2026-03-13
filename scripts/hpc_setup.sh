#!/bin/bash
# One-time HPC setup script. Run interactively on the HPC login node AFTER
# sourcing your .bashrc:  source ~/.bashrc
#
# Usage: bash hpc_setup.sh
#
# This script:
#   1. Clones the feature/spatial-mae-pretraining branch
#   2. Creates the micromamba 'crism' environment
#   3. Writes config.local.yaml
#   4. Runs a smoke test (2 epochs × 512 patches)

set -e

XDISK="/xdisk/${USER}"
WORKDIR="${XDISK}/crism_classification"
DATA_ROOT="${XDISK}/CRISM_MRDR"

echo "=== Setting up CRISM spatial MAE pre-training on UArizona HPC ==="
echo "  User:      $USER"
echo "  Work dir:  $WORKDIR"
echo "  Data root: $DATA_ROOT (rsync tiles here before running the full job)"
echo ""

# ── micromamba ────────────────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"

# ── 1. Clone the repo ─────────────────────────────────────────────────────────
if [ ! -d "$WORKDIR" ]; then
    git clone --branch feature/spatial-mae-pretraining \
        https://github.com/Michael-S-Phillips/crism-classification.git \
        "$WORKDIR"
    echo "Cloned repo to $WORKDIR"
else
    echo "Repo already exists at $WORKDIR — pulling latest"
    cd "$WORKDIR" && git pull
fi
cd "$WORKDIR"

# ── 2. Create config.local.yaml ───────────────────────────────────────────────
cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
echo "Created config.local.yaml"

# ── 3. Create micromamba environment ─────────────────────────────────────────
if micromamba env list | grep -q "^crism "; then
    echo "Environment 'crism' already exists — skipping"
else
    echo "Creating micromamba env 'crism' (takes a few minutes)..."
    micromamba create -n crism python=3.12 -y -c conda-forge
    micromamba activate crism
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install rasterio numpy wandb pyyaml tqdm
    echo "Environment created"
fi
micromamba activate crism

# ── 4. Smoke test (no data needed — uses first available tile) ────────────────
echo ""
echo "Running smoke test (2 epochs × 512 patches)..."
mkdir -p logs checkpoints

# Find at least one tile to verify the pipeline end-to-end
SMOKE_TILES=$(find "$DATA_ROOT" -name "*mrral*.hdr" 2>/dev/null | head -5 || true)
if [ -z "$SMOKE_TILES" ]; then
    echo "WARNING: No tiles found at $DATA_ROOT — skipping smoke test."
    echo "Transfer data first, then re-run:  bash scripts/hpc_setup.sh"
else
    python -u scripts/pretrain_spatial_mae.py \
        --epochs 2 \
        --patches_per_epoch 512 \
        --batch_size 32 \
        --embed_dim 64 \
        --n_heads 4 \
        --n_layers 2 \
        --num_workers 2 \
        --no_wandb
    echo "Smoke test passed!"
fi

echo ""
echo "=== Setup complete! Next steps: ==="
echo "  1. Transfer CRISM tiles:  rsync -avhP --include='*mrral*.img' --include='*mrral*.hdr' \\"
echo "         --include='mc*/' --exclude='*' \\"
echo "         YOUR_MACHINE:/mnt/crism/MRDR/ $DATA_ROOT/"
echo "  2. Submit the job:  sbatch scripts/hpc_pretrain.slurm"
