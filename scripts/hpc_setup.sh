#!/bin/bash
# One-time HPC setup script. Run this interactively on the HPC login node.
# Usage: bash hpc_setup.sh YOUR_PI_GROUP
#
# This script:
#   1. Creates the conda environment from environment.yml (or pip requirements)
#   2. Clones the repo worktree branch to /xdisk/$USER/crism_classification
#   3. Verifies the setup with a smoke test (1 epoch, 512 patches)

set -e
PI_GROUP="${1:?Usage: bash hpc_setup.sh YOUR_PI_GROUP}"
XDISK="/xdisk/${USER}"
WORKDIR="${XDISK}/crism_classification"
DATA_ROOT="${XDISK}/CRISM_MRDR"

echo "=== Setting up CRISM spatial MAE pre-training on UArizona HPC ==="
echo "  User:      $USER"
echo "  PI group:  $PI_GROUP"
echo "  Work dir:  $WORKDIR"
echo "  Data root: $DATA_ROOT"
echo ""

# ── 1. Clone the repo ─────────────────────────────────────────────────────────
module load git 2>/dev/null || true
if [ ! -d "$WORKDIR" ]; then
    git clone --branch feature/spatial-mae-pretraining \
        YOUR_GIT_REMOTE_URL \
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

# ── 3. Create conda environment ───────────────────────────────────────────────
module load cuda/12.0
module load conda

if conda env list | grep -q "^crism "; then
    echo "Conda env 'crism' already exists — skipping"
else
    echo "Creating conda env 'crism' (this takes a few minutes)..."
    conda create -n crism python=3.12 -y
    conda activate crism
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install rasterio numpy wandb pyyaml tqdm
    echo "Conda env created"
fi
conda activate crism

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "Running smoke test (2 epochs × 512 patches = ~2 minutes on HPC)..."
mkdir -p logs checkpoints

python -u scripts/pretrain_spatial_mae.py \
    --epochs 2 \
    --patches_per_epoch 512 \
    --batch_size 32 \
    --embed_dim 64 \
    --n_heads 4 \
    --n_layers 2 \
    --num_workers 4 \
    --no_wandb

echo ""
echo "=== Setup complete! Submit the full job with: ==="
echo "  # Edit hpc_pretrain.slurm: set --account=${PI_GROUP}"
echo "  sbatch scripts/hpc_pretrain.slurm"
