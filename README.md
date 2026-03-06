# CRISM Mineral Classification Pipeline

Multi-label pixel classification for Mars CRISM MRDR mrrsu tiles using 60 spectral parameter bands.

**Classes:** olivine_t1, olivine_t2, lcp, hcp, plagioclase, other

## Setup

```bash
# Install dependencies into existing crism conda env
conda run -n crism pip install scikit-learn xgboost lightgbm torch torchvision wandb pyarrow tqdm pyyaml pytest

# Configure W&B (one-time)
conda run -n crism python scripts/setup_wandb.py
```

## Data Preparation

```bash
# Build pixel dataset from geopackages + mrrsu tiles (~5-10 min)
conda run -n crism python scripts/build_dataset.py
```

Produces `data/pixels.parquet` — ~899k pixels × 66 columns (60 bands + labels + metadata + split).

## Training

```bash
# Linear baselines
conda run -n crism python scripts/train.py --model logreg
conda run -n crism python scripts/train.py --model svc

# Tree ensembles
conda run -n crism python scripts/train.py --model rf
conda run -n crism python scripts/train.py --model xgb
conda run -n crism python scripts/train.py --model lgbm

# Neural networks
conda run -n crism python scripts/train.py --model mlp --lr 1e-3 --epochs 100
conda run -n crism python scripts/train.py --model cnn --patch_size 7
conda run -n crism python scripts/train.py --model vit --embed_dim 128

# Skip W&B logging
conda run -n crism python scripts/train.py --model rf --no_wandb
```

## Hyperparameter Sweeps (W&B)

```bash
conda run -n crism wandb sweep config/sweep_mlp.yaml
conda run -n crism wandb agent <sweep_id>
```

## Inference

```bash
conda run -n crism python scripts/predict_tile.py \
    --tile_id t0503 \
    --model logreg \
    --checkpoint checkpoints/logreg_model.pkl
```

Outputs: `predictions/t0503/{class}_prob.tif` + `best_class.tif`

## Tests

```bash
conda run -n crism python -m pytest tests/ -v
```

## Confidence Tiers

Labels carry High/Moderate/Low confidence. Training uses sample weights (1.0/0.5/0.25). Test metrics are reported separately per tier.

## Label Encoding

Mixed-mineral labels (e.g. `hcp + olivine (High)`) produce multi-hot vectors. Untyped "olivine" splits 0.5/0.5 between olivine_t1 and olivine_t2. "Other" is a spectral denominator class (downsampled to ~400 polygons).
