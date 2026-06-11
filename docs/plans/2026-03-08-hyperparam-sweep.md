# Hyperparameter Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Assess baseline results, fix two root causes of premature early stopping (CNN dropout, class imbalance in loss), then run a structured hyperparameter sweep across all model families.

**Architecture:** Three parallel improvements feed into a single `scripts/sweep.py` that trains each config sequentially, logs to wandb with unique run names, and saves named checkpoints (e.g. `cnn_sweep3_best.pt`). The sweep is restartable — it skips configs whose checkpoint already exists.

**Tech Stack:** PyTorch, scikit-learn, XGBoost, LightGBM, wandb, numpy memmap cache

---

## Baseline Assessment (already done — reference only)

| Model | val_mAP | Worst classes |
|-------|---------|---------------|
| logreg | 0.560 | plagioclase=0.274, other=0.337 |
| svc | 0.558 | plagioclase=0.246, other=0.384 |
| rf | 0.608 | plagioclase=0.287, hcp=0.435 |
| xgb | 0.609 | plagioclase=0.264, hcp=0.497 |
| lgbm | 0.616 | plagioclase=0.272, hcp=0.488 |
| mlp | 0.613 | hcp=0.421, plagioclase=0.364 |
| cnn | 0.636 | stopped epoch 3 (no dropout!) |
| vit | 0.634 | stopped epoch 1 (low dropout=0.1) |

**Root causes:**
1. `SpectralSpatialCNN` has no dropout → overfits in 3 epochs → early stopping terminates the run
2. `WeightedBCEWithLogitsLoss` has no class weighting → underperforms on rare classes (hcp, plagioclase)
3. Sklearn sweep only explored one configuration per model family

---

## Task 1: Add dropout to SpectralSpatialCNN

**Files:**
- Modify: `models/cnn.py`

**Step 1: Read current implementation**

```bash
cat models/cnn.py
```

Confirm: no dropout parameter, no Dropout layers.

**Step 2: Write failing test**

Add to `tests/test_models.py` (create if it doesn't exist):

```python
def test_cnn_dropout_parameter():
    """CNN should accept a dropout parameter and apply it."""
    import torch
    from models.cnn import SpectralSpatialCNN
    model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=7, dropout=0.5)
    model.train()
    x = torch.randn(4, 60, 7, 7)
    out1 = model(x)
    out2 = model(x)
    # With dropout, outputs should differ in train mode
    assert not torch.allclose(out1, out2), "Dropout should cause stochastic outputs in train mode"
    model.eval()
    out3 = model(x)
    out4 = model(x)
    # In eval mode, outputs should be identical
    assert torch.allclose(out3, out4), "No dropout in eval mode"
```

**Step 3: Run test to verify it fails**

```bash
conda run -n crism python -m pytest tests/test_models.py::test_cnn_dropout_parameter -v
```
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'dropout'`

**Step 4: Update models/cnn.py**

Replace the entire file:

```python
import torch
import torch.nn as nn


class SpectralSpatialCNN(nn.Module):
    """
    2D CNN over a (patch_size x patch_size x n_bands) spatial-spectral patch.
    Input shape: (batch, n_bands, patch_size, patch_size)
    Returns logits of shape (batch, n_classes).
    """

    def __init__(
        self,
        n_bands: int = 60,
        n_classes: int = 6,
        patch_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
```

**Step 5: Run test to verify it passes**

```bash
conda run -n crism python -m pytest tests/test_models.py::test_cnn_dropout_parameter -v
```
Expected: PASS

**Step 6: Also run existing CNN test to ensure no regression**

```bash
conda run -n crism python -m pytest tests/test_train_torch.py::test_cnn_trains_with_cache -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add models/cnn.py tests/test_models.py
git commit -m "feat: add dropout to SpectralSpatialCNN to prevent early overfitting"
```

---

## Task 2: Add pos_weight class weighting to the loss

**Files:**
- Modify: `training/losses.py`
- Modify: `training/train_torch.py`

**Background:** HCP and plagioclase are rare. `pos_weight[c] = n_neg[c] / n_pos[c]` tells BCEWithLogitsLoss to upweight positive examples for rare classes. This is standard practice for imbalanced multi-label problems.

**Step 1: Write failing test**

Add to `tests/test_losses.py` (create if it doesn't exist):

```python
import torch
from training.losses import WeightedBCEWithLogitsLoss

def test_loss_accepts_pos_weight():
    """Loss should accept optional pos_weight tensor."""
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.randn(8, 6)
    targets = torch.randint(0, 2, (8, 6)).float()
    weights = torch.ones(8)
    pos_weight = torch.tensor([1.0, 1.0, 1.0, 4.0, 6.0, 2.0])  # upweight rare classes
    # Should not raise
    loss_val = loss_fn(logits, targets, weights, pos_weight=pos_weight)
    assert loss_val.item() > 0

def test_pos_weight_increases_loss_for_rare_class():
    """pos_weight should increase loss when rare class positive is missed."""
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.full((4, 6), -3.0)   # predicts all negative
    targets = torch.zeros(4, 6)
    targets[:, 4] = 1.0                  # class 4 is positive
    weights = torch.ones(4)
    loss_no_pw = loss_fn(logits, targets, weights)
    pos_weight = torch.ones(6)
    pos_weight[4] = 10.0
    loss_with_pw = loss_fn(logits, targets, weights, pos_weight=pos_weight)
    assert loss_with_pw > loss_no_pw
```

**Step 2: Run tests to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_losses.py -v
```
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'pos_weight'`

**Step 3: Update training/losses.py**

```python
import torch
import torch.nn as nn
from typing import Optional


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, weighted per sample by confidence weight.
    Averages over classes first, then takes confidence-weighted mean over samples.

    Optional pos_weight: (n_classes,) tensor of positive class weights.
    pos_weight[c] = n_neg[c] / n_pos[c] upweights rare classes.
    """

    def forward(
        self,
        logits: torch.Tensor,            # (batch, n_classes)
        targets: torch.Tensor,           # (batch, n_classes)
        weights: torch.Tensor,           # (batch,)
        pos_weight: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        # Per-sample, per-class BCE: shape (batch, n_classes)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
        # Mean over classes: shape (batch,)
        bce_per_sample = bce.mean(dim=1)
        # Weighted mean over samples
        return (bce_per_sample * weights).sum() / (weights.sum() + 1e-8)
```

**Step 4: Update training/train_torch.py to compute and use pos_weight**

In `train_torch_model`, add `use_pos_weight: bool = False` parameter, then compute pos_weight from training labels and pass to the loss.

Add after `use_patches = mrrsu_map is not None`:

```python
    # Compute pos_weight from training label prevalence
    pos_weight = None
    if use_pos_weight:
        train_sub = df[df['split'] == 'train']
        from data.dataset import LABEL_COLS
        y_tr = train_sub[LABEL_COLS].values.astype('float32')
        n_pos = (y_tr > 0.4).sum(axis=0).clip(min=1)
        n_neg = len(y_tr) - n_pos
        pw = (n_neg / n_pos).clip(max=20.0)   # cap at 20x to prevent instability
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)
```

And update the loss call in the training loop:
```python
loss = loss_fn(logits, labels, weights, pos_weight=pos_weight)
```

Also add `use_pos_weight: bool = False` to the `train_torch_model` signature.

**Step 5: Run loss tests**

```bash
conda run -n crism python -m pytest tests/test_losses.py -v
```
Expected: 2 PASSED

**Step 6: Run full train_torch test to check no regression**

```bash
conda run -n crism python -m pytest tests/test_train_torch.py -v
```
Expected: All PASSED

**Step 7: Commit**

```bash
git add training/losses.py training/train_torch.py tests/test_losses.py
git commit -m "feat: add optional pos_weight class weighting to loss for rare class uplift"
```

---

## Task 3: Extend scripts/train.py with new hyperparameter arguments

**Files:**
- Modify: `scripts/train.py`

**Background:** The sweep script will call `train.py` with different hyperparams. We need CLI args for: `--dropout` (CNN/MLP), `--hidden_dims` (MLP), `--use_pos_weight` (all torch), `--num_leaves` (lgbm), `--subsample` (lgbm/xgb), `--weight_decay` (torch), `--run_name` (custom wandb name for sweep runs), `--num_workers` (DataLoader).

**Step 1: No test needed** — this is pure CLI extension, integration-tested by sweep.py

**Step 2: Add arguments to scripts/train.py**

In the `argparse` block, add after existing args:

```python
    # Sweep / architecture kwargs
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate for CNN/MLP (default: model default)')
    parser.add_argument('--hidden_dims', type=str, default=None,
                        help='MLP hidden dims as comma-separated ints, e.g. 512,256,128')
    parser.add_argument('--use_pos_weight', action='store_true',
                        help='Use pos_weight in loss to upweight rare classes')
    parser.add_argument('--num_leaves', type=int, default=31,
                        help='LightGBM num_leaves')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Row subsample ratio for XGB/LGBM')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay for torch models')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom wandb run name (default: model name)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers')
```

**Step 3: Thread new args through model construction and training**

For MLP:
```python
            hidden_dims = tuple(int(x) for x in args.hidden_dims.split(',')) \
                if args.hidden_dims else (256, 128)
            dropout = args.dropout if args.dropout is not None else 0.3
            model = MLP(n_features=60, n_classes=6,
                        hidden_dims=hidden_dims, dropout=dropout)
            metrics = train_torch_model(
                model=model, df=df,
                model_name=args.run_name or 'mlp',
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
            )
```

For CNN:
```python
            dropout = args.dropout if args.dropout is not None else 0.3
            model = SpectralSpatialCNN(n_bands=60, n_classes=6,
                                       patch_size=args.patch_size, dropout=dropout)
            metrics = train_torch_model(
                model=model, df=df,
                model_name=args.run_name or args.model,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrrsu_map=mrrsu_map, patch_size=args.patch_size,
                cache_dir=cache_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
            )
```

For ViT (same pattern as CNN).

For LGBM — pass `num_leaves` and `subsample` through existing kwargs mechanism.

Also update `train_torch_model` to accept `weight_decay` (currently hardcoded as `1e-4`):

In `training/train_torch.py`, change:
```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```
And add `weight_decay: float = 1e-4` to the signature.

**Step 4: Smoke test with --help**

```bash
conda run -n crism python scripts/train.py --help | grep -E "dropout|hidden_dims|pos_weight|num_leaves"
```
Expected: all four args appear in help output.

**Step 5: Commit**

```bash
git add scripts/train.py training/train_torch.py
git commit -m "feat: extend train.py CLI with dropout, hidden_dims, pos_weight, num_leaves, subsample, weight_decay args for hyperparameter sweep"
```

---

## Task 4: Write scripts/sweep.py

**Files:**
- Create: `scripts/sweep.py`

**Background:** Defines all sweep configurations as a Python list. Each config is a dict mapping to `train.py` CLI args. For each config, `sweep.py`:
1. Checks if the checkpoint already exists (restartability)
2. Runs `conda run -n crism python scripts/train.py ...`
3. Logs result to a local summary CSV

This is ~19 configs total. Estimated runtime: lgbm/xgb ~5 min each, MLP ~5 min, CNN ~30 min, ViT ~30 min → ~4.5 hours total.

**Step 1: Create scripts/sweep.py**

```python
"""
Sequential hyperparameter sweep across all model families.
Each config trains one model variant and logs to wandb.
Restartable: skips configs whose checkpoint already exists.

Usage:
    conda run -n crism python scripts/sweep.py
    conda run -n crism python scripts/sweep.py --dry_run   # print configs only
"""
import argparse
import os
import subprocess
import sys
import csv
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(PROJ, 'scripts', 'train.py')
CKPT_DIR = os.path.join(PROJ, 'checkpoints')
LOG_DIR = os.path.join(PROJ, 'logs')

# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------
# Each entry: dict with keys matching train.py args.
# 'run_name' is used for --run_name and determines checkpoint filename.
# Checkpoint path: {CKPT_DIR}/{run_name}_best.pt  (sklearn: {run_name}_model.pkl)

SWEEP_CONFIGS = [
    # --- lgbm variants ---
    dict(model='lgbm', run_name='lgbm_sw1',
         n_estimators=500, learning_rate=0.05, num_leaves=63),
    dict(model='lgbm', run_name='lgbm_sw2',
         n_estimators=500, learning_rate=0.03, num_leaves=127),
    dict(model='lgbm', run_name='lgbm_sw3',
         n_estimators=300, learning_rate=0.1,  num_leaves=63, subsample=0.8),
    dict(model='lgbm', run_name='lgbm_sw4',
         n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8),

    # --- xgb variants ---
    dict(model='xgb', run_name='xgb_sw1',
         n_estimators=500, learning_rate=0.05, max_depth=4),
    dict(model='xgb', run_name='xgb_sw2',
         n_estimators=500, learning_rate=0.05, max_depth=8),
    dict(model='xgb', run_name='xgb_sw3',
         n_estimators=300, learning_rate=0.1,  max_depth=6, subsample=0.8),
    dict(model='xgb', run_name='xgb_sw4',
         n_estimators=500, learning_rate=0.03, max_depth=6),

    # --- MLP variants ---
    dict(model='mlp', run_name='mlp_sw1',
         epochs=200, patience=15, lr=1e-3, batch_size=512,
         hidden_dims='512,256,128', dropout=0.3, use_pos_weight=True),
    dict(model='mlp', run_name='mlp_sw2',
         epochs=200, patience=15, lr=5e-4, batch_size=512,
         hidden_dims='256,128', dropout=0.5, use_pos_weight=True),
    dict(model='mlp', run_name='mlp_sw3',
         epochs=200, patience=15, lr=1e-3, batch_size=256,
         hidden_dims='512,256', dropout=0.3, use_pos_weight=False),
    dict(model='mlp', run_name='mlp_sw4',
         epochs=200, patience=15, lr=2e-3, batch_size=1024,
         hidden_dims='256,128', dropout=0.2, use_pos_weight=True),

    # --- CNN variants (dropout now parameterised) ---
    dict(model='cnn', run_name='cnn_sw1', patch_size=7,
         epochs=200, patience=20, lr=5e-4, batch_size=256,
         dropout=0.3, use_pos_weight=True, weight_decay=1e-4),
    dict(model='cnn', run_name='cnn_sw2', patch_size=7,
         epochs=200, patience=20, lr=3e-4, batch_size=256,
         dropout=0.5, use_pos_weight=True, weight_decay=1e-4),
    dict(model='cnn', run_name='cnn_sw3', patch_size=7,
         epochs=200, patience=20, lr=1e-4, batch_size=256,
         dropout=0.3, use_pos_weight=False, weight_decay=1e-3),
    dict(model='cnn', run_name='cnn_sw4', patch_size=7,
         epochs=200, patience=20, lr=5e-4, batch_size=256,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4),

    # --- ViT variants ---
    dict(model='vit', run_name='vit_sw1', patch_size=7,
         epochs=200, patience=20, lr=5e-4, batch_size=256,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.3, use_pos_weight=True, weight_decay=1e-4),
    dict(model='vit', run_name='vit_sw2', patch_size=7,
         epochs=200, patience=20, lr=3e-4, batch_size=256,
         embed_dim=64, n_heads=4, n_layers=4,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4),
    dict(model='vit', run_name='vit_sw3', patch_size=7,
         epochs=200, patience=20, lr=1e-4, batch_size=256,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.3, use_pos_weight=False, weight_decay=1e-4),
]


def ckpt_exists(run_name: str, model: str) -> bool:
    if model in ('logreg', 'svc', 'rf', 'xgb', 'lgbm'):
        return os.path.exists(os.path.join(CKPT_DIR, f'{run_name}_model.pkl'))
    return os.path.exists(os.path.join(CKPT_DIR, f'{run_name}_best.pt'))


def config_to_args(cfg: dict) -> list:
    args = ['python', TRAIN]
    for k, v in cfg.items():
        if k == 'use_pos_weight':
            if v:
                args.append('--use_pos_weight')
        elif v is not None:
            args += [f'--{k}', str(v)]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    summary_path = os.path.join(LOG_DIR, f'sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results = []

    for i, cfg in enumerate(SWEEP_CONFIGS):
        run_name = cfg['run_name']
        model = cfg['model']
        print(f'\n[{i+1}/{len(SWEEP_CONFIGS)}] {run_name} — {cfg}')

        if ckpt_exists(run_name, model):
            print(f'  SKIPPING — checkpoint exists')
            continue

        if args.dry_run:
            cmd = config_to_args(cfg)
            print(f'  DRY RUN: {" ".join(cmd)}')
            continue

        cmd = config_to_args(cfg)
        print(f'  CMD: {" ".join(cmd)}')
        result = subprocess.run(
            ['conda', 'run', '-n', 'crism'] + cmd,
            cwd=PROJ,
            capture_output=False,
        )
        exit_code = result.returncode
        status = 'ok' if exit_code == 0 else f'FAILED({exit_code})'
        results.append({'run_name': run_name, 'model': model, 'status': status,
                        'config': str(cfg)})
        print(f'  {status}')

    if results and not args.dry_run:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_name', 'model', 'status', 'config'])
            writer.writeheader()
            writer.writerows(results)
        print(f'\nSweep summary written to {summary_path}')


if __name__ == '__main__':
    main()
```

**Step 2: Dry-run to verify config list prints correctly**

```bash
conda run -n crism python scripts/sweep.py --dry_run 2>&1 | head -60
```
Expected: prints 19 configs with their CLI args, no errors.

**Step 3: Also update train_sklearn.py to accept `num_leaves` and `subsample` kwargs**

In `_build_model` for lgbm, add `num_leaves` kwarg:
```python
    elif model_type == 'lgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', -1),
            num_leaves=kwargs.get('num_leaves', 31),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 1.0),
            random_state=42, n_jobs=-1, verbose=-1
        )
```

For xgb, add `subsample`:
```python
    elif model_type == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 1.0),
            eval_metric='logloss',
            tree_method='hist',
            random_state=42
        )
```

Also update `train_and_evaluate_sklearn` to pass `num_leaves` and `subsample` through:
- Add `num_leaves: int = 31` and `subsample: float = 1.0` to its signature
- Pass them in the `_build_model` call kwargs

And update `scripts/train.py` to pass `num_leaves=args.num_leaves, subsample=args.subsample` in the sklearn call.

**Step 4: Commit**

```bash
git add scripts/sweep.py training/train_sklearn.py
git commit -m "feat: add sweep.py with 19-config hyperparameter sweep and extend sklearn models with num_leaves/subsample"
```

---

## Task 5: Run the sweep

**Step 1: Create log directory if needed**

```bash
mkdir -p /mnt/gigas/CRISM/MRDR/crism_classification/logs
```

**Step 2: Launch sweep in background**

```bash
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
nohup conda run -n crism python -u scripts/sweep.py \
    > logs/sweep_${TIMESTAMP}.out 2>&1 &
echo "Sweep PID: $!"
```

**Step 3: Verify first config starts**

```bash
sleep 30 && tail -20 logs/sweep_*.out | tail -25
```
Expected: shows `[1/19] lgbm_sw1` and the lgbm training starting.

**Step 4: Monitor progress**

Check wandb online at https://wandb.ai/space-imagery-center/crism-mineral-classification.
Look for runs named `lgbm_sw1`, `lgbm_sw2`, etc.

**Step 5: Update memory with sweep status**

After confirming it's running, update MEMORY.md with the sweep PID and expected completion time.

---

## Success Criteria

- [ ] `test_cnn_dropout_parameter` passes
- [ ] `test_loss_accepts_pos_weight` and `test_pos_weight_increases_loss_for_rare_class` pass
- [ ] `scripts/sweep.py --dry_run` prints all 19 configs without error
- [ ] Sweep process running in background, first few lgbm configs completing in ~5 min each
- [ ] Wandb shows new runs: `lgbm_sw1`, `lgbm_sw2`, ...
