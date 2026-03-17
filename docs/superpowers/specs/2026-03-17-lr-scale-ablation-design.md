# LR Scale Ablation Study Design

**Date:** 2026-03-17
**Goal:** Determine whether fine-tuning the spatial MAE encoder helps or hurts downstream mineral classification, by sweeping encoder LR scales including a true frozen baseline.

---

## Scientific Question

We have a pre-trained `SpatialSpectralTransformer` encoder (128d, 6L, epoch 52, mae_loss=0.01793). When fine-tuning the downstream `SpatialSpectralClassifier`, should we:
- Freeze the encoder entirely and only train the linear head?
- Allow the encoder to update slowly (low LR scale)?
- Allow it to update more aggressively?

The four conditions isolate this axis cleanly, with all other hyperparameters fixed.

---

## Conditions

| Condition | Run name | Mechanism |
|-----------|----------|-----------|
| Frozen encoder | `spvit_frozen` | `--freeze_encoder` → `requires_grad=False` on all encoder params |
| LR scale 0.001 | `spvit_lrscale0001` | `--encoder_lr_scale 0.001` → encoder LR = 5e-7 |
| LR scale 0.01 | `spvit_lrscale001` | `--encoder_lr_scale 0.01` → encoder LR = 5e-6 |
| LR scale 0.1 | `spvit_lrscale01` | `--encoder_lr_scale 0.1` → encoder LR = 5e-5 |

Fixed across all runs: pretrain_ckpt=`spatial_mae_128d_6l_best.pt`, epochs=50, patience=10, batch_size=256, lr=5e-4, asl_loss (gamma_neg=4, gamma_pos=0, clip=0.05), embed_dim=128, n_heads=4, n_layers=6, patch_size=7.

---

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/train.py` | Modify | Add `--freeze_encoder` flag; apply freeze after loading pretrain ckpt |
| `training/train_torch.py` | Modify | Filter optimizer to only trainable params (handles frozen encoder correctly) |
| `scripts/hpc_ablation_lr_scale.slurm` | Create | SLURM array job (tasks 0–3, one per condition) |

All files are in the spatial-mae worktree: `/mnt/mrdr/crism_classification/.worktrees/spatial-mae/`.

---

## Code Changes

### `train.py` — `--freeze_encoder` flag

Add argument:
```python
parser.add_argument('--freeze_encoder', action='store_true',
                    help='Freeze encoder weights entirely (requires_grad=False). '
                         'Only effective when --pretrain_ckpt is set.')
```

After loading the pretrained checkpoint (in the `spatial_vit` branch), apply:
```python
if args.freeze_encoder:
    for p in model.encoder.parameters():
        p.requires_grad = False
    logging.info("Encoder frozen (requires_grad=False)")
```

Pass `freeze_encoder=args.freeze_encoder` (or nothing — the freeze is applied before `train_torch_model` is called, so `train_torch.py` just needs to handle frozen params transparently via the optimizer fix below).

### `train_torch.py` — Optimizer only over trainable params

Current optimizer construction (in `train_torch_model`):
```python
if hasattr(model, 'get_param_groups') and encoder_lr_scale is not None:
    param_groups = model.get_param_groups(lr, lr * encoder_lr_scale)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

Change the `else` branch to filter frozen params:
```python
else:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
```

This ensures frozen encoder params are excluded from optimizer state entirely.

---

## SLURM Array Job

**File:** `scripts/hpc_ablation_lr_scale.slurm`

```
#SBATCH --job-name=spvit_lr_ablation
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lr_ablation_%a_%j.out
#SBATCH --error=logs/lr_ablation_%a_%j.err
```

Task → condition mapping via bash array:
```bash
RUN_NAMES=("spvit_frozen" "spvit_lrscale0001" "spvit_lrscale001" "spvit_lrscale01")
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")
```

Config on HPC:
```
WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
```

The script writes `config.local.yaml` (identical pattern to `hpc_finetune.slurm`) pointing to `/groups/sbyrne/phillipsm/crism_classification` paths, then invokes `train.py` with the condition-specific args.

**Patch cache check:** The script checks for the existence of `mrral_train_patches_p7.npy` and exits with a clear error if not found, rather than silently failing mid-training.

---

## HPC Paths

| Resource | Path |
|----------|------|
| Work dir | `/groups/sbyrne/phillipsm/crism_classification` |
| Data / patch cache | `/groups/sbyrne/phillipsm/crism_classification/data/patch_cache/` |
| Checkpoints | `/groups/sbyrne/phillipsm/crism_classification/checkpoints/` |
| Logs | `/groups/sbyrne/phillipsm/crism_classification/logs/` |

Note: xdisk (`/xdisk/sbyrne/phillipsm/`) is at quota limit; all paths use `/groups/sbyrne/phillipsm/` instead.

---

## Usage

```bash
# On HPC, from work dir:
sbatch scripts/hpc_ablation_lr_scale.slurm

# Monitor:
squeue -u phillipsm

# Check a specific task log:
tail -f logs/lr_ablation_0_<jobid>.out   # frozen
tail -f logs/lr_ablation_3_<jobid>.out   # lr_scale=0.1
```

The four jobs run in parallel. Results appear in wandb project `crism-mineral-classification` under run names `spvit_frozen`, `spvit_lrscale0001`, `spvit_lrscale001`, `spvit_lrscale01`.

---

## Out of Scope

- Random-init baseline (user is convinced pre-training helps)
- Model scale sweep (captured in future work memory)
- Hyperparameter tuning of LR, batch size, loss params
