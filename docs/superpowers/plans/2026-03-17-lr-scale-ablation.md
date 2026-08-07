# LR Scale Ablation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--freeze_encoder` support to the spatial_vit training pipeline and create a SLURM array job that sweeps encoder LR scales (frozen, 0.001, 0.01, 0.1) for the HPC ablation study.

**Architecture:** Two co-dependent code changes (train.py adds the flag + freeze logic; train_torch.py filters frozen params from the optimizer and logs them to wandb), followed by a new SLURM array job script. All work is in the spatial-mae worktree on branch `feature/spatial-mae-pretraining`.

**Tech Stack:** Python, PyTorch, argparse, SLURM/sbatch, wandb

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `scripts/train.py` | Modify | Add `--freeze_encoder` flag, mutual-exclusivity guard, freeze loop, pass `freeze_encoder` to `train_torch_model` |
| `training/train_torch.py` | Modify | Add `freeze_encoder` param, filter optimizer to trainable params, add `encoder_lr_scale`+`freeze_encoder` to wandb config |
| `scripts/hpc_ablation_lr_scale.slurm` | Create | SLURM array job, 4 tasks, all paths under `/groups/sbyrne/phillipsm/crism_classification` |
| `tests/test_train_torch.py` | Modify | Add tests for frozen-encoder optimizer behaviour and training completion |

All paths below are relative to the worktree root:
`/mnt/mrdr/crism_classification/.worktrees/spatial-mae/`

---

## Chunk 1: Code changes (train_torch.py + train.py) with tests

### Task 1: Extend `train_torch.py` — freeze_encoder param + optimizer fix + wandb logging

**Files:**
- Modify: `training/train_torch.py:41-93` (signature + wandb init), `training/train_torch.py:150-157` (optimizer)
- Test: `tests/test_train_torch.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train_torch.py`:

```python
import copy

def make_fake_mrral_df_spatial(n=300):
    """Minimal mrral-format DataFrame (m0..m58) for spatial_vit / freeze tests.

    Named distinctly from the existing make_fake_mrral_df (mrrsu b0..b59 format)
    to avoid shadowing it.
    """
    rng = np.random.default_rng(42)
    data = {f'm{i}': rng.random(n).astype(np.float32) for i in range(59)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = (rng.random(n) > 0.7).astype(np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    n_train = int(n * 0.67)
    n_val = int(n * 0.165)
    n_test = n - n_train - n_val
    data['split'] = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    return pd.DataFrame(data)


def test_freeze_encoder_optimizer_only_has_head_params():
    """When encoder is frozen, optimizer must not contain encoder params."""
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    import unittest.mock as mock
    import torch.optim as _optim

    model = SpatialSpectralClassifier(n_bands=59, patch_size=3, n_classes=5,
                                      embed_dim=16, n_heads=2, n_layers=1)
    # Freeze encoder externally (as train.py does before calling train_torch_model)
    for p in model.encoder.parameters():
        p.requires_grad = False

    df = make_fake_mrral_df_spatial()

    # Capture optimizer params by patching AdamW at its usage point
    captured = {}
    _orig_adamw = _optim.AdamW
    def _mock_adamw(params, **kw):
        captured['params'] = list(params)
        return _orig_adamw(params, **kw)

    with mock.patch('torch.optim.AdamW', side_effect=_mock_adamw):
        train_torch_model(
            model=model, df=df, model_name='test_freeze',
            max_epochs=1, batch_size=32, lr=1e-3,
            use_wandb=False, checkpoint_dir=None,
            freeze_encoder=True,
        )

    assert captured, "AdamW was not called — optimizer was not constructed"
    # All optimized params must require grad (none should be frozen encoder params)
    for p in captured['params']:
        assert p.requires_grad, "Frozen param found in optimizer"


def test_freeze_encoder_weights_unchanged():
    """Encoder weights must not change after training with freeze_encoder=True."""
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    model = SpatialSpectralClassifier(n_bands=59, patch_size=3, n_classes=5,
                                      embed_dim=16, n_heads=2, n_layers=1)
    for p in model.encoder.parameters():
        p.requires_grad = False
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}

    df = make_fake_mrral_df_spatial()
    train_torch_model(
        model=model, df=df, model_name='test_freeze_weights',
        max_epochs=2, batch_size=32, lr=1e-3,
        use_wandb=False, checkpoint_dir=None,
        freeze_encoder=True,
    )

    for k, v_before in encoder_before.items():
        v_after = model.encoder.state_dict()[k]
        assert torch.allclose(v_before, v_after), f"Encoder param {k} changed during frozen training"


def test_freeze_encoder_head_params_do_change():
    """With encoder frozen, the head (linear layer) must still be trained."""
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    model = SpatialSpectralClassifier(n_bands=59, patch_size=3, n_classes=5,
                                      embed_dim=16, n_heads=2, n_layers=1)
    for p in model.encoder.parameters():
        p.requires_grad = False
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}

    df = make_fake_mrral_df_spatial()
    train_torch_model(
        model=model, df=df, model_name='test_freeze_head_trains',
        max_epochs=3, batch_size=32, lr=1e-2,  # higher LR to ensure visible change
        use_wandb=False, checkpoint_dir=None,
        freeze_encoder=True,
    )

    any_changed = any(
        not torch.allclose(head_before[k], model.head.state_dict()[k])
        for k in head_before
    )
    assert any_changed, "Head params did not change — training may not have occurred"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
conda run -n crism pytest tests/test_train_torch.py::test_freeze_encoder_optimizer_only_has_head_params tests/test_train_torch.py::test_freeze_encoder_weights_unchanged tests/test_train_torch.py::test_freeze_encoder_head_params_do_change -v 2>&1 | tail -20
```

Expected: FAIL — `train_torch_model` has no `freeze_encoder` parameter yet.

- [ ] **Step 3: Implement changes in `training/train_torch.py`**

**3a. Add `freeze_encoder` to function signature** (line 71, after `encoder_lr_scale`):

```python
    encoder_lr_scale: Optional[float] = None,
    freeze_encoder: bool = False,
    device: Optional[str] = None,
```

**3b. Add `encoder_lr_scale` and `freeze_encoder` to wandb config** (lines 85-93, update the config dict):

```python
    if use_wandb:
        import wandb as wb
        wb.init(
            project='crism-mineral-classification',
            name=model_name,
            config={'model': model_name, 'lr': lr, 'batch_size': batch_size,
                    'max_epochs': max_epochs, 'use_asl_loss': use_asl_loss,
                    'asl_gamma_neg': asl_gamma_neg,
                    'encoder_lr_scale': encoder_lr_scale,
                    'freeze_encoder': freeze_encoder,
                    **wandb_config}
        )
```

**3c. Fix optimizer `else` branch** (lines 150-157):

```python
    if encoder_lr_scale is not None and hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(
            head_lr=lr,
            encoder_lr=lr * encoder_lr_scale,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
conda run -n crism pytest tests/test_train_torch.py::test_freeze_encoder_optimizer_only_has_head_params tests/test_train_torch.py::test_freeze_encoder_weights_unchanged tests/test_train_torch.py::test_freeze_encoder_head_params_do_change -v 2>&1 | tail -20
```

Expected: 3 PASSED

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
conda run -n crism pytest tests/test_train_torch.py -v 2>&1 | tail -20
```

Expected: All existing tests still pass.

---

### Task 2: Extend `scripts/train.py` — `--freeze_encoder` flag, mutual-exclusivity guard, freeze loop

**Files:**
- Modify: `scripts/train.py:87-91` (args block), `scripts/train.py:342-370` (spatial_vit branch)

- [ ] **Step 1: Add `--freeze_encoder` argument** (after the `--encoder_lr_scale` argument at line 91):

```python
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights (requires_grad=False). '
                             'Mutually exclusive with --encoder_lr_scale. '
                             'Only effective when --pretrain_ckpt is set.')
```

- [ ] **Step 2: Add mutual-exclusivity guard** (after `args = parser.parse_args()` — verify exact line number in the live file before editing; it is near line 92, before the `cfg_path = ...` line):

```python
    if args.freeze_encoder and args.encoder_lr_scale is not None:
        parser.error('--freeze_encoder and --encoder_lr_scale are mutually exclusive.')
```

- [ ] **Step 3: Add freeze loop and pass freeze_encoder to train_torch_model** in the `spatial_vit` branch.

After the pretrain_ckpt loading block (after line 348):

```python
            if args.freeze_encoder:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                logging.info('Encoder frozen (requires_grad=False on all encoder params)')
```

And in the `train_torch_model` call (after `encoder_lr_scale=args.encoder_lr_scale,` at line 370), add:

```python
                freeze_encoder=args.freeze_encoder,
```

- [ ] **Step 4: Smoke-test the CLI flag (dry run)**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
conda run -n crism python scripts/train.py --model spatial_vit --freeze_encoder --encoder_lr_scale 0.1 --epochs 1 2>&1 | head -5
```

Expected: `error: argument --freeze_encoder: not allowed with argument --encoder_lr_scale` (argparse error)

```bash
conda run -n crism python scripts/train.py --model spatial_vit --freeze_encoder --help 2>&1 | grep freeze
```

Expected: `--freeze_encoder` appears in help text.

- [ ] **Step 5: Commit both code changes together**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
git add training/train_torch.py scripts/train.py tests/test_train_torch.py
git commit -m "feat: add --freeze_encoder flag with true parameter freeze

- Add --freeze_encoder to train.py spatial_vit branch
- Mutual exclusivity guard against --encoder_lr_scale
- train_torch.py optimizer filters to requires_grad=True params only
- Log encoder_lr_scale and freeze_encoder to wandb config
- Tests: optimizer contains no frozen params, weights unchanged, training completes"
```

---

## Chunk 2: SLURM array job

### Task 3: Create `scripts/hpc_ablation_lr_scale.slurm`

**Files:**
- Create: `scripts/hpc_ablation_lr_scale.slurm`

- [ ] **Step 1: Create the SLURM array job script**

```bash
cat > /mnt/mrdr/crism_classification/.worktrees/spatial-mae/scripts/hpc_ablation_lr_scale.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=spvit_lr_ablation
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lr_ablation_%a_%j.out
#SBATCH --error=logs/lr_ablation_%a_%j.err

# ── Condition arrays (index matches SLURM_ARRAY_TASK_ID) ──────────────────────
RUN_NAMES=("spvit_frozen" "spvit_lrscale0001" "spvit_lrscale001" "spvit_lrscale01")
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")

RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
LR_ARG="${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (${LR_ARG}) ==="

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/groups/sbyrne/phillipsm/crism_classification"
CACHE_DIR="${WORKDIR}/data/patch_cache"
CKPT_DIR="${WORKDIR}/checkpoints"
PRETRAIN_CKPT="${CKPT_DIR}/spatial_mae_128d_6l_best.pt"

# ── micromamba environment ─────────────────────────────────────────────────────
export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

# Must run from WORKDIR so relative log paths resolve correctly
cd "$WORKDIR"
mkdir -p logs checkpoints

# ── Write config.local.yaml ───────────────────────────────────────────────────
cat > config.local.yaml <<YAML
data_root: ${WORKDIR}
output_dir: ${WORKDIR}/data
patch_cache_dir: ${CACHE_DIR}
checkpoints_dir: ${CKPT_DIR}
YAML

# ── Patch cache check ─────────────────────────────────────────────────────────
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7.npy" ]; then
    echo "ERROR: mrral patch cache not found at ${CACHE_DIR}/mrral_train_patches_p7.npy"
    echo "Transfer the patch cache from qidu before submitting this job."
    echo "  rsync -av qidu:/mnt/mrdr/crism_classification/data/patch_cache/ ${CACHE_DIR}/"
    exit 1
fi

# ── Pre-training checkpoint check ─────────────────────────────────────────────
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: Pretrained checkpoint not found at ${PRETRAIN_CKPT}"
    exit 1
fi

# ── Run fine-tuning ───────────────────────────────────────────────────────────
echo "Starting training: ${RUN_NAME}"
python -u scripts/train.py \
    --model spatial_vit \
    --run_name "${RUN_NAME}" \
    --epochs 50 \
    --patience 10 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_ARG}

echo "=== ${RUN_NAME} complete ==="
EOF
```

- [ ] **Step 2: Verify the script looks correct**

```bash
cat /mnt/mrdr/crism_classification/.worktrees/spatial-mae/scripts/hpc_ablation_lr_scale.slurm
```

Check:
- All 4 `RUN_NAMES` and `LR_SCALE_ARGS` entries present
- `--freeze_encoder` is the arg for task 0 (no `--encoder_lr_scale`)
- `--n_layers 6` is explicit
- `exit 1` in both error guards
- Paths use `/groups/sbyrne/phillipsm/crism_classification`

- [ ] **Step 3: Commit the SLURM script**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
git add scripts/hpc_ablation_lr_scale.slurm
git commit -m "feat: add SLURM array job for encoder LR scale ablation

4-task array job sweeping: frozen encoder, lr_scale=0.001/0.01/0.1.
All paths under /groups/sbyrne/phillipsm/crism_classification.
Fails fast (exit 1) if patch cache or pretrain checkpoint missing."
```

---

## Chunk 3: Push to remote for HPC pull

### Task 4: Push branch so HPC can pull the changes

- [ ] **Step 1: Verify the worktree is on the correct branch**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
git status
git log --oneline -5
```

Expected: branch `feature/spatial-mae-pretraining`, 2 new commits visible.

- [ ] **Step 2: Push to remote**

```bash
git push origin feature/spatial-mae-pretraining
```

- [ ] **Step 3: Confirm remote is up to date**

```bash
git log --oneline origin/feature/spatial-mae-pretraining -5
```

Expected: same 2 commits as local.

---

## Usage on HPC (after pulling)

```bash
# On HPC
cd /groups/sbyrne/phillipsm/crism_classification
git pull origin feature/spatial-mae-pretraining

# Transfer patch cache from qidu if not already done:
# rsync -av qidu:/mnt/mrdr/crism_classification/data/patch_cache/ data/patch_cache/

# Submit the array job (from the worktree root):
sbatch scripts/hpc_ablation_lr_scale.slurm

# Monitor:
squeue -u phillipsm

# Check logs:
tail -f logs/lr_ablation_0_<jobid>.out   # spvit_frozen
tail -f logs/lr_ablation_1_<jobid>.out   # spvit_lrscale0001
tail -f logs/lr_ablation_2_<jobid>.out   # spvit_lrscale001
tail -f logs/lr_ablation_3_<jobid>.out   # spvit_lrscale01
```
