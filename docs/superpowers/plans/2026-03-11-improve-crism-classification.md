# Improve CRISM Classification mAP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push val mAP from 0.61 to ≥0.70 by fixing the MAE fine-tuning setup, replacing focal loss with asymmetric loss (ASL), adding combined mrral+mrrsu features, and wiring it all into a sweep.

**Architecture:** Three compounding improvements: (1) AsymmetricLoss decouples positive/negative focusing for extreme multi-label imbalance; (2) differential learning rate prevents catastrophic forgetting of MAE-pretrained encoder weights; (3) SpectralHybridClassifier concatenates the MAE-pretrained spectral encoder with a separate MLP branch over mrrsu summary parameters (domain-scientist-designed mineral indices). A structured ablation sweep (sweep_v6.py) tests each improvement independently and in combination.

**Tech Stack:** PyTorch, wandb, pandas/pyarrow, pytest. Conda env `crism`. Data at `data/mrral_pixels.parquet` (1.97M rows, m0..m58) and `data/pixels.parquet` (899K rows, b0..b59). Config: `config.yaml`. Checkpoints: `checkpoints/`. MAE checkpoint (4-layer): `checkpoints/mae_pretrain_128d_4l_best.pt`.

**Critical domain facts:**
- `n_classes=5`: ['olivine', 'lcp', 'hcp', 'plagioclase', 'other'] — olivine_t1/t2 collapsed
- mrral bands: m0..m58 (59 bands, 410–2457 nm)
- mrrsu bands: b0..b59 (60 bands — summary parameters including OLINDEX3=b15, BD1300=b17, LCPINDEX2=b18, HCPINDEX2=b19)
- HCP AP = 0.19 in last run (extreme class imbalance, biggest improvement target)
- MAE was pretrained with n_layers=4; classifier used n_layers=6 → partial init mismatch
- Existing 4-layer MAE checkpoint: `checkpoints/mae_pretrain_128d_4l_best.pt` ← use for v6 sweep
- All test runs: `conda run -n crism pytest tests/ -x -q`

---

## Chunk 1: Asymmetric Loss

### Task 1: Add AsymmetricLoss to training/losses.py

**Files:**
- Modify: `training/losses.py`
- Modify: `tests/test_losses.py`

AsymmetricLoss (Wang et al. 2021) decouples positive/negative gamma. For negatives it applies
a probability margin (clip) to hard-zero very easy negatives, and uses a higher gamma to further
suppress remaining easy negatives. For positives it uses gamma=0 (no down-weighting — every
missed positive hurts). This is better than symmetric focal for extreme multi-label imbalance like HCP.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_losses.py`:

```python
def test_asl_output_is_scalar():
    from training.losses import AsymmetricLoss
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 2, (8, 5)).float()
    weights = torch.ones(8)
    val = loss_fn(logits, targets, weights)
    assert val.shape == (), f"Expected scalar, got {val.shape}"
    assert val.item() > 0


def test_asl_accepts_pos_weight_kwarg():
    """ASL must accept pos_weight= kwarg for API compatibility with training loop."""
    from training.losses import AsymmetricLoss
    loss_fn = AsymmetricLoss()
    logits = torch.randn(4, 5)
    targets = torch.zeros(4, 5)
    weights = torch.ones(4)
    # Should not raise
    val = loss_fn(logits, targets, weights, pos_weight=torch.ones(5))
    assert val.item() >= 0


def test_asl_clip_zeroes_easy_negatives():
    """With clip=0.5, confident negatives (p << 0.5) should incur near-zero loss."""
    from training.losses import AsymmetricLoss
    loss_fn = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.5)
    # Very confident negatives: logits=-10 → p≈0, p_m=max(0-0.5,0)=0, log(1-0)≈0
    logits = torch.full((4, 5), -10.0)
    targets = torch.zeros(4, 5)
    weights = torch.ones(4)
    val = loss_fn(logits, targets, weights)
    assert val.item() < 0.01, f"Expected near-zero loss for confident negatives, got {val.item()}"


def test_asl_higher_gamma_neg_suppresses_negatives():
    """Higher gamma_neg should reduce loss on easy negatives more than lower gamma_neg."""
    from training.losses import AsymmetricLoss
    torch.manual_seed(0)
    logits = torch.full((8, 5), -2.0)   # moderate negative confidence
    targets = torch.zeros(8, 5)
    weights = torch.ones(8)
    loss_low  = AsymmetricLoss(gamma_neg=1.0, gamma_pos=0.0, clip=0.0)(logits, targets, weights)
    loss_high = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.0)(logits, targets, weights)
    assert loss_high < loss_low, "Higher gamma_neg should down-weight easy negatives more"


def test_asl_gamma_zero_clip_zero_equals_bce():
    """ASL with gamma_neg=0, gamma_pos=0, clip=0 should equal WeightedBCE."""
    from training.losses import AsymmetricLoss, WeightedBCEWithLogitsLoss
    torch.manual_seed(42)
    logits = torch.randn(8, 5)
    targets = (torch.randn(8, 5) > 0).float()
    weights = torch.ones(8)
    bce = WeightedBCEWithLogitsLoss()(logits, targets, weights)
    asl = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.0)(logits, targets, weights)
    assert abs(bce.item() - asl.item()) < 1e-4, f"Expected BCE≈ASL(γ=0,clip=0): {bce.item()} vs {asl.item()}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n crism pytest tests/test_losses.py::test_asl_output_is_scalar -x -q
```
Expected: `ImportError` or `AttributeError: module 'training.losses' has no attribute 'AsymmetricLoss'`

- [ ] **Step 3: Implement AsymmetricLoss**

Add to `training/losses.py` (after `FocalBCEWithLogitsLoss`):

```python
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Wang et al. 2021).

    Decouples positive/negative focusing via separate gamma values.
    A probability margin (clip) hard-zeros very easy negatives before
    computing the log, strongly suppressing trivial negative samples.

    Typical settings: gamma_neg=4, gamma_pos=0, clip=0.05.
    Setting gamma_neg=0, gamma_pos=0, clip=0 recovers standard BCE.

    API matches WeightedBCEWithLogitsLoss (pos_weight accepted but not used —
    ASL handles imbalance via the asymmetric focusing terms instead).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(
        self,
        logits: torch.Tensor,                       # (batch, n_classes)
        targets: torch.Tensor,                      # (batch, n_classes)
        weights: torch.Tensor,                      # (batch,)
        pos_weight: Optional[torch.Tensor] = None,  # accepted for API compat, not used
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)

        # Asymmetric clip: shift negative probability down by margin before log
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p

        # Log probabilities (clamped for numerical stability)
        log_p_pos = torch.log(p.clamp(min=1e-8))
        log_p_neg = torch.log((1 - p_neg).clamp(min=1e-8))

        # BCE with asymmetric log probs
        bce = targets * log_p_pos + (1 - targets) * log_p_neg

        # Asymmetric focal weights
        p_t = p * targets + p_neg * (1 - targets)
        focal_weight = torch.where(
            targets.bool(),
            (1 - p_t) ** self.gamma_pos,
            p_t ** self.gamma_neg,
        )

        loss = (-focal_weight * bce).mean(dim=1)   # (batch,)
        return (loss * weights).sum() / (weights.sum() + 1e-8)
```

- [ ] **Step 4: Run all loss tests**

```bash
conda run -n crism pytest tests/test_losses.py -x -q
```
Expected: all 9 tests pass (5 original + 5 new — one may deduplicate)

- [ ] **Step 5: Commit**

```bash
git add training/losses.py tests/test_losses.py
git commit -m "feat: add AsymmetricLoss for multi-label imbalance"
```

---

### Task 2: Wire ASL into train_torch.py and train.py

**Files:**
- Modify: `training/train_torch.py`
- Modify: `scripts/train.py`
- Modify: `tests/test_train_torch.py`

- [ ] **Step 1: Write failing test**

Read `tests/test_train_torch.py` first, then add:

```python
def test_train_with_asl_loss():
    """Training loop should run without error when use_asl_loss=True."""
    import torch, pandas as pd, numpy as np
    from training.train_torch import train_torch_model
    from models.spectral_transformer import SpectralTransformer

    n = 120
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':        rng.integers(0, 2, n).astype('float32'),
        'hcp':        rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other':      rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 80 + ['val'] * 40,
    })
    model = SpectralTransformer(n_bands=59, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    metrics = train_torch_model(
        model=model, df=df, model_name='test_asl',
        max_epochs=2, batch_size=32, lr=1e-3,
        patience=5, use_wandb=False, checkpoint_dir=None,
        use_asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0,
    )
    assert 'val_mAP' in metrics
```

- [ ] **Step 2: Run test to confirm failure**

```bash
conda run -n crism pytest tests/test_train_torch.py::test_train_with_asl_loss -x -q
```
Expected: `TypeError: train_torch_model() got an unexpected keyword argument 'use_asl_loss'`

- [ ] **Step 3: Add ASL args to train_torch.py**

In `training/train_torch.py`, add parameters to `train_torch_model` signature:

```python
def train_torch_model(
    ...
    use_asl_loss: bool = False,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
    ...
```

Replace the loss function selection block (currently near line 148):

```python
    if use_asl_loss:
        from training.losses import AsymmetricLoss
        loss_fn = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
    elif use_focal_loss:
        from training.losses import FocalBCEWithLogitsLoss
        loss_fn = FocalBCEWithLogitsLoss(gamma=focal_gamma)
    else:
        loss_fn = WeightedBCEWithLogitsLoss()
```

- [ ] **Step 4: Add ASL args to scripts/train.py**

In the argparse block:

```python
    parser.add_argument('--asl_loss', action='store_true',
                        help='Use asymmetric loss (Wang et al. 2021) instead of focal/BCE')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=0.0)
    parser.add_argument('--asl_clip', type=float, default=0.05)
```

In `BOOL_FLAGS` set in sweep scripts, add `'asl_loss'`.

Pass through to `train_torch_model` for `spectral_cnn` and `spectral_vit` branches:

```python
        use_asl_loss=args.asl_loss,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
```

Also add these to the wandb config dict in `train_torch.py`:

```python
        config={'model': model_name, 'lr': lr, 'batch_size': batch_size,
                'max_epochs': max_epochs, 'use_asl_loss': use_asl_loss,
                'asl_gamma_neg': asl_gamma_neg, **wandb_config}
```

- [ ] **Step 5: Run tests**

```bash
conda run -n crism pytest tests/test_train_torch.py -x -q
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add training/train_torch.py scripts/train.py tests/test_train_torch.py
git commit -m "feat: wire AsymmetricLoss into training loop (--asl_loss flag)"
```

---

## Chunk 2: Differential Learning Rate

### Task 3: Add differential LR to SpectralTransformer + train_torch.py

**Files:**
- Modify: `models/spectral_transformer.py`
- Modify: `training/train_torch.py`
- Modify: `scripts/train.py`
- Modify: `tests/test_models.py`

Differential LR sets the pretrained encoder to a low LR (e.g. lr×0.1) while the classification
head trains at the full LR. This prevents catastrophic forgetting of MAE-pretrained representations.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_models.py`:

```python
def test_spectral_transformer_get_param_groups():
    from models.spectral_transformer import SpectralTransformer
    model = SpectralTransformer(n_bands=59, n_classes=5, embed_dim=64, n_heads=2, n_layers=2)
    groups = model.get_param_groups(head_lr=3e-4, encoder_lr=3e-5)
    assert len(groups) == 2
    assert groups[0]['lr'] == 3e-5, "encoder group should get slow LR"
    assert groups[1]['lr'] == 3e-4, "head group should get fast LR"
    # Encoder params should not include head params
    head_param_ids = {id(p) for p in model.head.parameters()}
    encoder_param_ids = {id(p) for p in groups[0]['params']}
    assert not (head_param_ids & encoder_param_ids), "head params must not appear in encoder group"
    # All params covered
    all_group_ids = encoder_param_ids | {id(p) for p in groups[1]['params']}
    all_model_ids = {id(p) for p in model.parameters()}
    assert all_group_ids == all_model_ids, "All parameters must be in exactly one group"


def test_train_torch_differential_lr():
    """Training with encoder_lr_scale should not raise and should produce valid metrics."""
    import torch, pandas as pd, numpy as np
    from training.train_torch import train_torch_model
    from models.spectral_transformer import SpectralTransformer

    n = 120
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':  rng.integers(0, 2, n).astype('float32'),
        'hcp':  rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other': rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 80 + ['val'] * 40,
    })
    model = SpectralTransformer(n_bands=59, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    metrics = train_torch_model(
        model=model, df=df, model_name='test_diffr',
        max_epochs=2, batch_size=32, lr=3e-4,
        patience=5, use_wandb=False, checkpoint_dir=None,
        encoder_lr_scale=0.1,
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
conda run -n crism pytest tests/test_models.py::test_spectral_transformer_get_param_groups -x -q
```
Expected: `AttributeError: 'SpectralTransformer' object has no attribute 'get_param_groups'`

- [ ] **Step 3: Add get_param_groups to SpectralTransformer**

Add method to `SpectralTransformer` class in `models/spectral_transformer.py`:

```python
    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """
        Return two optimizer param groups for differential LR fine-tuning.

        When loaded from a MAE checkpoint, the encoder (band_embed, pos_embed,
        cls_token, encoder layers, norm) has pretrained weights that should be
        fine-tuned gently. The classification head is randomly initialized and
        can use a higher LR.

        Returns:
            [{'params': encoder_params, 'lr': encoder_lr},
             {'params': head_params,    'lr': head_lr}]
        """
        head_params = list(self.head.parameters())
        head_param_ids = {id(p) for p in head_params}
        encoder_params = [p for p in self.parameters() if id(p) not in head_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]
```

- [ ] **Step 4: Add encoder_lr_scale to train_torch.py**

Add parameter to `train_torch_model` signature:

```python
    encoder_lr_scale: Optional[float] = None,   # e.g. 0.1 → encoder LR = lr * 0.1
```

Replace the optimizer construction line (currently `optimizer = torch.optim.AdamW(model.parameters(), lr=lr, ...)`):

```python
    if encoder_lr_scale is not None and hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(
            head_lr=lr,
            encoder_lr=lr * encoder_lr_scale,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

Note: `CosineAnnealingLR` and `LinearLR` schedulers correctly handle param groups — each group's `initial_lr` is used independently. No other scheduler changes needed.

- [ ] **Step 5: Add encoder_lr_scale arg to scripts/train.py**

```python
    parser.add_argument('--encoder_lr_scale', type=float, default=None,
                        help='LR multiplier for pretrained encoder (e.g. 0.1 → 10× slower than head). '
                             'Only effective when --pretrain_ckpt is set and model has get_param_groups.')
```

Pass through to `train_torch_model`:

```python
        encoder_lr_scale=args.encoder_lr_scale,
```

- [ ] **Step 6: Run all model tests**

```bash
conda run -n crism pytest tests/test_models.py tests/test_train_torch.py -x -q
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add models/spectral_transformer.py training/train_torch.py scripts/train.py tests/test_models.py
git commit -m "feat: add differential LR support (get_param_groups + encoder_lr_scale)"
```

---

## Chunk 3: Hybrid Combined Features Model

### Task 4: Add CRISMCombinedDataset to data/dataset.py

**Files:**
- Modify: `data/dataset.py`
- Modify: `tests/test_dataset.py`

The combined dataset merges mrral_pixels.parquet (m0..m58) with pixels.parquet (b0..b59) on
`[tile_id, polygon_id, pixel_row, pixel_col]`. Result: 119-dim feature vector
`[mrral_59 | mrrsu_60]`. The hybrid model splits this back into two branches.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dataset.py`:

```python
MRRSU_PARQUET = '/mnt/gigas/CRISM/MRDR/crism_classification/data/pixels.parquet'

@pytest.fixture
def combined_df():
    """Synthetic dataframe with both mrral and mrrsu columns for unit tests."""
    import torch
    import numpy as np
    n = 40
    rng = np.random.default_rng(99)
    data = {
        'tile_id': ['t0001'] * n,
        'polygon_id': list(range(n)),
        'pixel_row': list(range(n)),
        'pixel_col': [0] * n,
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        **{f'b{i}': rng.random(n).astype('float32') for i in range(60)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':         rng.integers(0, 2, n).astype('float32'),
        'hcp':         rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other':       rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 30 + ['val'] * 10,
    }
    return pd.DataFrame(data)


def test_combined_dataset_feature_shape(combined_df):
    """CRISMCombinedDataset should return 119-dim feature tensor."""
    import torch
    from data.dataset import CRISMCombinedDataset
    ds = CRISMCombinedDataset(combined_df)
    feat, label, weight = ds[0]
    assert feat.shape == (119,), f"Expected (119,), got {feat.shape}"
    assert label.shape == (5,), f"Expected (5,) classes, got {label.shape}"
    assert feat.dtype == torch.float32


def test_combined_dataset_splits_correctly(combined_df):
    """First 59 dims should match mrral, last 60 should match mrrsu."""
    from data.dataset import CRISMCombinedDataset, MRRAL_BAND_COLS, BAND_COLS
    ds = CRISMCombinedDataset(combined_df)
    feat, _, _ = ds[0]
    # Compare against raw df values (after _collapse_labels which doesn't change band cols)
    expected_mrral = combined_df[MRRAL_BAND_COLS].iloc[0].values
    expected_mrrsu = combined_df[BAND_COLS].iloc[0].values
    import numpy as np
    np.testing.assert_allclose(feat[:59].numpy(), expected_mrral, rtol=1e-5)
    np.testing.assert_allclose(feat[59:].numpy(), expected_mrrsu, rtol=1e-5)


def test_combined_dataset_raises_on_missing_mrral(combined_df):
    """Should raise ValueError if mrral columns missing."""
    from data.dataset import CRISMCombinedDataset
    df_no_mrral = combined_df.drop(columns=[f'm{i}' for i in range(59)])
    with pytest.raises(ValueError, match="mrral"):
        CRISMCombinedDataset(df_no_mrral)


def test_combined_dataset_raises_on_missing_mrrsu(combined_df):
    """Should raise ValueError if mrrsu columns missing."""
    from data.dataset import CRISMCombinedDataset
    df_no_mrrsu = combined_df.drop(columns=[f'b{i}' for i in range(60)])
    with pytest.raises(ValueError, match="mrrsu"):
        CRISMCombinedDataset(df_no_mrrsu)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
conda run -n crism pytest tests/test_dataset.py::test_combined_dataset_feature_shape -x -q
```
Expected: `ImportError` or `cannot import name 'CRISMCombinedDataset'`

- [ ] **Step 3: Implement CRISMCombinedDataset**

Add to `data/dataset.py` (after `CRISMSpectralDataset`):

```python
class CRISMCombinedDataset(Dataset):
    """
    Per-pixel dataset combining mrral 59-band reflectance with mrrsu 60-band
    summary parameters into a single 119-dim feature vector.

    Requires a merged DataFrame with both m0..m58 (mrral) and b0..b59 (mrrsu)
    columns present. Build via:

        mrral_df = pd.read_parquet('data/mrral_pixels.parquet')
        mrrsu_df = pd.read_parquet('data/pixels.parquet')
        MERGE_KEYS = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
        combined = mrral_df.merge(mrrsu_df[MERGE_KEYS + BAND_COLS], on=MERGE_KEYS, how='inner')

    Features layout: features[:59] = mrral bands, features[59:] = mrrsu bands.
    This layout matches SpectralHybridClassifier.forward() which splits on dim 59.
    """

    N_MRRAL = 59
    N_MRRSU = 60
    N_FEATURES = N_MRRAL + N_MRRSU  # 119

    def __init__(self, df: pd.DataFrame):
        missing_mrral = [c for c in MRRAL_BAND_COLS if c not in df.columns]
        if missing_mrral:
            raise ValueError(
                f"DataFrame missing mrral columns: {missing_mrral[:5]}... "
                "Merge mrral_pixels.parquet with pixels.parquet first."
            )
        missing_mrrsu = [c for c in BAND_COLS if c not in df.columns]
        if missing_mrrsu:
            raise ValueError(
                f"DataFrame missing mrrsu columns: {missing_mrrsu[:5]}... "
                "Merge mrral_pixels.parquet with pixels.parquet first."
            )
        df = _collapse_labels(df)
        mrral = df[MRRAL_BAND_COLS].values.astype('float32')
        mrrsu = df[BAND_COLS].values.astype('float32')
        self.features = torch.tensor(
            np.concatenate([mrral, mrrsu], axis=1), dtype=torch.float32
        )
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]
```

Also add `import numpy as np` at the top of `data/dataset.py` if not already present.

- [ ] **Step 4: Run dataset tests**

```bash
conda run -n crism pytest tests/test_dataset.py -x -q
```
Expected: all pass (including new combined dataset tests)

- [ ] **Step 5: Commit**

```bash
git add data/dataset.py tests/test_dataset.py
git commit -m "feat: add CRISMCombinedDataset (mrral 59 + mrrsu 60 = 119 features)"
```

---

### Task 5: Create SpectralHybridClassifier

**Files:**
- Create: `models/hybrid_classifier.py`
- Modify: `tests/test_models.py`

The hybrid classifier routes mrral spectra through the SpectralTransformer encoder (pretrained
via MAE) and mrrsu summary parameters through a small 2-layer MLP, then concatenates and
classifies. The feature split at dimension 59 mirrors CRISMCombinedDataset's layout.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_models.py`:

```python
def test_hybrid_classifier_output_shape():
    from models.hybrid_classifier import SpectralHybridClassifier
    model = SpectralHybridClassifier(
        n_mrral=59, n_mrrsu=60, n_classes=5,
        embed_dim=64, n_heads=2, n_layers=2,
    )
    x = torch.randn(4, 119)   # 59 mrral + 60 mrrsu
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"


def test_hybrid_classifier_get_param_groups():
    from models.hybrid_classifier import SpectralHybridClassifier
    model = SpectralHybridClassifier(
        n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=64, n_heads=2, n_layers=2,
    )
    groups = model.get_param_groups(head_lr=3e-4, encoder_lr=3e-5)
    assert len(groups) == 2
    assert groups[0]['lr'] == 3e-5
    assert groups[1]['lr'] == 3e-4
    # All parameters covered, no overlap
    all_group_ids = set()
    for g in groups:
        ids = {id(p) for p in g['params']}
        assert not (ids & all_group_ids), "Param groups must not overlap"
        all_group_ids |= ids
    all_model_ids = {id(p) for p in model.parameters()}
    assert all_group_ids == all_model_ids


def test_hybrid_classifier_load_encoder_state_dict():
    """load_encoder_state_dict should load pretrained encoder weights without error."""
    import copy
    from models.hybrid_classifier import SpectralHybridClassifier
    # Create two models; copy encoder state from one to the other
    m1 = SpectralHybridClassifier(n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    m2 = SpectralHybridClassifier(n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    # Extract encoder state from m1 (simulates MAE checkpoint)
    encoder_state = {k: v for k, v in m1.encoder.state_dict().items()
                     if not k.startswith('head.')}
    missing, unexpected = m2.load_encoder_state_dict(encoder_state)
    assert len(missing) == 0, f"Unexpected missing keys: {missing}"
    # Verify encoder weights actually loaded (m2 encoder == m1 encoder)
    for k, v in m1.encoder.state_dict().items():
        if not k.startswith('head.'):
            assert torch.allclose(m2.encoder.state_dict()[k], v), f"Weight {k} not loaded"


def test_hybrid_classifier_returns_logits_not_probs():
    from models.hybrid_classifier import SpectralHybridClassifier
    model = SpectralHybridClassifier(n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    model.eval()
    x = torch.zeros(2, 119)
    out = model(x)
    # If sigmoid were applied to zero logits, output would be exactly 0.5
    # Raw logits from zero input will not all be 0.5
    assert not torch.allclose(out, torch.full_like(out, 0.5))
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
conda run -n crism pytest tests/test_models.py::test_hybrid_classifier_output_shape -x -q
```
Expected: `ModuleNotFoundError: No module named 'models.hybrid_classifier'`

- [ ] **Step 3: Create models/hybrid_classifier.py**

```python
"""
SpectralHybridClassifier: combines mrral spectral encoder with mrrsu branch.

Processes a 119-dim input [mrral_59 | mrrsu_60] by routing mrral bands through
a SpectralTransformer encoder (MAE-pretrained) and mrrsu summary parameters
through a shallow MLP, then concatenating and classifying.

This architecture leverages:
  - MAE-pretrained spectral representation (mrral 59 bands)
  - Domain-scientist mineral indices (mrrsu summary params: OLINDEX3, BD1300, etc.)
"""
import torch
import torch.nn as nn

from models.spectral_transformer import SpectralTransformer


class SpectralHybridClassifier(nn.Module):
    """
    Hybrid classifier for CRISM mineral classification.

    Input:  (batch, n_mrral + n_mrrsu)  — first n_mrral dims are mrral, rest mrrsu
    Output: (batch, n_classes) — raw logits

    Args:
        n_mrral:      Number of mrral spectral bands (default: 59)
        n_mrrsu:      Number of mrrsu summary parameter bands (default: 60)
        n_classes:    Number of output classes (default: 5)
        embed_dim:    SpectralTransformer embedding dimension (default: 128)
        n_heads:      Number of attention heads (default: 4)
        n_layers:     Number of transformer encoder layers (default: 4)
        dropout:      Dropout rate applied in both branches (default: 0.1)
        mrrsu_hidden: Hidden dimension of the mrrsu MLP branch (default: 64)
    """

    N_MRRAL = 59  # expected mrral feature count — validated in forward

    def __init__(
        self,
        n_mrral: int = 59,
        n_mrrsu: int = 60,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        mrrsu_hidden: int = 64,
    ):
        super().__init__()
        self.n_mrral = n_mrral
        self.n_mrrsu = n_mrrsu

        # Spectral encoder — use SpectralTransformer with Identity head
        # so forward() returns the raw CLS embedding (embed_dim,) not logits.
        self.encoder = SpectralTransformer(
            n_bands=n_mrral, n_classes=embed_dim,
            embed_dim=embed_dim, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )
        self.encoder.head = nn.Identity()

        # mrrsu branch: lightweight MLP
        self.mrrsu_branch = nn.Sequential(
            nn.Linear(n_mrrsu, mrrsu_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mrrsu_hidden, mrrsu_hidden),
            nn.GELU(),
        )

        # Joint classification head
        self.head = nn.Linear(embed_dim + mrrsu_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_mrral + n_mrrsu) — layout matches CRISMCombinedDataset
        """
        mrral = x[:, :self.n_mrral]           # (batch, 59)
        mrrsu = x[:, self.n_mrral:]           # (batch, 60)
        cls_embed = self.encoder(mrral)       # (batch, embed_dim)
        mrrsu_feat = self.mrrsu_branch(mrrsu) # (batch, mrrsu_hidden)
        combined = torch.cat([cls_embed, mrrsu_feat], dim=1)
        return self.head(combined)            # (batch, n_classes)

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """
        Differential LR: pretrained encoder gets slow LR, mrrsu branch and
        classification head get fast LR.

        Returns:
            [{'params': encoder_params, 'lr': encoder_lr},
             {'params': new_params,     'lr': head_lr}]
        """
        encoder_param_ids = {id(p) for p in self.encoder.parameters()}
        encoder_params = list(self.encoder.parameters())
        new_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': new_params,     'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        """
        Load SpectralTransformer encoder weights from a MAE checkpoint.
        Delegates to the inner encoder's load_encoder_state_dict method.

        Args:
            state: dict from SpectralMAE.encoder_state_dict() — keys like
                   'band_embed.weight', 'pos_embed.weight', 'encoder.layers.N.*'

        Returns:
            (missing_keys, unexpected_keys)
        """
        return self.encoder.load_encoder_state_dict(state)
```

- [ ] **Step 4: Run model tests**

```bash
conda run -n crism pytest tests/test_models.py -x -q
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add models/hybrid_classifier.py tests/test_models.py
git commit -m "feat: add SpectralHybridClassifier (mrral encoder + mrrsu branch)"
```

---

### Task 6: Wire SpectralHybridClassifier into scripts/train.py

**Files:**
- Modify: `scripts/train.py`
- Modify: `tests/test_train_torch.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_train_torch.py`:

```python
def test_train_hybrid_model_e2e():
    """SpectralHybridClassifier should train end-to-end through train_torch_model."""
    import torch, pandas as pd, numpy as np
    from training.train_torch import train_torch_model
    from models.hybrid_classifier import SpectralHybridClassifier

    n = 120
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        **{f'b{i}': rng.random(n).astype('float32') for i in range(60)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':  rng.integers(0, 2, n).astype('float32'),
        'hcp':  rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other': rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 80 + ['val'] * 40,
    })
    model = SpectralHybridClassifier(
        n_mrral=59, n_mrrsu=60, n_classes=5,
        embed_dim=32, n_heads=2, n_layers=2,
    )
    metrics = train_torch_model(
        model=model, df=df, model_name='test_hybrid',
        max_epochs=2, batch_size=32, lr=3e-4,
        patience=5, use_wandb=False, checkpoint_dir=None,
        use_asl_loss=True,
    )
    assert 'val_mAP' in metrics
```

- [ ] **Step 2: Run test to confirm failure**

```bash
conda run -n crism pytest tests/test_train_torch.py::test_train_hybrid_model_e2e -x -q
```
Expected: pass already (train_torch_model is model-agnostic) OR fail with a KeyError from dataset
selection. The test depends on `CRISMCombinedDataset` being chosen when both m* and b* cols exist.

Update `make_dataset` in `training/train_torch.py` to detect combined features:

```python
    def make_dataset(sub_df, split_name='train'):
        from data.dataset import MRRAL_BAND_COLS, BAND_COLS, CRISMSpectralDataset, CRISMCombinedDataset
        if use_patches:
            return CRISMPatchDataset(sub_df, mrrsu_map, patch_size=patch_size,
                                     cache_dir=cache_dir, split=split_name)
        has_mrral = MRRAL_BAND_COLS[0] in sub_df.columns
        has_mrrsu = BAND_COLS[0] in sub_df.columns
        if has_mrral and has_mrrsu:
            return CRISMCombinedDataset(sub_df)
        if has_mrral:
            return CRISMSpectralDataset(sub_df)
        return CRISMPixelDataset(sub_df)
```

- [ ] **Step 3: Add spectral_hybrid model to scripts/train.py**

In `TORCH_MODELS` set:
```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit', 'spectral_hybrid'}
```

In the `elif args.model in ('spectral_cnn', 'spectral_vit'):` branch, after the existing model
construction, add a new `elif args.model == 'spectral_hybrid':` block:

```python
        elif args.model == 'spectral_hybrid':
            import yaml as _yaml
            from models.hybrid_classifier import SpectralHybridClassifier
            mrral_parquet = os.path.join(os.path.dirname(parquet_path), 'mrral_pixels.parquet')
            mrrsu_parquet = parquet_path  # pixels.parquet has b0..b59

            df_mrral = pd.read_parquet(mrral_parquet)
            df_mrrsu = pd.read_parquet(mrrsu_parquet)
            from data.dataset import BAND_COLS
            MERGE_KEYS = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
            df_combined = df_mrral.merge(
                df_mrrsu[MERGE_KEYS + BAND_COLS],
                on=MERGE_KEYS,
                how='inner',
            )
            logging.info(
                f"Combined dataset: {len(df_combined)} pixels "
                f"({len(df_mrral)} mrral ∩ {len(df_mrrsu)} mrrsu)"
            )

            dropout = args.dropout if args.dropout is not None else 0.1
            model = SpectralHybridClassifier(
                n_mrral=59, n_mrrsu=60, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f"Loaded MAE encoder from {args.pretrain_ckpt}. "
                    f"Missing: {missing}, Unexpected: {unexpected}"
                )

            metrics = train_torch_model(
                model=model, df=df_combined, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                use_spectral_aug=args.spectral_aug,
                aug_noise_std=args.aug_noise_std,
                aug_band_dropout=args.aug_band_dropout,
                aug_shift_std=args.aug_shift_std,
                encoder_lr_scale=args.encoder_lr_scale,
            )
```

- [ ] **Step 4: Run all tests**

```bash
conda run -n crism pytest tests/ -x -q
```
Expected: all pass

- [ ] **Step 5: Smoke-test the hybrid model**

```bash
conda run -n crism python scripts/train.py \
    --model spectral_hybrid \
    --run_name shybrid_smoke \
    --epochs 2 --patience 5 \
    --embed_dim 64 --n_heads 2 --n_layers 2 \
    --asl_loss --no_wandb
```
Expected: completes 2 epochs, prints metrics, no crash.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py training/train_torch.py tests/test_train_torch.py
git commit -m "feat: wire SpectralHybridClassifier into training pipeline (--model spectral_hybrid)"
```

---

## Chunk 4: Sweep v6

### Task 7: Create scripts/sweep_v6.py

**Files:**
- Create: `scripts/sweep_v6.py`

This sweep tests each improvement independently and in the full combination. It uses the
existing 4-layer MAE checkpoint (`mae_pretrain_128d_4l_best.pt`) so all configs except those
requiring the new 6-layer MAE can run immediately without waiting for pretraining.

The configs are designed as a clean ablation:
- A: SpectralCNN + ASL (CNN baseline with new loss)
- B: SpectralViT + ASL (transformer + new loss, no pretrain)
- C: SpectralViT + ASL + differential LR + 4-layer MAE pretrain (tests diff LR benefit)
- D: Hybrid (mrral+mrrsu) + ASL (tests combined features, no pretrain)
- E: Hybrid + ASL + differential LR + 4-layer MAE pretrain (full combination)

Configs C and E use `n_layers=4` to match the existing 4-layer MAE checkpoint.

- [ ] **Step 1: Create the sweep file**

Create `scripts/sweep_v6.py`:

```python
"""
Sweep v6: ASL + differential LR + combined mrral/mrrsu features.

Changes from v5:
  - AsymmetricLoss (gamma_neg=4, gamma_pos=0) replaces focal loss
  - Differential LR (encoder_lr = lr * 0.1) for MAE-pretrained configs
  - SpectralHybridClassifier: mrral encoder + mrrsu summary param branch
  - spectral_aug re-enabled with moderate settings
  - All configs use n_layers=4 when loading 4-layer MAE checkpoint

Configs:
  A. scnn_asl_v6:         SpectralCNN  + ASL                          (CNN baseline)
  B. svit_asl_v6:         SpectralViT  + ASL               (6L, no pretrain)
  C. svit_asl_diffr_v6:   SpectralViT  + ASL + diff LR  + 4L MAE  ← target (mrral only)
  D. shybrid_asl_v6:      Hybrid       + ASL               (4L, no pretrain)
  E. shybrid_asl_diffr_v6: Hybrid      + ASL + diff LR  + 4L MAE  ← primary target

Usage:
    python scripts/sweep_v6.py
    python scripts/sweep_v6.py --dry_run
    python scripts/sweep_v6.py --only scnn_asl_v6 svit_asl_v6
"""
import argparse
import csv
import os
import subprocess
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(PROJ, 'scripts', 'train.py')
CKPT_DIR = os.path.join(PROJ, 'checkpoints')
LOG_DIR = os.path.join(PROJ, 'logs')
MAE4_CKPT = os.path.join(CKPT_DIR, 'mae_pretrain_128d_4l_best.pt')

SWEEP_CONFIGS = [
    # A: SpectralCNN + ASL
    dict(model='spectral_cnn', run_name='scnn_asl_v6',
         epochs=200, patience=30, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         warmup_epochs=0, lr_t_max=50),

    # B: SpectralViT + ASL (6L, no pretrain)
    dict(model='spectral_vit', run_name='svit_asl_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         warmup_epochs=5, lr_t_max=50),

    # C: SpectralViT + ASL + differential LR + 4-layer MAE (n_layers=4 to match checkpoint)
    dict(model='spectral_vit', run_name='svit_asl_diffr_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         encoder_lr_scale=0.1,
         pretrain_ckpt=MAE4_CKPT,
         warmup_epochs=5, lr_t_max=50),

    # D: Hybrid (mrral+mrrsu) + ASL (4L, no pretrain)
    dict(model='spectral_hybrid', run_name='shybrid_asl_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         warmup_epochs=5, lr_t_max=50),

    # E: Hybrid + ASL + differential LR + 4-layer MAE (primary target)
    dict(model='spectral_hybrid', run_name='shybrid_asl_diffr_v6',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=4,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
         spectral_aug=True, aug_noise_std=0.01, aug_band_dropout=0.10, aug_shift_std=0.01,
         encoder_lr_scale=0.1,
         pretrain_ckpt=MAE4_CKPT,
         warmup_epochs=5, lr_t_max=50),
]

BOOL_FLAGS = {
    'use_pos_weight', 'high_conf_only', 'focal_loss',
    'balanced_sampling', 'spectral_aug', 'asl_loss',
}


def ckpt_exists(run_name: str) -> bool:
    return os.path.exists(os.path.join(CKPT_DIR, f'{run_name}_best.pt'))


def config_to_args(cfg: dict) -> list:
    args = ['python', TRAIN]
    for k, v in cfg.items():
        if k in BOOL_FLAGS:
            if v:
                args.append(f'--{k}')
        elif v is not None:
            args += [f'--{k}', str(v)]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these run_names (space-separated)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if checkpoint already exists')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    configs = SWEEP_CONFIGS
    if args.only:
        configs = [c for c in configs if c['run_name'] in args.only]

    total = len(configs)
    results = []

    for i, cfg in enumerate(configs):
        run_name = cfg['run_name']
        print(f'\n[{i+1}/{total}] {run_name}', flush=True)
        if not args.force and ckpt_exists(run_name):
            print('  SKIPPING — checkpoint exists (use --force to rerun)', flush=True)
            continue
        pretrain = cfg.get('pretrain_ckpt')
        if pretrain and not os.path.exists(pretrain):
            print(f'  SKIPPING — MAE checkpoint not found: {pretrain}', flush=True)
            continue
        cmd = config_to_args(cfg)
        if args.dry_run:
            print(f'  DRY RUN: {" ".join(cmd)}', flush=True)
            continue
        print(f'  CMD: {" ".join(cmd)}', flush=True)
        result = subprocess.run(['conda', 'run', '-n', 'crism'] + cmd, cwd=PROJ)
        status = 'ok' if result.returncode == 0 else f'FAILED({result.returncode})'
        results.append({'run_name': run_name, 'status': status})
        print(f'  {status}', flush=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results and not args.dry_run:
        out = os.path.join(LOG_DIR, f'sweep_v6_{stamp}.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'status'])
            w.writeheader()
            w.writerows(results)
        print(f'\nSweep summary: {out}')

    print(f'\nDone. {len(results)} ran, {sum(1 for r in results if r["status"] == "ok")} ok.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Dry-run to verify arg generation**

```bash
conda run -n crism python scripts/sweep_v6.py --dry_run
```
Expected: prints 5 CMD lines, all valid python scripts/train.py invocations with correct flags.

- [ ] **Step 3: Verify existing MAE checkpoint is present**

```bash
ls -lh checkpoints/mae_pretrain_128d_4l_best.pt
```
Expected: file exists, ~4MB. If missing, retrain with:
```bash
conda run -n crism python scripts/pretrain_mae.py --n_layers 4 --no_wandb
```

- [ ] **Step 4: Run the full test suite before sweep**

```bash
conda run -n crism pytest tests/ -x -q
```
Expected: all pass.

- [ ] **Step 5: Commit sweep script**

```bash
git add scripts/sweep_v6.py
git commit -m "feat: add sweep_v6.py ablation (ASL + diff LR + hybrid model)"
```

- [ ] **Step 6: Launch sweep**

```bash
nohup conda run -n crism python scripts/sweep_v6.py \
    > logs/sweep_v6_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Sweep PID: $!"
```

Expected runtime: ~3 hours (5 configs × ~35 min each).
Monitor: `tail -f logs/sweep_v6_*.log`

---

## Expected Outcomes

| Config | Key change | Expected mAP |
|--------|-----------|--------------|
| scnn_asl_v6 | CNN + ASL | 0.64–0.67 |
| svit_asl_v6 | ViT + ASL | 0.65–0.68 |
| svit_asl_diffr_v6 | + differential LR + MAE | 0.68–0.72 |
| shybrid_asl_v6 | combined features + ASL | 0.67–0.71 |
| shybrid_asl_diffr_v6 | full combination | **0.70–0.75** |

If `shybrid_asl_diffr_v6` does not reach 0.70, the next lever is: retrain MAE with n_layers=6
(`python scripts/pretrain_mae.py --n_layers 6`), then re-run config E with n_layers=6.
