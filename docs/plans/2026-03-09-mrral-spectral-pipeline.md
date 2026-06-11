# CRISM mrral Spectral Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace mrrsu summary-product features with raw mrral hyperspectral reflectance spectra (59 bands, 410–2457 nm), add label-quality filtering, improved loss functions, and MAE pre-training to push mAP toward 0.90.

**Architecture:** New `mrral_pixels.parquet` stores 59-band reflectance spectra extracted from mrral rasters using the same GPKG polygons as the mrrsu pipeline. Per-pixel spectral classification models (1D CNN, Spectral Transformer) operate on this spectrum. A masked-autoencoder (MAE) is pre-trained on the full mrral corpus then fine-tuned on labeled pixels. All approaches are compared in a structured ablation sweep.

**Tech Stack:** rasterio, spectral, numpy, pandas/pyarrow, PyTorch, wandb. All runs use conda env `crism`. Config at `config.yaml`. Data at `/mnt/gigas/CRISM/MRDR/`.

**Critical domain facts:**
- mrral bands 0-58 (0-indexed) cover 410–2457 nm — the 13 bands from index 59 onward (2530–3923 nm) are too noisy and must be excluded.
- Only **9,452 High-confidence plagioclase pixels** exist in training. Label quality (% High-conf) directly predicts per-class AP. Training on High-conf-only removes ~93% noisy plagioclase labels.
- mrrsu `b0..b59` are existing parquet columns. mrral columns are named `m0..m58` to avoid collision.
- NODATA value is 65535 in both products.
- Classes: `['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']`
- All test runs: `conda run -n crism pytest tests/ -x -q`

---

## Task 1: mrral data extraction

**Files:**
- Modify: `data/extract_pixels.py`
- Create: `scripts/build_mrral_dataset.py`
- Test: `tests/test_mrral_extraction.py`

**Context:** `extract_pixels.py` already extracts mrrsu pixels. We need an mrral variant that (a) finds mrral files, (b) reads only the first 59 bands, (c) writes `data/mrral_pixels.parquet` with columns `m0..m58` plus the same metadata columns.

**Step 1: Write failing tests**

```python
# tests/test_mrral_extraction.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

def test_find_mrral_pairs_returns_mrral_paths():
    from data.extract_pixels import find_mrral_pairs
    cfg_data_root = '/mnt/gigas/CRISM/MRDR'
    cfg_gpkg_dir = '/mnt/gigas/CRISM/MRDR/categorized_mineral_units'
    pairs = find_mrral_pairs(cfg_gpkg_dir, cfg_data_root)
    assert len(pairs) > 0
    for tile_id, gpkg_path, mrral_path in pairs:
        assert 'mrral' in mrral_path.lower()
        assert os.path.exists(mrral_path)

def test_mrral_records_have_59_spectral_columns():
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair
    cfg_data_root = '/mnt/gigas/CRISM/MRDR'
    cfg_gpkg_dir = '/mnt/gigas/CRISM/MRDR/categorized_mineral_units'
    pairs = find_mrral_pairs(cfg_gpkg_dir, cfg_data_root)
    tile_id, gpkg_path, mrral_path = pairs[0]
    records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
    assert len(records) > 0
    r = records[0]
    for i in range(59):
        assert f'm{i}' in r, f'm{i} missing'
    for i in range(59, 72):
        assert f'm{i}' not in r, f'm{i} should not be present (> 2500nm)'

def test_mrral_records_no_nodata():
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair
    cfg_data_root = '/mnt/gigas/CRISM/MRDR'
    cfg_gpkg_dir = '/mnt/gigas/CRISM/MRDR/categorized_mineral_units'
    pairs = find_mrral_pairs(cfg_gpkg_dir, cfg_data_root)
    tile_id, gpkg_path, mrral_path = pairs[0]
    records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
    for r in records[:500]:
        vals = [r[f'm{i}'] for i in range(59)]
        assert all(v < 65535 for v in vals), "NODATA not filtered"
```

**Step 2: Run to verify failure**

```bash
conda run -n crism pytest tests/test_mrral_extraction.py -x -q
```
Expected: `ImportError` or `AttributeError: module has no attribute 'find_mrral_pairs'`

**Step 3: Implement `find_mrral_pairs` and `extract_mrral_pixels_from_pair` in `data/extract_pixels.py`**

Add after the existing `find_tile_pairs` function:

```python
MRRAL_N_BANDS = 59  # bands 0-58, wavelengths 410-2457 nm; bands 59-71 (>2500 nm) excluded

def find_mrral_pairs(
    gpkg_dir: str,
    data_root: str
) -> List[Tuple[str, str, str]]:
    """
    Find (tile_id, gpkg_path, mrral_path) triples.
    Mirrors find_tile_pairs but matches mrral instead of mrrsu files.
    """
    pairs = []
    for fname in sorted(os.listdir(gpkg_dir)):
        if not fname.endswith('.gpkg'):
            continue
        tile_id = fname.replace('.gpkg', '').lower()
        gpkg_path = os.path.join(gpkg_dir, fname)
        matches = glob.glob(
            os.path.join(data_root, '**', f'{tile_id}_mrral*.img'),
            recursive=True
        )
        if not matches:
            logger.warning(f"No mrral file found for {tile_id}, skipping.")
            continue
        pairs.append((tile_id, gpkg_path, sorted(matches)[0]))
    return pairs


def extract_mrral_pixels_from_pair(
    tile_id: str,
    mrral_path: str,
    gpkg_path: str,
    other_polygon_ids: Optional[Set] = None,
    gdf=None,
) -> List[Dict[str, Any]]:
    """
    Extract per-pixel mrral spectra (59 bands, 410-2457 nm) from one tile.

    Identical logic to extract_pixels_from_pair but:
    - reads mrral file instead of mrrsu
    - stores bands as m0..m58 (not b0..b59)
    - reads exactly MRRAL_N_BANDS=59 bands regardless of file band count
    """
    records = []

    with rasterio.open(mrral_path) as src:
        raster_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        actual_bands = min(MRRAL_N_BANDS, src.count)

        if gdf is None:
            gdf = gpd.read_file(gpkg_path)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        for poly_idx, row in gdf.iterrows():
            category = row.get('Category', '')
            if not category or isinstance(category, float):
                continue
            if other_polygon_ids is not None and 'other' in category.lower():
                if poly_idx not in other_polygon_ids:
                    continue
            try:
                label, conf_weight = parse_category(str(category))
            except ValueError:
                logger.warning(f"Could not parse {category!r} in {tile_id}, skipping.")
                continue
            conf_tier = get_confidence_tier(str(category))
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                mask = rasterize(
                    [(geom, 1)], out_shape=(height, width),
                    transform=transform, fill=0, dtype=np.uint8
                ).astype(bool)
            except Exception as e:
                logger.warning(f"Rasterize failed polygon {poly_idx} in {tile_id}: {e}")
                continue

            pixel_rows, pixel_cols = np.where(mask)
            if len(pixel_rows) == 0:
                continue

            row_min, row_max = int(pixel_rows.min()), int(pixel_rows.max()) + 1
            col_min, col_max = int(pixel_cols.min()), int(pixel_cols.max()) + 1
            window = rasterio.windows.Window(
                col_min, row_min, col_max - col_min, row_max - row_min
            )
            chunk = src.read(list(range(1, actual_bands + 1)), window=window)

            for r, c in zip(pixel_rows, pixel_cols):
                pixel_vals = chunk[:, r - row_min, c - col_min]
                if np.any(pixel_vals >= NODATA_VALUE):
                    continue
                if np.any(np.isnan(pixel_vals)):
                    pixel_vals = np.nan_to_num(pixel_vals, nan=0.0)
                record: Dict[str, Any] = {
                    'tile_id': tile_id,
                    'polygon_id': int(poly_idx),
                    'pixel_row': int(r),
                    'pixel_col': int(c),
                }
                for b_idx in range(actual_bands):
                    record[f'm{b_idx}'] = float(pixel_vals[b_idx])
                for cls_idx, cls_name in enumerate(CLASSES):
                    record[cls_name] = float(label[cls_idx])
                record['confidence_weight'] = float(conf_weight)
                record['confidence_tier'] = conf_tier
                records.append(record)

    return records
```

**Step 4: Create `scripts/build_mrral_dataset.py`**

```python
"""
Extract mrral (59-band, 410-2457 nm) spectra for all labeled polygons.
Writes data/mrral_pixels.parquet.

Usage:
    conda run -n crism python scripts/build_mrral_dataset.py
"""
import os, sys, logging, yaml
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'config.yaml')))
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair

    pairs = find_mrral_pairs(cfg['gpkg_dir'], cfg['data_root'])
    logging.info(f"Found {len(pairs)} mrral tile pairs")

    # Re-use same train/val/test split as mrrsu parquet — join on tile+polygon+row+col
    mrrsu_df = pd.read_parquet(os.path.join(cfg['output_dir'], 'pixels.parquet'))
    split_map = mrrsu_df.set_index(
        ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
    )['split'].to_dict()

    all_records = []
    for tile_id, gpkg_path, mrral_path in pairs:
        logging.info(f"Processing {tile_id}")
        records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
        logging.info(f"  {len(records)} pixels")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    # Assign same split as mrrsu dataset
    df['split'] = df.apply(
        lambda r: split_map.get(
            (r['tile_id'], r['polygon_id'], r['pixel_row'], r['pixel_col']), 'train'
        ), axis=1
    )
    out = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df.to_parquet(out, index=False)
    logging.info(f"Wrote {len(df)} pixels to {out}")
    logging.info(f"Splits: {df['split'].value_counts().to_dict()}")

if __name__ == '__main__':
    main()
```

**Step 5: Run tests**

```bash
conda run -n crism pytest tests/test_mrral_extraction.py -x -q
```
Expected: all 3 tests PASS.

**Step 6: Run extraction (background, ~20 min)**

```bash
nohup conda run -n crism python scripts/build_mrral_dataset.py \
    > logs/build_mrral_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

Verify when done:
```bash
conda run -n crism python -c "
import pandas as pd
df = pd.read_parquet('data/mrral_pixels.parquet')
print(df.shape, df.columns[:10].tolist(), df['split'].value_counts().to_dict())
"
```
Expected: ~900k rows, columns include `m0..m58`.

**Step 7: Commit**

```bash
git add data/extract_pixels.py scripts/build_mrral_dataset.py tests/test_mrral_extraction.py
git commit -m "feat: extract mrral 59-band spectra for all labeled polygons"
```

---

## Task 2: High-confidence-only training mode

**Files:**
- Modify: `data/dataset.py`
- Modify: `training/train_torch.py`
- Modify: `scripts/train.py`
- Test: `tests/test_dataset.py`

**Context:** 93% of plagioclase labels are Low/Moderate confidence and confuse the model. Adding a `high_conf_only=True` flag filters the training split to `confidence_tier == 'High'` pixels only. Validation/test always use all confidence tiers.

**Step 1: Add tests to `tests/test_dataset.py`**

```python
def test_crism_pixel_dataset_high_conf_only():
    import pandas as pd, torch
    from data.dataset import CRISMPixelDataset
    parquet = '/mnt/gigas/CRISM/MRDR/crism_classification/data/pixels.parquet'
    df = pd.read_parquet(parquet)
    train_all = df[df['split'] == 'train']
    train_high = df[(df['split'] == 'train') & (df['confidence_tier'] == 'High')]
    ds_all = CRISMPixelDataset(train_all)
    ds_high = CRISMPixelDataset(train_high)
    assert len(ds_high) < len(ds_all)
    assert len(ds_high) > 0

def test_high_conf_has_no_low_conf_pixels():
    import pandas as pd
    parquet = '/mnt/gigas/CRISM/MRDR/crism_classification/data/pixels.parquet'
    df = pd.read_parquet(parquet)
    high = df[df['confidence_tier'] == 'High']
    assert 'Low' not in high['confidence_tier'].values
```

**Step 2: Run to verify tests pass already** (they should — `CRISMPixelDataset` already accepts any DataFrame)

```bash
conda run -n crism pytest tests/test_dataset.py -x -q
```

**Step 3: Add `high_conf_only` parameter to `train_torch_model` in `training/train_torch.py`**

In `train_torch_model` signature, add:
```python
high_conf_only: bool = False,
```

In the body, after loading datasets, insert this filter before `make_dataset`:
```python
# Filter training split to High-confidence only if requested
train_df = df[df['split'] == 'train']
if high_conf_only:
    train_df = train_df[train_df['confidence_tier'] == 'High']
    logger.info(f"high_conf_only: training on {len(train_df)} High-conf pixels "
                f"(down from {(df['split']=='train').sum()})")
val_df = df[df['split'] == 'val']
```

Then update `make_dataset` to use `train_df` and `val_df` directly:
```python
def make_dataset(sub_df):
    if use_patches:
        return CRISMPatchDataset(sub_df, mrrsu_map, patch_size=patch_size,
                                 cache_dir=cache_dir, split=...)
    return CRISMPixelDataset(sub_df)

train_ds = make_dataset(train_df)
val_ds = make_dataset(val_df)
```

Also update `pos_weight` computation to use `train_df` instead of `df[df['split']=='train']`.

**Step 4: Add `--high_conf_only` CLI arg to `scripts/train.py`**

```python
parser.add_argument('--high_conf_only', action='store_true',
                    help='Train on High-confidence pixels only')
```

Pass to `train_torch_model`:
```python
high_conf_only=args.high_conf_only,
```

**Step 5: Run full test suite**

```bash
conda run -n crism pytest tests/ -x -q
```
Expected: all tests pass.

**Step 6: Commit**

```bash
git add data/dataset.py training/train_torch.py scripts/train.py tests/test_dataset.py
git commit -m "feat: add high_conf_only training mode to filter noisy labels"
```

---

## Task 3: mrral spectral dataset class

**Files:**
- Modify: `data/dataset.py`
- Test: `tests/test_dataset.py`

**Context:** `CRISMPixelDataset` uses mrrsu bands `b0..b59`. We need `CRISMSpectralDataset` for mrral using columns `m0..m58` (59 bands). Also expose `MRRAL_BAND_COLS` constant. This class is used by all mrral-based models.

**Step 1: Add tests**

```python
def test_crism_spectral_dataset_shape():
    import pandas as pd, torch
    parquet = '/mnt/gigas/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet'
    if not os.path.exists(parquet):
        pytest.skip("mrral_pixels.parquet not yet built")
    df = pd.read_parquet(parquet)
    train = df[df['split'] == 'train']
    from data.dataset import CRISMSpectralDataset
    ds = CRISMSpectralDataset(train)
    feat, label, weight = ds[0]
    assert feat.shape == (59,), f"Expected (59,), got {feat.shape}"
    assert label.shape == (6,)
    assert weight.shape == ()

def test_crism_spectral_dataset_high_conf_only():
    import pandas as pd
    parquet = '/mnt/gigas/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet'
    if not os.path.exists(parquet):
        pytest.skip("mrral_pixels.parquet not yet built")
    df = pd.read_parquet(parquet)
    train_all = df[df['split'] == 'train']
    train_high = df[(df['split'] == 'train') & (df['confidence_tier'] == 'High')]
    from data.dataset import CRISMSpectralDataset
    ds_all = CRISMSpectralDataset(train_all)
    ds_high = CRISMSpectralDataset(train_high)
    assert len(ds_high) < len(ds_all)
```

**Step 2: Add to `data/dataset.py`**

```python
MRRAL_BAND_COLS = [f'm{i}' for i in range(59)]  # 59 bands, 410-2457 nm


class CRISMSpectralDataset(Dataset):
    """
    Per-pixel dataset using mrral 59-band reflectance spectra (m0..m58).
    Used for all mrral-based models (SpectralMLP, SpectralCNN1D, SpectralTransformer, MAE).
    """

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in MRRAL_BAND_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"mrral parquet missing columns: {missing[:5]}... "
                             f"Run scripts/build_mrral_dataset.py first.")
        self.features = torch.tensor(df[MRRAL_BAND_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]
```

**Step 3: Run tests**

```bash
conda run -n crism pytest tests/test_dataset.py -x -q
```
Expected: all pass (spectral tests skip until mrral parquet built).

**Step 4: Commit**

```bash
git add data/dataset.py tests/test_dataset.py
git commit -m "feat: add CRISMSpectralDataset for mrral 59-band spectra"
```

---

## Task 4: Focal loss and class-balanced sampler

**Files:**
- Modify: `training/losses.py`
- Modify: `training/train_torch.py`
- Modify: `scripts/train.py`
- Test: `tests/test_losses.py`

**Context:** Standard BCE treats all examples equally. Focal loss down-weights easy negatives (γ=2), forcing the model to focus on hard plagioclase/HCP examples. Class-balanced sampling oversamples rare classes by giving each pixel a weight proportional to the rarity of its most-rare positive class.

**Step 1: Add tests to `tests/test_losses.py`**

```python
def test_focal_loss_down_weights_easy_examples():
    import torch
    from training.losses import FocalBCEWithLogitsLoss, WeightedBCEWithLogitsLoss
    # Very confident correct prediction — focal should give much lower loss than BCE
    logits = torch.tensor([[5.0, -5.0, 5.0, -5.0, 5.0, -5.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    weights = torch.ones(1)
    bce_loss = WeightedBCEWithLogitsLoss()(logits, targets, weights)
    focal_loss = FocalBCEWithLogitsLoss(gamma=2.0)(logits, targets, weights)
    assert focal_loss < bce_loss, "Focal loss should be lower for easy (confident correct) examples"

def test_focal_loss_same_as_bce_when_gamma_zero():
    import torch
    from training.losses import FocalBCEWithLogitsLoss, WeightedBCEWithLogitsLoss
    torch.manual_seed(0)
    logits = torch.randn(8, 6)
    targets = (torch.randn(8, 6) > 0).float()
    weights = torch.ones(8)
    bce = WeightedBCEWithLogitsLoss()(logits, targets, weights)
    focal0 = FocalBCEWithLogitsLoss(gamma=0.0)(logits, targets, weights)
    assert abs(bce.item() - focal0.item()) < 1e-5

def test_build_class_balanced_weights():
    import numpy as np, pandas as pd
    from training.train_torch import build_class_balanced_weights
    # Construct fake df with imbalanced labels
    n = 1000
    labels = np.zeros((n, 6))
    labels[:10, 4] = 1.0   # 10 plagioclase positives (very rare)
    labels[:500, 0] = 1.0  # 500 olivine_t1 positives (common)
    df = pd.DataFrame(labels, columns=['olivine_t1','olivine_t2','lcp','hcp','plagioclase','other'])
    df['confidence_weight'] = 1.0
    weights = build_class_balanced_weights(df)
    # Plagioclase-positive pixels should have higher weight
    plag_idx = np.where(labels[:, 4] > 0.4)[0]
    non_plag_idx = np.where(labels[:, 4] <= 0.4)[0]
    assert weights[plag_idx].mean() > weights[non_plag_idx].mean() * 5
```

**Step 2: Run to verify failure**

```bash
conda run -n crism pytest tests/test_losses.py -x -q
```
Expected: `ImportError: cannot import name 'FocalBCEWithLogitsLoss'`

**Step 3: Add `FocalBCEWithLogitsLoss` to `training/losses.py`**

```python
class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal binary cross-entropy, weighted per sample.

    Applies focal modulation (1 - p_t)^gamma to standard BCE, down-weighting
    easy examples and focusing training on hard/rare ones (e.g. plagioclase).

    gamma=2.0 is standard; gamma=0.0 reduces to weighted BCE.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,       # (batch, n_classes)
        targets: torch.Tensor,      # (batch, n_classes)
        weights: torch.Tensor,      # (batch,)
        pos_weight: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).mean(dim=1)
        return (loss * weights).sum() / (weights.sum() + 1e-8)
```

**Step 4: Add `build_class_balanced_weights` to `training/train_torch.py`**

Add this function before `train_torch_model`:

```python
def build_class_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Build per-pixel sampling weights to oversample rare-class positives.

    Each pixel receives weight = max imbalance ratio of any class it is
    positive for. Plagioclase/HCP pixels get ~20–50x the weight of common
    olivine pixels.
    """
    from data.dataset import LABEL_COLS
    labels = df[LABEL_COLS].values.astype('float32')
    n_pos = (labels > 0.4).sum(axis=0).clip(min=1)
    n_neg = len(labels) - n_pos
    imbalance = n_neg / n_pos  # higher = rarer class

    pixel_weights = np.ones(len(labels), dtype=np.float32)
    is_pos = labels > 0.4  # (n, 6)
    for i in range(len(labels)):
        if is_pos[i].any():
            pixel_weights[i] = float(imbalance[is_pos[i]].max())
    return pixel_weights
```

**Step 5: Wire up `focal_loss` and `balanced_sampling` in `train_torch_model`**

In the signature add:
```python
use_focal_loss: bool = False,
focal_gamma: float = 2.0,
use_balanced_sampling: bool = False,
```

Replace loss_fn construction:
```python
if use_focal_loss:
    from training.losses import FocalBCEWithLogitsLoss
    loss_fn = FocalBCEWithLogitsLoss(gamma=focal_gamma)
else:
    loss_fn = WeightedBCEWithLogitsLoss()
```

Replace `train_loader` construction:
```python
if use_balanced_sampling:
    from torch.utils.data import WeightedRandomSampler
    pw = build_class_balanced_weights(train_df)
    sampler = WeightedRandomSampler(pw, num_samples=len(pw), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
else:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
```

**Step 6: Add CLI args to `scripts/train.py`**

```python
parser.add_argument('--focal_loss', action='store_true')
parser.add_argument('--focal_gamma', type=float, default=2.0)
parser.add_argument('--balanced_sampling', action='store_true')
```
Pass all three to `train_torch_model`.

**Step 7: Run tests**

```bash
conda run -n crism pytest tests/test_losses.py -x -q
```
Expected: all 3 tests PASS.

**Step 8: Commit**

```bash
git add training/losses.py training/train_torch.py scripts/train.py tests/test_losses.py
git commit -m "feat: add focal loss and class-balanced sampler for plagioclase/hcp"
```

---

## Task 5: Spectral augmentation

**Files:**
- Create: `training/augmentations.py`
- Modify: `training/train_torch.py`
- Modify: `scripts/train.py`
- Test: `tests/test_augmentations.py`

**Context:** Spectral augmentation improves generalization on hyperspectral data. Three transforms: (1) Gaussian noise on all bands, (2) random band dropout (zero out k random bands), (3) spectral shift (add constant offset). Applied only during training, not validation.

**Step 1: Write tests**

```python
# tests/test_augmentations.py
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_spectral_augmentation_preserves_shape():
    from training.augmentations import SpectralAugmentation
    aug = SpectralAugmentation(noise_std=0.005, band_dropout=0.15, shift_std=0.01)
    x = torch.ones(59)
    out = aug(x)
    assert out.shape == (59,), f"Shape changed: {out.shape}"

def test_spectral_augmentation_applies_noise():
    import torch
    from training.augmentations import SpectralAugmentation
    torch.manual_seed(42)
    aug = SpectralAugmentation(noise_std=0.1, band_dropout=0.0, shift_std=0.0)
    x = torch.zeros(59)
    out = aug(x)
    assert not torch.allclose(out, x), "Noise should modify the spectrum"

def test_spectral_augmentation_band_dropout():
    from training.augmentations import SpectralAugmentation
    import torch
    torch.manual_seed(0)
    aug = SpectralAugmentation(noise_std=0.0, band_dropout=0.5, shift_std=0.0)
    x = torch.ones(59)
    out = aug(x)
    n_zeros = (out == 0).sum().item()
    assert n_zeros > 0, "Band dropout should zero some bands"
    assert n_zeros < 59, "Band dropout should not zero all bands"

def test_no_augmentation_in_eval_mode():
    from training.augmentations import SpectralAugmentation
    import torch
    aug = SpectralAugmentation(noise_std=1.0, band_dropout=0.5, shift_std=1.0)
    aug.eval()
    x = torch.ones(59)
    out = aug(x)
    assert torch.allclose(out, x), "No augmentation should be applied in eval mode"
```

**Step 2: Create `training/augmentations.py`**

```python
"""
Spectral augmentation transforms for mrral hyperspectral data.
Applied only in training mode (aug.train()); identity in eval mode.
"""
import torch
import torch.nn as nn


class SpectralAugmentation(nn.Module):
    """
    Stochastic spectral augmentation for 1D reflectance spectra.

    Three independent transforms applied in sequence:
    1. Gaussian noise:   x += N(0, noise_std)
    2. Band dropout:     randomly zero out each band with probability band_dropout
    3. Spectral shift:   x += N(0, shift_std) (same offset all bands)

    In eval mode, all transforms are disabled (returns input unchanged).
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        band_dropout: float = 0.10,
        shift_std: float = 0.005,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.band_dropout = band_dropout
        self.shift_std = shift_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_bands,) or (batch, n_bands)"""
        if not self.training:
            return x
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.band_dropout > 0:
            mask = torch.bernoulli(
                torch.full(x.shape, 1 - self.band_dropout, device=x.device)
            )
            x = x * mask
        if self.shift_std > 0:
            shift_shape = x.shape[:-1] + (1,) if x.dim() > 1 else (1,)
            x = x + torch.randn(shift_shape, device=x.device) * self.shift_std
        return x
```

**Step 3: Wire augmentation into `train_torch_model`**

In `training/train_torch.py`, signature add:
```python
use_spectral_aug: bool = False,
aug_noise_std: float = 0.005,
aug_band_dropout: float = 0.10,
aug_shift_std: float = 0.005,
```

Create augmentation object before training loop:
```python
augment = None
if use_spectral_aug:
    from training.augmentations import SpectralAugmentation
    augment = SpectralAugmentation(
        noise_std=aug_noise_std,
        band_dropout=aug_band_dropout,
        shift_std=aug_shift_std,
    ).to(device)
```

In the training loop, after `features = features.to(device)`:
```python
if augment is not None:
    augment.train()
    features = augment(features)
```

**Step 4: Add CLI args to `scripts/train.py`**

```python
parser.add_argument('--spectral_aug', action='store_true')
parser.add_argument('--aug_noise_std', type=float, default=0.005)
parser.add_argument('--aug_band_dropout', type=float, default=0.10)
parser.add_argument('--aug_shift_std', type=float, default=0.005)
```

**Step 5: Run tests**

```bash
conda run -n crism pytest tests/test_augmentations.py -x -q
```
Expected: all 4 tests PASS.

**Step 6: Commit**

```bash
git add training/augmentations.py training/train_torch.py scripts/train.py tests/test_augmentations.py
git commit -m "feat: spectral augmentation (noise, band dropout, shift) for mrral training"
```

---

## Task 6: 1D Spectral CNN and Spectral Transformer models

**Files:**
- Create: `models/spectral_cnn.py`
- Create: `models/spectral_transformer.py`
- Modify: `scripts/train.py`
- Test: `tests/test_models.py`

**Context:** Two new per-pixel spectral models operating on the 59-band mrral spectrum. `SpectralCNN1D` uses 1D convolutions over the spectral axis (analogous to processing an audio waveform). `SpectralTransformer` treats each band as a token with positional encoding. Both output 6-class logits.

**Step 1: Add tests to `tests/test_models.py`**

```python
def test_spectral_cnn1d_forward_shape():
    import torch
    from models.spectral_cnn import SpectralCNN1D
    model = SpectralCNN1D(n_bands=59, n_classes=6)
    x = torch.randn(4, 59)
    out = model(x)
    assert out.shape == (4, 6)

def test_spectral_cnn1d_dropout_parameter():
    from models.spectral_cnn import SpectralCNN1D
    m = SpectralCNN1D(n_bands=59, n_classes=6, dropout=0.4)
    assert m is not None

def test_spectral_transformer_forward_shape():
    import torch
    from models.spectral_transformer import SpectralTransformer
    model = SpectralTransformer(n_bands=59, n_classes=6, embed_dim=64, n_heads=4, n_layers=4)
    x = torch.randn(4, 59)
    out = model(x)
    assert out.shape == (4, 6)

def test_spectral_transformer_mask_token():
    import torch
    from models.spectral_transformer import SpectralTransformer
    model = SpectralTransformer(n_bands=59, n_classes=6)
    # Should accept masked input (zeros for masked bands)
    x = torch.randn(2, 59)
    x[:, 10:20] = 0.0   # simulate masked bands
    out = model(x)
    assert out.shape == (2, 6)
```

**Step 2: Create `models/spectral_cnn.py`**

```python
"""
1D Spectral CNN for per-pixel mineral classification.
Treats the 59-band mrral spectrum as a 1D signal and applies convolutional
feature extraction along the spectral dimension.
"""
import torch
import torch.nn as nn


class SpectralCNN1D(nn.Module):
    """
    1D CNN operating on a single pixel's reflectance spectrum.

    Input:  (batch, n_bands)  — e.g. (batch, 59) for mrral
    Output: (batch, n_classes) — raw logits

    Architecture:
        spectrum → unsqueeze → Conv1d stack → global avg pool → dropout → linear
    """

    def __init__(self, n_bands: int = 59, n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: local spectral patterns (kernel 5 covers ~100nm)
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            # Block 2: medium-range absorption features
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            # Block 3: broad spectral shape
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands)
        x = x.unsqueeze(1)           # (batch, 1, n_bands)
        x = self.features(x)         # (batch, 256, n_bands)
        x = self.pool(x).squeeze(2)  # (batch, 256)
        return self.classifier(x)    # (batch, n_classes)
```

**Step 3: Create `models/spectral_transformer.py`**

```python
"""
Spectral Transformer for per-pixel mineral classification.
Treats each spectral band as a token (with learned positional embedding).
Used both for classification and as the MAE encoder backbone.
"""
import torch
import torch.nn as nn
import math


class SpectralTransformer(nn.Module):
    """
    Transformer operating on a pixel's reflectance spectrum.

    Each of the 59 bands is projected to embed_dim, positional embeddings added,
    then processed through n_layers Transformer encoder blocks. A CLS token
    aggregates the sequence for classification.

    Input:  (batch, n_bands)
    Output: (batch, n_classes) — raw logits
    """

    def __init__(
        self,
        n_bands: int = 59,
        n_classes: int = 6,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.embed_dim = embed_dim

        # Project each scalar band value to embed_dim
        self.band_embed = nn.Linear(1, embed_dim)
        # Learned positional embedding for each band position
        self.pos_embed = nn.Embedding(n_bands + 1, embed_dim)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands)
        B = x.shape[0]
        # Embed each band: (batch, n_bands, embed_dim)
        tokens = self.band_embed(x.unsqueeze(-1))
        # Add positional embeddings for band positions 1..n_bands
        pos_ids = torch.arange(1, self.n_bands + 1, device=x.device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)
        # Prepend CLS token (position 0)
        cls = self.cls_token.expand(B, -1, -1)
        cls = cls + self.pos_embed(torch.zeros(1, device=x.device, dtype=torch.long))
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_bands+1, embed_dim)
        # Transformer
        out = self.encoder(tokens)
        cls_out = self.norm(out[:, 0])             # CLS token
        return self.head(cls_out)
```

**Step 4: Add `spectral_cnn` and `spectral_vit` to `scripts/train.py`**

In `TORCH_MODELS`:
```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit'}
```

In the model construction block, add:
```python
elif args.model == 'spectral_cnn':
    from models.spectral_cnn import SpectralCNN1D
    dropout = args.dropout if args.dropout is not None else 0.3
    model = SpectralCNN1D(n_bands=59, n_classes=6, dropout=dropout)
    df_mrral = pd.read_parquet(os.path.join(
        os.path.dirname(parquet_path), 'mrral_pixels.parquet'))
    metrics = train_torch_model(
        model=model, df=df_mrral, model_name=run_name,
        max_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
        use_pos_weight=args.use_pos_weight, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, lr_t_max=args.lr_t_max,
        use_focal_loss=args.focal_loss, focal_gamma=args.focal_gamma,
        use_balanced_sampling=args.balanced_sampling,
        use_spectral_aug=args.spectral_aug,
        high_conf_only=args.high_conf_only,
    )

elif args.model == 'spectral_vit':
    from models.spectral_transformer import SpectralTransformer
    dropout = args.dropout if args.dropout is not None else 0.1
    model = SpectralTransformer(
        n_bands=59, n_classes=6,
        embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=dropout,
    )
    df_mrral = pd.read_parquet(os.path.join(
        os.path.dirname(parquet_path), 'mrral_pixels.parquet'))
    metrics = train_torch_model(
        model=model, df=df_mrral, model_name=run_name,
        max_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
        use_pos_weight=args.use_pos_weight, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, lr_t_max=args.lr_t_max,
        use_focal_loss=args.focal_loss, focal_gamma=args.focal_gamma,
        use_balanced_sampling=args.balanced_sampling,
        use_spectral_aug=args.spectral_aug,
        high_conf_only=args.high_conf_only,
    )
```

Also update `train_torch_model` to use `CRISMSpectralDataset` when `model` is a `SpectralCNN1D` or `SpectralTransformer`. The cleanest way: check if `MRRAL_BAND_COLS[0]` is in `df.columns`; if so, use `CRISMSpectralDataset`, else use `CRISMPixelDataset`.

In `make_dataset` inside `train_torch_model`:
```python
def make_dataset(sub_df):
    if use_patches:
        return CRISMPatchDataset(sub_df, mrrsu_map, patch_size=patch_size,
                                 cache_dir=cache_dir, split=split_name)
    from data.dataset import MRRAL_BAND_COLS, CRISMSpectralDataset
    if MRRAL_BAND_COLS[0] in sub_df.columns:
        return CRISMSpectralDataset(sub_df)
    return CRISMPixelDataset(sub_df)
```

**Step 5: Run model tests**

```bash
conda run -n crism pytest tests/test_models.py -x -q
```
Expected: all tests PASS.

**Step 6: Commit**

```bash
git add models/spectral_cnn.py models/spectral_transformer.py scripts/train.py tests/test_models.py
git commit -m "feat: SpectralCNN1D and SpectralTransformer for mrral 59-band classification"
```

---

## Task 7: MAE pre-training

**Files:**
- Create: `models/mae.py`
- Create: `scripts/pretrain_mae.py`
- Test: `tests/test_mae.py`

**Context:** Masked Autoencoder (MAE) pre-training: randomly mask 40% of spectral bands and train the model to predict the masked values. This forces the encoder to learn meaningful spectral representations. The pre-trained encoder weights are then used to initialise the `SpectralTransformer` before fine-tuning on labeled data. We also train a pre-trained version of `SpectralCNN1D` for comparison.

MAE uses 899k labeled + any unlabeled pixels available. For now, train on labeled pixels only (sufficient). Use the `SpectralTransformer` encoder architecture from Task 6, add a small linear decoder head.

**Step 1: Write tests**

```python
# tests/test_mae.py
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_spectral_mae_forward():
    from models.mae import SpectralMAE
    model = SpectralMAE(n_bands=59, embed_dim=128, n_heads=4, n_layers=4, mask_ratio=0.4)
    x = torch.randn(4, 59)
    loss, pred, mask = model(x)
    assert loss.shape == (), "Loss should be scalar"
    assert pred.shape == (4, 59), f"Pred shape {pred.shape}"
    assert mask.shape == (4, 59), f"Mask shape {mask.shape}"
    assert 0.3 < mask.float().mean().item() < 0.5, "~40% should be masked"

def test_spectral_mae_encoder_extract():
    from models.mae import SpectralMAE
    model = SpectralMAE(n_bands=59, embed_dim=128, n_heads=4, n_layers=4)
    x = torch.randn(4, 59)
    embed = model.encode(x)
    assert embed.shape == (4, 128), f"Encoder output shape {embed.shape}"

def test_mae_pretrained_weights_loadable_into_spectral_transformer():
    from models.mae import SpectralMAE
    from models.spectral_transformer import SpectralTransformer
    mae = SpectralMAE(n_bands=59, embed_dim=128, n_heads=4, n_layers=4)
    classifier = SpectralTransformer(n_bands=59, n_classes=6,
                                     embed_dim=128, n_heads=4, n_layers=4)
    # Load encoder weights from MAE into classifier encoder
    state = mae.encoder_state_dict()
    missing, unexpected = classifier.load_encoder_state_dict(state)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
```

**Step 2: Create `models/mae.py`**

```python
"""
Spectral Masked Autoencoder (MAE) for CRISM mrral data.

Pre-trains a SpectralTransformer encoder to reconstruct randomly masked bands.
The encoder can then be loaded into SpectralTransformer for fine-tuning.

Reference: He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
           adapted for 1D spectral data.
"""
import torch
import torch.nn as nn
from models.spectral_transformer import SpectralTransformer


class SpectralMAE(nn.Module):
    """
    Masked Autoencoder for spectral data.

    Forward pass:
      1. Randomly mask mask_ratio fraction of bands (set to 0)
      2. Encode masked spectrum with SpectralTransformer encoder
      3. Decode CLS embedding to predict ALL 59 band values
      4. Compute MSE loss on masked bands only

    After pre-training:
      - Call encoder_state_dict() to extract encoder weights
      - Load into SpectralTransformer.load_encoder_state_dict()
    """

    def __init__(
        self,
        n_bands: int = 59,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        decoder_dim: int = 64,
        mask_ratio: float = 0.40,
        dropout: float = 0.0,   # no dropout during MAE pre-training
    ):
        super().__init__()
        self.n_bands = n_bands
        self.mask_ratio = mask_ratio

        # Encoder: shared with downstream SpectralTransformer
        self.encoder = SpectralTransformer(
            n_bands=n_bands, n_classes=embed_dim,  # head output = embed_dim (replaced below)
            embed_dim=embed_dim, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )
        # Replace classification head with identity (we use CLS embed directly)
        self.encoder.head = nn.Identity()

        # Decoder: lightweight MLP that predicts all band values from CLS token
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, n_bands),
        )

    def _random_mask(self, x: torch.Tensor) -> tuple:
        """Returns (masked_x, mask) where mask=True means band was masked."""
        B, N = x.shape
        n_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def forward(self, x: torch.Tensor):
        """
        Returns: (loss, pred, mask)
          loss: scalar MSE on masked bands
          pred: (B, n_bands) reconstructed spectrum
          mask: (B, n_bands) bool, True = was masked
        """
        x_masked, mask = self._random_mask(x)
        cls_embed = self.encoder(x_masked)  # (B, embed_dim) — from encoder.head=Identity
        pred = self.decoder(cls_embed)       # (B, n_bands)
        # MSE only on masked bands
        loss = ((pred - x) ** 2)[mask].mean()
        return loss, pred, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS embedding without masking. Shape: (B, embed_dim)."""
        return self.encoder(x)

    def encoder_state_dict(self) -> dict:
        """Return encoder weights (excluding the replaced head)."""
        return {k: v for k, v in self.encoder.state_dict().items()
                if not k.startswith('head.')}


# Add to SpectralTransformer in models/spectral_transformer.py:
# def load_encoder_state_dict(self, state: dict):
#     missing, unexpected = [], []
#     own = self.state_dict()
#     for k, v in state.items():
#         if k in own:
#             own[k] = v
#         else:
#             unexpected.append(k)
#     for k in own:
#         if k not in state and not k.startswith('head.'):
#             missing.append(k)
#     self.load_state_dict(own)
#     return missing, unexpected
```

**Step 3: Add `load_encoder_state_dict` to `models/spectral_transformer.py`**

```python
def load_encoder_state_dict(self, state: dict):
    """
    Load encoder weights from a pre-trained SpectralMAE.
    Skips the classification head (which is randomly initialized).
    Returns (missing_keys, unexpected_keys).
    """
    own = self.state_dict()
    unexpected = [k for k in state if k not in own]
    missing = [k for k in own if k not in state and not k.startswith('head.')]
    for k, v in state.items():
        if k in own:
            own[k] = v
    self.load_state_dict(own)
    return missing, unexpected
```

**Step 4: Create `scripts/pretrain_mae.py`**

```python
"""
MAE pre-training on mrral spectral data.

Usage:
    conda run -n crism python scripts/pretrain_mae.py
    conda run -n crism python scripts/pretrain_mae.py --epochs 100 --embed_dim 256

Saves checkpoint to: checkpoints/mae_pretrain_{embed_dim}d_{n_layers}l.pt
"""
import argparse, os, sys, logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.40)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'config.yaml')))
    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    ckpt_dir = cfg['checkpoints_dir']

    df = pd.read_parquet(parquet)
    # Use all pixels (train+val+test) for pretraining — no labels used
    from data.dataset import CRISMSpectralDataset
    ds = CRISMSpectralDataset(df)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.mae import SpectralMAE
    model = SpectralMAE(
        n_bands=59, embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, mask_ratio=args.mask_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = f'mae_pretrain_{args.embed_dim}d_{args.n_layers}l'
        wandb.init(project='crism-mineral-classification', name=run_name,
                   config=vars(args))

    best_loss = float('inf')
    run_name = f'mae_pretrain_{args.embed_dim}d_{args.n_layers}l'

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, _, _ in loader:
            features = features.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(features)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        mean_loss = np.mean(losses)
        logging.info(f"Epoch {epoch}/{args.epochs} | mae_loss={mean_loss:.5f}")
        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'mae_loss': mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({'encoder_state': model.encoder_state_dict(),
                        'mae_loss': best_loss, 'config': vars(args)}, ckpt_path)

    logging.info(f"Best MAE loss: {best_loss:.5f}")
    logging.info(f"Checkpoint: {ckpt_dir}/{run_name}_best.pt")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
```

**Step 5: Add MAE-pretrained fine-tuning to `scripts/train.py`**

Add `--pretrain_ckpt` argument:
```python
parser.add_argument('--pretrain_ckpt', type=str, default=None,
                    help='Path to MAE pretrain checkpoint; loads encoder weights into spectral_vit')
```

In the `spectral_vit` model construction block, after building `model`:
```python
if args.pretrain_ckpt:
    ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
    missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
    logging.info(f"Loaded MAE encoder. Missing: {missing}, Unexpected: {unexpected}")
```

**Step 6: Run MAE tests**

```bash
conda run -n crism pytest tests/test_mae.py -x -q
```
Expected: all 3 tests PASS.

**Step 7: Commit**

```bash
git add models/mae.py models/spectral_transformer.py scripts/pretrain_mae.py scripts/train.py tests/test_mae.py
git commit -m "feat: spectral MAE pre-training for SpectralTransformer encoder"
```

---

## Task 8: Ablation sweep

**Files:**
- Create: `scripts/sweep_v3.py`

**Context:** Structured ablation comparing all new components against each other. Run sequentially, skip already-checkpointed runs. Each run uses `--model spectral_cnn` or `--model spectral_vit`. Pre-run MAE pretraining once and reference its checkpoint.

**Step 1: Run MAE pretraining (background, ~30 min)**

```bash
nohup conda run -n crism python scripts/pretrain_mae.py \
    --epochs 100 --embed_dim 128 --n_heads 4 --n_layers 4 \
    > logs/mae_pretrain_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

Verify:
```bash
ls checkpoints/mae_pretrain_128d_4l_best.pt
```

**Step 2: Create `scripts/sweep_v3.py`**

```python
"""
Ablation sweep comparing mrral-based spectral models.
Answers: what combination of components gets us to >0.90 mAP?

Ablation groups:
  A. Input data: mrrsu vs mrral (spectral_cnn baseline vs cnn_sw4)
  B. Label quality: all conf vs high_conf_only
  C. Loss function: BCE vs focal loss
  D. Sampler: unweighted vs balanced
  E. Pre-training: no pretrain vs MAE pretrain
  F. Kitchen sink: all improvements combined

Usage:
    conda run -n crism python scripts/sweep_v3.py
"""
import argparse, os, subprocess, csv
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(PROJ, 'scripts', 'train.py')
CKPT_DIR = os.path.join(PROJ, 'checkpoints')
LOG_DIR = os.path.join(PROJ, 'logs')
MAE_CKPT = os.path.join(CKPT_DIR, 'mae_pretrain_128d_4l_best.pt')

SWEEP_CONFIGS = [
    # --- Group A: raw mrral spectral data vs mrrsu ---
    # Baseline CNN on mrral (compare to cnn_sw4 mAP=0.652 on mrrsu)
    dict(model='spectral_cnn', run_name='scnn_base',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         warmup_epochs=0, lr_t_max=50),

    # --- Group B: high_conf_only ---
    dict(model='spectral_cnn', run_name='scnn_highconf',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         high_conf_only=True, warmup_epochs=0, lr_t_max=50),

    # --- Group C: focal loss ---
    dict(model='spectral_cnn', run_name='scnn_focal',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         focal_loss=True, focal_gamma=2.0,
         warmup_epochs=0, lr_t_max=50),

    # --- Group D: balanced sampler ---
    dict(model='spectral_cnn', run_name='scnn_balanced',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         balanced_sampling=True, warmup_epochs=0, lr_t_max=50),

    # --- Group E: spectral augmentation ---
    dict(model='spectral_cnn', run_name='scnn_aug',
         epochs=200, patience=25, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         spectral_aug=True, warmup_epochs=0, lr_t_max=50),

    # --- Group F: SpectralTransformer (no pretrain) ---
    dict(model='spectral_vit', run_name='svit_base',
         epochs=200, patience=25, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         warmup_epochs=5, lr_t_max=50),

    # --- Group G: SpectralTransformer + MAE pretrain ---
    dict(model='spectral_vit', run_name='svit_mae',
         epochs=200, patience=25, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         warmup_epochs=5, lr_t_max=50,
         pretrain_ckpt=MAE_CKPT),

    # --- Group H: Kitchen sink (all improvements) ---
    dict(model='spectral_cnn', run_name='scnn_best',
         epochs=200, patience=30, lr=5e-4, batch_size=512,
         dropout=0.2, use_pos_weight=True, weight_decay=1e-4,
         high_conf_only=True, focal_loss=True, focal_gamma=2.0,
         balanced_sampling=True, spectral_aug=True,
         warmup_epochs=0, lr_t_max=50),
    dict(model='spectral_vit', run_name='svit_best',
         epochs=200, patience=30, lr=3e-4, batch_size=512,
         embed_dim=128, n_heads=4, n_layers=6,
         dropout=0.1, use_pos_weight=True, weight_decay=1e-4,
         high_conf_only=True, focal_loss=True, focal_gamma=2.0,
         balanced_sampling=True, spectral_aug=True,
         warmup_epochs=5, lr_t_max=50,
         pretrain_ckpt=MAE_CKPT),
]

BOOL_FLAGS = {'use_pos_weight', 'high_conf_only', 'focal_loss', 'balanced_sampling', 'spectral_aug'}


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
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    total = len(SWEEP_CONFIGS)
    results = []

    for i, cfg in enumerate(SWEEP_CONFIGS):
        run_name = cfg['run_name']
        print(f'\n[{i+1}/{total}] {run_name}', flush=True)
        if ckpt_exists(run_name):
            print('  SKIPPING — checkpoint exists', flush=True)
            continue
        if 'pretrain_ckpt' in cfg and not os.path.exists(cfg.get('pretrain_ckpt', '')):
            print(f'  SKIPPING — MAE checkpoint not found: {cfg["pretrain_ckpt"]}', flush=True)
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
        out = os.path.join(LOG_DIR, f'sweep_v3_{stamp}.csv')
        import csv
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'status'])
            w.writeheader(); w.writerows(results)
        print(f'\nSweep summary: {out}')

    print(f'\nDone. {len(results)} run, {sum(1 for r in results if r["status"]=="ok")} ok.')


if __name__ == '__main__':
    main()
```

**Step 3: Dry run to validate configs**

```bash
conda run -n crism python scripts/sweep_v3.py --dry_run
```
Expected: prints 9 DRY RUN lines with valid CLI args, no errors.

**Step 4: Launch sweep (background)**

```bash
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
nohup conda run -n crism python -u scripts/sweep_v3.py \
    > logs/sweep_v3_${TIMESTAMP}.log 2>&1 &
echo "PID=$!"
```

**Step 5: Update watcher to include v3 checkpoints**

Edit `/tmp/watch_sweep.sh` (or create a new `/tmp/watch_v3.sh`) checking for:
`scnn_base`, `scnn_highconf`, `scnn_focal`, `scnn_balanced`, `scnn_aug`,
`svit_base`, `svit_mae`, `scnn_best`, `svit_best`

Then auto-run `generate_report.py` when done.

**Step 6: Commit**

```bash
git add scripts/sweep_v3.py scripts/pretrain_mae.py
git commit -m "feat: ablation sweep v3 comparing mrral spectral models + MAE pretrain"
```

---

## Task 9: Ensemble and test-set evaluation

**Files:**
- Create: `scripts/evaluate_ensemble.py`
- Test: `tests/test_evaluate_ensemble.py`

**Context:** Once sweep_v3 is complete, combine the top-3 models into a simple average ensemble and evaluate on the held-out test split. This gives the final, honest performance estimate.

**Step 1: Create `scripts/evaluate_ensemble.py`**

```python
"""
Evaluate top-N model ensemble on the test split.

Usage:
    conda run -n crism python scripts/evaluate_ensemble.py \
        --checkpoints checkpoints/scnn_best_best.pt checkpoints/svit_best_best.pt checkpoints/scnn_focal_best.pt
"""
import argparse, os, sys, logging
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model_from_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Infer model type from checkpoint filename
    name = os.path.basename(ckpt_path)
    if 'scnn' in name or 'spectral_cnn' in name:
        from models.spectral_cnn import SpectralCNN1D
        model = SpectralCNN1D(n_bands=59, n_classes=6)
    elif 'svit' in name or 'spectral_vit' in name:
        from models.spectral_transformer import SpectralTransformer
        model = SpectralTransformer(n_bands=59, n_classes=6)
    elif 'cnn' in name:
        from models.cnn import SpectralSpatialCNN
        model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=7)
    elif 'mlp' in name:
        from models.mlp import MLP
        model = MLP(n_features=60, n_classes=6)
    else:
        raise ValueError(f"Cannot infer model type from: {name}")
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model.to(device)


def predict(model, df, device, batch_size=1024):
    from torch.utils.data import DataLoader
    from data.dataset import CRISMSpectralDataset, CRISMPixelDataset, MRRAL_BAND_COLS
    if MRRAL_BAND_COLS[0] in df.columns:
        ds = CRISMSpectralDataset(df)
    else:
        ds = CRISMPixelDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for feats, _, _ in loader:
            logits = model(feats.to(device))
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(os.path.join(PROJ, 'config.yaml')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data (prefer mrral if available)
    mrral_path = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    mrrsu_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    if os.path.exists(mrral_path):
        df = pd.read_parquet(mrral_path)
    else:
        df = pd.read_parquet(mrrsu_path)
    test_df = df[df['split'] == args.split]

    from data.label_parser import CLASSES
    y_true = test_df[CLASSES].values.astype('float32')
    conf_tiers = test_df['confidence_tier'].tolist()

    # Collect predictions from each model
    all_scores = []
    for ckpt_path in args.checkpoints:
        logging.info(f"Loading {ckpt_path}")
        model = load_model_from_checkpoint(ckpt_path, device)
        scores = predict(model, test_df, device)
        all_scores.append(scores)
        from evaluation.metrics import compute_full_metrics
        m = compute_full_metrics(y_true, scores, conf_tiers)
        logging.info(f"  {os.path.basename(ckpt_path)}: mAP={m['mAP']:.4f}")
        for cls, ap in m['per_class_ap'].items():
            logging.info(f"    {cls}: AP={ap:.4f}")

    # Ensemble average
    if len(all_scores) > 1:
        ensemble_scores = np.mean(all_scores, axis=0)
        from evaluation.metrics import compute_full_metrics
        m = compute_full_metrics(y_true, ensemble_scores, conf_tiers)
        logging.info(f"\nEnsemble ({len(all_scores)} models): mAP={m['mAP']:.4f}")
        for cls, ap in m['per_class_ap'].items():
            logging.info(f"  {cls}: AP={ap:.4f}")


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add scripts/evaluate_ensemble.py
git commit -m "feat: ensemble evaluation script for test-split final results"
```

---

## Quick Reference: Running the Full Pipeline

```bash
# 1. Build mrral parquet (run once, ~20 min)
nohup conda run -n crism python scripts/build_mrral_dataset.py > logs/build_mrral.log 2>&1 &

# 2. MAE pretraining (run once after mrral parquet done, ~30 min)
nohup conda run -n crism python scripts/pretrain_mae.py > logs/mae_pretrain.log 2>&1 &

# 3. Ablation sweep (run after MAE done, ~8-12 hours total)
nohup conda run -n crism python -u scripts/sweep_v3.py > logs/sweep_v3.log 2>&1 &

# 4. Generate report
conda run -n crism python scripts/generate_report.py

# 5. Ensemble test eval (after sweep done, pick top checkpoints)
conda run -n crism python scripts/evaluate_ensemble.py \
    --checkpoints checkpoints/scnn_best_best.pt checkpoints/svit_best_best.pt

# Run all tests at any point
conda run -n crism pytest tests/ -x -q
```

### When Hellas data arrives:
1. Copy new GPKG files to `categorized_mineral_units/`
2. Re-run `scripts/build_mrral_dataset.py` (and `scripts/build_dataset.py` for mrrsu)
3. Re-run sweep_v3 (existing checkpoints skipped; new runs use richer plagioclase data)
