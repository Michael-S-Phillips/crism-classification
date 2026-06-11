# Patch Pre-Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pre-extract all (60, 7, 7) patches from mrrsu rasters once and save as numpy memmaps so CNN/ViT training reads a single array index per sample instead of a random SAMBA window-read.

**Architecture:** `scripts/cache_patches.py` writes `{split}_patches_p{patch_size}.npy` memmaps into `data/patch_cache/`. `CRISMPatchDataset` gains `cache_dir` + `split` params; when a matching cache file exists it loads via `np.memmap` instead of opening rasterio handles. `run_all_models.sh` runs the caching script before CNN/ViT if cache files are missing.

**Tech Stack:** numpy memmap, rasterio (write path only), pandas, tqdm, PyTorch

---

### Task 1: Add `patch_cache_dir` to config.yaml

**Files:**
- Modify: `config.yaml`

**Step 1: Add the key**

Open `config.yaml` and add after `output_dir`:
```yaml
patch_cache_dir: /mnt/gigas/CRISM/MRDR/crism_classification/data/patch_cache
```

**Step 2: Verify**
```bash
grep patch_cache_dir config.yaml
```
Expected output: `patch_cache_dir: /mnt/gigas/CRISM/MRDR/crism_classification/data/patch_cache`

**Step 3: Commit**
```bash
git add config.yaml
git commit -m "config: add patch_cache_dir for pre-cached patch memmaps"
```

---

### Task 2: Write `scripts/cache_patches.py`

**Files:**
- Create: `scripts/cache_patches.py`

**Step 1: Write the script**

```python
"""
Pre-extract spatial patches from mrrsu rasters and save as numpy memmaps.

Creates {cache_dir}/{split}_patches_p{patch_size}.npy for each split.
Shape: (n_split_samples, n_bands, patch_size, patch_size) float32.
Row order matches pixels.parquet row order within each split.

Usage:
    conda run -n crism python scripts/cache_patches.py
    conda run -n crism python scripts/cache_patches.py --patch_size 7 --config config.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CRISMPatchDataset, BAND_COLS
from data.extract_pixels import find_tile_pairs


def cache_split(split: str, df: pd.DataFrame, mrrsu_map: dict,
                patch_size: int, cache_dir: str) -> None:
    out_path = os.path.join(cache_dir, f'{split}_patches_p{patch_size}.npy')
    if os.path.exists(out_path):
        print(f'[cache] {split}: already exists at {out_path}, skipping.')
        return

    sub = df[df['split'] == split].reset_index(drop=True)
    n = len(sub)
    n_bands = len(BAND_COLS)
    print(f'[cache] {split}: {n:,} samples → {out_path}')

    fp = np.memmap(out_path, dtype='float32', mode='w+',
                   shape=(n, n_bands, patch_size, patch_size))

    # Use live CRISMPatchDataset to extract patches (no cache_dir → rasterio path)
    ds = CRISMPatchDataset(sub, mrrsu_map, patch_size=patch_size)

    for idx in tqdm(range(n), desc=split, unit='patch'):
        patch, _labels, _weight = ds[idx]
        fp[idx] = patch.numpy()

    fp.flush()
    del fp
    ds.close()
    print(f'[cache] {split}: done.')


def main():
    parser = argparse.ArgumentParser(description='Pre-cache CRISM spatial patches.')
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    cache_dir = cfg['patch_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)

    print(f'Loading {parquet_path} ...')
    df = pd.read_parquet(parquet_path)

    print(f'Finding tile pairs in {cfg["data_root"]} ...')
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}
    print(f'Found {len(mrrsu_map)} tiles.')

    for split in args.splits:
        cache_split(split, df, mrrsu_map, args.patch_size, cache_dir)

    print('All splits cached.')


if __name__ == '__main__':
    main()
```

**Step 2: Commit**
```bash
git add scripts/cache_patches.py
git commit -m "feat: add cache_patches.py to pre-extract spatial patches to memmap"
```

---

### Task 3: Modify `CRISMPatchDataset` to use cache when available

**Files:**
- Modify: `data/dataset.py:33-136`

**Step 1: Write a failing test first**

Add to `tests/test_dataset.py`:
```python
def test_patch_dataset_uses_cache(tmp_path):
    """CRISMPatchDataset loads from memmap cache instead of rasterio when available."""
    import torch
    from data.dataset import CRISMPatchDataset, BAND_COLS

    n = 4
    patch_size = 7
    n_bands = len(BAND_COLS)

    # Build minimal fake df (no actual raster needed when cache exists)
    data = {f'b{i}': np.zeros(n, dtype=np.float32) for i in range(n_bands)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = np.zeros(n, dtype=np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    data['tile_id'] = ['t0001'] * n
    data['pixel_row'] = [0] * n
    data['pixel_col'] = [0] * n
    data['split'] = ['train'] * n
    df = pd.DataFrame(data)

    # Write a fake cache file
    sentinel = np.arange(n * n_bands * patch_size * patch_size,
                         dtype=np.float32).reshape(n, n_bands, patch_size, patch_size)
    cache_file = tmp_path / 'train_patches_p7.npy'
    fp = np.memmap(str(cache_file), dtype='float32', mode='w+',
                   shape=(n, n_bands, patch_size, patch_size))
    fp[:] = sentinel
    fp.flush()
    del fp

    ds = CRISMPatchDataset(df, mrrsu_map={}, patch_size=7,
                           cache_dir=str(tmp_path), split='train')
    patch, labels, weight = ds[0]

    assert patch.shape == (n_bands, patch_size, patch_size)
    assert patch.dtype == torch.float32
    # Patch values should come from cache, not rasterio
    np.testing.assert_allclose(patch.numpy(), sentinel[0])
```

**Step 2: Run it to confirm it fails**
```bash
cd /mnt/gigas/CRISM/MRDR/crism_classification
conda run -n crism python -m pytest tests/test_dataset.py::test_patch_dataset_uses_cache -v
```
Expected: `FAILED` (AttributeError or TypeError — CRISMPatchDataset doesn't accept cache_dir yet)

**Step 3: Modify `CRISMPatchDataset.__init__`**

In `data/dataset.py`, update the `__init__` signature and add cache loading at the end of `__init__`:

```python
def __init__(
    self,
    df: pd.DataFrame,
    mrrsu_map: Dict[str, str],
    patch_size: int = 7,
    cache_dir: Optional[str] = None,
    split: Optional[str] = None,
):
    assert patch_size % 2 == 1, "patch_size must be odd"
    df = df.reset_index(drop=True)
    self.mrrsu_map = mrrsu_map
    self.patch_size = patch_size
    self.half = patch_size // 2
    self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
    self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
    self._tile_ids = df['tile_id'].values
    self._pixel_rows = df['pixel_row'].values.astype(np.int64)
    self._pixel_cols = df['pixel_col'].values.astype(np.int64)
    self._n = len(df)
    self._handles: Dict[str, rasterio.DatasetReader] = {}
    self._pid = os.getpid()
    # Load memmap cache if available
    self._cache = None
    if cache_dir and split:
        cache_file = os.path.join(cache_dir, f'{split}_patches_p{patch_size}.npy')
        if os.path.exists(cache_file):
            self._cache = np.memmap(
                cache_file, dtype='float32', mode='r',
                shape=(self._n, len(BAND_COLS), patch_size, patch_size)
            )
```

Also add `from typing import Optional` to imports if not already present.

**Step 4: Update `__getitem__` to short-circuit when cache is loaded**

Replace the opening lines of `__getitem__` to add a cache check before the rasterio logic:

```python
def __getitem__(self, idx):
    if self._cache is not None:
        patch = torch.from_numpy(self._cache[idx].copy())
        return patch, self.labels[idx], self.weights[idx]

    # Re-open handles if we've been forked into a DataLoader worker
    current_pid = os.getpid()
    # ... rest of existing rasterio code unchanged ...
```

**Step 5: Run test to confirm it passes**
```bash
conda run -n crism python -m pytest tests/test_dataset.py::test_patch_dataset_uses_cache -v
```
Expected: `PASSED`

**Step 6: Run existing dataset tests to confirm no regression**
```bash
conda run -n crism python -m pytest tests/test_dataset.py -v -k "not patch_dataset_shape"
```
Note: `test_patch_dataset_shape` is skipped because it references the old `/mnt/crism` path.

**Step 7: Commit**
```bash
git add data/dataset.py tests/test_dataset.py
git commit -m "feat: CRISMPatchDataset loads from memmap cache when cache_dir+split provided"
```

---

### Task 4: Thread `cache_dir` through `train_torch_model`

**Files:**
- Modify: `training/train_torch.py:19-33`

**Step 1: Write a failing test**

Add to `tests/test_train_torch.py`:
```python
def test_cnn_trains_with_cache(tmp_path):
    """train_torch_model passes cache_dir to CRISMPatchDataset."""
    import numpy as np
    from models.cnn import SpectralSpatialCNN
    from data.dataset import BAND_COLS

    n = 120
    patch_size = 7
    n_bands = len(BAND_COLS)
    df = make_fake_df(n)

    # Write fake cache for both splits
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        n_s = len(sub)
        fp = np.memmap(
            str(tmp_path / f'{split}_patches_p{patch_size}.npy'),
            dtype='float32', mode='w+',
            shape=(n_s, n_bands, patch_size, patch_size)
        )
        fp[:] = np.random.rand(n_s, n_bands, patch_size, patch_size).astype(np.float32)
        fp.flush()
        del fp

    model = SpectralSpatialCNN(n_bands=n_bands, n_classes=6, patch_size=patch_size)
    # mrrsu_map has dummy entry; cache means rasterio is never called
    mrrsu_map = {'t0001': '/nonexistent/path.img'}
    metrics = train_torch_model(
        model=model, df=df, model_name='cnn_cache_test',
        max_epochs=2, batch_size=32, lr=1e-3,
        use_wandb=False, checkpoint_dir=None,
        mrrsu_map=mrrsu_map, patch_size=patch_size,
        cache_dir=str(tmp_path),
    )
    assert 'val_mAP' in metrics
```

**Step 2: Run test to confirm it fails**
```bash
conda run -n crism python -m pytest tests/test_train_torch.py::test_cnn_trains_with_cache -v
```
Expected: `FAILED` — `train_torch_model` doesn't accept `cache_dir` yet.

**Step 3: Update `train_torch_model` signature and `make_dataset`**

In `training/train_torch.py`, add `cache_dir` parameter and pass it through:

```python
def train_torch_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    model_name: str,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    use_wandb: bool = True,
    checkpoint_dir: Optional[str] = None,
    mrrsu_map: Optional[Dict[str, str]] = None,
    patch_size: int = 7,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    **wandb_config
) -> Dict[str, Any]:
```

And update `make_dataset`:
```python
def make_dataset(split):
    sub = df[df['split'] == split]
    if use_patches:
        return CRISMPatchDataset(sub, mrrsu_map, patch_size=patch_size,
                                 cache_dir=cache_dir, split=split)
    return CRISMPixelDataset(sub)
```

**Step 4: Run test to confirm it passes**
```bash
conda run -n crism python -m pytest tests/test_train_torch.py::test_cnn_trains_with_cache -v
```
Expected: `PASSED`

**Step 5: Run full test suite**
```bash
conda run -n crism python -m pytest tests/test_train_torch.py -v
```
Expected: all existing tests still pass.

**Step 6: Commit**
```bash
git add training/train_torch.py tests/test_train_torch.py
git commit -m "feat: thread cache_dir through train_torch_model to CRISMPatchDataset"
```

---

### Task 5: Read `patch_cache_dir` from config in `scripts/train.py`

**Files:**
- Modify: `scripts/train.py:91-112`

**Step 1: Update the CNN/ViT block to pass `cache_dir`**

In `scripts/train.py`, in the `elif args.model in ('cnn', 'vit'):` block, add one line after `mrrsu_map = ...`:

```python
cache_dir = cfg.get('patch_cache_dir')
```

And pass it to `train_torch_model`:
```python
metrics = train_torch_model(
    model=model, df=df, model_name=args.model,
    max_epochs=args.epochs, batch_size=args.batch_size,
    lr=args.lr, patience=args.patience,
    use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
    mrrsu_map=mrrsu_map, patch_size=args.patch_size,
    cache_dir=cache_dir,
)
```

**Step 2: Verify the change looks right**
```bash
grep -A3 "cache_dir" scripts/train.py
```

**Step 3: Commit**
```bash
git add scripts/train.py
git commit -m "feat: train.py reads patch_cache_dir from config and passes to train_torch_model"
```

---

### Task 6: Add cache-generation step to `run_all_models.sh`

**Files:**
- Modify: `scripts/run_all_models.sh:64-68`

**Step 1: Insert cache check before the CNN line**

Before the `run_model cnn ...` line, add:
```bash
# --- Generate patch cache if needed (used by CNN and ViT) ---
PATCH_CACHE_DIR="$PROJ_DIR/data/patch_cache"
if [[ ! -f "$PATCH_CACHE_DIR/train_patches_p7.npy" ]]; then
    log "===== Generating patch cache (one-time, ~1-2 hrs) ====="
    conda run -n crism python "$PROJ_DIR/scripts/cache_patches.py" \
        2>&1 | tee -a "$LOG_FILE"
    log "===== Patch cache complete ====="
fi
```

**Step 2: Verify placement — the block should be between `run_model mlp` and `run_model cnn`**
```bash
grep -n "cache\|run_model cnn\|run_model mlp\|run_model vit" scripts/run_all_models.sh
```
Expected ordering: mlp line → cache block → cnn line → vit line.

**Step 3: Commit**
```bash
git add scripts/run_all_models.sh
git commit -m "feat: run_all_models.sh auto-generates patch cache before CNN/ViT"
```

---

### Task 7: Run the cache script

Now build the actual cache. This is a one-time operation (~1–2 hrs due to SAMBA I/O for 726k+ samples).

**Step 1: Start cache generation in background**
```bash
cd /mnt/gigas/CRISM/MRDR/crism_classification
nohup conda run -n crism python scripts/cache_patches.py \
    > logs/cache_patches.out 2>&1 &
echo $! > logs/cache_patches.pid
echo "Cache generation PID: $(cat logs/cache_patches.pid)"
```

**Step 2: Monitor progress**
```bash
tail -f logs/cache_patches.out
```
Expected output: tqdm progress bars for train (726k), val (98k), test splits.

**Step 3: Verify cache files created**
```bash
ls -lh data/patch_cache/
```
Expected:
```
train_patches_p7.npy   ~8.1 GB   (726033 × 60 × 7 × 7 × 4 bytes)
val_patches_p7.npy     ~1.1 GB
test_patches_p7.npy    ~1.1 GB
```

**Step 4: Quick sanity check**
```bash
conda run -n crism python -c "
import numpy as np
fp = np.memmap('data/patch_cache/train_patches_p7.npy', dtype='float32', mode='r',
               shape=(726033, 60, 7, 7))
print('shape:', fp.shape)
print('sample patch[0] mean:', fp[0].mean())
print('no nan:', not np.isnan(fp[:100]).any())
"
```
Expected: shape `(726033, 60, 7, 7)`, reasonable float mean, no NaN.
