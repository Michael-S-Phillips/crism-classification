"""
PyTorch Dataset classes and sklearn array loaders for the CRISM pixel dataset.
"""
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

LABEL_COLS_RAW = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']


def _collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with olivine_t1/t2 merged into a single 'olivine' column.

    olivine = max(olivine_t1, olivine_t2), so:
      - Hard labels (1.0/0.0): result is 1.0 if either type is positive.
      - Soft labels (0.5/0.5 for Hellas): result is 0.5 (still present but uncertain).
    Uniform confidence weights (1.0) are assigned regardless of confidence_tier,
    so all pixels contribute equally during training.
    """
    out = df.copy()
    out['olivine'] = out[['olivine_t1', 'olivine_t2']].max(axis=1)
    out['confidence_weight'] = 1.0
    return out
BAND_COLS = [f'b{i}' for i in range(60)]
MRRAL_BAND_COLS = [f'm{i}' for i in range(59)]  # 59 bands, 410-2457 nm (< 2500 nm cutoff)
NODATA_VALUE = 65535


class CRISMPixelDataset(Dataset):
    """Per-pixel dataset for MLP and linear models."""

    def __init__(self, df: pd.DataFrame):
        df = _collapse_labels(df)
        self.features = torch.tensor(df[BAND_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


class CRISMSpectralDataset(Dataset):
    """
    Per-pixel dataset using mrral 59-band reflectance spectra (m0..m58).
    Used by all mrral-based models: SpectralCNN1D, SpectralTransformer, MAE.

    Requires mrral_pixels.parquet — run scripts/build_mrral_dataset.py first.
    """

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in MRRAL_BAND_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"mrral parquet missing columns: {missing[:5]}... "
                "Run scripts/build_mrral_dataset.py first."
            )
        df = _collapse_labels(df)
        self.features = torch.tensor(df[MRRAL_BAND_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


class CRISMPatchDataset(Dataset):
    """
    Spatial patch dataset for CNN and ViT.

    Extracts a (bands × patch_size × patch_size) neighbourhood around each
    pixel from the corresponding mrrsu raster at runtime. Border pixels are
    zero-padded. File handles are cached per tile and re-opened safely after
    DataLoader fork (detected via os.getpid() comparison).

    Call close() when done, or use as a context manager.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mrrsu_map: Dict[str, str],
        patch_size: int = 7,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        df = _collapse_labels(df).reset_index(drop=True)
        self.mrrsu_map = mrrsu_map
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        # Extract hot-path columns as arrays to avoid per-item DataFrame overhead
        self._tile_ids = df['tile_id'].values
        self._pixel_rows = df['pixel_row'].values.astype(np.int64)
        self._pixel_cols = df['pixel_col'].values.astype(np.int64)
        self._n = len(df)
        # File handles cached per tile; cleared on DataLoader fork (pid check)
        self._handles: Dict[str, rasterio.DatasetReader] = {}
        self._pid = os.getpid()
        # Load memmap cache if available — bypasses rasterio reads at item time
        self._cache = None
        if cache_dir and split:
            cache_file = os.path.join(cache_dir, f'{split}_patches_p{patch_size}.npy')
            if os.path.exists(cache_file):
                self._cache = np.memmap(
                    cache_file, dtype='float32', mode='r',
                    shape=(self._n, len(BAND_COLS), patch_size, patch_size)
                )

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if self._cache is not None:
            patch = torch.from_numpy(self._cache[idx].copy())
            return patch, self.labels[idx], self.weights[idx]

        # Re-open handles if we've been forked into a DataLoader worker
        current_pid = os.getpid()
        if current_pid != self._pid:
            self._handles.clear()
            self._pid = current_pid

        tile_id = self._tile_ids[idx]
        pr = int(self._pixel_rows[idx])
        pc = int(self._pixel_cols[idx])

        if tile_id not in self._handles:
            if tile_id not in self.mrrsu_map:
                raise KeyError(
                    f"tile_id {tile_id!r} not found in mrrsu_map. "
                    f"Available tiles (first 5): {sorted(self.mrrsu_map)[:5]}"
                )
            try:
                self._handles[tile_id] = rasterio.open(self.mrrsu_map[tile_id])
            except Exception as e:
                raise OSError(f"Failed to open raster for tile {tile_id!r}: {e}") from e
        src = self._handles[tile_id]

        if src.count != len(BAND_COLS):
            raise ValueError(
                f"Tile {tile_id!r} has {src.count} bands, expected {len(BAND_COLS)}"
            )

        h = self.half
        r0 = max(0, pr - h)
        r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h)
        c1 = min(src.width, pc + h + 1)

        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        chunk = src.read(window=window).astype(np.float32)  # (bands, h, w)

        # Replace nodata and NaN with 0
        chunk[chunk >= NODATA_VALUE] = 0.0
        chunk = np.nan_to_num(chunk, nan=0.0)

        # Place chunk into zero-padded full patch
        full = np.zeros((len(BAND_COLS), self.patch_size, self.patch_size), dtype=np.float32)
        dst_r = h - (pr - r0)  # offset in output array (= max(0, h - pr) at borders)
        dst_c = h - (pc - c0)
        full[:, dst_r:dst_r + chunk.shape[1], dst_c:dst_c + chunk.shape[2]] = chunk

        return torch.from_numpy(full), self.labels[idx], self.weights[idx]

    def close(self):
        """Close all open rasterio file handles."""
        for src in getattr(self, '_handles', {}).values():
            try:
                src.close()
            except Exception:
                pass
        if hasattr(self, '_handles'):
            self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


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


def load_sklearn_arrays(parquet_path: str):
    """
    Load train/val/test arrays for sklearn models.

    Returns
    -------
    X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test
    All as numpy arrays. y arrays are shape (n, 5) float32.
    """
    df = _collapse_labels(pd.read_parquet(parquet_path))

    def _split(split_name):
        sub = df[df['split'] == split_name]
        X = sub[BAND_COLS].values.astype(np.float32)
        y = sub[LABEL_COLS].values.astype(np.float32)
        w = sub['confidence_weight'].values.astype(np.float32)
        return X, y, w

    return (*_split('train'), *_split('val'), *_split('test'))


class CRISMSpectralPatchDataset(Dataset):
    """
    Spatial patch dataset for SpatialSpectralClassifier fine-tuning.

    Reads 7×7×59 mrral reflectance patches around labeled pixel centers.
    Applies the same normalization as CRISMGlobalPatchDataset (clip to [0, 0.5]).
    Border pixels are zero-padded. File handles are cached per tile, pid-safe.
    """

    CLIP_MAX = 0.5
    NODATA = 65535.0

    def __init__(
        self,
        df: pd.DataFrame,
        mrral_map: Dict[str, str],
        patch_size: int = 7,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        df = _collapse_labels(df).reset_index(drop=True)
        self.mrral_map = mrral_map
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
        # Load memmap cache if available — bypasses rasterio reads at item time
        self._cache = None
        if cache_dir and split:
            cache_file = os.path.join(cache_dir, f'mrral_{split}_patches_p{patch_size}.npy')
            if os.path.exists(cache_file):
                self._cache = np.memmap(
                    cache_file, dtype='float32', mode='r',
                    shape=(self._n, patch_size, patch_size, 59)
                )

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if self._cache is not None:
            patch = torch.from_numpy(self._cache[idx].copy())
            return patch, self.labels[idx], self.weights[idx]

        current_pid = os.getpid()
        if current_pid != self._pid:
            self._handles.clear()
            self._pid = current_pid

        tile_id = self._tile_ids[idx]
        pr = int(self._pixel_rows[idx])
        pc = int(self._pixel_cols[idx])

        if tile_id not in self._handles:
            if tile_id not in self.mrral_map:
                raise KeyError(f"tile_id {tile_id!r} not found in mrral_map")
            self._handles[tile_id] = rasterio.open(self.mrral_map[tile_id])
        src = self._handles[tile_id]

        h = self.half
        r0 = max(0, pr - h);  r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h);  c1 = min(src.width,  pc + h + 1)
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)

        # Read first 59 bands (1-indexed for rasterio)
        chunk = src.read(list(range(1, 60)), window=window).astype(np.float32)  # (59, h, w)

        # Zero-pad to (59, patch_size, patch_size)
        patch = np.zeros((59, self.patch_size, self.patch_size), dtype=np.float32)
        dr0 = pr - h - max(0, pr - h) + (max(0, pr - h) - (pr - h))
        dc0 = pc - h - max(0, pc - h) + (max(0, pc - h) - (pc - h))
        # Simplified: pad offset is how far the actual window start is from intended
        dr0 = max(0, h - pr)
        dc0 = max(0, h - pc)
        ph, pw = chunk.shape[1], chunk.shape[2]
        patch[:, dr0:dr0 + ph, dc0:dc0 + pw] = chunk

        # Handle nodata
        patch[(patch == self.NODATA) | ~np.isfinite(patch)] = 0.0

        # Normalize: clip to [0, CLIP_MAX]
        patch = np.clip(patch, 0.0, self.CLIP_MAX)

        # (59, 7, 7) → (7, 7, 59)
        patch = patch.transpose(1, 2, 0)

        return torch.from_numpy(patch.copy()), self.labels[idx], self.weights[idx]

    def close(self):
        for src in self._handles.values():
            src.close()
        self._handles.clear()
