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

LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
BAND_COLS = [f'b{i}' for i in range(60)]
MRRAL_BAND_COLS = [f'm{i}' for i in range(59)]  # 59 bands, 410-2457 nm (< 2500 nm cutoff)
NODATA_VALUE = 65535


class CRISMPixelDataset(Dataset):
    """Per-pixel dataset for MLP and linear models."""

    def __init__(self, df: pd.DataFrame):
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
        df = df.reset_index(drop=True)
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


def load_sklearn_arrays(parquet_path: str):
    """
    Load train/val/test arrays for sklearn models.

    Returns
    -------
    X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test
    All as numpy arrays. y arrays are shape (n, 6) float32.
    """
    df = pd.read_parquet(parquet_path)

    def _split(split_name):
        sub = df[df['split'] == split_name]
        X = sub[BAND_COLS].values.astype(np.float32)
        y = sub[LABEL_COLS].values.astype(np.float32)
        w = sub['confidence_weight'].values.astype(np.float32)
        return X, y, w

    return (*_split('train'), *_split('val'), *_split('test'))
