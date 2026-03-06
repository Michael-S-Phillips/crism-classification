"""
PyTorch Dataset classes and sklearn array loaders for the pixel dataset.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
BAND_COLS = [f'b{i}' for i in range(60)]


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


class CRISMPatchDataset(Dataset):
    """
    Spatial patch dataset for CNN and ViT.
    Extracts a (patch_size x patch_size x 60) neighbourhood around each pixel
    from the corresponding mrrsu raster at runtime.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mrrsu_map: Dict[str, str],
        patch_size: int = 7,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.df = df.reset_index(drop=True)
        self.mrrsu_map = mrrsu_map
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

        # Cache open rasterio file handles per tile
        self._handles: Dict[str, rasterio.DatasetReader] = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tile_id = row['tile_id']
        pr, pc = int(row['pixel_row']), int(row['pixel_col'])

        if tile_id not in self._handles:
            self._handles[tile_id] = rasterio.open(self.mrrsu_map[tile_id])
        src = self._handles[tile_id]

        h = self.half
        r0 = max(0, pr - h)
        r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h)
        c1 = min(src.width, pc + h + 1)

        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        patch = src.read(window=window).astype(np.float32)  # (bands, h, w)

        # Replace NODATA
        patch[patch >= 65535] = 0.0
        patch = np.nan_to_num(patch, nan=0.0)

        # Pad to patch_size x patch_size if near border
        full = np.zeros((src.count, self.patch_size, self.patch_size), dtype=np.float32)
        full[:, h - (pr - r0):h - (pr - r0) + patch.shape[1],
                h - (pc - c0):h - (pc - c0) + patch.shape[2]] = patch

        return torch.tensor(full, dtype=torch.float32), self.labels[idx], self.weights[idx]

    def __del__(self):
        for src in self._handles.values():
            src.close()


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
