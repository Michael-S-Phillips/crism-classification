"""Shared helpers for v5 figure scripts."""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CRISM_LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
CLASS_COLORS = {
    'olivine':    '#2ca02c',
    'lcp':        '#1f77b4',
    'hcp':        '#d62728',
    'plagioclase':'#ff7f0e',
    'other':      '#7f7f7f',
}
TIER_COLORS = {'High': '#2c7bb6', 'Moderate': '#abd9e9', 'Low': '#fdae61'}
SPLIT_COLORS = {'train': '#4daf4a', 'val': '#377eb8', 'test': '#e41a1c'}

WAVELENGTHS_59 = None  # cached on first access


def get_wavelengths_59() -> np.ndarray:
    """Wavelengths (nm) for the 59 mrral bands used in training. Read from
    the first available .hdr file under /mnt/mrdr/mc*/."""
    global WAVELENGTHS_59
    if WAVELENGTHS_59 is not None:
        return WAVELENGTHS_59
    import spectral.io.envi as envi
    hdrs = sorted(glob.glob('/mnt/mrdr/mc*/t*mrral*.hdr'))
    if not hdrs:
        raise RuntimeError('No mrral .hdr files found under /mnt/mrdr/mc*/.')
    hdr = envi.open(hdrs[0])
    WAVELENGTHS_59 = np.array(
        [float(w) for w in hdr.metadata['wavelength']][:59], dtype=np.float32,
    )
    return WAVELENGTHS_59


def load_mrral_parquet(path: str = '/mnt/mrdr/crism_classification/data/mrral_pixels.parquet') -> pd.DataFrame:
    """Load the post-collapse parquet, with `olivine` aggregated and a
    consistent confidence_weight column applied (matches training behavior)."""
    df = pd.read_parquet(path)
    import sys
    sys.path.insert(0, '/mnt/mrdr/crism_classification')
    from data.dataset import _collapse_labels
    return _collapse_labels(df)


def build_mrral_map(data_root: str = '/mnt/mrdr') -> Dict[str, str]:
    """tile_id → .img path map. Supports nested (mc*/) and flat layouts."""
    hdrs = sorted(set(
        glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
        + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))
    ))
    mp = {}
    for h in hdrs:
        tid = os.path.basename(h).split('_mrral_')[0]
        mp[tid] = h.replace('.hdr', '.img')
    return mp


def read_patch_from_tile(
    mrral_path: str,
    pixel_row: int,
    pixel_col: int,
    patch_size: int = 7,
    n_bands: int = 59,
) -> np.ndarray:
    """Read a (patch_size, patch_size, n_bands) patch around (pixel_row, pixel_col).
    Border pixels zero-padded. NODATA (65535) → 0. Clipped to [0, 0.5] to match
    CRISMSpectralPatchDataset.CLIP_MAX preprocessing.
    """
    import rasterio
    half = patch_size // 2
    with rasterio.open(mrral_path) as src:
        h, w = src.height, src.width
        actual_bands = min(n_bands, src.count)
        r0 = max(pixel_row - half, 0)
        r1 = min(pixel_row + half + 1, h)
        c0 = max(pixel_col - half, 0)
        c1 = min(pixel_col + half + 1, w)
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        # Read explicit band indices (1-based for rasterio) within the window.
        raw = src.read(list(range(1, actual_bands + 1)), window=window)
    # raw shape: (actual_bands, h_w, w_w) — transpose to (H, W, B)
    arr = np.transpose(raw, (1, 2, 0)).astype(np.float32)
    if arr.shape[2] < n_bands:
        pad_b = n_bands - arr.shape[2]
        arr = np.concatenate(
            [arr, np.zeros((arr.shape[0], arr.shape[1], pad_b), dtype=np.float32)],
            axis=2,
        )
    arr = arr[:, :, :n_bands]
    arr[arr >= 65535.0] = 0.0
    arr = np.clip(arr, 0.0, 0.5)
    # Zero-pad to full patch_size at tile edges
    if arr.shape[0] < patch_size or arr.shape[1] < patch_size:
        out = np.zeros((patch_size, patch_size, n_bands), dtype=np.float32)
        rb = patch_size - (r1 - r0) if pixel_row - half < 0 else 0
        cb = patch_size - (c1 - c0) if pixel_col - half < 0 else 0
        out[rb:rb + arr.shape[0], cb:cb + arr.shape[1], :] = arr
        return out
    return arr


def find_representative_pixels(
    df: pd.DataFrame,
    n_per_class: int = 1,
    seed: int = 0,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """Return dict class_name → list of (tile_id, pixel_row, pixel_col)
    for confidently positive pixels of that class.

    Picks pixels with the class label > 0.9 AND high confidence_tier."""
    rng = np.random.default_rng(seed)
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    for cls in CRISM_LABEL_COLS:
        cand = df[(df[cls] > 0.9) & (df.get('confidence_tier', 'High') == 'High')]
        if len(cand) == 0:
            cand = df[df[cls] > 0.9]
        if len(cand) == 0:
            out[cls] = []
            continue
        idxs = rng.choice(len(cand), size=min(n_per_class, len(cand)), replace=False)
        rows = cand.iloc[idxs]
        out[cls] = [(r['tile_id'], int(r['pixel_row']), int(r['pixel_col']))
                    for _, r in rows.iterrows()]
    return out
