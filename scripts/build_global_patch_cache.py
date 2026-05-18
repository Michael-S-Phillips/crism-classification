"""
Build the global pre-training patch cache.

Reads all mrral tiles under `data_root` (config), samples ~2834 valid patches
per tile, writes 50 sharded `.npy` files of 100k patches each (~58 GB total)
to `--output`, plus a `shard_index.json` recording build provenance.

Usage (HPC):
    python scripts/build_global_patch_cache.py \\
        --output /xdisk/sbyrne/phillipsm/crism_patch_cache/ \\
        --workers 16 --seed 42

Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Tuple

import numpy as np
import rasterio
import rasterio.windows

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.cached_patch_dataset import compute_valid_centers


N_BANDS = 59          # mrral bands 1–59 (matches data/global_patch_dataset.py)
CLIP_MAX = 0.5
MIN_VALID_FRAC = 0.8
NODATA_VALUE = 65535
PATCH_SIZE = 7

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def extract_patches_from_tile(
    hdr_path: str,
    n_target: int,
    patch_size: int = PATCH_SIZE,
    min_valid_frac: float = MIN_VALID_FRAC,
    clip_max: float = CLIP_MAX,
    nodata_value: float = NODATA_VALUE,
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """Sample up to n_target valid patches from one mrral tile.

    Returns:
        patches: (n, patch_size, patch_size, n_bands) float32 where n ≤ n_target
        n_skipped_short: max(0, n_target - n) — how many fewer than requested
    """
    img_path = hdr_path.replace('.hdr', '.img')

    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        # Load band 1 to compute nodata mask (proxy for which positions are valid).
        band1 = src.read(1).astype(np.float32)
        nodata = (band1 == nodata_value) | ~np.isfinite(band1)

        valid_centers = compute_valid_centers(nodata, patch_size, min_valid_frac)
        valid_rs, valid_cs = np.where(valid_centers)

        n_valid = len(valid_rs)
        rng = np.random.default_rng(seed)
        if n_valid == 0:
            return np.zeros((0, patch_size, patch_size, N_BANDS), dtype=np.float32), n_target
        n_take = min(n_target, n_valid)
        choice = rng.choice(n_valid, size=n_take, replace=False)
        sampled_rs = valid_rs[choice]
        sampled_cs = valid_cs[choice]

        half = patch_size // 2
        out = np.zeros((n_take, patch_size, patch_size, N_BANDS), dtype=np.float32)
        for i in range(n_take):
            r, c = int(sampled_rs[i]), int(sampled_cs[i])
            window = rasterio.windows.Window(c - half, r - half, patch_size, patch_size)
            patch = src.read(list(range(1, N_BANDS + 1)), window=window).astype(np.float32)
            # (N_BANDS, P, P) -> (P, P, N_BANDS)
            patch = patch.transpose(1, 2, 0)
            # Replace nodata with 0.0 and clip.
            mask = (patch == nodata_value) | ~np.isfinite(patch)
            patch[mask] = 0.0
            np.clip(patch, 0.0, clip_max, out=patch)
            out[i] = patch

    n_skipped_short = max(0, n_target - n_take)
    return out, n_skipped_short
