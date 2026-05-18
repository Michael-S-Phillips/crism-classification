"""
CRISMCachedPatchDataset — mmap-backed reader for the pre-built global patch cache.

Replaces CRISMGlobalPatchDataset (deleted). Reads 7×7×59 float32 patches from
sharded `.npy` files produced by `scripts/build_global_patch_cache.py`. Per-patch
zero-mean / unit-variance normalization happens on read.

Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md
"""
from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


def compute_valid_centers(
    nodata: np.ndarray,
    patch_size: int = 7,
    min_valid_frac: float = 0.8,
) -> np.ndarray:
    """Return a (H, W) bool array: True where a patch_size×patch_size patch
    centered at (r, c) has ≥ min_valid_frac valid pixels.

    Centers within `patch_size // 2` of any edge are False (the patch would
    fall off the tile).

    Vectorized via scipy.signal.convolve2d on the nodata mask — one pass over
    the tile regardless of how many candidate centers are evaluated.
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    H, W = nodata.shape
    half = patch_size // 2
    n_pix = patch_size * patch_size

    # Count of nodata pixels in each patch_size×patch_size window, centered.
    kernel = np.ones((patch_size, patch_size), dtype=np.int32)
    nodata_counts = convolve2d(
        nodata.astype(np.int32), kernel, mode='same', boundary='fill', fillvalue=1,
    )
    valid_counts = n_pix - nodata_counts
    valid_frac = valid_counts / n_pix

    valid = valid_frac >= min_valid_frac

    # Force False at edges where a patch would fall off.
    valid[:half, :] = False
    valid[H - half:, :] = False
    valid[:, :half] = False
    valid[:, W - half:] = False
    return valid
