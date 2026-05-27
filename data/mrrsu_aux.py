# data/mrrsu_aux.py
"""Helpers for building the smoothed mrrsu auxiliary features (RPEAK1, BD1300).

The plag-vs-olivine discriminant (RPEAK1) is regional, not per-pixel, so we feed
the classifier a 7x7-mean of the mrrsu parameter rasters. NODATA pixels are
excluded from each window mean.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

NODATA = 65535.0
RPEAK1_BAND = 8      # 0-indexed mrrsu band
BD1300_BAND = 17     # 0-indexed mrrsu band


def mean_pool_nodata(raster: np.ndarray, patch_size: int = 7,
                     nodata: float = NODATA) -> np.ndarray:
    """KxK windowed mean of a 2-D raster, excluding NODATA / non-finite pixels.

    Returns a float32 array the same shape as `raster`. Windows with no valid
    pixels are NaN. Uses two box filters (sum of values / count of valid) so the
    cost is O(H*W) regardless of window size.
    """
    r = raster.astype(np.float64)
    valid = np.isfinite(r) & (r != nodata)
    vals = np.where(valid, r, 0.0)
    counts = valid.astype(np.float64)
    # uniform_filter computes the MEAN over the window; multiply by area to get sums
    area = patch_size * patch_size
    sum_vals = uniform_filter(vals, size=patch_size, mode='constant', cval=0.0) * area
    sum_cnt = uniform_filter(counts, size=patch_size, mode='constant', cval=0.0) * area
    out = np.full(r.shape, np.nan, dtype=np.float32)
    nz = sum_cnt > 0
    out[nz] = (sum_vals[nz] / sum_cnt[nz]).astype(np.float32)
    return out
