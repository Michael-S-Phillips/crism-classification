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

# Aux column ordering used everywhere downstream: column 0 = RPEAK1, column 1 = BD1300.
AUX_BAND_ORDER = ("RPEAK1", "BD1300")

# Physically plausible ranges for each aux band. Pixels outside these bounds are
# treated as NODATA (CRISM continuum-removal / peak-finding failure modes can
# produce e.g. RPEAK1 < 0.5 um or > 1.0 um; real plagioclase RPEAK1 sits
# ~0.7-0.8 um). BD1300 is a continuum-removed band depth, occasionally slightly
# negative due to noise; |BD1300| > 0.5 indicates a calibration artifact.
BAND_VALID_RANGES = {
    "RPEAK1": (0.5, 1.0),
    "BD1300": (-0.5, 0.5),
}


def physically_valid_mask(arr: np.ndarray, band: str) -> np.ndarray:
    """Return a boolean mask of physically-valid entries in ``arr``.

    A value is valid iff it is finite, not the NODATA sentinel, and within
    ``BAND_VALID_RANGES[band]`` (inclusive).
    """
    if band not in BAND_VALID_RANGES:
        raise KeyError(
            f"unknown band {band!r}; expected one of {sorted(BAND_VALID_RANGES)}"
        )
    lo, hi = BAND_VALID_RANGES[band]
    a = np.asarray(arr)
    finite = np.isfinite(a)
    not_sentinel = a != NODATA
    in_range = (a >= lo) & (a <= hi)
    return finite & not_sentinel & in_range


def apply_invalid_to_nan(arr: np.ndarray, band: str) -> np.ndarray:
    """Return a float32 copy of ``arr`` with physically-invalid entries set to NaN.

    Valid entries are preserved bit-exact (modulo dtype). Downstream code that
    already handles NaN (e.g. ``mean_pool_nodata`` via its ``isfinite`` mask, or
    NumPy reductions like ``nanmean``) can then ignore the invalid pixels
    uniformly.
    """
    a = np.asarray(arr, dtype=np.float32).copy()
    valid = physically_valid_mask(a, band)
    a[~valid] = np.nan
    return a


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
