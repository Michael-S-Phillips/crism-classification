# data/synthetic_plag.py
"""Build synthetic plagioclase training patches from ENVI mean-spectra libraries.

The 30 plagioclase spectra in /mnt/mrdr/plagioclase-targeted/ are mean spectra per
ROI (545 bands, 364-3937 nm) with no spatial info. This module (a) resamples each
spectrum to the 59 mrral bands and (b) synthesizes spatial 7x7x59 patches by tiling
the spectrum and adding per-pixel noise — so they can train the spatial encoder.
"""
from __future__ import annotations

import numpy as np

WL_SENTINEL = 65535.0   # invalid-wavelength marker in the ENVI band centers
CLIP_MAX = 0.5          # matches CRISMSpectralPatchDataset.CLIP_MAX


def interp_to_mrral_wavelengths(
    lib_wl: np.ndarray,
    lib_refl: np.ndarray,
    target_wl: np.ndarray,
) -> np.ndarray:
    """Linearly resample one library spectrum onto the target mrral wavelengths.

    Drops invalid library samples (wavelength == 65535 sentinel, or non-finite
    wavelength/reflectance) before interpolating. Linear interp; values outside
    the valid library range are held at the nearest endpoint (np.interp default).
    """
    lib_wl = np.asarray(lib_wl, dtype=np.float64)
    lib_refl = np.asarray(lib_refl, dtype=np.float64)
    valid = (
        np.isfinite(lib_wl) & np.isfinite(lib_refl) & (lib_wl < WL_SENTINEL)
    )
    if valid.sum() < 2:
        raise ValueError("Need >=2 valid library samples to interpolate")
    wl = lib_wl[valid]
    refl = lib_refl[valid]
    order = np.argsort(wl)
    return np.interp(np.asarray(target_wl, dtype=np.float64), wl[order], refl[order])
