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


def synthesize_patches(
    spectrum: np.ndarray,
    n_aug: int,
    rng: np.random.Generator,
    patch_size: int = 7,
    noise_sigma: float = 0.005,
    jitter_sigma: float = 0.003,
    continuum_scale_range: tuple[float, float] = (0.97, 1.03),
) -> np.ndarray:
    """Tile one 59-band spectrum into n_aug augmented (patch_size, patch_size, 59) patches.

    Each augmentation applies, independently per patch:
      - a global continuum scale (multiplicative, uniform in continuum_scale_range)
      - per-band jitter (additive, same across the 49 pixels — a spectral wobble)
      - per-pixel Gaussian noise (additive, independent per pixel & band)
    Per-pixel noise is what prevents a flat-tile shortcut: the 49 pixels differ.
    Output is clipped to [0, CLIP_MAX] to match the real patch normalization.
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    n_bands = spectrum.shape[0]
    P = patch_size
    out = np.empty((n_aug, P, P, n_bands), dtype=np.float32)
    for i in range(n_aug):
        scale = rng.uniform(*continuum_scale_range)
        band_jitter = rng.normal(0.0, jitter_sigma, size=n_bands).astype(np.float32)
        base = spectrum * scale + band_jitter                      # (59,)
        tile = np.broadcast_to(base, (P, P, n_bands)).copy()       # (7,7,59)
        tile += rng.normal(0.0, noise_sigma, size=tile.shape).astype(np.float32)
        out[i] = np.clip(tile, 0.0, CLIP_MAX)
    return out
