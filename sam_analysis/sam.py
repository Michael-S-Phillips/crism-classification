"""Pure-numpy Spectral Angle Mapper (SAM) core.

The SAM angle between two non-negative spectra is
    theta = arccos( <t, r> / (||t|| ||r||) )
and lies in [0, pi/2] for reflectance vectors with t, r >= 0. We do not
assume non-negativity here (the math also handles arbitrary real vectors,
giving [0, pi]), since the same routine is reused for embedding-space
cosine-angle comparisons in `embedding_sam`.

Conventions:
- Pixels with all-NaN bands (or zero-norm) return NaN angle.
- NaN bands are dropped pairwise (target-NaN OR ref-NaN -> ignored on
  both sides of the dot product and the norms).
- All math is float64 internally; outputs are float32 to keep the angle
  rasters small on disk.
"""
from __future__ import annotations

import numpy as np

# Pure-numpy: do not import sklearn / spectral / scipy here.

EPS = 1e-12


def _angle_from_dot_norms(dot: np.ndarray, n_t: np.ndarray, n_r: np.ndarray) -> np.ndarray:
    """Return arccos(dot / (n_t * n_r)) with safe handling for zero norms.

    Returns NaN where either norm is zero (no spectral information).
    Clamps the cosine to [-1, 1] before arccos to avoid NaN due to FP slop.
    """
    denom = n_t * n_r
    invalid = denom <= EPS
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(invalid, np.nan, dot / np.where(invalid, 1.0, denom))
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    return angle


def spectral_angle(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute SAM angle in radians between target spectrum/spectra and a reference.

    Args:
        target: shape (N, B) or (B,) — pixel spectra. NaN values are treated as
                missing for that pixel/band pair.
        ref:    shape (B,) — reference endmember spectrum (no NaNs expected; any
                NaNs in ref are dropped pairwise like target NaNs).
    Returns:
        Per-pixel angle (radians). Shape (N,) for 2D input, scalar for 1D.
        NaN where the pixel has < 2 valid bands or zero norm.
    """
    target = np.asarray(target, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    if ref.ndim != 1:
        raise ValueError(f"ref must be 1D, got shape {ref.shape}")

    if target.ndim == 1:
        out = spectral_angle(target[None, :], ref)
        return float(out[0])

    if target.ndim != 2:
        raise ValueError(f"target must be 1D or 2D, got shape {target.shape}")

    n, b = target.shape
    if b != ref.shape[0]:
        raise ValueError(
            f"band dimension mismatch: target has {b} bands, ref has {ref.shape[0]}"
        )

    ref_finite = np.isfinite(ref)
    target_finite = np.isfinite(target)
    valid = target_finite & ref_finite[None, :]   # (N, B)

    # Replace NaNs with 0 so they contribute nothing to dot/norms after masking.
    t_masked = np.where(valid, target, 0.0)
    r_row = np.where(ref_finite, ref, 0.0)        # (B,)
    # Reference is the same per pixel; mask it pairwise via `valid`.
    dot = np.einsum("nb,b->n", t_masked, r_row)
    # Norms must be computed using only the bands that are valid for THAT pixel.
    n_t = np.sqrt(np.einsum("nb,nb->n", t_masked, t_masked))
    n_r = np.sqrt(((r_row[None, :] ** 2) * valid).sum(axis=1))

    angle = _angle_from_dot_norms(dot, n_t, n_r)

    # If <2 valid bands, the angle is meaningless: force NaN.
    valid_count = valid.sum(axis=1)
    angle = np.where(valid_count < 2, np.nan, angle)
    return angle


def sam_raster(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Apply spectral_angle to every pixel in a (H, W, B) cube.

    Args:
        cube: (H, W, B) float; NaN where NoData.
        ref:  (B,) reference endmember.
    Returns:
        (H, W) float32 angle raster. NaN where pixel is all-invalid.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D (H, W, B); got {cube.shape}")
    h, w, b = cube.shape
    flat = cube.reshape(h * w, b)
    angles = spectral_angle(flat, ref)
    return angles.reshape(h, w).astype(np.float32)


def cosine_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience wrapper: angle in radians between row vectors of `a` and
    a single reference vector `b`. Identical math to `spectral_angle`; this is
    here so embedding-space code reads more naturally.
    """
    return spectral_angle(a, b)
