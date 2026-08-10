"""Continuum removal for mrral spectra (59-band model input).

The classifier/encoder consume the 59-band mrral window (m0..m58, 410-2457 nm).
Raw-reflectance variance is dominated by albedo/continuum, not mineralogy, which
entangles brightness with mineral signal and breaks cross-terrain generalization
(see docs/superpowers/specs/2026-07-15-cr-mrral-representation-design.md).

This module maps a raw spectrum to its continuum-removed (CR) form using an
**upper convex hull** continuum over the good-band window (the 1 µm VNIR/IR
detector-overlap bands, indices 16-19 / 1021-1056 nm, are excluded). CR isolates
absorption bands (values <= 1, band depth = 1 - CR). Absolute albedo, discarded by
CR but a real cue for bland/dust, is preserved separately as a brightness scalar.

Parameter-free (no polynomial degree or tie-points) and NaN/Inf-safe.
"""
from __future__ import annotations

import json
import os

import numpy as np

_SIDECAR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'mrral_wavelengths_59.json')
with open(_SIDECAR) as _f:
    _meta = json.load(_f)

WAVELENGTHS_59 = np.asarray(_meta['wavelengths_nm'], dtype=np.float64)
assert WAVELENGTHS_59.shape == (59,), WAVELENGTHS_59.shape
N_BANDS = 59
_EXCL_LO, _EXCL_HI = _meta['exclusion_window_nm']
_GOOD = ~((WAVELENGTHS_59 >= _EXCL_LO) & (WAVELENGTHS_59 <= _EXCL_HI))
_GOOD_IDX = np.where(_GOOD)[0]
_GOOD_WL = WAVELENGTHS_59[_GOOD_IDX]


def good_band_mask_59() -> np.ndarray:
    """Bool mask over the 59 model bands; False inside the 1 µm overlap window."""
    return _GOOD.copy()


def _upper_hull_continuum(y: np.ndarray) -> np.ndarray:
    """Upper convex hull of (good_wl, y) interpolated back onto the good bands.

    y: (n_good,) reflectance on the good-band wavelengths (increasing). Returns the
    continuum (>= y everywhere) on those same bands.
    """
    x = _GOOD_WL
    n = len(x)
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            ox, oy = x[hull[-2]], y[hull[-2]]
            ax, ay = x[hull[-1]], y[hull[-1]]
            # cross((a-o),(i-o)); >=0 is a left turn/collinear -> pop to keep the
            # hull above every point (upper hull).
            cross = (ax - ox) * (y[i] - oy) - (ay - oy) * (x[i] - ox)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    hull = np.asarray(hull)
    return np.interp(x, x[hull], y[hull])


def _cr_one(spec59: np.ndarray) -> np.ndarray:
    """Continuum-remove a single 59-band spectrum -> (59,), CR<=1, excluded->1."""
    out = np.ones(59, dtype=np.float32)
    y = spec59[_GOOD_IDX].astype(np.float64)
    # Degenerate/near-empty pixel (NODATA->0, flat): no bands.
    if not np.all(np.isfinite(y)) or float(np.max(y)) <= 1e-6:
        return out
    cont = _upper_hull_continuum(y)
    cont = np.where(cont <= 1e-6, 1.0, cont)
    cr = y / cont
    cr = np.clip(np.nan_to_num(cr, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    out[_GOOD_IDX] = cr.astype(np.float32)
    return out


def continuum_removed(spec: np.ndarray) -> np.ndarray:
    """CR of a spectrum or batch. spec: (..., 59) -> same shape, CR reflectance.

    NaN/Inf-safe; excluded and degenerate bands -> 1.0.
    """
    spec = np.asarray(spec)
    if spec.shape[-1] != 59:
        raise ValueError(f'expected last dim 59, got {spec.shape}')
    flat = spec.reshape(-1, 59)
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = _cr_one(flat[i])
    return out.reshape(spec.shape).astype(np.float32)


LIN_CR_CLIP = (0.0, 2.0)   # p99.99 of real data is 1.415; the tails reach +-10


def _linear_continuum(y: np.ndarray) -> np.ndarray:
    """Least-squares straight line through (good_wl, y). y: (n, n_good)."""
    x = (_GOOD_WL - _GOOD_WL.mean()) / (_GOOD_WL.max() - _GOOD_WL.min())
    X = np.stack([np.ones_like(x), x], axis=1)            # (n_good, 2)
    coef, *_ = np.linalg.lstsq(X, y.T, rcond=None)        # (2, n)
    return (X @ coef).T                                   # (n, n_good)


def linear_continuum_removed(spec: np.ndarray) -> np.ndarray:
    """Divide out a per-spectrum LINEAR continuum. spec: (..., 59) -> same shape.

    Removes overall level and slope but is mathematically incapable of removing
    curvature, because a line has none. That is the difference from
    continuum_removed(): upper-hull CR divides by the convex hull, and a broad
    convex arch IS approximately the hull, so hull CR destroys it (41% of
    alteration's 1-2um arch retained, vs 84% for bland). Per-pixel, the arch
    alone separates alteration from every other class at AUC 0.990 under linear
    CR against 0.856 under hull CR.

    Excluded bands and degenerate spectra -> 1.0, matching continuum_removed.
    Output is clipped to LIN_CR_CLIP: unclipped values reach +-10 and would
    dominate gradients.
    """
    spec = np.asarray(spec)
    if spec.shape[-1] != N_BANDS:
        raise ValueError(f'expected last dim {N_BANDS}, got {spec.shape}')
    flat = spec.reshape(-1, N_BANDS).astype(np.float64)
    out = np.ones_like(flat, dtype=np.float32)

    y = flat[:, _GOOD_IDX]
    ok = np.isfinite(y).all(axis=1) & (np.max(np.abs(y), axis=1) > 1e-6)
    if ok.any():
        cont = _linear_continuum(y[ok])
        cont = np.where(np.abs(cont) < 1e-6, 1.0, cont)
        r = np.nan_to_num(y[ok] / cont, nan=1.0, posinf=1.0, neginf=1.0)
        out[np.ix_(ok, _GOOD_IDX)] = np.clip(
            r, LIN_CR_CLIP[0], LIN_CR_CLIP[1]).astype(np.float32)
    return out.reshape(spec.shape)


def brightness_scalar(spec: np.ndarray) -> np.ndarray:
    """Mean good-band reflectance (pre-CR). spec: (..., 59) -> (...)."""
    spec = np.asarray(spec, dtype=np.float32)
    if spec.shape[-1] != 59:
        raise ValueError(f'expected last dim 59, got {spec.shape}')
    return spec[..., _GOOD_IDX].mean(axis=-1)


def cr_patch(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CR a (P, P, 59) patch. Returns (CR patch (P,P,59), brightness (P,P))."""
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim != 3 or patch.shape[-1] != 59:
        raise ValueError(f'expected (P,P,59), got {patch.shape}')
    cr = continuum_removed(patch)
    bright = brightness_scalar(patch)
    return cr, bright
