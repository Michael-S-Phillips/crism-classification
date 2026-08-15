# -*- coding: utf-8 -*-
"""Continuum removal for the 59-band mrral grid — standalone port for QGIS.

This plugin runs inside QGIS's own Python interpreter, which has no access to the
`crism` conda env and therefore cannot import ``data/continuum_removal.py``. The
maths is reimplemented here, **numpy only** (no QGIS imports, no repo imports, no
scipy), so the module is importable and testable outside QGIS.

The whole point of a viewer that draws CR spectra is that the plot shows what the
model consumed. A divergent port would draw a plausible but wrong spectrum, so
``tests/test_plugin_cr_parity.py`` (run in the `crism` env, outside QGIS) asserts
this module matches ``data.continuum_removal`` to float tolerance on real CRISM
spectra and on the degenerate inputs. If ``data/continuum_removal.py`` ever
changes, that test fails and the two must be reconciled deliberately.

Ported behaviours, exactly:
  * upper-convex-hull CR over the good bands, excluded bands -> 1.0;
  * degenerate spectra (non-finite, or max <= 1e-6) -> all ones, never raising;
  * continuum floored at 1e-6 before dividing; hull CR clipped to [0, 1];
  * per-spectrum least-squares linear continuum, clipped to LIN_CR_CLIP.
"""

import numpy as np

# Wavelength grid of the 59-band model input (mrral 0327_4, first 59 of 72 bands,
# 410-2457 nm). Embedded verbatim from data/mrral_wavelengths_59.json because the
# plugin cannot read the repo's metadata sidecar.
WAVELENGTHS_59 = np.asarray([
    410.12, 442.63, 533.74, 598.86, 650.99, 683.59, 709.68, 742.3, 774.92,
    801.04, 833.68, 859.81, 892.48, 925.16, 951.31, 984.01, 1021.0, 1023.27,
    1047.2, 1055.99, 1079.96, 1152.06, 1211.09, 1250.45, 1257.01, 1263.57,
    1276.7, 1329.21, 1368.61, 1394.89, 1427.73, 1467.16, 1500.03, 1506.61,
    1559.21, 1625.0, 1657.91, 1690.82, 1750.09, 1809.39, 1875.3, 1928.06,
    1974.24, 1980.84, 2007.23, 2066.64, 2119.48, 2139.3, 2165.72, 2205.38,
    2231.82, 2251.65, 2291.33, 2317.79, 2331.02, 2350.87, 2390.58, 2430.3,
    2456.79,
], dtype=np.float64)

N_BANDS = 59

# The 1 um VNIR/IR detector-overlap window; bands inside it are excluded from the
# continuum fit (indices 16, 17, 18, 19 on this grid).
EXCLUSION_WINDOW_NM = (1000.0, 1065.0)

_GOOD = ~((WAVELENGTHS_59 >= EXCLUSION_WINDOW_NM[0])
          & (WAVELENGTHS_59 <= EXCLUSION_WINDOW_NM[1]))
_GOOD_IDX = np.where(_GOOD)[0]
_GOOD_WL = WAVELENGTHS_59[_GOOD_IDX]

LIN_CR_CLIP = (0.0, 2.0)

# Bands hidden by the viewer's "bad bands" toggle: band 0 is the 410.1 nm blue
# edge artefact (reaches ~1180 I/F and compresses every autoscaled y-axis), and
# 16-19 are the detector-overlap window, which hull CR sets to exactly 1.0 by
# construction — a flat plateau that is not a measurement.
BAD_BAND_INDICES = (0, 16, 17, 18, 19)


def good_band_mask():
    """(59,) bool — False inside the 1000-1065 nm detector-overlap window."""
    return _GOOD.copy()


def bad_band_mask():
    """(59,) bool — True for bands the viewer masks when 'bad bands' is on."""
    m = np.zeros(N_BANDS, dtype=bool)
    m[list(BAD_BAND_INDICES)] = True
    return m


def _upper_hull_continuum(y, x=None):
    """Upper convex hull of (good_wl, y) interpolated back onto the good bands.

    y: (n_good,) reflectance on the good-band wavelengths (increasing). Returns
    the continuum (>= y everywhere) on those same bands.
    """
    if x is None:
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


def _cr_one(spec59, good_idx=None):
    """Continuum-remove a single 59-band spectrum -> (59,), CR<=1, excluded->1."""
    out = np.ones(N_BANDS, dtype=np.float32)
    if good_idx is None:
        good_idx = _GOOD_IDX
    y = spec59[good_idx].astype(np.float64)
    # Degenerate/near-empty pixel (NODATA->0, flat): no bands.
    if not np.all(np.isfinite(y)) or float(np.max(y)) <= 1e-6:
        return out
    cont = _upper_hull_continuum(y, WAVELENGTHS_59[good_idx])
    cont = np.where(cont <= 1e-6, 1.0, cont)
    cr = y / cont
    cr = np.clip(np.nan_to_num(cr, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    out[good_idx] = cr.astype(np.float32)
    return out


def hull_cr(spec, extra_exclude=None):
    """Upper-hull CR of a spectrum or batch. spec: (..., 59) -> same shape.

    NaN/Inf-safe; excluded and degenerate bands -> 1.0.

    extra_exclude: optional (59,) bool mask of ADDITIONAL bands to keep out of
    the hull fit. Default None reproduces data/continuum_removal.py exactly and
    is what the parity test pins.

    Why this exists. An upper hull is anchored by its extremes, so a single
    artefact band drags the whole continuum. Band 0 (410.1 nm) carries the
    blue-edge artefact, and the vectorizer CLIPS it to CLIP_MAX = 0.5 rather
    than discarding it, so polygon means routinely carry band_00 = 0.5000 beside
    a band_01 of 0.04. Feeding that to the hull inflated the deepest apparent
    band from 0.043 to 0.416 -- a 10x exaggeration, max deviation 0.639 across
    the spectrum. The pipeline never sees this because load_tile marks any pixel
    with a band above 1.0 I/F as INVALID, so the artefact never reaches the
    model; it survives only into the display means the viewer reads. Masking
    such a band for DISPLAY after the fit is useless -- it has already set the
    continuum. It must be excluded from the fit itself.
    """
    spec = np.asarray(spec)
    if spec.shape[-1] != N_BANDS:
        raise ValueError('expected last dim %d, got %s' % (N_BANDS, spec.shape))
    flat = spec.reshape(-1, N_BANDS)
    out = np.empty_like(flat, dtype=np.float32)
    if extra_exclude is None:
        idx = _GOOD_IDX
    else:
        keep = _GOOD & ~np.asarray(extra_exclude, dtype=bool)
        idx = np.where(keep)[0]
        if idx.size < 3:
            raise ValueError(
                'extra_exclude leaves %d usable bands; a hull needs at least 3'
                % idx.size)
    for i in range(flat.shape[0]):
        out[i] = _cr_one(flat[i], idx)
    return out.reshape(spec.shape).astype(np.float32)


def _linear_continuum(y):
    """Least-squares straight line through (good_wl, y). y: (n, n_good)."""
    x = (_GOOD_WL - _GOOD_WL.mean()) / (_GOOD_WL.max() - _GOOD_WL.min())
    X = np.stack([np.ones_like(x), x], axis=1)            # (n_good, 2)
    coef = np.linalg.lstsq(X, y.T, rcond=None)[0]         # (2, n)
    return (X @ coef).T                                   # (n, n_good)


def linear_cr(spec):
    """Divide out a per-spectrum LINEAR continuum. spec: (..., 59) -> same shape.

    Removes overall level and slope but cannot remove curvature, so a broad
    convex arch (which upper-hull CR destroys, being approximately the hull
    itself) survives. Excluded bands and degenerate spectra -> 1.0, matching
    hull_cr; output clipped to LIN_CR_CLIP.
    """
    spec = np.asarray(spec)
    if spec.shape[-1] != N_BANDS:
        raise ValueError('expected last dim %d, got %s' % (N_BANDS, spec.shape))
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
