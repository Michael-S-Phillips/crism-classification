import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.review.app import make_spectrum_figure
from data.continuum_removal import WAVELENGTHS_59, good_band_mask_59

def _center_trace(fig):
    # the visible central line is the only width-2 trace (envelope traces are width 0)
    return next(t for t in fig.data if getattr(t.line, 'width', 0) == 2)

def _center_y(fig):
    return np.array(_center_trace(fig).y, dtype=float)

def _envelope_ys(fig):
    up = next(t for t in fig.data if getattr(t, 'name', None) == 'envelope_upper')
    lo = next(t for t in fig.data if getattr(t, 'name', None) == 'envelope_lower')
    return np.array(up.y, dtype=float), np.array(lo.y, dtype=float)

def test_cr_toggle_per_pixel_median_and_gapped_overlap():
    rng = np.random.default_rng(0)
    spectra = (0.1 + 0.25 * rng.random((40, 59))).astype(np.float32)
    wl = WAVELENGTHS_59
    raw = _center_y(make_spectrum_figure(spectra, wl, continuum_removed=False))
    cr_fig = make_spectrum_figure(spectra, wl, continuum_removed=True)
    cr = _center_y(cr_fig)
    upper, lower = _envelope_ys(cr_fig)
    gb = good_band_mask_59()
    # 1 µm overlap bands (m16-19) are gapped (NaN) so they don't render a notch
    assert np.all(np.isnan(cr[~gb]))
    assert np.all(np.isnan(upper[~gb])) and np.all(np.isnan(lower[~gb]))
    # CR property holds on the good bands
    assert np.nanmax(cr) <= 1.0001
    # envelope is the 16-84 percentile band: bounded above by 1.0 (never the
    # unphysical mean+σ > 1.0), and ordered lower <= centre <= upper
    assert np.nanmax(upper[gb]) <= 1.0001
    assert np.all(lower[gb] <= cr[gb] + 1e-6)
    assert np.all(cr[gb] <= upper[gb] + 1e-6)
    # actually transformed vs raw on the good bands
    assert not np.allclose(raw[gb], cr[gb])
    # raw path unchanged (full-band mean, no gaps)
    raw_fig = make_spectrum_figure(spectra, wl, continuum_removed=False)
    assert _center_trace(raw_fig).name == 'mean'
    assert np.allclose(raw, spectra.mean(0), atol=1e-4)
    assert not np.isnan(raw).any()
