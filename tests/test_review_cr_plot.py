import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.review.app import make_spectrum_figure
from data.continuum_removal import WAVELENGTHS_59

def _mean_trace_y(fig):
    return np.array(next(t.y for t in fig.data if getattr(t, 'name', None) == 'mean'))

def test_cr_toggle_changes_plot_and_bounds_below_one():
    rng = np.random.default_rng(0)
    spectra = (0.1 + 0.25 * rng.random((20, 59))).astype(np.float32)
    wl = WAVELENGTHS_59
    raw = _mean_trace_y(make_spectrum_figure(spectra, wl, continuum_removed=False))
    cr = _mean_trace_y(make_spectrum_figure(spectra, wl, continuum_removed=True))
    assert np.nanmax(cr) <= 1.0001            # CR property
    assert not np.allclose(raw, cr)            # actually transformed
    assert np.allclose(raw, spectra.mean(0), atol=1e-4)  # raw path unchanged
