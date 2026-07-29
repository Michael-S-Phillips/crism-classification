import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.review.app import make_spectrum_figure
from data.continuum_removal import WAVELENGTHS_59, good_band_mask_59

def _mean_trace_y(fig):
    return np.array(next(t.y for t in fig.data if getattr(t, 'name', None) == 'mean'), dtype=float)

def test_cr_toggle_per_pixel_median_and_gapped_overlap():
    rng = np.random.default_rng(0)
    spectra = (0.1 + 0.25 * rng.random((40, 59))).astype(np.float32)
    wl = WAVELENGTHS_59
    raw = _mean_trace_y(make_spectrum_figure(spectra, wl, continuum_removed=False))
    cr = _mean_trace_y(make_spectrum_figure(spectra, wl, continuum_removed=True))
    gb = good_band_mask_59()
    # 1 µm overlap bands (m16-19) are gapped (NaN) so they don't render a notch
    assert np.all(np.isnan(cr[~gb]))
    # CR property holds on the good bands
    assert np.nanmax(cr) <= 1.0001
    # actually transformed vs raw on the good bands
    assert not np.allclose(raw[gb], cr[gb])
    # raw path unchanged (full-band mean, no gaps)
    assert np.allclose(raw, spectra.mean(0), atol=1e-4)
    assert not np.isnan(raw).any()
