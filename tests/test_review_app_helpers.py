import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from scripts.review.app import compute_progress, make_spectrum_figure


def test_compute_progress_aggregates_by_mineral(tmp_path):
    # Synthetic decisions.csv
    decisions = pd.DataFrame([
        # 2 confirms for hcp (n_pixels 300 + 200), 1 reject for hcp, 1 skip for hcp
        {'predicted_class': 'hcp', 'decision': 'confirm', 'n_pixels': 300, 'corrected_class': ''},
        {'predicted_class': 'hcp', 'decision': 'confirm', 'n_pixels': 200, 'corrected_class': ''},
        {'predicted_class': 'hcp', 'decision': 'reject', 'n_pixels': 99, 'corrected_class': ''},
        {'predicted_class': 'hcp', 'decision': 'skip', 'n_pixels': 50, 'corrected_class': ''},
        # 1 confirm for lcp
        {'predicted_class': 'lcp', 'decision': 'confirm', 'n_pixels': 150, 'corrected_class': ''},
    ])
    csv = tmp_path / 'decisions.csv'
    decisions.to_csv(csv, index=False)

    prog_hcp = compute_progress(str(csv), mineral='hcp', target_pixels=30000)
    assert prog_hcp['confirmed_pixels'] == 500
    assert prog_hcp['reviewed'] == 4
    assert prog_hcp['confirm_count'] == 2
    assert prog_hcp['reject_count'] == 1
    assert prog_hcp['skip_count'] == 1
    assert prog_hcp['target_pixels'] == 30000
    assert prog_hcp['fraction'] == pytest.approx(500 / 30000)
    assert prog_hcp['target_reached'] is False

    prog_lcp = compute_progress(str(csv), mineral='lcp', target_pixels=100)
    assert prog_lcp['confirmed_pixels'] == 150
    assert prog_lcp['target_reached'] is True


def test_compute_progress_handles_missing_csv(tmp_path):
    prog = compute_progress(str(tmp_path / 'no.csv'), mineral='hcp', target_pixels=30000)
    assert prog['confirmed_pixels'] == 0
    assert prog['reviewed'] == 0
    assert prog['target_reached'] is False


def test_make_spectrum_figure_has_mean_and_envelope_traces():
    n_pixels, n_bands = 12, 59
    rng = np.random.default_rng(0)
    spectra = rng.normal(0.2, 0.01, size=(n_pixels, n_bands)).astype(np.float32)
    wavelengths_nm = np.linspace(410, 2457, n_bands)
    fig = make_spectrum_figure(spectra, wavelengths_nm)
    assert isinstance(fig, Figure)
    # At least: 1 mean line + 1 lower-envelope + 1 upper-envelope (3+ traces);
    # use trace names to assert presence.
    names = {tr.name for tr in fig.data}
    assert 'mean' in names
    assert any('envelope' in (n or '') for n in names)


def test_make_spectrum_figure_zero_pixels_returns_empty_figure():
    fig = make_spectrum_figure(
        np.zeros((0, 59), dtype=np.float32),
        np.linspace(410, 2457, 59),
    )
    assert isinstance(fig, Figure)
    # No data crash; ok to be empty


def test_make_spectrum_figure_default_xrange_450_2500():
    spectra = np.full((10, 59), 0.3, dtype=np.float32)
    fig = make_spectrum_figure(spectra, np.linspace(410, 2457, 59))
    assert list(fig.layout.xaxis.range) == [450.0, 2500.0]


def test_make_spectrum_figure_yrange_robust_to_spurious_band():
    # Flat 0.3 spectra with ONE in-window band at 0.9 (survives the [-0.5,1.5]
    # pixel filter and the <=1.05 clamp): the y-axis must not stretch to 0.9.
    spectra = np.full((10, 59), 0.3, dtype=np.float32)
    spectra[:, 30] = 0.9
    fig = make_spectrum_figure(spectra, np.linspace(410, 2457, 59))
    y_lo, y_hi = fig.layout.yaxis.range
    assert y_hi < 0.6, f'spurious band stretched y to {y_hi}'
    assert y_lo < 0.3 < y_hi


def test_make_spectrum_figure_yrange_ignores_out_of_window_band():
    # Band 0 (~410 nm, outside the 450-2500 display window) is noisy-high:
    # it must not drive the y-range.
    spectra = np.full((10, 59), 0.3, dtype=np.float32)
    spectra[:, 0] = 0.95
    fig = make_spectrum_figure(spectra, np.linspace(410, 2457, 59))
    y_lo, y_hi = fig.layout.yaxis.range
    assert y_hi < 0.6, f'out-of-window band stretched y to {y_hi}'
