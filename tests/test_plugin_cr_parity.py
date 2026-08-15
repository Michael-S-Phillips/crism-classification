"""Parity between the QGIS plugin's CR port and the pipeline's CR.

The spectrum-viewer plugin runs inside QGIS's Python, which cannot import
``data/continuum_removal.py``, so ``qgis_plugins/crism_spectrum_viewer/crism_cr.py``
reimplements the maths. A divergent port would draw spectra that look entirely
plausible but differ from what the model consumed — the failure is silent by
construction, because both implementations produce smooth curves in [0, 1].

This test is the only thing that catches it. Everything here compares the port
against the pipeline **value by value**; an assertion that merely checked
finiteness or range would pass against a hull computed over the wrong bands, a
missing continuum floor, or a wrong wavelength grid, and is worthless here.

Runs outside QGIS, in the `crism` env. crism_cr.py is loaded by file path (not as
a package) because the plugin package's ``__init__`` imports qgis.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data import continuum_removal as pipeline  # noqa: E402

_PLUGIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'qgis_plugins', 'crism_spectrum_viewer')
_CR_PATH = os.path.join(_PLUGIN_DIR, 'crism_cr.py')


def _load_port():
    spec = importlib.util.spec_from_file_location('_crism_cr_port', _CR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


port = _load_port()

N_REAL_SPECTRA = 600      # enough that a subtle hull bug cannot hide
ATOL = 1e-6


# ── the port's contract with its environment ────────────────────────────────

def test_port_imports_numpy_only():
    """It must run inside QGIS's interpreter: numpy and nothing else.

    A stray `from data...` or scipy import would work in this env and fail in
    QGIS, where the module is actually used.
    """
    with open(_CR_PATH) as fh:
        tree = ast.parse(fh.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imported.add('<relative import>')
            elif node.module:
                imported.add(node.module.split('.')[0])
    assert imported == {'numpy'}, f'port imports beyond numpy: {sorted(imported)}'


# ── grid and mask parity ────────────────────────────────────────────────────

def test_wavelength_grid_identical():
    """The embedded grid must be the pipeline's, bit for bit — the hull is
    interpolated over these x-values, so any drift silently bends the continuum."""
    assert port.WAVELENGTHS_59.shape == (59,)
    np.testing.assert_array_equal(port.WAVELENGTHS_59, pipeline.WAVELENGTHS_59)


def test_good_band_mask_identical():
    np.testing.assert_array_equal(port.good_band_mask(),
                                  pipeline.good_band_mask_59())


def test_good_band_mask_excludes_exactly_16_to_19():
    """Pin the indices themselves, not just agreement with the pipeline: if both
    grids were wrong together, the mask test above would still pass."""
    excluded = np.where(~port.good_band_mask())[0]
    np.testing.assert_array_equal(excluded, np.array([16, 17, 18, 19]))
    # ...and those are the bands inside 1000-1065 nm, with the neighbours out.
    wl = port.WAVELENGTHS_59
    assert np.all((wl[excluded] >= 1000.0) & (wl[excluded] <= 1065.0))
    assert wl[15] < 1000.0 and wl[20] > 1065.0


def test_bad_band_mask_is_blue_edge_plus_overlap():
    """The viewer's default masking: band 0 (410.1 nm) plus the overlap window."""
    bad = np.where(port.bad_band_mask())[0]
    np.testing.assert_array_equal(bad, np.array([0, 16, 17, 18, 19]))
    assert abs(port.WAVELENGTHS_59[0] - 410.12) < 1e-9


def test_lin_cr_clip_matches_pipeline():
    assert tuple(port.LIN_CR_CLIP) == tuple(pipeline.LIN_CR_CLIP)


# ── real-spectrum parity ────────────────────────────────────────────────────

def _real_spectra():
    """(N, 59) float32 spectra sampled deterministically from a real mrral tile."""
    rasterio = pytest.importorskip('rasterio')
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config_loader import load_config
    root = load_config()['data_root']
    img = os.path.join(root, 'mc13', 't1250_mrral_20n078_0327_4.img')
    if not os.path.exists(img):
        pytest.skip('real mrral tile not available')
    with rasterio.open(img) as src:
        cube = src.read(list(range(1, 60))).astype(np.float32)   # (59, H, W)
    flat = cube.reshape(59, -1).T                                # (H*W, 59)
    # Prefer real, non-NODATA spectra: those exercise the hull. Degenerate ones
    # are covered explicitly below.
    valid = flat[np.all(np.isfinite(flat) & (flat != 65535.0), axis=1)]
    if len(valid) < N_REAL_SPECTRA:
        pytest.skip('too few valid spectra in tile')
    idx = np.linspace(0, len(valid) - 1, N_REAL_SPECTRA).astype(np.int64)
    return np.ascontiguousarray(valid[idx])


@pytest.fixture(scope='module')
def real_spectra():
    return _real_spectra()


def test_real_spectra_are_actually_structured(real_spectra):
    """Guard the guard: if the sample were flat or degenerate, every CR value
    would be 1.0 and the parity tests below would pass against anything."""
    ref = pipeline.continuum_removed(real_spectra)
    good = pipeline.good_band_mask_59()
    depth = 1.0 - ref[:, good]
    assert depth.max() > 0.05, 'sample has no absorption features to disagree on'
    assert (depth.max(axis=1) > 0.01).mean() > 0.5, 'most spectra look degenerate'


def test_hull_cr_matches_pipeline_on_real_spectra(real_spectra):
    got = port.hull_cr(real_spectra)
    ref = pipeline.continuum_removed(real_spectra)
    np.testing.assert_allclose(got, ref, rtol=0, atol=ATOL)


def test_linear_cr_matches_pipeline_on_real_spectra(real_spectra):
    got = port.linear_cr(real_spectra)
    ref = pipeline.linear_continuum_removed(real_spectra)
    np.testing.assert_allclose(got, ref, rtol=0, atol=ATOL)


def test_parity_on_pipeline_preprocessed_spectra(real_spectra):
    """The pipeline clips raw reflectance to [0, 0.5] and zeroes NODATA before
    CR; the plugin sees vectorizer means. Both value ranges must agree."""
    clipped = np.clip(real_spectra, 0.0, 0.5)
    np.testing.assert_allclose(port.hull_cr(clipped),
                               pipeline.continuum_removed(clipped),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(port.linear_cr(clipped),
                               pipeline.linear_continuum_removed(clipped),
                               rtol=0, atol=ATOL)


def test_parity_one_spectrum_at_a_time(real_spectra):
    """The plugin calls the transforms on a single (59,) spectrum, not a batch —
    the shape the batch tests above never exercise."""
    for i in range(0, N_REAL_SPECTRA, 37):
        s = real_spectra[i]
        assert port.hull_cr(s).shape == (59,)
        np.testing.assert_allclose(port.hull_cr(s),
                                   pipeline.continuum_removed(s),
                                   rtol=0, atol=ATOL)
        np.testing.assert_allclose(port.linear_cr(s),
                                   pipeline.linear_continuum_removed(s),
                                   rtol=0, atol=ATOL)


# ── degenerate inputs ───────────────────────────────────────────────────────

def _degenerate_cases():
    wl = pipeline.WAVELENGTHS_59
    x = (wl - wl.min()) / (wl.max() - wl.min())
    return {
        'all_zero': np.zeros(59, dtype=np.float32),
        'all_nan': np.full(59, np.nan, dtype=np.float32),
        'one_nan': np.where(np.arange(59) == 30, np.nan,
                            0.2).astype(np.float32),
        'pos_inf': np.where(np.arange(59) == 5, np.inf, 0.2).astype(np.float32),
        'neg_inf': np.where(np.arange(59) == 44, -np.inf,
                            0.2).astype(np.float32),
        'flat': np.full(59, 0.2, dtype=np.float32),
        'below_epsilon': np.full(59, 1e-9, dtype=np.float32),
        'at_epsilon': np.full(59, 1e-6, dtype=np.float32),
        'negative': np.full(59, -0.3, dtype=np.float32),
        'nodata_65535': np.full(59, 65535.0, dtype=np.float32),
        'ramp': (0.05 + 0.3 * x).astype(np.float32),
    }


@pytest.mark.parametrize('name', sorted(_degenerate_cases()))
def test_degenerate_parity(name):
    """Degenerate handling is where two implementations diverge most quietly:
    the pipeline returns all-ones rather than raising, and the two 'all ones'
    branches (hull vs linear) have DIFFERENT trigger conditions (max <= 1e-6 vs
    max|.| > 1e-6), which a port is likely to conflate."""
    spec = _degenerate_cases()[name]
    np.testing.assert_allclose(port.hull_cr(spec),
                               pipeline.continuum_removed(spec),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(port.linear_cr(spec),
                               pipeline.linear_continuum_removed(spec),
                               rtol=0, atol=ATOL)


def test_degenerate_batch_parity():
    """Mixed batch: the linear path is vectorised over an `ok` mask, so a valid
    spectrum sharing a batch with a NaN one is a distinct code path."""
    cases = _degenerate_cases()
    batch = np.stack([cases[k] for k in sorted(cases)])
    np.testing.assert_allclose(port.hull_cr(batch),
                               pipeline.continuum_removed(batch),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(port.linear_cr(batch),
                               pipeline.linear_continuum_removed(batch),
                               rtol=0, atol=ATOL)


def test_all_nan_returns_ones_not_raises():
    """Pin the documented degenerate contract itself, so a port that 'fixed' it
    by raising or propagating NaN fails here as well as in the parity tests."""
    out = port.hull_cr(np.full(59, np.nan, dtype=np.float32))
    assert np.all(out == 1.0)
    out = port.linear_cr(np.full(59, np.nan, dtype=np.float32))
    assert np.all(out == 1.0)


def test_wrong_band_count_raises():
    """The plugin disables CR off the 59-band grid; the module refuses too, so a
    UI regression surfaces as an error rather than a plausible wrong hull."""
    for n in (58, 60, 87):
        with pytest.raises(ValueError):
            port.hull_cr(np.full(n, 0.2, dtype=np.float32))
        with pytest.raises(ValueError):
            port.linear_cr(np.full(n, 0.2, dtype=np.float32))


# ── random-spectrum sweep (shape variety the tile may not contain) ──────────

def test_parity_on_random_spectra():
    """Rough, spiky and monotone shapes stress hull vertex selection where real
    smooth spectra may not: ties, collinear runs and a single dominating peak."""
    rng = np.random.default_rng(0)
    specs = [
        rng.uniform(0.0, 0.5, size=(200, 59)),
        rng.uniform(0.0, 0.02, size=(100, 59)),                    # near-floor
        np.sort(rng.uniform(0.0, 0.5, size=(50, 59)), axis=1),     # monotone up
        -np.sort(-rng.uniform(0.0, 0.5, size=(50, 59)), axis=1),   # monotone down
        np.round(rng.uniform(0.0, 0.5, size=(100, 59)), 2),        # many ties
    ]
    for arr in specs:
        arr = arr.astype(np.float32)
        np.testing.assert_allclose(port.hull_cr(arr),
                                   pipeline.continuum_removed(arr),
                                   rtol=0, atol=ATOL)
        np.testing.assert_allclose(port.linear_cr(arr),
                                   pipeline.linear_continuum_removed(arr),
                                   rtol=0, atol=ATOL)


# --------------------------------------------------------------------------
# Bad bands must leave the CR FIT, not merely the plot
# --------------------------------------------------------------------------

def test_extra_exclude_removes_a_band_from_the_hull_fit():
    """An upper hull is anchored by its extremes, so one artefact band drags the
    whole continuum. Band 0 (410.1 nm) carries the blue-edge artefact and the
    vectorizer CLIPS it to 0.5 rather than discarding it, so polygon means
    routinely pair band_00 = 0.5 with a band_01 near 0.04. Masking it for
    display AFTER the fit is useless -- it has already set the continuum.

    Measured on a real deployed polygon: deepest band 0.4161 -> 0.0426, a 10x
    exaggeration removed.
    """
    hull_cr, bad_band_mask = port.hull_cr, port.bad_band_mask
    WAVELENGTHS_59 = port.WAVELENGTHS_59

    wl = np.asarray(WAVELENGTHS_59)
    # Baseline near 0.05 with band 0 clipped to 0.5, matching a real deployed
    # polygon (band_00 0.5000 beside band_01 0.0403). Measured ratio 5.7x here,
    # 9.8x on that polygon. Under the bug -- masking after the fit -- the two
    # numbers are IDENTICAL, so any threshold above 1.0 discriminates; 2.5x is
    # comfortably inside the real effect and clear of the boundary.
    spec = 0.045 + 0.01 * (wl - wl.min()) / (wl.max() - wl.min())
    spec[wl > 1400] -= 0.008 * np.exp(-0.5 * ((wl[wl > 1400] - 1900) / 200) ** 2)
    spec[0] = 0.5                      # the clipped blue-edge artefact

    diag = wl >= 1000.0
    with_artefact = 1.0 - hull_cr(spec)[diag].min()
    without = 1.0 - hull_cr(spec, extra_exclude=bad_band_mask())[diag].min()
    assert with_artefact > 2.5 * without, (
        f'band 0 did not distort the hull as expected: {with_artefact:.4f} vs '
        f'{without:.4f} — is it still entering the fit?')


def test_extra_exclude_none_is_byte_identical_to_the_pipeline_path():
    """The default path must remain exactly what the parity tests pin."""
    hull_cr = port.hull_cr
    rng = np.random.default_rng(3)
    spec = (0.08 + 0.15 * rng.random((16, 59))).astype(np.float64)
    assert np.array_equal(hull_cr(spec), hull_cr(spec, extra_exclude=None))


def test_a_clean_band0_spectrum_is_unaffected_by_the_exclusion():
    """The fix must not silently change spectra that were never broken."""
    hull_cr, bad_band_mask = port.hull_cr, port.bad_band_mask
    WAVELENGTHS_59 = port.WAVELENGTHS_59
    wl = np.asarray(WAVELENGTHS_59)
    spec = 0.10 + 0.02 * (wl - wl.min()) / (wl.max() - wl.min())
    spec[wl > 1400] -= 0.03 * np.exp(-0.5 * ((wl[wl > 1400] - 1900) / 200) ** 2)
    diag = wl >= 1000.0
    a = 1.0 - hull_cr(spec)[diag].min()
    b = 1.0 - hull_cr(spec, extra_exclude=bad_band_mask())[diag].min()
    assert abs(a - b) < 0.01, f'clean spectrum changed materially: {a:.4f} vs {b:.4f}'
