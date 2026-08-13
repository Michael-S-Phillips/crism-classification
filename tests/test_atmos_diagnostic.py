import numpy as np

from scripts.atmos_diagnostic import air_mass, detection_rate_by_decile


def test_detects_a_planted_elevation_dependence():
    """If detections concentrate at low elevation that is residual CO2, not
    clinopyroxene. The diagnostic must surface exactly that."""
    rng = np.random.default_rng(0)
    elev = rng.uniform(-4000, 2000, size=(100, 100)).astype(np.float32)
    prob = (elev < -2000).astype(np.float32) * 0.9      # planted: low only
    valid = np.ones_like(prob, bool)
    rows = detection_rate_by_decile(prob, valid, elev, threshold=0.5, n=10)
    assert len(rows) == 10
    assert rows[0]['rate'] > 0.5, 'lowest elevation decile should be hot'
    assert rows[-1]['rate'] == 0.0, 'highest elevation decile should be cold'


def test_flat_dependence_reports_flat():
    """Detections uncorrelated with the covariate must not read as a false
    positive elevation dependence -- a real risk if bin edges are computed
    with unequal population sizes, which inflates per-bin sampling noise."""
    rng = np.random.default_rng(1)
    elev = rng.uniform(-4000, 2000, size=(100, 100)).astype(np.float32)
    prob = rng.random(elev.shape).astype(np.float32)
    valid = np.ones_like(prob, bool)
    rows = detection_rate_by_decile(prob, valid, elev, threshold=0.5, n=10)
    ns = [r['n'] for r in rows]
    assert max(ns) - min(ns) <= 1, f'quantile bins should be equal-population: {ns}'
    rates = [r['rate'] for r in rows]
    assert max(rates) - min(rates) < 0.15, f'spurious dependence: {rates}'


def test_air_mass_increases_with_incidence_angle():
    assert air_mass(np.float32(60.0), np.float32(0.0)) > \
           air_mass(np.float32(0.0), np.float32(0.0))


def test_nodata_covariate_pixels_are_excluded():
    """65535 is the mrrde sentinel. A pixel whose covariate is NaN must not
    contribute to any decile's rate or count -- otherwise a nodata-heavy tile
    edge would masquerade as a real low- or high-covariate detection rate."""
    rng = np.random.default_rng(2)
    elev = rng.uniform(-4000, 2000, size=(50, 50)).astype(np.float32)
    prob = rng.random(elev.shape).astype(np.float32)
    valid = np.ones_like(prob, bool)

    elev_with_nans = elev.copy()
    elev_with_nans[:10, :] = np.nan
    n_nan = int(np.isnan(elev_with_nans).sum())

    rows = detection_rate_by_decile(prob, valid, elev_with_nans, threshold=0.5, n=10)
    total_n = sum(r['n'] for r in rows)
    assert total_n == elev.size - n_nan
