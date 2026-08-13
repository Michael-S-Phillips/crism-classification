import os

import numpy as np
import pytest

from scripts.atmos_diagnostic import (MRRDE_EMA, MRRDE_ELEVATION, MRRDE_INA,
                                      NODATA, air_mass,
                                      derive_mrrde_path,
                                      detection_rate_by_decile,
                                      load_mrrde_covariates, main)


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


# ─────────────────────────────────────────────────────────────────────────────
# The 65535 -> NaN masking in load_mrrde_covariates.
#
# Deleting these three masks leaves every test above green, because they all
# feed load_mrrde_covariates' OUTPUT in by hand. The masks only exist on the
# read path, so they can only be pinned by reading a raster that carries the
# sentinel. What an unmasked sentinel does: 65535 deg clips to 89 deg, giving
# air_mass ~= 114.6 against a typical ~3.15 — so EVERY nodata pixel lands in the
# top air-mass decile and manufactures exactly the high-air-mass detection
# concentration this diagnostic exists to detect. Each mask is checked
# INDEPENDENTLY: removing any one of the three must fail a test.
# ─────────────────────────────────────────────────────────────────────────────

TYPICAL_INA, TYPICAL_EMA = 45.0, 10.0     # ~2.43 air mass, an ordinary pixel


def _write_mrrde(path, elev, ina, ema):
    """A real 19-band mrrde raster, so the sentinel travels the actual read
    path (rasterio -> astype(float32) -> mask) rather than a stubbed one."""
    rasterio = pytest.importorskip('rasterio')
    h, w = elev.shape
    with rasterio.open(path, 'w', driver='GTiff', height=h, width=w,
                       count=19, dtype='float32') as dst:
        for b in range(1, 20):
            dst.write(np.zeros((h, w), np.float32), b)
        dst.write(elev.astype(np.float32), MRRDE_ELEVATION + 1)
        dst.write(ina.astype(np.float32), MRRDE_INA + 1)
        dst.write(ema.astype(np.float32), MRRDE_EMA + 1)
    return str(path)


def _bands(h=2, w=2):
    elev = np.full((h, w), -1500.0, np.float32)
    ina = np.full((h, w), TYPICAL_INA, np.float32)
    ema = np.full((h, w), TYPICAL_EMA, np.float32)
    return elev, ina, ema


def test_nodata_elevation_becomes_nan_on_read(tmp_path):
    """Without the elevation mask a 65535 sentinel is a real-looking +65 km
    elevation and drags the top elevation decile with it."""
    elev, ina, ema = _bands()
    elev[0, 0] = NODATA
    p = _write_mrrde(tmp_path / 'mrrde_elev.tif', elev, ina, ema)
    out_elev, _ = load_mrrde_covariates(p)
    assert np.isnan(out_elev[0, 0]), (
        f'elevation sentinel survived the read as {out_elev[0, 0]}')
    assert np.isfinite(out_elev[1, 1]) and out_elev[1, 1] == pytest.approx(-1500.0)


def test_nodata_incidence_angle_becomes_nan_air_mass(tmp_path):
    """The INA mask, independently. An unmasked 65535 deg clips to 89 deg and
    yields a finite, bogus air mass instead of dropping out."""
    elev, ina, ema = _bands()
    ina[0, 1] = NODATA
    p = _write_mrrde(tmp_path / 'mrrde_ina.tif', elev, ina, ema)
    _, am = load_mrrde_covariates(p)
    assert np.isnan(am[0, 1]), (
        f'INA sentinel produced a finite air mass {am[0, 1]} instead of NaN')
    assert am[1, 1] == pytest.approx(1 / np.cos(np.deg2rad(TYPICAL_INA))
                                     + 1 / np.cos(np.deg2rad(TYPICAL_EMA)),
                                     rel=1e-5)


def test_nodata_emission_angle_becomes_nan_air_mass(tmp_path):
    """The EMA mask, independently: masking INA alone would still leave a
    65535 emission angle clipping to 89 deg."""
    elev, ina, ema = _bands()
    ema[1, 0] = NODATA
    p = _write_mrrde(tmp_path / 'mrrde_ema.tif', elev, ina, ema)
    _, am = load_mrrde_covariates(p)
    assert np.isnan(am[1, 0]), (
        f'EMA sentinel produced a finite air mass {am[1, 0]} instead of NaN')
    assert np.isfinite(am[1, 1])


def test_nodata_pixels_do_not_manufacture_a_high_air_mass_detection_peak(tmp_path):
    """The impact, at the boundary that matters. Detections that sit ONLY on
    nodata pixels must not read as a high-air-mass concentration: unmasked,
    those pixels all score air_mass ~114.6 against a typical ~2.4, land in the
    top decile together, and fabricate the exact residual-CO2 signature the
    diagnostic is built to find — a false positive in the false-positive
    detector."""
    h = w = 20
    elev, ina, ema = _bands(h, w)
    nodata = np.zeros((h, w), bool)
    nodata[:2, :] = True                 # a nodata-heavy tile edge
    # Only INA carries the sentinel here: with EMA also masked the air mass
    # would come out NaN via the OTHER mask, and removing the INA mask alone
    # would leave this test green.
    ina[nodata] = NODATA
    elev[nodata] = NODATA
    # vary the good pixels so the deciles are not degenerate
    ina[~nodata] = np.linspace(20, 60, int((~nodata).sum())).astype(np.float32)
    p = _write_mrrde(tmp_path / 'mrrde_edge.tif', elev, ina, ema)
    _, am = load_mrrde_covariates(p)

    prob = nodata.astype(np.float32)      # detections ONLY on the nodata edge
    valid = np.ones((h, w), bool)
    rows = detection_rate_by_decile(prob, valid, am, threshold=0.5, n=10)
    assert sum(r['n'] for r in rows) == int((~nodata).sum()), (
        'nodata pixels were binned as if they had a real air mass')
    assert max(r['rate'] for r in rows) == 0.0, (
        f'nodata detections leaked into an air-mass decile: '
        f'{[r["rate"] for r in rows]}')
    assert max(r['hi'] for r in rows) < 10.0, (
        'a nodata sentinel reached the air-mass bin edges')


def test_derive_mrrde_path_swaps_only_the_product_code():
    """Smoke test for the path derivation. The product code appears once in a
    real MRDR basename, but the directory can contain anything, so the
    replacement must be scoped to the basename."""
    p = derive_mrrde_path('/data/mrral/mc13/t1250_mrral_20n078_0327_4.img')
    assert p == '/data/mrral/mc13/t1250_mrrde_20n078_0327_4.img'
    assert os.path.dirname(p) == '/data/mrral/mc13'
    # a directory named after the other product is untouched
    assert derive_mrrde_path('/x/_mrral_/t1_mrral_a.img') == '/x/_mrral_/t1_mrrde_a.img'


def test_main_runs_end_to_end_on_a_synthetic_tile(tmp_path, monkeypatch, capsys):
    """Smoke test for main(): npz -> class lookup -> mrrde derivation ->
    co-registration check -> decile tables."""
    h = w = 8
    elev, ina, ema = _bands(h, w)
    elev[:] = np.linspace(-3000, 1000, h * w).reshape(h, w)
    _write_mrrde(tmp_path / 't1250_mrrde_20n078_0327_4.img', elev, ina, ema)
    tile = str(tmp_path / 't1250_mrral_20n078_0327_4.img')

    probs = np.zeros((h, w, 2), np.float32)
    probs[..., 1] = (elev < -2000).astype(np.float32)
    npz = str(tmp_path / 'probs.npz')
    np.savez_compressed(npz, probs=probs, valid_mask=np.ones((h, w), bool),
                        class_names=np.array(['olivine', 'hcp']))

    monkeypatch.setattr('sys.argv', ['atmos_diagnostic.py', '--probs', npz,
                                     '--tile', tile, '--klass', 'hcp'])
    main()
    out = capsys.readouterr().out
    assert 'hcp detection rate by elevation (m) decile' in out
    assert 'hcp detection rate by air mass decile' in out
    assert 'Reported, not gated' in out


def test_main_rejects_a_class_the_npz_does_not_carry(tmp_path, monkeypatch):
    """An unknown --klass must fail loudly rather than index a wrong channel."""
    h = w = 4
    elev, ina, ema = _bands(h, w)
    _write_mrrde(tmp_path / 't1_mrrde_a_0327_4.img', elev, ina, ema)
    npz = str(tmp_path / 'probs.npz')
    np.savez_compressed(npz, probs=np.zeros((h, w, 1), np.float32),
                        valid_mask=np.ones((h, w), bool),
                        class_names=np.array(['olivine']))
    monkeypatch.setattr('sys.argv', ['atmos_diagnostic.py', '--probs', npz,
                                     '--tile', str(tmp_path / 't1_mrral_a_0327_4.img'),
                                     '--klass', 'hcp'])
    with pytest.raises(SystemExit, match='hcp'):
        main()
