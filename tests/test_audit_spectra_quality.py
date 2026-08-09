"""Tests for scripts/audit_spectra_quality.py.

The audit exists because of the 2026-08-08 t1444 incident: a review session
extracted a tile mid-download and froze 537,525 rows with reflectance 0.0 across
2251-2457 nm. Nothing crashed, nothing warned, and because bland is review-only
in the hand-core build ~72% of that class carried a zero tail no other class had.

These tests lock the detectors that would have caught it, and — just as
important — lock the fact that the KNOWN blue-edge artifact does NOT fail the
run. A gate that fires on a condition the pipeline already handles gets ignored,
and then it catches nothing.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from audit_spectra_quality import audit_block, CHECKS, INFO_CHECKS  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PROJ, 'scripts', 'audit_spectra_quality.py')
BANDS = [f'm{i}' for i in range(59)]


def _clean(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.35, size=(n, 59)).astype(np.float32)


def test_clean_spectra_trip_nothing():
    flags = audit_block(_clean(), min_run=5, min_tail=3)
    for c in CHECKS + INFO_CHECKS:
        assert not flags[c].any(), f'{c} fired on clean spectra'


def test_zero_tail_detected():
    """The exact t1444 signature: trailing bands frozen at 0.0."""
    X = _clean()
    X[3:7, 51:] = 0.0
    flags = audit_block(X, min_run=5, min_tail=3)
    assert flags['zero_tail'].sum() == 4
    assert not flags['all_zero'].any()
    assert not flags['nodata'].any()


def test_interior_zero_run_detected():
    X = _clean()
    X[2, 20:30] = 0.0
    flags = audit_block(X, min_run=5, min_tail=3)
    assert flags['zero_run'][2]
    assert not flags['zero_tail'][2], 'interior run must not be read as a tail'


def test_short_zero_run_below_threshold_ignored():
    X = _clean()
    X[1, 20:22] = 0.0          # 2 bands, under min_run=5
    flags = audit_block(X, min_run=5, min_tail=3)
    assert not flags['zero_run'][1]


def test_nodata_and_nonfinite_and_flat():
    X = _clean()
    X[0, 5] = 65535.0
    X[1, 9] = np.nan
    X[2, :] = 0.2
    flags = audit_block(X, min_run=5, min_tail=3)
    assert flags['nodata'][0]
    assert flags['nonfinite'][1]
    assert flags['flat'][2]


def test_blue_edge_is_informational_not_a_defect():
    """Band-0 spikes are the known MRRAL artifact the training reader masks.

    If this ever starts failing the run, the gate becomes noise and people stop
    reading it — which is how t1444 would slip through a second time.
    """
    X = _clean()
    X[4, 0] = 1180.0                       # band 0 only
    flags = audit_block(X, min_run=5, min_tail=3)
    assert flags['blue_edge'][4]
    assert not flags['over_phys'][4], 'band 0 must not count as a real defect'


def test_over_phys_fires_for_non_blue_edge_bands():
    X = _clean()
    X[5, 30] = 4.0                         # a real, unhandled corruption
    flags = audit_block(X, min_run=5, min_tail=3)
    assert flags['over_phys'][5]
    assert not flags['blue_edge'][5]


def _write(tmp_path, X, name='frag.parquet'):
    d = {c: X[:, i].astype(np.float64) for i, c in enumerate(BANDS)}
    d['tile_id'] = ['t9999'] * len(X)
    d['pixel_row'] = np.arange(len(X))
    d['pixel_col'] = np.arange(len(X))
    p = tmp_path / name
    pd.DataFrame(d).to_parquet(p, index=False)
    return str(p)


def _run(path, *extra):
    return subprocess.run([sys.executable, SCRIPT, path, *extra],
                          capture_output=True, text=True, cwd=PROJ)


def test_cli_exits_zero_on_clean_parquet(tmp_path):
    r = _run(_write(tmp_path, _clean(30)))
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'RESULT: PASS' in r.stdout


def test_cli_exits_nonzero_and_names_the_tile(tmp_path):
    X = _clean(30)
    X[:15, 51:] = 0.0
    r = _run(_write(tmp_path, X))
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'RESULT: FAIL' in r.stdout
    assert 't9999' in r.stdout, 'per-tile breakdown must name the offending tile'


def test_cli_blue_edge_alone_still_passes(tmp_path):
    X = _clean(30)
    X[:10, 0] = 900.0
    r = _run(_write(tmp_path, X))
    assert r.returncode == 0, 'known blue-edge artifact must not fail the gate'
    assert 'blue_edge' in r.stdout


def test_cli_rejects_a_parquet_without_band_columns(tmp_path):
    p = tmp_path / 'nobands.parquet'
    pd.DataFrame({'tile_id': ['t1'], 'x': [1.0]}).to_parquet(p, index=False)
    r = _run(str(p))
    assert r.returncode == 2, 'bad invocation must be distinguishable from defects'
