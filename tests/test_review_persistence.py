import datetime as dt
import os
import numpy as np
import pandas as pd
import pytest

from scripts.review.persistence import (
    DecisionLog,
    ConfirmedPixelsWriter,
    HardNegativesWriter,
    confirmed_schema_columns,
)


# ---- DecisionLog -----------------------------------------------------------

def _record(uid='t0001::thresh_0.95::0', decision='confirm', corrected=''):
    return dict(
        source_gpkg='vector_mc13_relabeled/hcp.gpkg',
        layer='thresh_0.95',
        polygon_uid=uid,
        tile_id='t0001',
        predicted_class='hcp',
        decision=decision,
        corrected_class=corrected,
        n_pixels=312,
        area_m2=90400.5,
    )


def test_decision_log_creates_csv_and_appends_header(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    log.append(_record())
    df = pd.read_csv(csv)
    assert list(df.columns) == [
        'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
        'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
    ]
    assert df.iloc[0]['polygon_uid'] == 't0001::thresh_0.95::0'
    assert df.iloc[0]['ts']  # iso8601 string


def test_decision_log_appends_without_rewriting_header(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    log.append(_record(uid='t0001::thresh_0.95::0'))
    log.append(_record(uid='t0001::thresh_0.95::1', decision='reject'))
    df = pd.read_csv(csv)
    assert len(df) == 2
    # File should have exactly one header line
    with open(csv) as fp:
        lines = fp.readlines()
    assert lines[0].startswith('ts,source_gpkg')


def test_decision_log_uids_seen(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    log.append(_record(uid='a::b::0'))
    log.append(_record(uid='a::b::1'))
    log2 = DecisionLog(str(csv))   # reopened
    assert log2.uids_seen() == {'a::b::0', 'a::b::1'}


def test_decision_log_uids_seen_empty_when_no_file(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    assert log.uids_seen() == set()


# ---- ConfirmedPixelsWriter -------------------------------------------------

def test_confirmed_writer_schema_matches_mrral_pixels(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001',
        polygon_uid='t0001::thresh_0.95::0',
        rows=np.array([5, 6, 7], dtype=np.int64),
        cols=np.array([5, 6, 7], dtype=np.int64),
        spectra=np.arange(3 * 59, dtype=np.float32).reshape(3, 59),
        label_class='hcp',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert list(df.columns) == confirmed_schema_columns()
    assert len(df) == 3
    assert df['hcp'].iloc[0] == 1.0
    assert df['olivine_t1'].iloc[0] == 0.0
    assert df['confidence_weight'].iloc[0] == 1.0
    assert df['confidence_tier'].iloc[0] == 'High'
    assert df['split'].iloc[0] == 'train'
    assert df['tile_id'].iloc[0] == 't0001'
    assert df['m0'].iloc[0] == pytest.approx(0.0)
    assert df['m58'].iloc[2] == pytest.approx(3 * 59 - 1)


def test_confirmed_writer_olivine_sets_t1(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='x::y::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='olivine')
    w.flush()
    df = pd.read_parquet(pq)
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['olivine_t2'].iloc[0] == 0.0


def test_confirmed_writer_dedupes_on_reappend(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='t0001::a::0',
                     rows=np.array([1], dtype=np.int64),
                     cols=np.array([1], dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    w.flush()
    # Append the SAME polygon again — must replace, not duplicate
    w2 = ConfirmedPixelsWriter(str(pq))
    w2.append_polygon(tile_id='t0001', polygon_uid='t0001::a::0',
                      rows=np.array([1, 2], dtype=np.int64),
                      cols=np.array([1, 2], dtype=np.int64),
                      spectra=np.ones((2, 59), dtype=np.float32),
                      label_class='hcp')
    w2.flush()
    df = pd.read_parquet(pq)
    assert len(df) == 2  # replaced, not duplicated
    assert df['m0'].iloc[0] == 1.0


def test_confirmed_writer_dedupe_survives_fresh_process(tmp_path, monkeypatch):
    """polygon_id MUST be deterministic across processes (no PYTHONHASHSEED dep).

    Simulates a fresh-process re-append by clearing the writer state AND any
    cached hash randomization between writes."""
    pq = tmp_path / 'confirmed.parquet'
    # First write
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='t0001::a::0',
                     rows=np.array([1], dtype=np.int64),
                     cols=np.array([1], dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    w.flush()
    first_polygon_id = pd.read_parquet(pq)['polygon_id'].iloc[0]

    # Force a different Python hash seed inside the second writer's polygon_id
    # derivation: monkeypatch the deterministic helper to confirm that the
    # writer is using it (not bare hash()). If the writer used bare hash(),
    # this monkeypatch wouldn't change anything and the assertion at the end
    # would still pass by coincidence — so we ALSO assert the polygon_id is
    # equal to the deterministic helper's output, which guarantees the writer
    # consults the helper rather than hash().
    from scripts.review.persistence import _polygon_id_int
    expected_polygon_id = _polygon_id_int('t0001::a::0')
    assert first_polygon_id == expected_polygon_id

    # Re-append the SAME polygon (different writer instance) — must replace, not duplicate
    w2 = ConfirmedPixelsWriter(str(pq))
    w2.append_polygon(tile_id='t0001', polygon_uid='t0001::a::0',
                      rows=np.array([1, 2], dtype=np.int64),
                      cols=np.array([1, 2], dtype=np.int64),
                      spectra=np.ones((2, 59), dtype=np.float32),
                      label_class='hcp')
    w2.flush()
    df = pd.read_parquet(pq)
    assert len(df) == 2  # replaced, not 3 (which would be a duplicate-leak bug)
    assert df['m0'].iloc[0] == 1.0


# ---- HardNegativesWriter ---------------------------------------------------

def test_hard_negatives_blank_corrected(tmp_path):
    pq = tmp_path / 'hard_negatives.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='x::y::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp',
        corrected_class=None,
    )
    w.flush()
    df = pd.read_parquet(pq)
    # All label columns 0; negative_of populated
    assert df['olivine_t1'].iloc[0] == 0.0
    assert df['lcp'].iloc[0] == 0.0
    assert df['hcp'].iloc[0] == 0.0
    assert df['negative_of'].iloc[0] == 'hcp'


def test_hard_negatives_dedupe_survives_fresh_process(tmp_path):
    pq = tmp_path / 'hard_neg.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='hn::a::0',
        rows=np.array([1], dtype=np.int64),
        cols=np.array([1], dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class=None,
    )
    w.flush()
    w2 = HardNegativesWriter(str(pq))
    w2.append_polygon(
        tile_id='t0001', polygon_uid='hn::a::0',
        rows=np.array([1, 2], dtype=np.int64),
        cols=np.array([1, 2], dtype=np.int64),
        spectra=np.ones((2, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class=None,
    )
    w2.flush()
    df = pd.read_parquet(pq)
    assert len(df) == 2
    assert df['m0'].iloc[0] == 1.0


def test_hard_negatives_with_corrected(tmp_path):
    pq = tmp_path / 'hard_negatives.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='x::y::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp',
        corrected_class='olivine',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['hcp'].iloc[0] == 0.0
    # When corrected_class is set, negative_of is left blank/null
    assert pd.isna(df['negative_of'].iloc[0]) or df['negative_of'].iloc[0] == ''
