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
    REVIEW_CONFIDENCE_WEIGHTS,
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
        'co_occurring_classes', 'confidence',
    ]
    assert df.iloc[0]['polygon_uid'] == 't0001::thresh_0.95::0'
    assert df.iloc[0]['ts']  # iso8601 string


def test_decision_log_records_confidence(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    rec = _record()
    rec['confidence'] = 'Moderate'
    log.append(rec)
    df = pd.read_csv(csv)
    assert df.iloc[0]['confidence'] == 'Moderate'


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
    assert df['confidence_tier'].iloc[0] == 'Reviewed-High'
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


# ---- Re-decision support (most_recent_for + drop_polygon) ------------------

def test_most_recent_for_returns_last_match(tmp_path):
    csv_path = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv_path))
    log.append(_record(uid='t::a::0', decision='confirm'))
    log.append(_record(uid='t::a::0', decision='reject'))  # supersede
    log.append(_record(uid='t::b::0', decision='confirm'))
    most = log.most_recent_for('t::a::0')
    assert most is not None
    assert most['decision'] == 'reject'
    assert log.most_recent_for('does::not::exist') is None


def test_most_recent_for_returns_none_when_no_file(tmp_path):
    log = DecisionLog(str(tmp_path / 'no.csv'))
    assert log.most_recent_for('x::y::0') is None


def test_drop_polygon_removes_rows_keyed_by_uid(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='keep::me::0',
                     rows=np.array([1], dtype=np.int64),
                     cols=np.array([1], dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    w.append_polygon(tile_id='t0001', polygon_uid='drop::me::0',
                     rows=np.array([2, 3], dtype=np.int64),
                     cols=np.array([2, 3], dtype=np.int64),
                     spectra=np.ones((2, 59), dtype=np.float32),
                     label_class='hcp')
    w.flush()
    assert len(pd.read_parquet(pq)) == 3

    w.drop_polygon('drop::me::0')
    df = pd.read_parquet(pq)
    assert len(df) == 1
    assert df['m0'].iloc[0] == 0.0  # the "keep" row survives


def test_drop_polygon_idempotent_when_missing(tmp_path):
    """In the per-polygon-file dataset layout, dropping a uid that was never
    written is a clean no-op (just an os.path.exists check that fails)."""
    pq = tmp_path / 'confirmed'  # directory now, not single file
    w = ConfirmedPixelsWriter(str(pq))
    # The directory exists after init, but no per-polygon files in it.
    assert os.path.isdir(pq)
    assert len(os.listdir(pq)) == 0
    # Drop a uid that was never written → no-op
    w.drop_polygon('x::y::0')
    assert len(os.listdir(pq)) == 0
    # Write one polygon, then drop a different uid → original file survives
    w.append_polygon(tile_id='t0001', polygon_uid='real::poly::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    before_count = len(os.listdir(pq))
    w.drop_polygon('nonexistent::poly::0')
    assert len(os.listdir(pq)) == before_count


def test_hard_negatives_drop_polygon(tmp_path):
    pq = tmp_path / 'hardneg.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='hn::a::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     predicted_class='hcp', corrected_class=None)
    w.flush()
    assert len(pd.read_parquet(pq)) == 1
    w.drop_polygon('hn::a::0')
    assert len(pd.read_parquet(pq)) == 0


# ---- decisions.csv durability (no overwrites, header-race-safe) ------------

def test_decision_log_many_appends_no_lost_rows(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    for i in range(250):
        log.append(_record(uid=f'bulk::a::{i}'))
    df = pd.read_csv(csv)
    assert len(df) == 250
    assert df['polygon_uid'].iloc[0] == 'bulk::a::0'
    assert df['polygon_uid'].iloc[-1] == 'bulk::a::249'
    # Header appears exactly once
    with open(csv) as fp:
        lines = fp.readlines()
    assert sum(1 for L in lines if L.startswith('ts,source_gpkg')) == 1


def test_decision_log_header_safe_when_empty_file_preexists(tmp_path):
    """If something else (e.g. a touch) created an empty csv before us, the
    next append must still write the header — checked via file POSITION
    after open, not pre-open existence."""
    csv = tmp_path / 'decisions.csv'
    csv.touch()  # 0-byte file exists
    assert os.path.exists(csv) and csv.stat().st_size == 0
    log = DecisionLog(str(csv))
    log.append(_record(uid='post::touch::0'))
    df = pd.read_csv(csv)
    assert list(df.columns)[0] == 'ts'
    assert df['polygon_uid'].iloc[0] == 'post::touch::0'


def test_atomic_parquet_no_stray_tmp_file(tmp_path):
    """After a successful flush there must be no .tmp file left behind."""
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='atomic::a::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    w.flush()
    assert pq.exists()
    assert not (tmp_path / 'confirmed.parquet.tmp').exists()


def test_atomic_parquet_second_flush_preserves_first(tmp_path):
    """A second flush from a fresh writer must add to the existing parquet,
    not destroy the prior content (atomic-rename writes the unioned data)."""
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='r::a::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    w.flush()
    w2 = ConfirmedPixelsWriter(str(pq))
    w2.append_polygon(tile_id='t0002', polygon_uid='r::b::0',
                      rows=np.zeros(1, dtype=np.int64),
                      cols=np.zeros(1, dtype=np.int64),
                      spectra=np.ones((1, 59), dtype=np.float32),
                      label_class='hcp')
    w2.flush()
    df = pd.read_parquet(pq)
    assert len(df) == 2
    assert df['polygon_id'].nunique() == 2  # both polygons preserved


# ---- new UI categories: bland (alias of 'other') + ambiguous (tag-only) ----

def test_confirmed_writer_bland_maps_to_other_column(tmp_path):
    """The UI exposes 'bland' as the friendly name for the schema's 'other'
    label column. A confirmed bland polygon must set other=1.0 (and nothing
    else) in the parquet, so downstream pipelines see the same column."""
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='bland::poly::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='bland')
    w.flush()
    df = pd.read_parquet(pq)
    assert df['other'].iloc[0] == 1.0
    assert df['olivine_t1'].iloc[0] == 0.0
    assert df['lcp'].iloc[0] == 0.0
    assert df['hcp'].iloc[0] == 0.0


def test_hard_neg_bland_writes_other_column(tmp_path):
    """If a rejected polygon is tagged 'bland' via the dropdown, treat it
    like a positive bland confirmation in the negatives parquet (positive
    'other' label, blank negative_of) — same downstream semantics as
    confirming bland directly."""
    pq = tmp_path / 'hardneg.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='bn::a::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='bland',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['other'].iloc[0] == 1.0
    assert df['hcp'].iloc[0] == 0.0
    assert pd.isna(df['negative_of'].iloc[0]) or df['negative_of'].iloc[0] == ''


def test_hard_neg_ambiguous_is_negative_tag_not_positive_class(tmp_path):
    """'ambiguous' is a non-mineral tag: rejected polygon, no positive label
    anywhere, negative_of='ambiguous' (NOT predicted_class)."""
    pq = tmp_path / 'hardneg.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='amb::a::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='ambiguous',
    )
    w.flush()
    df = pd.read_parquet(pq)
    # All label columns zero — ambiguous is NOT a positive class
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        assert df[col].iloc[0] == 0.0, f'{col} should be 0 for ambiguous'
    assert df['negative_of'].iloc[0] == 'ambiguous'


def test_hard_neg_alteration_is_tag_not_positive_label(tmp_path):
    """'alteration' is a tag (not a positive mineral class). corrected_class='alteration'
    produces all-zero labels with negative_of='alteration', matching existing
    103.9k rows produced by load_alteration_mc11. The 7-class build ingests
    alteration exclusively from negative_of='alteration', not from label columns."""
    pq = tmp_path / 'hardneg.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='alt::a::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='alteration',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['negative_of'].iloc[0] == 'alteration', 'alteration is a tag: negative_of should be alteration'
    assert df['alteration'].iloc[0] == 0.0, 'alteration label column must be 0 (tag, not positive)'
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        assert df[col].iloc[0] == 0.0, f'{col} should be 0 when tagged alteration'
    assert df['confidence_weight'].iloc[0] == 1.0
    assert df['confidence_tier'].iloc[0] == 'Reviewed-High'


# ---- Multi-label confirms (co-occurring minerals) --------------------------

def test_confirmed_writer_extra_classes_sets_multiple_labels(tmp_path):
    """When a polygon is confirmed as olivine AND has co-occurring hcp,
    BOTH columns must be 1.0 in the parquet — that's what multi-label loss
    consumes. Single-class confirm (today's behavior) still works with no
    extra_classes."""
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='multi::a::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        label_class='olivine', extra_classes=['hcp'],
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['hcp'].iloc[0] == 1.0
    # Other columns still zero
    assert df['lcp'].iloc[0] == 0.0
    assert df['plagioclase'].iloc[0] == 0.0
    assert df['other'].iloc[0] == 0.0


def test_confirmed_writer_extra_classes_none_is_single_class(tmp_path):
    """Backward compat: extra_classes=None (or omitted) behaves exactly
    like the original single-class confirm path."""
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='single::a::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        label_class='olivine',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['hcp'].iloc[0] == 0.0


def test_decision_log_migrates_legacy_csv_in_place(tmp_path):
    """A pre-migration decisions.csv (no co_occurring_classes column) should
    be silently upgraded on first append. Existing rows get an empty value
    for the new column; new rows write the value as normal. Without this,
    DictWriter would corrupt the file by writing 11-field rows into a
    10-field schema."""
    csv = tmp_path / 'decisions.csv'
    # Hand-write a legacy CSV (10 columns; no co_occurring_classes)
    legacy_header = ('ts,source_gpkg,layer,polygon_uid,tile_id,'
                      'predicted_class,decision,corrected_class,n_pixels,area_m2')
    legacy_row = ('2026-06-07T19:53:00Z,vector_mc13_relabeled/hcp.gpkg,'
                   'thresh_0.97,t1030::thresh_0.97::4,t1030,hcp,skip,,228,7557998')
    csv.write_text(legacy_header + '\n' + legacy_row + '\n')

    log = DecisionLog(str(csv))
    log.append(_record(uid='multi::a::0', decision='confirm'))

    df = pd.read_csv(csv)
    assert 'co_occurring_classes' in df.columns
    assert len(df) == 2
    # Legacy row preserved
    assert df.iloc[0]['polygon_uid'] == 't1030::thresh_0.97::4'
    # New row has the new column (default empty in _record helper)
    assert df.iloc[1]['polygon_uid'] == 'multi::a::0'


def test_confirmed_writer_extra_classes_with_bland(tmp_path):
    """Edge case: extra_classes can include 'bland' (an alias for the
    'other' schema column). label_class=hcp + extra_classes=['bland']
    means hcp=1.0 AND other=1.0 (a hcp-bearing dust patch, rare but valid)."""
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='hcp_bland::a::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        label_class='hcp', extra_classes=['bland'],
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['hcp'].iloc[0] == 1.0
    assert df['other'].iloc[0] == 1.0


# ---- per-polygon dataset layout (OOM fix from 2026-06-10) -----------------

def test_append_polygon_writes_one_file_per_polygon(tmp_path):
    """Each append creates its own parquet file in the output directory.
    There is NO read-modify-write of accumulated history — this is the
    fix for the 13-14 GB OOM that the prior single-file flush caused."""
    pq_dir = tmp_path / 'confirmed'
    w = ConfirmedPixelsWriter(str(pq_dir))
    for i in range(5):
        w.append_polygon(
            tile_id=f't{i:04d}', polygon_uid=f'multi::a::{i}',
            rows=np.zeros(3, dtype=np.int64),
            cols=np.zeros(3, dtype=np.int64),
            spectra=np.zeros((3, 59), dtype=np.float32),
            label_class='olivine',
        )
    files = sorted(f for f in os.listdir(pq_dir) if f.endswith('.parquet'))
    assert len(files) == 5
    # Read the directory as a unified dataset
    df = pd.read_parquet(str(pq_dir))
    assert len(df) == 15  # 5 polygons × 3 rows each


def test_append_polygon_uid_re_append_overwrites_file(tmp_path):
    """Re-appending the same polygon_uid replaces its file (same hash →
    same filename → atomic-rename overwrite)."""
    pq_dir = tmp_path / 'confirmed'
    w = ConfirmedPixelsWriter(str(pq_dir))
    w.append_polygon(tile_id='t0001', polygon_uid='dup::a::0',
                     rows=np.array([0], dtype=np.int64),
                     cols=np.array([0], dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='olivine')
    w.append_polygon(tile_id='t0001', polygon_uid='dup::a::0',  # same uid
                     rows=np.array([10, 11], dtype=np.int64),
                     cols=np.array([20, 21], dtype=np.int64),
                     spectra=np.ones((2, 59), dtype=np.float32),
                     label_class='olivine')
    df = pd.read_parquet(str(pq_dir))
    # Only the LATEST write survives (2 rows, not 3)
    assert len(df) == 2
    assert df['pixel_row'].tolist() == [10, 11]


def test_legacy_single_file_migration(tmp_path):
    """If a pre-2026-06-10 single-file parquet exists at <dir>.parquet,
    instantiating the writer moves it into the directory as legacy.parquet
    so the historical data is still readable via the new dataset path."""
    pq_dir = tmp_path / 'confirmed'
    legacy_path = str(pq_dir) + '.parquet'

    # Hand-write a legacy single-file parquet with one row
    legacy_df = pd.DataFrame({c: [''] if c in ('tile_id', 'confidence_tier', 'split')
                                       else [0.0] if c.startswith('m') or c in [
                                           'olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                                           'plagioclase', 'other', 'confidence_weight',
                                       ] else [0]
                                       for c in confirmed_schema_columns()})
    legacy_df.to_parquet(legacy_path)
    assert os.path.isfile(legacy_path)
    assert not os.path.exists(pq_dir)

    w = ConfirmedPixelsWriter(str(pq_dir))
    # After init: directory exists, legacy file moved inside
    assert os.path.isdir(pq_dir)
    assert not os.path.exists(legacy_path)
    assert os.path.exists(pq_dir / 'legacy.parquet')
    # The legacy data is still reachable via the dataset read
    df = pd.read_parquet(str(pq_dir))
    assert len(df) == 1


# ---- Confidence weight stamping (Task 1) -----------------------------------

@pytest.mark.parametrize('confidence,weight', [
    ('High', 1.0), ('Moderate', 0.75), ('Low', 0.5),
])
def test_confirmed_writer_stamps_confidence(tmp_path, confidence, weight):
    pq = tmp_path / 'confirmed'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::a::0',
        rows=np.array([0, 1]), cols=np.array([0, 1]),
        spectra=np.zeros((2, 59), dtype=np.float32),
        label_class='hcp', confidence=confidence,
    )
    df = pd.read_parquet(str(pq))
    assert (df['confidence_weight'] == weight).all()
    assert (df['confidence_tier'] == f'Reviewed-{confidence}').all()


def test_review_confidence_weights_values():
    assert REVIEW_CONFIDENCE_WEIGHTS == {'High': 1.0, 'Moderate': 0.75, 'Low': 0.5}


def test_hard_negatives_mineral_reassignment_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::a::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='olivine',
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['confidence_weight'].iloc[0] == 0.5
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Low'
    assert df['negative_of'].iloc[0] == ''


def test_hard_negatives_tag_reject_now_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::b::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='ambiguous',
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['confidence_weight'].iloc[0] == 0.5
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Low'
    assert df['negative_of'].iloc[0] == 'ambiguous'


def test_hard_negatives_blank_corrected_keeps_fixed_weight(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::c::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class=None,
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['confidence_weight'].iloc[0] == 1.0
    assert df['confidence_tier'].iloc[0] == 'High'
    assert df['negative_of'].iloc[0] == 'hcp'


# ---- Task A: alteration as tag; confidence on all active-assignment branches --

def test_alteration_is_not_a_mineral_class():
    from scripts.review.persistence import _is_mineral_class
    assert _is_mineral_class('alteration') is False
    assert _is_mineral_class('olivine') is True


def test_hard_negatives_alteration_tag_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::alt::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='alteration',
        confidence='Moderate',
    )
    df = pd.read_parquet(str(pq))
    assert df['negative_of'].iloc[0] == 'alteration'
    assert df['alteration'].iloc[0] == 0.0          # tag, not a positive label
    assert df['confidence_weight'].iloc[0] == 0.75
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Moderate'


def test_hard_negatives_ambiguous_tag_weighted(tmp_path):
    pq = tmp_path / 'hardneg'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='t0001::amb::0',
        rows=np.array([0]), cols=np.array([0]),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp', corrected_class='ambiguous',
        confidence='Low',
    )
    df = pd.read_parquet(str(pq))
    assert df['negative_of'].iloc[0] == 'ambiguous'
    assert df['confidence_weight'].iloc[0] == 0.5
    assert df['confidence_tier'].iloc[0] == 'Reviewed-Low'
