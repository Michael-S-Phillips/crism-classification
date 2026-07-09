"""Tests for the labeled-spectra corpus assembler (Component 1).

Strict TDD: these are written before the implementation. They build synthetic
parquets in tmp_path that mimic each source's on-disk schema (59 band cols
m0..m58; hard_negatives carry a ``negative_of`` column) and exercise the
assembler end-to-end via ``assemble``.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.label_quant.assemble_labeled_spectra import (
    assemble,
    BAND_COLS,
    OUTPUT_COLS,
)

ALL_BANDS = [f"m{i}" for i in range(0, 59)]  # m0..m58 on-disk
MINERALS = ["olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase"]


def _make_frame(rows):
    """Build a source-like DataFrame from a list of row dicts.

    Fills all m0..m58 band columns and the five mineral columns (defaulting to
    0.0) plus ``other`` (default 0.0) and ``confidence_weight`` (default 1.0)
    unless overridden per row.
    """
    out = []
    for i, r in enumerate(rows):
        rec = {
            "tile_id": r.get("tile_id", "T1"),
            "polygon_id": r.get("polygon_id", "poly0"),
            "pixel_row": r.get("pixel_row", i),
            "pixel_col": r.get("pixel_col", 0),
            "other": r.get("other", 0.0),
            "confidence_weight": r.get("confidence_weight", 1.0),
        }
        for b, bcol in enumerate(ALL_BANDS):
            rec[bcol] = float(b)
        for m in MINERALS:
            rec[m] = r.get(m, 0.0)
        for k, v in r.items():
            if k not in rec and k in ("alteration", "negative_of"):
                rec[k] = v
        out.append(rec)
    return pd.DataFrame(out)


def _write_hand(tmp_path, rows, name="mrral_pixels.parquet"):
    p = tmp_path / name
    _make_frame(rows).to_parquet(p)
    return p


def _write_dir(tmp_path, dirname, rows, with_alteration=True,
               with_negative_of=False):
    d = tmp_path / dirname
    d.mkdir(parents=True, exist_ok=True)
    df = _make_frame(rows)
    if with_alteration and "alteration" not in df.columns:
        df["alteration"] = [r.get("alteration", 0.0) for r in rows]
    if not with_alteration and "alteration" in df.columns:
        df = df.drop(columns=["alteration"])
    if with_negative_of and "negative_of" not in df.columns:
        df["negative_of"] = [r.get("negative_of", "") for r in rows]
    df.to_parquet(d / "part0.parquet")
    return d


def test_schema_and_band_window(tmp_path):
    hand = _write_hand(tmp_path, [{"lcp": 1.0}])
    full, viz = assemble(hand_path=hand, confirmed_dirs=[], reassigned_dirs=[],
                         bland_path=None, write=False)
    assert list(full.columns) == OUTPUT_COLS
    assert list(viz.columns) == OUTPUT_COLS
    # band cols are exactly m2..m58 (no m0/m1)
    assert BAND_COLS == [f"m{i}" for i in range(2, 59)]
    assert "m0" not in full.columns and "m1" not in full.columns
    assert "m2" in full.columns and "m58" in full.columns
    band_cols = [c for c in full.columns if c[0] == "m" and c[1:].isdigit()]
    assert len(band_cols) == 57
    # real pixel coordinates are carried through both outputs (int dtype)
    for df in (full, viz):
        assert "pixel_row" in df.columns and "pixel_col" in df.columns
        assert np.issubdtype(df["pixel_row"].dtype, np.integer)
        assert np.issubdtype(df["pixel_col"].dtype, np.integer)


def test_class_collapse_and_multi(tmp_path):
    hand = _write_hand(tmp_path, [
        {"pixel_row": 0, "olivine_t1": 1.0, "hcp": 1.0},   # A: two classes
        {"pixel_row": 1, "lcp": 1.0},                       # B: pure lcp
    ])
    full, _ = assemble(hand_path=hand, confirmed_dirs=[], reassigned_dirs=[],
                       bland_path=None, write=False)
    a = full[full.pixel_row_key == 0] if "pixel_row_key" in full else None
    # A -> two rows (olivine, hcp), both multi=True
    a_rows = full[(full["class"].isin(["olivine", "hcp"]))]
    assert set(a_rows["class"]) == {"olivine", "hcp"}
    assert a_rows["multi"].all()
    # B -> one lcp row, multi=False
    b_rows = full[full["class"] == "lcp"]
    assert len(b_rows) == 1
    assert not b_rows["multi"].iloc[0]
    # olivine_t2 alone also collapses to olivine
    assert "olivine" in set(full["class"])


def test_precedence_reassigned_over_confirmed_over_hand(tmp_path):
    # Pixel P: hcp in all three sources -> reassigned wins.
    # Pixel Q: hcp in hand + confirmed -> confirmed wins.
    hand = _write_hand(tmp_path, [
        {"tile_id": "T1", "pixel_row": 10, "pixel_col": 5, "hcp": 1.0},  # P
        {"tile_id": "T1", "pixel_row": 20, "pixel_col": 6, "hcp": 1.0},  # Q
    ])
    conf = _write_dir(tmp_path, "confirmed", [
        {"tile_id": "T1", "pixel_row": 10, "pixel_col": 5, "hcp": 1.0},  # P
        {"tile_id": "T1", "pixel_row": 20, "pixel_col": 6, "hcp": 1.0},  # Q
    ])
    reas = _write_dir(tmp_path, "hard_negatives", [
        {"tile_id": "T1", "pixel_row": 10, "pixel_col": 5, "hcp": 1.0,
         "negative_of": ""},  # P
    ], with_negative_of=True)
    full, _ = assemble(hand_path=hand, confirmed_dirs=[conf],
                       reassigned_dirs=[reas], bland_path=None, write=False)
    hcp = full[full["class"] == "hcp"]
    assert len(hcp) == 2  # deduped to one row per pixel
    src_by_tile = dict(zip(zip(hcp["tile_id"]), hcp["source"]))  # noqa
    # locate P and Q via band-independent uniqueness: use source counts
    assert set(hcp["source"]) == {"reassigned", "confirmed"}
    assert (hcp["source"] == "reassigned").sum() == 1  # P
    assert (hcp["source"] == "confirmed").sum() == 1   # Q


def test_reassigned_negative_of_filter(tmp_path):
    reas = _write_dir(tmp_path, "hard_negatives", [
        {"tile_id": "T1", "pixel_row": 1, "olivine_t1": 1.0,
         "negative_of": "ambiguous"},   # excluded
        {"tile_id": "T1", "pixel_row": 2, "olivine_t1": 1.0,
         "negative_of": ""},            # included
    ], with_negative_of=True)
    full, _ = assemble(hand_path=None, confirmed_dirs=[],
                       reassigned_dirs=[reas], bland_path=None, write=False)
    oliv = full[full["class"] == "olivine"]
    assert len(oliv) == 1
    assert oliv["source"].iloc[0] == "reassigned"


def test_alteration_and_junk_tag_sources(tmp_path):
    # negative_of='alteration' -> class='alteration' (source='tag');
    # negative_of='ambiguous' -> class='junk' (source='tag'). Both from tags.
    hn = _write_dir(tmp_path, "hard_negatives", [
        {"tile_id": "T1", "pixel_row": 1, "negative_of": "alteration",
         "confidence_weight": 0.75},          # -> class='alteration', tag
        {"tile_id": "T1", "pixel_row": 2, "negative_of": "ambiguous",
         "confidence_weight": 0.5},           # -> class='junk', tag
    ], with_negative_of=True)
    full, _ = assemble(hand_path=None, confirmed_dirs=[],
                       reassigned_dirs=[hn], bland_path=None, write=False)
    alt = full[full["class"] == "alteration"]
    assert len(alt) == 1
    assert alt["source"].iloc[0] == "tag"
    assert not bool(alt["multi"].iloc[0])
    assert alt["confidence_weight"].iloc[0] == 0.75
    junk = full[full["class"] == "junk"]
    assert len(junk) == 1
    assert junk["source"].iloc[0] == "tag"
    assert not bool(junk["multi"].iloc[0])
    assert junk["confidence_weight"].iloc[0] == 0.5
    assert len(full) == 2


def test_base_bland_dust_in_full_parquet(tmp_path):
    # mrral_pixels rows with other>0.5 become class='bland_dust',
    # source='hand', in the FULL corpus (not just viz), weight forced 1.0.
    hand = _write_hand(tmp_path, [
        {"pixel_row": 0, "lcp": 1.0},                       # mineral
        {"pixel_row": 1, "other": 1.0, "confidence_weight": 0.25},  # dust
        {"pixel_row": 2, "other": 1.0, "confidence_weight": 0.5},   # dust
    ])
    full, viz = assemble(hand_path=hand, confirmed_dirs=[], reassigned_dirs=[],
                         write=False)
    dust = full[full["class"] == "bland_dust"]
    assert len(dust) == 2
    assert set(dust["source"]) == {"hand"}
    assert (dust["confidence_weight"] == 1.0).all()   # forced
    assert not dust["multi"].any()
    assert (full["class"] == "lcp").sum() == 1
    # bland_dust reaches the viz subsample too
    assert (viz["class"] == "bland_dust").sum() == 2


def test_review_bland_reject_not_collapsed(tmp_path):
    # negative_of='' rows with other>0.5 and NO mineral -> class='bland_reject',
    # source='reassigned' (reject->bland). Must NOT be collapsed into a mineral.
    hn = _write_dir(tmp_path, "hard_negatives", [
        {"tile_id": "T1", "pixel_row": 1, "other": 1.0, "negative_of": "",
         "confidence_weight": 0.75},          # review-bland (reject)
        {"tile_id": "T1", "pixel_row": 2, "olivine_t1": 1.0,
         "negative_of": ""},                  # mineral reassign
    ], with_negative_of=True)
    full, _ = assemble(hand_path=None, confirmed_dirs=[],
                       reassigned_dirs=[hn], bland_path=None, write=False)
    rej = full[full["class"] == "bland_reject"]
    assert len(rej) == 1
    assert rej["source"].iloc[0] == "reassigned"
    assert rej["confidence_weight"].iloc[0] == 0.75
    # the reject-bland pixel did NOT leak into any mineral class
    assert (full["class"].isin(["olivine", "lcp", "hcp", "plagioclase",
                                "alteration"])).sum() == 1
    assert (full["class"] == "olivine").sum() == 1


def test_no_plain_bland_class_remains(tmp_path):
    # After the dust/reject split, no row may carry class='bland' anywhere.
    hand = _write_hand(tmp_path, [
        {"pixel_row": 0, "lcp": 1.0},
        {"pixel_row": 1, "other": 1.0},       # -> bland_dust
    ])
    hn = _write_dir(tmp_path, "hard_negatives", [
        {"tile_id": "T1", "pixel_row": 2, "other": 1.0, "negative_of": ""},
    ], with_negative_of=True)
    full, viz = assemble(hand_path=hand, confirmed_dirs=[],
                         reassigned_dirs=[hn], write=False)
    assert "bland" not in set(full["class"])
    assert "bland" not in set(viz["class"])
    assert "bland_dust" in set(full["class"])
    assert "bland_reject" in set(full["class"])


def test_band_dtype_is_float32(tmp_path):
    hand = _write_hand(tmp_path, [
        {"pixel_row": 0, "lcp": 1.0},
        {"pixel_row": 1, "other": 1.0},
    ])
    full, viz = assemble(hand_path=hand, confirmed_dirs=[], reassigned_dirs=[],
                         write=False)
    for col in BAND_COLS:
        assert full[col].dtype == np.float32
        assert viz[col].dtype == np.float32


def test_viz_per_polygon_cap_and_class_total(tmp_path):
    rows = []
    # lcp: one polygon, 1000 px -> capped to 200
    for i in range(1000):
        rows.append({"tile_id": "T1", "polygon_id": "lcp_poly",
                     "pixel_row": i, "pixel_col": 0, "lcp": 1.0})
    # hcp: 40 polygons x 200 px = 8000 px -> subsample to 5000
    for pg in range(40):
        for i in range(200):
            rows.append({"tile_id": "T1", "polygon_id": f"hcp_{pg}",
                         "pixel_row": 10000 + pg * 1000 + i, "pixel_col": 1,
                         "hcp": 1.0})
    hand = _write_hand(tmp_path, rows)
    full, viz = assemble(hand_path=hand, confirmed_dirs=[], reassigned_dirs=[],
                         bland_path=None, write=False, seed=42,
                         viz_per_class=5000, viz_polygon_cap=200)
    # full corpus keeps everything
    assert (full["class"] == "lcp").sum() == 1000
    assert (full["class"] == "hcp").sum() == 8000
    # viz: lcp capped to 200 (single polygon)
    viz_lcp = viz[viz["class"] == "lcp"]
    assert len(viz_lcp) == 200
    # viz: hcp subsampled to <= viz_per_class
    viz_hcp = viz[viz["class"] == "hcp"]
    assert len(viz_hcp) == 5000
    # per-polygon cap holds within viz
    assert viz_hcp.groupby("polygon_id").size().max() <= 200


def test_alteration_nan_when_column_absent(tmp_path):
    # Old confirmed schema lacks 'alteration' entirely.
    conf = _write_dir(tmp_path, "confirmed_old", [
        {"tile_id": "T1", "pixel_row": 1, "lcp": 1.0},
    ], with_alteration=False)
    full, _ = assemble(hand_path=None, confirmed_dirs=[conf],
                       reassigned_dirs=[], bland_path=None, write=False)
    assert (full["class"] == "lcp").sum() == 1
    assert "alteration" not in set(full["class"])       # no bogus alt rows
    assert not full["class"].isna().any()                # no NaN class rows
    assert full["multi"].dtype == bool
