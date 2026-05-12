# tests/test_categorize_sup_gpkg.py
"""Tests for the supplementary-gpkg categorization script."""
import pandas as pd
import pytest

from scripts.categorize_sup_gpkg import (
    categorize_minerals,
    is_contaminated_denom,
)


def _row(id1="", id2="", id3="", id4=""):
    """Build a row dict with the four Mineral ID columns."""
    return pd.Series({
        "Mineral ID 1": id1,
        "Mineral ID 2": id2,
        "Mineral ID 3": id3,
        "Mineral ID 4": id4,
    })


# --- categorize_minerals --------------------------------------------------

def test_clean_primary_class_high():
    assert categorize_minerals(_row(id1="hcp")) == "hcp (High)"


def test_two_clean_classes_sorted_high():
    # Categories are sorted alphabetically: hcp < olivine
    assert categorize_minerals(_row(id1="olivine", id2="hcp")) == "hcp + olivine (High)"


def test_pm_in_id1_drops_to_low():
    assert categorize_minerals(_row(id1="±hcp")) == "hcp (Low)"


def test_pm_in_id2_drops_to_moderate():
    assert categorize_minerals(_row(id1="hcp", id2="±olivine")) == "hcp + olivine (Moderate)"


def test_pm_in_id1_then_id2_stays_low():
    # Once tier is Low it cannot upgrade.
    assert categorize_minerals(_row(id1="±hcp", id2="±olivine")) == "hcp + olivine (Low)"


def test_uncertain_in_id1_alone_returns_other_low():
    # 'uncertain' is not in the minerals list → no categories collected → 'Other'.
    assert categorize_minerals(_row(id1="uncertain")) == "Other (Low)"


def test_uncertain_in_id1_with_secondary_class():
    # Uncertain in ID1 drops tier to Low; secondary clean class contributes the only category.
    assert categorize_minerals(_row(id1="uncertain", id2="olivine")) == "olivine (Low)"


def test_uncertain_in_id2_drops_to_moderate():
    assert categorize_minerals(_row(id1="hcp", id2="uncertain")) == "hcp (Moderate)"


def test_no_minerals_recognized_returns_other_high():
    # Empty / unknown tokens → 'Other' at default High tier.
    assert categorize_minerals(_row(id1="bland")) == "Other (High)"


def test_all_empty_row_returns_other_high():
    assert categorize_minerals(_row()) == "Other (High)"


def test_denom_only_returns_other_high():
    # Lone denom (no contamination) categorizes as Other (High) — bland.
    assert categorize_minerals(_row(id1="denom")) == "Other (High)"


def test_felsic_forces_low_tier():
    # 'felsic' is in the minerals list AND triggers Low tier.
    assert categorize_minerals(_row(id1="felsic")) == "felsic (Low)"


def test_alteration_in_id2_drops_to_low():
    assert categorize_minerals(_row(id1="olivine", id2="alteration")) == "alteration + olivine (Low)"


def test_slope_substring_forces_low():
    assert categorize_minerals(_row(id1="red slope")) == "red slope (Low)"


def test_pm_prefix_stripped_in_output():
    # 'plagioclase' should appear without ± in the final Category.
    out = categorize_minerals(_row(id1="±plagioclase"))
    assert out == "plagioclase (Low)"


def test_nan_id_is_skipped():
    # pd.NA / None / NaN in a cell must not crash.
    row = _row(id1="hcp")
    row["Mineral ID 2"] = float("nan")
    assert categorize_minerals(row) == "hcp (High)"


def test_alteration_in_id1_stays_high():
    # 'alteration' in ID 1 does NOT drop tier (notebook rule:
    # alteration only forces Low when in non-ID1 cells).
    assert categorize_minerals(_row(id1="alteration")) == "alteration (High)"


def test_substring_triggers_dont_append():
    # 'felsic uncertain' is not an exact MINERALS token, so no category
    # is appended; both substrings still trigger Low.
    assert categorize_minerals(_row(id1="felsic uncertain")) == "Other (Low)"


# --- is_contaminated_denom ------------------------------------------------

def test_denom_alone_is_not_contaminated():
    assert is_contaminated_denom(_row(id1="denom")) is False


def test_denom_with_text_in_id2_is_contaminated():
    assert is_contaminated_denom(_row(id1="denom", id2="probably has olivine")) is True


def test_denom_with_pm_in_id2_is_contaminated():
    assert is_contaminated_denom(_row(id1="denom", id2="±pyroxene")) is True


def test_denom_with_clean_class_in_id2_is_contaminated():
    # Even a clean class secondary is considered contamination for a denom polygon.
    assert is_contaminated_denom(_row(id1="denom", id2="hcp")) is True


def test_non_denom_row_is_not_contaminated():
    assert is_contaminated_denom(_row(id1="hcp", id2="±olivine")) is False


def test_denom_case_insensitive():
    assert is_contaminated_denom(_row(id1="DENOM", id2="hcp")) is True


def test_denom_with_whitespace_is_not_contaminated():
    # Whitespace-only secondary cells don't count as contamination.
    assert is_contaminated_denom(_row(id1="denom", id2="  ")) is False


def test_nan_id1_is_not_contaminated():
    # NaN in Mineral ID 1 short-circuits the denom check.
    assert is_contaminated_denom(_row(id1=float("nan"), id2="hcp")) is False


def test_denom_with_whitespace_around_token_still_matches():
    # Strip + lowercase on Mineral ID 1 means '  DENOM  ' is still treated as denom.
    assert is_contaminated_denom(_row(id1=" denom ", id2="hcp")) is True


# --- process_gpkg ---------------------------------------------------------

import geopandas as gpd
from shapely.geometry import Polygon

from scripts.categorize_sup_gpkg import process_gpkg


def _make_synthetic_gpkg(path: str) -> None:
    """Write a small synthetic GPKG that exercises every code path."""
    rows = [
        # idx 0: clean primary class → "hcp (High)"
        {"Mineral ID 1": "hcp",      "Mineral ID 2": "",          "Mineral ID 3": "", "Mineral ID 4": ""},
        # idx 1: ± in ID2 → Moderate
        {"Mineral ID 1": "hcp",      "Mineral ID 2": "±olivine",  "Mineral ID 3": "", "Mineral ID 4": ""},
        # idx 2: clean denom → "Other (High)" (kept)
        {"Mineral ID 1": "denom",    "Mineral ID 2": "",          "Mineral ID 3": "", "Mineral ID 4": ""},
        # idx 3: contaminated denom → DROPPED
        {"Mineral ID 1": "denom",    "Mineral ID 2": "probably has olivine", "Mineral ID 3": "", "Mineral ID 4": ""},
        # idx 4: uncertain alone → "Other (Low)" (kept)
        {"Mineral ID 1": "uncertain","Mineral ID 2": "",          "Mineral ID 3": "", "Mineral ID 4": ""},
    ]
    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(len(rows))]
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(path, driver="GPKG")


def test_process_gpkg_writes_category_and_drops_contaminated(tmp_path):
    src = tmp_path / "T9999.gpkg"
    dst = tmp_path / "out" / "T9999.gpkg"
    dst.parent.mkdir()

    _make_synthetic_gpkg(str(src))
    stats = process_gpkg(str(src), str(dst))

    assert stats["rows_in"] == 5
    assert stats["rows_out"] == 4
    assert stats["contaminated_dropped"] == 1

    out = gpd.read_file(str(dst))
    assert "Category" in out.columns
    assert sorted(out["Category"].tolist()) == sorted([
        "hcp (High)",
        "hcp + olivine (Moderate)",
        "Other (High)",     # clean denom
        "Other (Low)",      # uncertain alone
    ])


def test_process_gpkg_preserves_existing_columns(tmp_path):
    src = tmp_path / "T9998.gpkg"
    dst = tmp_path / "out" / "T9998.gpkg"
    dst.parent.mkdir()

    _make_synthetic_gpkg(str(src))
    process_gpkg(str(src), str(dst))

    out = gpd.read_file(str(dst))
    for col in ("Mineral ID 1", "Mineral ID 2", "Mineral ID 3", "Mineral ID 4", "geometry"):
        assert col in out.columns


# --- find_conflicts -------------------------------------------------------

from scripts.categorize_sup_gpkg import find_conflicts


def test_find_conflicts_empty_when_no_overlap(tmp_path):
    src_dir = tmp_path / "in"; src_dir.mkdir()
    dst_dir = tmp_path / "out"; dst_dir.mkdir()
    (src_dir / "A.gpkg").touch()
    (src_dir / "B.gpkg").touch()

    assert find_conflicts(str(src_dir), str(dst_dir)) == []


def test_find_conflicts_returns_collisions(tmp_path):
    src_dir = tmp_path / "in"; src_dir.mkdir()
    dst_dir = tmp_path / "out"; dst_dir.mkdir()
    (src_dir / "A.gpkg").touch()
    (src_dir / "B.gpkg").touch()
    (dst_dir / "A.gpkg").touch()   # collides

    conflicts = find_conflicts(str(src_dir), str(dst_dir))
    assert conflicts == [("A.gpkg", str(dst_dir / "A.gpkg"))]


def test_find_conflicts_ignores_non_gpkg(tmp_path):
    src_dir = tmp_path / "in"; src_dir.mkdir()
    dst_dir = tmp_path / "out"; dst_dir.mkdir()
    (src_dir / "A.gpkg").touch()
    (src_dir / "notes.txt").touch()  # not a gpkg
    (dst_dir / "notes.txt").touch()  # collides but irrelevant

    assert find_conflicts(str(src_dir), str(dst_dir)) == []


# --- verify_categories_parsable -------------------------------------------

from scripts.categorize_sup_gpkg import verify_categories_parsable


def test_verify_categories_passes_for_clean_output(tmp_path):
    """An output produced by process_gpkg should always pass verification."""
    src = tmp_path / "T9997.gpkg"
    dst = tmp_path / "out" / "T9997.gpkg"
    dst.parent.mkdir()
    _make_synthetic_gpkg(str(src))
    process_gpkg(str(src), str(dst))

    # Should not raise.
    verify_categories_parsable(str(dst))


def test_verify_categories_raises_on_unknown_token(tmp_path):
    """If somehow a row's Category produces an empty label parse, raise."""
    gpkg = tmp_path / "T9996.gpkg"
    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    # 'mystery_mineral' is not in the label_parser vocabulary → all-zero label.
    gdf = gpd.GeoDataFrame(
        [{"Category": "mystery_mineral (High)"}],
        geometry=geoms, crs="EPSG:4326",
    )
    gdf.to_file(str(gpkg), driver="GPKG")

    with pytest.raises(ValueError, match="unparseable Category"):
        verify_categories_parsable(str(gpkg))
