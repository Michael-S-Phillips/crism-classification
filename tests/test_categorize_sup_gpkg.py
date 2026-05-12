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
