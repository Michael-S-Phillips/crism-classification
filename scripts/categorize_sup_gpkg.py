# scripts/categorize_sup_gpkg.py
"""
Categorize raw supplementary GeoPackages so they match the schema of the
existing /mnt/mrdr/categorized_mineral_units/*.gpkg files.

Ports the `categorize_minerals` function from
/mnt/mrdr/categorize_gpkg_ratio_files.ipynb (the notebook used to produce
the original 40 categorized files), adds a contaminated-denom skip rule,
and writes outputs alongside the existing categorized files.

Usage:
    conda run -n crism python scripts/categorize_sup_gpkg.py
    conda run -n crism python scripts/categorize_sup_gpkg.py \\
        --input_dir /mnt/mrdr/categorized_mineral_units/sup \\
        --output_dir /mnt/mrdr/categorized_mineral_units
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Minerals the notebook recognises as appendable category tokens.
# Order is irrelevant; comparison is `value in MINERALS` after lowercasing/stripping.
MINERALS = {
    "olivine",
    "plagioclase",
    "lcp",
    "hcp",
    "red slope",
    "felsic",
    "alteration",
    "spinel",
}

# Mineral-ID column names used by both the sup files and the existing
# categorized files.
ID_COLS = ("Mineral ID 1", "Mineral ID 2", "Mineral ID 3", "Mineral ID 4")


def categorize_minerals(row) -> str:
    """
    Build a Category string ("hcp + olivine (Moderate)") from a row's
    Mineral ID 1-4 columns.

    Faithful port of the categorize_minerals function in
    /mnt/mrdr/categorize_gpkg_ratio_files.ipynb, with two safety guards added
    over the notebook source: NaN-before-.lower() (the notebook called
    row[col].lower() before its pd.isna check, which would crash on a NaN
    float; the guard is moved here to run first) and an explicit empty-string
    skip after stripping (the notebook would fall through with mineral="" and
    produce the same no-op, but the intent is now explicit). Neither guard
    changes output for any input the existing 40 categorized files were
    produced from.

    Behaviour summary:
      - Default tier = "High".
      - "±" in Mineral ID 1 → tier "Low"; in 2-4 → "Moderate" unless already Low.
      - "uncertain" in Mineral ID 1 → "Low"; in 2-4 → "Moderate" unless already Low.
      - "felsic" → "Low".
      - "alteration" in non-ID1 → "Low".
      - "slope" → "Low".
      - Each cell that matches a token in MINERALS (post-±-strip) is appended.
      - If no minerals matched → "Other (<tier>)".
      - Otherwise tokens are sorted, joined with " + ", suffixed with " (<tier>)".
    """
    categories: List[str] = []
    confidence = "High"

    for col in ID_COLS:
        mineral = row[col]
        # pandas NaN, None, or pd.NA → skip (matches notebook's `pd.isna` check).
        if pd.isna(mineral):
            continue
        mineral = str(mineral).lower()
        if mineral == "":
            continue

        if "±" in mineral:
            if col == "Mineral ID 1":
                confidence = "Low"
            else:
                if confidence != "Low":
                    confidence = "Moderate"
            mineral = mineral.replace("±", "").strip()

        if "uncertain" in mineral:
            if col == "Mineral ID 1":
                confidence = "Low"
            else:
                if confidence != "Low":
                    confidence = "Moderate"

        if "felsic" in mineral:
            confidence = "Low"
        if "alteration" in mineral:
            if col != "Mineral ID 1":
                confidence = "Low"
        if "slope" in mineral:
            confidence = "Low"

        if mineral in MINERALS:
            categories.append(mineral)

    if not categories:
        return f"Other ({confidence})"
    categories.sort()
    return f"{' + '.join(categories)} ({confidence})"


def is_contaminated_denom(row) -> bool:
    """
    Return True if Mineral ID 1 is exactly "denom" (case-insensitive, stripped)
    AND any of Mineral ID 2/3/4 has non-blank content.

    These rows are the contaminated denominator polygons — denoms that the
    annotator noted weren't fully bland (e.g., "probably has hcp", "±pyroxene").
    They should be dropped from the output entirely.
    """
    id1 = row.get("Mineral ID 1")
    if pd.isna(id1):
        return False
    if str(id1).strip().lower() != "denom":
        return False

    for col in ("Mineral ID 2", "Mineral ID 3", "Mineral ID 4"):
        val = row.get(col)
        if pd.isna(val):
            continue
        if str(val).strip() != "":
            return True
    return False
