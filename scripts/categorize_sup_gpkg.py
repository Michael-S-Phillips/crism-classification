# scripts/categorize_sup_gpkg.py
"""
Categorize raw supplementary GeoPackages so they match the schema of the
existing /Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/*.gpkg files.

Ports the `categorize_minerals` function from
/Volumes/Mars_GIS/CRISM/MRDR/categorize_gpkg_ratio_files.ipynb (the notebook used to produce
the original 40 categorized files), adds a contaminated-denom skip rule,
and writes outputs alongside the existing categorized files.

Usage:
    conda run -n crism python scripts/categorize_sup_gpkg.py
    conda run -n crism python scripts/categorize_sup_gpkg.py \\
        --input_dir /Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/sup \\
        --output_dir /Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.label_parser import _TOKEN_MAP  # noqa: E402, WPS437

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

# Compiled regex for stripping confidence parentheticals, e.g. "(High)".
_CONF_RE = re.compile(r"\(\w+\)")
# Tuple of all token keys known to the downstream label parser.
_KNOWN_TOKEN_KEYS = tuple(_TOKEN_MAP.keys())


def categorize_minerals(row) -> str:
    """
    Build a Category string ("hcp + olivine (Moderate)") from a row's
    Mineral ID 1-4 columns.

    Faithful port of the categorize_minerals function in
    /Volumes/Mars_GIS/CRISM/MRDR/categorize_gpkg_ratio_files.ipynb, with two safety guards added
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


def find_conflicts(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Return a list of (filename, existing_target_path) for every .gpkg in
    input_dir whose filename already exists in output_dir.

    Empty list means safe to write.
    """
    conflicts: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".gpkg"):
            continue
        target = os.path.join(output_dir, fname)
        if os.path.exists(target):
            conflicts.append((fname, target))
    return conflicts


def verify_categories_parsable(gpkg_path: str) -> None:
    """
    Open a written GeoPackage and confirm every Category string is a value
    the downstream pipeline understands — every "+"-separated token (after
    stripping the (Tier) suffix) must contain at least one substring from
    the parser's _TOKEN_MAP keys.

    The parser uses substring matching, so e.g. "hcp" matches the "hcp"
    key. Known non-target tokens ("alteration", "red slope", "spinel",
    "pyroxene") are also accepted — they're in _TOKEN_MAP with empty
    contributions and are deliberate "ignore me" markers.

    Raises ValueError on the first row with an unknown token.
    """
    gdf = gpd.read_file(gpkg_path)
    if "Category" not in gdf.columns:
        raise ValueError(f"{gpkg_path}: no Category column present")

    for idx, cat in enumerate(gdf["Category"].tolist()):
        if cat is None or (isinstance(cat, float) and np.isnan(cat)) or cat == "":
            raise ValueError(f"{gpkg_path}: row {idx} has empty Category")
        mineral_part = _CONF_RE.sub("", str(cat)).strip().lower()
        for tok in (t.strip() for t in mineral_part.split("+")):
            if not tok:
                continue
            if not any(key in tok for key in _KNOWN_TOKEN_KEYS):
                raise ValueError(
                    f"{gpkg_path}: row {idx} has unparseable Category {cat!r} "
                    f"— token {tok!r} not in label_parser vocabulary"
                )


def process_gpkg(input_path: str, output_path: str) -> dict:
    """
    Read a sup-style GeoPackage, drop contaminated-denom rows, synthesise the
    Category column with categorize_minerals, write the result.

    Returns a stats dict:
      {'rows_in': int, 'rows_out': int, 'contaminated_dropped': int}
    """
    gdf = gpd.read_file(input_path)
    rows_in = len(gdf)

    # Identify and drop contaminated denoms.
    contaminated_mask = gdf.apply(is_contaminated_denom, axis=1)
    n_contaminated = int(contaminated_mask.sum())
    gdf = gdf[~contaminated_mask].copy()

    # Apply the categorization rule to every surviving row.
    gdf["Category"] = gdf.apply(categorize_minerals, axis=1)

    # geopandas.to_file refuses to overwrite an existing GPKG cleanly; ensure
    # the parent dir exists. The caller is responsible for conflict detection
    # before reaching this function.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")

    return {
        "rows_in": rows_in,
        "rows_out": len(gdf),
        "contaminated_dropped": n_contaminated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize raw supplementary GPKG files (no Category column) "
            "into the schema used by the existing categorized_mineral_units/ files."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/sup",
        help="Directory containing raw sup .gpkg files (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units",
        help="Directory to write categorized .gpkg files (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error("Input dir does not exist: %s", args.input_dir)
        return 2
    if not os.path.isdir(args.output_dir):
        logger.error("Output dir does not exist: %s", args.output_dir)
        return 2

    # 1. Pre-flight conflict check. Abort before doing any work if any
    #    target filename already exists.
    conflicts = find_conflicts(args.input_dir, args.output_dir)
    if conflicts:
        logger.error(
            "Aborting: %d output filename(s) already exist in %s. "
            "Resolve manually (different files mean different annotation sessions; "
            "no safe auto-merge is possible).",
            len(conflicts), args.output_dir,
        )
        for fname, target in conflicts:
            logger.error("  %s already exists at %s", fname, target)
        return 1

    # 2. Process each input file.
    inputs = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".gpkg")
    )
    if not inputs:
        logger.warning("No .gpkg files found in %s — nothing to do.", args.input_dir)
        return 0

    grand_in = grand_out = grand_drop = 0
    for src in inputs:
        fname = os.path.basename(src)
        dst = os.path.join(args.output_dir, fname)
        stats = process_gpkg(src, dst)
        verify_categories_parsable(dst)
        grand_in += stats["rows_in"]
        grand_out += stats["rows_out"]
        grand_drop += stats["contaminated_dropped"]
        logger.info(
            "%s: %d rows → %d categorized, %d contaminated denoms skipped",
            fname, stats["rows_in"], stats["rows_out"], stats["contaminated_dropped"],
        )

    logger.info(
        "Done. %d files processed, %d total rows in, %d out, %d contaminated denoms dropped.",
        len(inputs), grand_in, grand_out, grand_drop,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
