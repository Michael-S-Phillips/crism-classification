# Supplementary GPKG Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert 10 new GeoPackage files in `/mnt/mrdr/categorized_mineral_units/sup/` into the `Category`-column schema the existing training pipeline already consumes, drop 5 contaminated denoms, then refresh the labeled-pixel parquets and patch caches so subsequent classifier training picks up the new tiles.

**Architecture:** One new standalone Python script (`scripts/categorize_sup_gpkg.py`) ports the `categorize_minerals` function from `/mnt/mrdr/categorize_gpkg_ratio_files.ipynb`, adds a contaminated-denom skip rule, pre-flights conflicts against the target directory, and writes new `.gpkg` files alongside the existing 40. Downstream uses the unchanged `scripts/build_dataset.py` → `scripts/build_mrral_dataset.py` → `scripts/cache_mrral_patches.py` pipeline. MAE checkpoint is **not** retrained.

**Tech Stack:** Python 3, geopandas, pandas, pytest, conda env `crism`.

**Conventions:**
- All Python commands run in the `crism` conda env: `conda run -n crism python …` (or activate the env directly).
- Working directory for all commands: `/mnt/mrdr/crism_classification`.
- Repo root is its own git repo (`/mnt/mrdr/crism_classification/.git`).
- Commit message prefixes follow recent history: `feat:`, `fix:`, `test:`, `data:`.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `scripts/categorize_sup_gpkg.py` | Create | Categorize sup/ GPKGs, write to main dir, pre-flight conflicts, verify outputs |
| `tests/test_categorize_sup_gpkg.py` | Create | Unit tests for `categorize_minerals`, `is_contaminated_denom`, and end-to-end `process_gpkg` against a synthetic GeoPackage |

No other source files are touched. The downstream pipeline (`scripts/build_dataset.py`, `scripts/build_mrral_dataset.py`, `scripts/cache_mrral_patches.py`, `data/extract_pixels.py`, `data/label_parser.py`) is run as-is — no edits.

---

## Reference: categorization rules (from the spec)

For each row of a sup `.gpkg`:

1. **Contaminated-denom skip (NEW):** if `Mineral ID 1` (stripped, lowercased) `== "denom"` AND any of `Mineral ID 2/3/4` is non-blank → drop the row from the output entirely.
2. Otherwise, run `categorize_minerals(row)` (verbatim port from the notebook):
   - Iterate `Mineral ID 1` → `Mineral ID 4`.
   - Tier starts at `"High"`.
   - `±` in cell → strip `±`; if in `Mineral ID 1` → tier `"Low"`; else tier `"Moderate"` (unless already `"Low"`).
   - `"uncertain"` substring in cell → same tier rule as `±`.
   - `"felsic"` substring → tier `"Low"`.
   - `"alteration"` substring in non-ID1 cell → tier `"Low"`.
   - `"slope"` substring → tier `"Low"`.
   - If the (post-strip) cell value equals one of `{olivine, plagioclase, lcp, hcp, red slope, felsic, alteration, spinel}`, append it to the category list.
   - After all four cells: sort categories alphabetically, join with `" + "`, suffix `" (<tier>)"`.
   - If no categories collected → return `"Other (<tier>)"`.

The resulting `Category` strings are a subset of the existing parser's vocabulary (verified by the post-write parity check in Task 7).

---

## Chunk 1: TDD the pure functions

### Task 1: Test scaffolding + tests for `categorize_minerals`

**Files:**
- Create: `tests/test_categorize_sup_gpkg.py`

- [ ] **Step 1: Write the failing test file**

```python
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
```

- [ ] **Step 2: Run the test file and verify it fails on import**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.categorize_sup_gpkg'` (or `ImportError: cannot import name 'categorize_minerals'`). This is the failing state we want before implementing.

---

### Task 2: Implement `categorize_minerals` and `is_contaminated_denom`

**Files:**
- Create: `scripts/categorize_sup_gpkg.py`

- [ ] **Step 1: Create the script skeleton with the two pure functions**

```python
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

    Verbatim port of the categorize_minerals function in
    /mnt/mrdr/categorize_gpkg_ratio_files.ipynb. Behaviour summary:
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
```

- [ ] **Step 2: Run the unit tests and verify they pass**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py -v
```

Expected: all 23 tests pass (16 for `categorize_minerals`, 7 for `is_contaminated_denom`). `scripts/` works as a Python 3 namespace package — no `__init__.py` needed. Pytest finds the project root via `tests/__init__.py` (already present) and adds it to `sys.path`, which is how peer tests already import `from data.extract_pixels …`.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/categorize_sup_gpkg.py tests/test_categorize_sup_gpkg.py
git commit -m "feat: port categorize_minerals + add denom-contamination check"
```

---

## Chunk 2: File-level processing and conflict detection

### Task 3: Add `process_gpkg` + integration test

**Files:**
- Modify: `scripts/categorize_sup_gpkg.py` (append new function)
- Modify: `tests/test_categorize_sup_gpkg.py` (append new tests)

- [ ] **Step 1: Append failing integration tests**

Append to `tests/test_categorize_sup_gpkg.py`:

```python
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
```

- [ ] **Step 2: Run the new tests and verify they fail**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py::test_process_gpkg_writes_category_and_drops_contaminated -v
```

Expected: `ImportError: cannot import name 'process_gpkg' from 'scripts.categorize_sup_gpkg'`.

- [ ] **Step 3: Append `process_gpkg` to the script**

Append to `scripts/categorize_sup_gpkg.py`:

```python
import geopandas as gpd


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
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py -v
```

Expected: all tests pass (the new two plus the 23 from Task 2 = 25 total).

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/categorize_sup_gpkg.py tests/test_categorize_sup_gpkg.py
git commit -m "feat: add process_gpkg with contaminated-denom drop"
```

---

### Task 4: Pre-flight conflict detection

**Files:**
- Modify: `scripts/categorize_sup_gpkg.py`
- Modify: `tests/test_categorize_sup_gpkg.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_categorize_sup_gpkg.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py::test_find_conflicts_returns_collisions -v
```

Expected: `ImportError: cannot import name 'find_conflicts'`.

- [ ] **Step 3: Append `find_conflicts` to the script**

Append to `scripts/categorize_sup_gpkg.py`:

```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py -v
```

Expected: all tests pass (28 total).

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/categorize_sup_gpkg.py tests/test_categorize_sup_gpkg.py
git commit -m "feat: pre-flight conflict detection for categorize_sup_gpkg"
```

---

### Task 5: Post-write parity check against `parse_category`

**Files:**
- Modify: `scripts/categorize_sup_gpkg.py`
- Modify: `tests/test_categorize_sup_gpkg.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_categorize_sup_gpkg.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py::test_verify_categories_passes_for_clean_output -v
```

Expected: `ImportError: cannot import name 'verify_categories_parsable'`.

- [ ] **Step 3: Append `verify_categories_parsable` to the script**

Append to `scripts/categorize_sup_gpkg.py`:

```python
import re
import numpy as np

# Re-use the existing label parser to ensure every emitted Category token is
# in its vocabulary. We need the token vocabulary itself (not just the
# parse function) because the parser silently returns all-zero labels for
# both unknown tokens AND known non-target tokens like "red slope" — we
# can't distinguish those by parse output alone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.label_parser import _TOKEN_MAP  # noqa: E402, WPS437

_CONF_RE = re.compile(r"\(\w+\)")
_KNOWN_TOKEN_KEYS = tuple(_TOKEN_MAP.keys())


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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py -v
```

Expected: all tests pass (30 total).

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/categorize_sup_gpkg.py tests/test_categorize_sup_gpkg.py
git commit -m "feat: verify Categories parse cleanly via existing label_parser"
```

---

## Chunk 3: CLI driver + execution

### Task 6: Wire up `main()` with argparse

**Files:**
- Modify: `scripts/categorize_sup_gpkg.py`

- [ ] **Step 1: Append `main()` and the `__main__` guard**

Append to `scripts/categorize_sup_gpkg.py`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize raw supplementary GPKG files (no Category column) "
            "into the schema used by the existing categorized_mineral_units/ files."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="/mnt/mrdr/categorized_mineral_units/sup",
        help="Directory containing raw sup .gpkg files (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/mrdr/categorized_mineral_units",
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
```

- [ ] **Step 2: Run the full test file once more to confirm nothing regressed**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_categorize_sup_gpkg.py -v
```

Expected: all 30 tests pass.

- [ ] **Step 3: Smoke-test the CLI against a single file (no writes to the real output dir yet)**

```bash
cd /mnt/mrdr/crism_classification
mkdir -p /tmp/sup_smoke/in /tmp/sup_smoke/out
cp /mnt/mrdr/categorized_mineral_units/sup/T0822.gpkg /tmp/sup_smoke/in/
conda run -n crism python scripts/categorize_sup_gpkg.py \
    --input_dir /tmp/sup_smoke/in \
    --output_dir /tmp/sup_smoke/out
```

Expected log line (counts may vary slightly): `T0822.gpkg: 10 rows → 10 categorized, 0 contaminated denoms skipped`.

- [ ] **Step 4: Inspect the smoke output**

```bash
conda run -n crism python -c "
import geopandas as gpd
g = gpd.read_file('/tmp/sup_smoke/out/T0822.gpkg')
print('rows:', len(g))
print('columns include Category:', 'Category' in g.columns)
print(g['Category'].value_counts().to_dict())
"
```

Expected: prints a non-empty `Category` distribution; all values look like `"<token>(s) (High|Moderate|Low)"` or `"Other (High)"`.

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/categorize_sup_gpkg.py
git commit -m "feat: CLI entry point for categorize_sup_gpkg with pre-flight + verify"
```

---

### Task 7: Run the conversion on the real sup files

**Files:** none modified — this task executes the script against the real data.

- [ ] **Step 1: Sanity-check the pre-flight before any writes**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -c "
from scripts.categorize_sup_gpkg import find_conflicts
c = find_conflicts('/mnt/mrdr/categorized_mineral_units/sup',
                    '/mnt/mrdr/categorized_mineral_units')
print('conflicts:', c)
"
```

Expected: prints `conflicts: []`. If any conflicts appear, STOP — investigate manually (these would be files in the main directory with the same name as one in `sup/`, which the spec says should not happen for this batch).

- [ ] **Step 2: Record pre-state file counts for the audit log**

```bash
ls /mnt/mrdr/categorized_mineral_units/*.gpkg | wc -l
```

Record this number — expected: 40.

- [ ] **Step 3: Run the conversion**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/categorize_sup_gpkg.py
```

Expected console output (exact totals: 146 in, 141 out, 5 contaminated dropped):

```
T0573.gpkg: 30 rows → 30 categorized, 0 contaminated denoms skipped
T0608.gpkg: 10 rows → 10 categorized, 0 contaminated denoms skipped
T0644.gpkg: 20 rows → 17 categorized, 3 contaminated denoms skipped
T0645.gpkg: 16 rows → 15 categorized, 1 contaminated denoms skipped
T0682.gpkg: 14 rows → 14 categorized, 0 contaminated denoms skipped
T0685.gpkg: 12 rows → 12 categorized, 0 contaminated denoms skipped
T0818.gpkg: 19 rows → 18 categorized, 1 contaminated denoms skipped
T0822.gpkg: 10 rows → 10 categorized, 0 contaminated denoms skipped
T0886.gpkg:  9 rows →  9 categorized, 0 contaminated denoms skipped
T1020.gpkg:  6 rows →  6 categorized, 0 contaminated denoms skipped
Done. 10 files processed, 146 total rows in, 141 out, 5 contaminated denoms dropped.
```

- [ ] **Step 4: Verify post-state file count**

```bash
ls /mnt/mrdr/categorized_mineral_units/*.gpkg | wc -l
```

Expected: 50 (was 40, + 10 new tiles).

- [ ] **Step 5: Spot-check one converted file looks right**

```bash
conda run -n crism python -c "
import geopandas as gpd
g = gpd.read_file('/mnt/mrdr/categorized_mineral_units/T0644.gpkg')
print('rows:', len(g))
print('has Category:', 'Category' in g.columns)
print('category distribution:')
for k, v in g['Category'].value_counts().items():
    print(f'  {k}: {v}')
"
```

Expected: 17 rows, `Category` present, distribution looks plausible (a mix of `Other (High)`, single-mineral, and multi-mineral categories with various tiers).

- [ ] **Step 6: Confirm the originals in `sup/` are still there**

```bash
ls /mnt/mrdr/categorized_mineral_units/sup/*.gpkg | wc -l
```

Expected: 10 (the originals are untouched).

**No commit in this task** — the converted `.gpkg` files live outside the repo at `/mnt/mrdr/categorized_mineral_units/`, so there's nothing in the repo to add. The script's behavior is already committed in Task 6.

---

## Chunk 4: Pipeline refresh

> **Heads-up about splits:** `scripts/build_dataset.py` calls `assign_tile_splits` which shuffles all tile IDs with seed 42 before assigning train/val/test. Adding 10 new tiles changes the shuffle order, so existing tiles may move between splits. The val_mAP=0.7175 baseline was measured on the old split — direct comparisons to it after this refresh are not strict. Treat the post-refresh run as a new baseline. (This is consistent with the spec's "no new evaluation splits" stance — we are using the existing split *logic*, not preserving the previous split *assignments*.)

### Task 8: Regenerate `pixels.parquet` (mrrsu)

**Files:** none modified — this task runs an existing script.

- [ ] **Step 1: Run the mrrsu dataset build**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/build_dataset.py
```

Expected: completes without errors. Final log line reports `Saved <N> total pixels to /mnt/mrdr/crism_classification/data/pixels.parquet`. `N` should be larger than the pre-refresh count (the wiki recorded ~1.97M).

- [ ] **Step 2: Confirm the 10 new tile IDs are present**

```bash
conda run -n crism python -c "
import pandas as pd
df = pd.read_parquet('/mnt/mrdr/crism_classification/data/pixels.parquet')
new_tiles = ['t0573','t0608','t0644','t0645','t0682','t0685','t0818','t0822','t0886','t1020']
present = df['tile_id'].str.lower().unique()
missing = [t for t in new_tiles if t not in present]
print('missing:', missing)
print('per-new-tile pixel counts:')
sub = df[df['tile_id'].str.lower().isin(new_tiles)]
print(sub.groupby('tile_id').size().sort_index().to_dict())
"
```

Expected: `missing: []` and a non-zero pixel count for each new tile.

---

### Task 9: Regenerate `mrral_pixels.parquet`

**Files:** none modified.

- [ ] **Step 1: Run the mrral dataset build**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/build_mrral_dataset.py
```

Expected: completes without errors. Final log line: `Wrote <N> pixels to /mnt/mrdr/crism_classification/data/mrral_pixels.parquet`.

- [ ] **Step 2: Confirm new tiles appear with consistent splits**

```bash
conda run -n crism python -c "
import pandas as pd
df = pd.read_parquet('/mnt/mrdr/crism_classification/data/mrral_pixels.parquet')
new_tiles = ['t0573','t0608','t0644','t0645','t0682','t0685','t0818','t0822','t0886','t1020']
sub = df[df['tile_id'].str.lower().isin(new_tiles)]
print('new-tile pixel count:', len(sub))
print('per-tile split:')
print(sub.groupby('tile_id')['split'].first().sort_index().to_dict())
print('confidence-tier counts (new tiles only):')
print(sub['confidence_tier'].value_counts().to_dict())
print('class coverage (label > 0, new tiles only):')
for c in ['olivine_t1','olivine_t2','lcp','hcp','plagioclase','other']:
    print(f'  {c}: {(sub[c] > 0).sum()}')
"
```

Expected: each new tile is assigned exactly one split (train/val/test), confidence tiers include some Moderate and/or Low entries (from `±` and `uncertain` rows), and the `other` count is dominated by the 67 clean denoms.

---

### Task 10: Regenerate the patch caches

**Files:** none modified.

- [ ] **Step 1: Cache 7×7 patches for all splits**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/cache_mrral_patches.py --patch_size 7
```

Expected: completes and writes (or overwrites) `data/patch_cache/mrral_train_patches_p7.npy`, `mrral_val_patches_p7.npy`, `mrral_test_patches_p7.npy`. The script prints the shape of each output array.

- [ ] **Step 2: Verify new patch cache sizes**

```bash
conda run -n crism python -c "
import numpy as np
for split in ('train','val','test'):
    arr = np.load(f'/mnt/mrdr/crism_classification/data/patch_cache/mrral_{split}_patches_p7.npy', mmap_mode='r')
    print(f'{split}: shape={arr.shape}, dtype={arr.dtype}')
"
```

Expected: each split prints a 4-D shape ending in `7, 7, 59` and dtype `float32`. The total number of samples (sum of the first dims across splits) should exceed the pre-refresh count.

---

### Task 11: Final acceptance check

**Files:** none modified.

- [ ] **Step 1: Acceptance criteria summary**

Confirm by inspection — no commands here, just a checklist over what previous tasks produced:

- [x] 10 new files in `/mnt/mrdr/categorized_mineral_units/` (verified in Task 7 Step 4).
- [x] Each new file has a `Category` column whose every value parses through `parse_category` (verified by `verify_categories_parsable` in Task 6 / Task 7 Step 3).
- [x] 5 contaminated denoms dropped (verified by Task 7 Step 3 log totals).
- [x] `pixels.parquet` and `mrral_pixels.parquet` regenerated with new tile IDs present (Tasks 8 & 9).
- [x] Patch caches regenerated (Task 10).

Classifier retraining is **not** part of this plan — that's a follow-up the user runs at their discretion using the existing `training/train_torch.py` and `config/sweep_*.yaml` workflows, against the refreshed parquets/patch caches. The MAE checkpoint (`spatial_mae_128d_6l_best.pt`) is reused as-is — no retraining.
