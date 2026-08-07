"""Endmember library loader for SAM analysis.

Loads four CRISM mineral classes as 59-band mean reflectance spectra:
- olivine      (union of Type1 + Type2 from xlsx if available, else parquet)
- lcp          (from xlsx)
- hcp          (from xlsx)
- plagioclase  (computed from data/mrral_pixels.parquet — xlsx lacks it)

Falls back to a parquet sibling of the xlsx if the openpyxl loader fails
(per the hard-rule in the task brief).
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

N_BANDS = 59

DEFAULT_XLSX = "/Volumes/Mars_GIS/CRISM/MRDR/endmember_extraction/crism_endmembers/crism_endmember_spectra.xlsx"
DEFAULT_PARQUET = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet"
XLSX_PARQUET_FALLBACK = (
    "/Volumes/Mars_GIS/CRISM/MRDR/endmember_extraction/crism_endmembers/crism_endmember_spectra.parquet"
)

# Map our class names to the column basename used in the xlsx 'Summary' sheet.
# 'olivine' is computed downstream as the mean of Type1 + Type2.
XLSX_COLUMNS = {
    "olivine_t1": "Type1_Olivine",
    "olivine_t2": "Type2_Olivine",
    "lcp": "LCP",
    "hcp": "HCP",
}


def _load_xlsx_summary(xlsx_path: str) -> pd.DataFrame:
    """Load the 'Summary' sheet from the endmember xlsx.

    Falls back to the parquet sibling if reading the xlsx fails (e.g. missing
    openpyxl). Raises FileNotFoundError if neither is available.
    """
    try:
        return pd.read_excel(xlsx_path, sheet_name="Summary")
    except ImportError as e:
        if os.path.exists(XLSX_PARQUET_FALLBACK):
            return pd.read_parquet(XLSX_PARQUET_FALLBACK)
        raise FileNotFoundError(
            f"Could not read {xlsx_path} ({e}) and fallback parquet not found "
            f"at {XLSX_PARQUET_FALLBACK}"
        ) from e
    except FileNotFoundError:
        if os.path.exists(XLSX_PARQUET_FALLBACK):
            return pd.read_parquet(XLSX_PARQUET_FALLBACK)
        raise


def _compute_plag_from_parquet(parquet_path: str) -> np.ndarray:
    """Compute plagioclase mean spectrum from high-confidence train pixels.

    Filters: split == 'train' AND plagioclase >= 0.7 AND confidence_tier == 'High'.
    Returns: (59,) float64 mean of m0..m58.
    """
    band_cols = [f"m{i}" for i in range(N_BANDS)]
    # Read only the columns we need to keep memory low.
    df = pd.read_parquet(
        parquet_path,
        columns=["split", "plagioclase", "confidence_tier"] + band_cols,
    )
    plag = df[
        (df["split"] == "train")
        & (df["plagioclase"] >= 0.7)
        & (df["confidence_tier"] == "High")
    ]
    if len(plag) == 0:
        raise RuntimeError(
            f"No plagioclase high-confidence train pixels found in {parquet_path}"
        )
    spec = plag[band_cols].mean(axis=0).to_numpy(dtype=np.float64)
    # The parquet values are clipped reflectance in [0, 0.5]; sanity check.
    if not np.isfinite(spec).all():
        raise RuntimeError(
            "NaN encountered in plagioclase mean spectrum — bad parquet values?"
        )
    return spec


def _olivine_from_parquet(parquet_path: str) -> np.ndarray:
    """Fallback olivine spectrum from parquet when xlsx-derived means are bad.

    Filters: split == 'train' AND (olivine_t1 + olivine_t2) >= 0.7 AND
             confidence_tier == 'High'.
    Returns: (59,) float64 mean of m0..m58.
    """
    band_cols = [f"m{i}" for i in range(N_BANDS)]
    df = pd.read_parquet(
        parquet_path,
        columns=["split", "olivine_t1", "olivine_t2", "confidence_tier"] + band_cols,
    )
    oli = df[
        (df["split"] == "train")
        & ((df["olivine_t1"] + df["olivine_t2"]) >= 0.7)
        & (df["confidence_tier"] == "High")
    ]
    if len(oli) == 0:
        raise RuntimeError(
            f"No olivine high-confidence train pixels found in {parquet_path}"
        )
    return oli[band_cols].mean(axis=0).to_numpy(dtype=np.float64)


def _column_mean(summary_df: pd.DataFrame, base: str) -> np.ndarray:
    """Extract the *_Mean column for a given basename from the Summary sheet."""
    col = f"{base}_Mean"
    if col not in summary_df.columns:
        raise KeyError(f"Column {col} not in summary sheet: {summary_df.columns.tolist()}")
    arr = summary_df[col].to_numpy(dtype=np.float64)
    if arr.shape[0] < N_BANDS:
        raise ValueError(
            f"{col} has only {arr.shape[0]} rows; need >= {N_BANDS}"
        )
    return arr[:N_BANDS]


def _looks_invalid(spec: np.ndarray) -> bool:
    """Heuristic: the xlsx 'Summary' Mean columns sometimes contain wildly negative
    values (e.g. -52.97) — clearly contaminated by NoData. Treat these as invalid
    and trigger a parquet fallback.

    A real reflectance spectrum should fall in roughly [0, 0.5].
    """
    if not np.isfinite(spec).all():
        return True
    return bool((spec.min() < -1.0) or (spec.max() > 5.0))


def load_endmember_library(
    xlsx_path: str = DEFAULT_XLSX,
    parquet_path: str = DEFAULT_PARQUET,
) -> Dict[str, np.ndarray]:
    """Return dict mapping class name -> mean reflectance spectrum (59,) float64.

    Classes returned: 'olivine', 'lcp', 'hcp', 'plagioclase'.

    Strategy:
    - LCP, HCP: from xlsx Summary *_Mean columns. Fall back to parquet means
      (filtered by the corresponding label column >= 0.7, confidence_tier == 'High')
      if the xlsx column looks contaminated (e.g. -52.97 NoData mean).
    - olivine: mean of (Type1, Type2) xlsx means; same parquet fallback if invalid.
    - plagioclase: ALWAYS computed from parquet (xlsx does not include it).
    """
    summary = _load_xlsx_summary(xlsx_path)

    lib: Dict[str, np.ndarray] = {}

    # LCP / HCP — preferred from xlsx, fallback to parquet by label name.
    for cls, base in (("lcp", "LCP"), ("hcp", "HCP")):
        spec = _column_mean(summary, base)
        if _looks_invalid(spec):
            spec = _label_mean_from_parquet(parquet_path, cls)
        lib[cls] = spec

    # olivine — mean of Type1 + Type2 xlsx means, fallback to parquet (olivine_t1+t2 >= 0.7).
    t1 = _column_mean(summary, "Type1_Olivine")
    t2 = _column_mean(summary, "Type2_Olivine")
    olivine = 0.5 * (t1 + t2)
    if _looks_invalid(t1) or _looks_invalid(t2) or _looks_invalid(olivine):
        olivine = _olivine_from_parquet(parquet_path)
    lib["olivine"] = olivine

    # plagioclase — always from parquet.
    lib["plagioclase"] = _compute_plag_from_parquet(parquet_path)

    # Sanity-check all spectra have the right length and finite values.
    for name, spec in lib.items():
        if spec.shape != (N_BANDS,):
            raise RuntimeError(
                f"endmember '{name}' has shape {spec.shape}, expected ({N_BANDS},)"
            )
        if not np.isfinite(spec).all():
            raise RuntimeError(f"endmember '{name}' contains NaN/Inf")
    return lib


def _label_mean_from_parquet(parquet_path: str, label: str) -> np.ndarray:
    """Generic parquet-based mean for a single label column."""
    band_cols = [f"m{i}" for i in range(N_BANDS)]
    df = pd.read_parquet(
        parquet_path, columns=["split", label, "confidence_tier"] + band_cols
    )
    sel = df[
        (df["split"] == "train")
        & (df[label] >= 0.7)
        & (df["confidence_tier"] == "High")
    ]
    if len(sel) == 0:
        raise RuntimeError(f"No high-confidence train pixels for label '{label}'")
    return sel[band_cols].mean(axis=0).to_numpy(dtype=np.float64)
