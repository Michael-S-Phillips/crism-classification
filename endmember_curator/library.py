"""Endmember library management — initialise, load, save, promote.

The library is a JSON dict mapping class name -> {
    "mean": [59 floats],
    "source": "labeled-xlsx:olivine-mean" | "polygon:T0433/14" | ...,
    "n_pixels": int,
    "promoted_at": ISO8601 timestamp,
}

A second JSONL log file records every user decision (correct/incorrect/promote) so
we can audit changes and replay history.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data" / "endmember_curator"
LIBRARY_PATH = DATA_DIR / "endmembers_current.json"
DECISIONS_PATH = DATA_DIR / "decisions.jsonl"
VERSIONS_DIR = DATA_DIR / "versions"
POLYGON_PARQUET = DATA_DIR / "polygon_spectra.parquet"

CLASSES = ("olivine", "lcp", "hcp", "plagioclase")


# ---- SAM ------------------------------------------------------------

def spectral_angle(target: np.ndarray, ref: np.ndarray) -> float:
    """Pure-numpy SAM (radians) between two equal-length spectra.

    Returns pi/2 ("no information") when either vector has zero magnitude.
    NaN bands in either input are ignored pairwise.
    """
    t = np.asarray(target, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {r.shape}")
    mask = np.isfinite(t) & np.isfinite(r)
    if mask.sum() < 3:
        return float("nan")
    tt = t[mask]
    rr = r[mask]
    nt = np.linalg.norm(tt)
    nr = np.linalg.norm(rr)
    if nt < 1e-12 or nr < 1e-12:
        return math.pi / 2
    cos = float(np.dot(tt, rr) / (nt * nr))
    cos = max(-1.0, min(1.0, cos))
    return math.acos(cos)


def angles_to_library(spectrum: Iterable[float], library: dict) -> dict[str, float]:
    """Return {class: SAM angle in radians} from a spectrum to every class endmember."""
    s = np.asarray(list(spectrum), dtype=np.float64)
    return {cls: spectral_angle(s, np.asarray(library[cls]["mean"])) for cls in CLASSES}


# ---- Library I/O ----------------------------------------------------

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def initial_library_from_xlsx() -> dict:
    """Bootstrap from the existing xlsx + computed plag endmember.

    Mirrors what sam_analysis.endmembers.load_endmember_library does, but writes the
    result as a JSON dict in the schema we use here.
    """
    from sam_analysis.endmembers import load_endmember_library
    lib_arrs = load_endmember_library()
    out = {}
    for cls in CLASSES:
        out[cls] = {
            "mean": lib_arrs[cls].astype(float).tolist(),
            "source": "bootstrap:xlsx+parquet" if cls != "plagioclase" else "bootstrap:parquet_train_high",
            "n_pixels": -1,
            "promoted_at": _now(),
        }
    return out


def load_library() -> dict:
    if LIBRARY_PATH.exists():
        return json.loads(LIBRARY_PATH.read_text())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lib = initial_library_from_xlsx()
    save_library(lib, snapshot=True)
    return lib


def save_library(library: dict, *, snapshot: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(json.dumps(library, indent=2))
    if snapshot:
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (VERSIONS_DIR / f"endmembers_{stamp}.json").write_text(json.dumps(library, indent=2))


def promote(library: dict, cls: str, polygon_uid: str, spectrum: list[float], n_pixels: int) -> dict:
    """Replace class endmember with the given polygon's spectrum. Snapshots prior version."""
    if cls not in CLASSES:
        raise ValueError(f"unknown class {cls}")
    library = dict(library)
    library[cls] = {
        "mean": list(map(float, spectrum)),
        "source": f"polygon:{polygon_uid}",
        "n_pixels": int(n_pixels),
        "promoted_at": _now(),
    }
    save_library(library, snapshot=True)
    return library


# ---- Decisions log --------------------------------------------------

def log_decision(record: dict) -> None:
    """Append one decision row to the JSONL log."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {**record, "logged_at": _now()}
    with DECISIONS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_decisions() -> pd.DataFrame:
    if not DECISIONS_PATH.exists():
        return pd.DataFrame(columns=["polygon_uid", "decision", "class", "notes", "logged_at"])
    rows = [json.loads(line) for line in DECISIONS_PATH.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


# ---- Polygon spectra ------------------------------------------------

def load_polygon_spectra() -> pd.DataFrame:
    if not POLYGON_PARQUET.exists():
        raise FileNotFoundError(
            f"{POLYGON_PARQUET} not found. Run: "
            f"PYTHONPATH={PROJ_ROOT} python -m endmember_curator.precompute"
        )
    return pd.read_parquet(POLYGON_PARQUET)
