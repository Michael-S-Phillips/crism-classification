"""Walk the categorized gpkgs and materialise one row per polygon.

Output: data/endmember_curator/polygon_spectra.parquet

Schema:
    polygon_uid      : str  ("T0433/14" — globally unique)
    tile             : str  ("T0433")
    polygon_number   : str
    category         : str  (raw gpkg category)
    minerals         : list[str]  (parsed: olivine / lcp / hcp / plagioclase / other)
    confidence_tier  : str  ("High" | "Moderate" | "Low" | "")
    is_pure          : bool (exactly one mineral in `minerals` and not 'other')
    n_pixels         : int
    spectrum_mean    : list[float]  (length 59)
    wavelengths      : list[float]  (length 59)
    ratio_spectrum   : list[float] | null
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
GPKG_DIR = "/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units"
OUT_PARQUET = PROJ_ROOT / "data" / "endmember_curator" / "polygon_spectra.parquet"

MINERAL_KEYWORDS = {
    "olivine": ["olivine"],
    "lcp": ["lcp"],
    "hcp": ["hcp"],
    "plagioclase": ["plagioclase", "plagiolcase"],  # typo seen in some gpkgs
    "spinel": ["spinel"],
    "alteration": ["alteration", "alter"],
}

TIER_PATTERN = re.compile(r"\((high|moderate|low)\)", flags=re.IGNORECASE)


def parse_arr(s: object) -> np.ndarray:
    """Parse a gpkg comma/space-separated numeric blob."""
    if not isinstance(s, str):
        return np.array([], dtype=float)
    s = s.strip().strip("[]").replace("\n", " ")
    for sep in (",", " "):
        try:
            a = np.fromstring(s, sep=sep)
            if a.size > 5:
                return a.astype(float)
        except Exception:
            pass
    return np.array([], dtype=float)


def parse_category(cat: str) -> tuple[list[str], str]:
    """Return (minerals_present, confidence_tier_or_empty)."""
    if not isinstance(cat, str) or not cat:
        return [], ""
    c = cat.lower()
    mins: list[str] = []
    for key, kws in MINERAL_KEYWORDS.items():
        if any(k in c for k in kws):
            mins.append(key)
    if not mins and "other" in c:
        mins = ["other"]
    tier_m = TIER_PATTERN.search(cat)
    tier = tier_m.group(1).capitalize() if tier_m else ""
    return mins, tier


def is_pure(minerals: list[str]) -> bool:
    """Pure = exactly one mineral and not 'other'."""
    return len(minerals) == 1 and minerals[0] not in ("other", "alteration", "spinel")


def walk_gpkgs(gpkg_dir: str = GPKG_DIR) -> pd.DataFrame:
    rows: list[dict] = []
    paths = sorted(glob.glob(os.path.join(gpkg_dir, "T*.gpkg")))
    for gp in paths:
        tile = os.path.basename(gp)[:-5]
        try:
            g = gpd.read_file(gp)
        except Exception as e:
            print(f"  skip {tile}: {e}", file=sys.stderr)
            continue
        for _, r in g.iterrows():
            cat = r.get("Category", "") or ""
            mins, tier = parse_category(cat)
            wvl = parse_arr(r.get("wvl"))
            spec_mean = parse_arr(r.get("Spectrum Mean"))
            ratio = parse_arr(r.get("Ratio Spectrum"))
            # gpkgs store full 72-band spectra; project pipeline uses first 59.
            if spec_mean.size < 59 or wvl.size < 59:
                continue
            spec_mean = spec_mean[:59]
            wvl = wvl[:59]
            if ratio.size >= 59:
                ratio = ratio[:59]
            else:
                ratio = np.array([], dtype=float)
            poly_num = str(r.get("Polygon Number"))
            n_pix = r.get("Number of Points")
            try:
                n_pix = int(n_pix) if n_pix is not None else 0
            except Exception:
                n_pix = 0
            rows.append({
                "polygon_uid": f"{tile}/{poly_num}",
                "tile": tile,
                "polygon_number": poly_num,
                "category": cat,
                "minerals": mins,
                "confidence_tier": tier,
                "is_pure": is_pure(mins),
                "n_pixels": n_pix,
                "spectrum_mean": spec_mean.tolist(),
                "wavelengths": wvl.tolist(),
                "ratio_spectrum": ratio.tolist() if ratio.size == 59 else None,
            })
    return pd.DataFrame(rows)


def main():
    df = walk_gpkgs()
    print(f"Loaded {len(df):,} polygons from {df['tile'].nunique()} tiles")
    print("Counts by primary mineral:")
    pure_only = df[df["is_pure"]]
    if len(pure_only):
        print(pure_only["minerals"].astype(str).value_counts().to_string())
    print(f"\nConfidence tier distribution:")
    print(df["confidence_tier"].value_counts().to_string())
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
