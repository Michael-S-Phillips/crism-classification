"""Find MTRDR scenes that spatially intersect the Argyre MRDR tiles.

Walks `/mnt/mrdr/categorized_mineral_units/FeldsReview/**` for `*_07_if*j_mtr3.img`
files, opens their `.hdr`/raster to read corner coordinates, and intersects
each MTRDR footprint with each Argyre tile footprint.

Writes a JSON mapping {tile_id: [matching_mtrdr_paths]} to
`sam_analysis/outputs/argyre/mtrdr_pairings.json`. The dict may be empty.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import math
import re

import rasterio
from shapely.geometry import box

# Mars 2000 IAU sphere parameters used by CRISM products. The "custom sphere"
# WKTs encode the std-parallel latitude in the spheroid name; we use a fixed
# canonical radius for the lonlat conversion since the difference between the
# per-WKT radii is < 0.3% and we only need a coarse intersection test.
MARS_R = 3396190.0  # IAU 2000 mean radius (m)

FELDSREVIEW_ROOT = "/mnt/mrdr/categorized_mineral_units/FeldsReview"
DEFAULT_TILES = {
    "t0434": "/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img",
    "t0435": "/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img",
}
OUTPUT_JSON = "sam_analysis/outputs/argyre/mtrdr_pairings.json"


def _equirect_to_lonlat(x: float, y: float, lat0_deg: float, lon0_deg: float) -> tuple[float, float]:
    """Inverse Mars equirectangular projection.

    x, y are in meters relative to (lon0_deg, 0); lat0_deg is the standard parallel.
    Uses a fixed sphere radius (MARS_R) — accurate to ~0.3% across the planet.
    """
    lat = math.degrees(y / MARS_R)
    lon = lon0_deg + math.degrees(x / (MARS_R * math.cos(math.radians(lat0_deg))))
    return lon, lat


def _parse_crs_centers(crs_wkt: str) -> tuple[float, float]:
    """Pull (standard_parallel_1, central_meridian) from a Mars Equirect WKT.

    Falls back to (0, 0) if the keys aren't found.
    """
    sp_match = re.search(r'PARAMETER\["standard_parallel_1",\s*(-?[0-9.]+)', crs_wkt)
    cm_match = re.search(r'PARAMETER\["central_meridian",\s*(-?[0-9.]+)', crs_wkt)
    sp = float(sp_match.group(1)) if sp_match else 0.0
    cm = float(cm_match.group(1)) if cm_match else 0.0
    return sp, cm


def _scene_bounds_lonlat(path: str):
    """Return (lon_left, lat_bottom, lon_right, lat_top) for a Mars Equirect raster.

    Uses a hand-rolled inverse projection (see _equirect_to_lonlat) so we don't
    rely on rasterio.warp.transform_bounds, which is flaky with custom Mars CRS.
    """
    with rasterio.open(path) as src:
        bnds = src.bounds
        crs = src.crs
        if crs is None:
            return tuple(bnds), None
    try:
        sp, cm = _parse_crs_centers(crs.to_wkt())
        lon1, lat1 = _equirect_to_lonlat(bnds.left, bnds.bottom, sp, cm)
        lon2, lat2 = _equirect_to_lonlat(bnds.right, bnds.top, sp, cm)
        return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)), crs.to_wkt()
    except Exception:
        return tuple(bnds), crs.to_wkt() if crs else None


def find_pairings(
    tile_paths: Dict[str, str],
    feldsreview_root: str = FELDSREVIEW_ROOT,
) -> Dict[str, List[str]]:
    """For each tile, return list of MTRDR .img paths whose footprint intersects it."""
    mtrdrs = sorted(
        glob.glob(os.path.join(feldsreview_root, "**", "*_07_if*j_mtr3.img"),
                  recursive=True)
    )
    print(f"  Found {len(mtrdrs)} MTRDR scenes under {feldsreview_root}")

    # Pre-compute MTRDR footprints in lon/lat.
    mtrdr_polys = []
    for mp in mtrdrs:
        try:
            bnds, _ = _scene_bounds_lonlat(mp)
            mtrdr_polys.append((mp, box(*bnds)))
        except Exception as e:
            print(f"  WARN: could not read bounds for {mp}: {e}")

    out: Dict[str, List[str]] = {}
    for tid, tpath in tile_paths.items():
        try:
            bnds, _ = _scene_bounds_lonlat(tpath)
        except Exception as e:
            print(f"  WARN: could not read bounds for tile {tid}: {e}")
            out[tid] = []
            continue
        tile_poly = box(*bnds)
        matches = [mp for mp, mpoly in mtrdr_polys if tile_poly.intersects(mpoly)]
        out[tid] = matches
        print(f"  {tid}: {len(matches)} overlapping MTRDR scene(s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=list(DEFAULT_TILES.keys()))
    ap.add_argument("--feldsreview", default=FELDSREVIEW_ROOT)
    ap.add_argument("--output", default=OUTPUT_JSON)
    args = ap.parse_args()

    tile_paths = {t: DEFAULT_TILES[t] for t in args.tiles if t in DEFAULT_TILES}
    if not tile_paths:
        raise SystemExit(f"No known tile paths for {args.tiles}; known: {list(DEFAULT_TILES.keys())}")

    pairings = find_pairings(tile_paths, args.feldsreview)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairings, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps({k: len(v) for k, v in pairings.items()}, indent=2))


if __name__ == "__main__":
    main()
