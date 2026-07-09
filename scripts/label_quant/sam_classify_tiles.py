"""Component 4 — tiered SAM classification of test tiles.

Classifies CRISM mrral tiles against the 8 MEDOID endmembers derived by
``sam_endmembers.py`` (reports/label_quantification/endmembers.csv), using the
spectral angle over the 57-band window m2..m58 (raster bands 3..59, 1-indexed).

Conservative rule: a pixel is labelled mineral M only if
  (a) argmin over ALL 8 endmembers is M — bland_dust / bland_reject / junk act
      as competing background absorbers, so pixels that hug the background get
      NO mineral label; and
  (b) its angle to M is <= the layer threshold.
Output classes: olivine, lcp, hcp, plagioclase, alteration.

An angle ladder (deg, tighter = higher-confidence tier) produces one gpkg layer
per angle per mineral. Output mirrors the classifier floor-test vectorize
conventions (vectorize_per_mineral_thresholds_nili_6cls.py): one gpkg per
mineral, one layer per threshold, min-polygon-size speckle filter, and embedded
QGIS layer_styles shaded dark(loose)->bright(tight) from MINERAL_BASE_RGB.

CLI: --tiles --tile_dir --endmembers --out_dir --angles
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from affine import Affine
from pyproj import CRS
from shapely.geometry import shape as shapely_shape

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NODATA = 65535
# m_i lives in raster band (i+1), 1-indexed. m2..m58 -> bands 3..59.
WINDOW_BAND_START = 3   # 1-indexed
WINDOW_BAND_END = 59    # inclusive
N_WINDOW_BANDS = WINDOW_BAND_END - WINDOW_BAND_START + 1  # 57

# Output mineral classes (order = display / table order).
MINERAL_NAMES = ["olivine", "lcp", "hcp", "plagioclase", "alteration"]

# Default angle ladder (deg); loosest -> tightest.
DEFAULT_ANGLES = [3.0, 2.0, 1.5, 1.25, 1.0, 0.8, 0.6, 0.5]

MIN_PIXELS = 4  # kill single/near-single-pixel speckle

MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

# Per-mineral base colours (puce alteration) — copied from the floor-test
# vectorize script so QGIS styling matches.
MINERAL_BASE_RGB = {
    "olivine":     (1.00, 0.00, 0.00),
    "hcp":         (1.00, 0.00, 1.00),
    "lcp":         (0.00, 1.00, 1.00),
    "plagioclase": (1.00, 0.84, 0.00),
    "alteration":  (0.80, 0.53, 0.60),  # puce
}
MIN_SAT = 0.30
MAX_SAT = 1.00


# --------------------------------------------------------------------------- #
# Core SAM math
# --------------------------------------------------------------------------- #
def angles_deg(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Spectral angles in DEGREES between rows of X (N, B) and rows of E (K, B).

    Returns (N, K). Zero-norm rows yield 90 deg (they are masked out upstream).
    """
    X = np.asarray(X, dtype=float)
    E = np.asarray(E, dtype=float)
    xn = np.linalg.norm(X, axis=1, keepdims=True)
    en = np.linalg.norm(E, axis=1, keepdims=True)
    xn = np.where(xn == 0.0, 1.0, xn)
    en = np.where(en == 0.0, 1.0, en)
    cos = np.clip((X / xn) @ (E / en).T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_valid_mask(cube: np.ndarray) -> np.ndarray:
    """Valid = no 65535 in the window, not all-zero, all finite. cube (B,H,W)."""
    nod = np.any(cube == NODATA, axis=0)
    allzero = np.all(cube == 0, axis=0)
    nonfinite = np.any(~np.isfinite(cube), axis=0)
    return ~(nod | allzero | nonfinite)


def assign_pixels(cube: np.ndarray, E: np.ndarray):
    """Per-pixel nearest endmember over ALL endmembers.

    cube (B,H,W), E (K,B). Returns (argmin (H,W) int with invalid=-1,
    min_angle_deg (H,W) float, valid (H,W) bool).
    """
    Bn, H, W = cube.shape
    valid = compute_valid_mask(cube)
    flat = cube.reshape(Bn, -1).T.astype(float)  # (H*W, B)
    ang = angles_deg(flat, E)                     # (H*W, K)
    argmin = np.argmin(ang, axis=1)
    minang = ang[np.arange(ang.shape[0]), argmin]
    argmin = argmin.reshape(H, W)
    minang = minang.reshape(H, W)
    argmin = np.where(valid, argmin, -1)
    return argmin, minang, valid


def mineral_mask(argmin, minang, valid, class_idx: int, angle: float) -> np.ndarray:
    """Conservative pixel mask for one mineral at one angle threshold."""
    return valid & (argmin == class_idx) & (minang <= angle)


# --------------------------------------------------------------------------- #
# Endmembers / rasters
# --------------------------------------------------------------------------- #
def load_medoid_endmembers(path: str):
    """Load the 8 medoid endmembers. Returns (names list, E array (K, 57))."""
    df = pd.read_csv(path)
    med = df[df["kind"] == "medoid"].reset_index(drop=True)
    band_cols = [f"m{i}" for i in range(2, 59)]
    E = med[band_cols].to_numpy(dtype=float)
    names = list(med["class"])
    return names, E


def read_window(mrral_path: str):
    """Read the 57-band window (bands 3..59). Returns (cube (57,H,W), transform, crs)."""
    with rasterio.open(mrral_path) as src:
        bands = list(range(WINDOW_BAND_START, WINDOW_BAND_END + 1))
        cube = src.read(bands).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return cube, transform, crs


# --------------------------------------------------------------------------- #
# Polygonization
# --------------------------------------------------------------------------- #
def polygonize_mask(mask_bool, valid_mask, transform, crs, min_px=MIN_PIXELS,
                    reproject_to: CRS | None = None) -> gpd.GeoDataFrame:
    """Polygonize a boolean mask; drop polygons smaller than ``min_px`` pixels."""
    if not mask_bool.any():
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    polygons = []
    for geom, value in rasterio.features.shapes(
        mask_bool.astype(np.uint8), mask=valid_mask,
        transform=transform, connectivity=4,
    ):
        if int(value) == 1:
            polygons.append(shapely_shape(geom))
    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    pixel_area = abs(transform.a * transform.e)
    gdf["count_px"] = (gdf.geometry.area / pixel_area).round().astype(int)
    gdf = gdf[gdf["count_px"] >= min_px].copy()
    if len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    if reproject_to is not None:
        gdf = gdf.to_crs(reproject_to)
    return gdf.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# QGIS styling (copied from vectorize_per_mineral_thresholds_nili_6cls.py)
# --------------------------------------------------------------------------- #
def _layer_color_rgb255(mineral: str, layer_idx: int, n_layers: int):
    base = MINERAL_BASE_RGB[mineral]
    sat = MAX_SAT if n_layers <= 1 else \
        MIN_SAT + (MAX_SAT - MIN_SAT) * layer_idx / (n_layers - 1)
    return tuple(int(round(c * sat * 255)) for c in base)


_QML_TEMPLATE = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <renderer-v2 forceraster="0" type="singleSymbol" enableorderby="0">
    <symbols>
      <symbol alpha="1" type="fill" name="0" clip_to_extent="1">
        <layer pass="0" class="SimpleFill" enabled="1" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="{r},{g},{b},255"/>
            <Option name="outline_color" type="QString" value="35,35,35,0"/>
            <Option name="outline_style" type="QString" value="no"/>
            <Option name="outline_width" type="QString" value="0"/>
            <Option name="style" type="QString" value="solid"/>
            <Option name="joinstyle" type="QString" value="bevel"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""


def add_qgis_layer_styles(gpkg_path: str, layer_colors: dict):
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS layer_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                f_table_catalog TEXT(256), f_table_schema TEXT(256),
                f_table_name TEXT(256), f_geometry_column TEXT(256),
                styleName TEXT(30), styleQML TEXT, styleSLD TEXT,
                useAsDefault BOOLEAN, description TEXT, owner TEXT(30),
                ui TEXT(30), update_time TIMESTAMP DEFAULT (datetime('now')))
        """)
        cur.execute(
            "DELETE FROM layer_styles WHERE f_table_name IN ({})".format(
                ",".join("?" * len(layer_colors))),
            list(layer_colors.keys()))
        for layer_name, (r, g, b) in layer_colors.items():
            qml = _QML_TEMPLATE.format(r=r, g=g, b=b)
            cur.execute(
                """INSERT INTO layer_styles
                   (f_table_catalog, f_table_schema, f_table_name,
                    f_geometry_column, styleName, styleQML,
                    useAsDefault, description, owner)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("", "", layer_name, "geom", layer_name, qml, 1,
                 f"SAM angle ladder color for {layer_name}", ""))
        conn.commit()
    finally:
        conn.close()


def _afmt(angle: float) -> str:
    """3.0 -> '3.0', 1.25 -> '1.25', 0.5 -> '0.5' (matches spec layer names)."""
    return str(float(angle))


def _layer_name(angle: float) -> str:
    return f"ang_{_afmt(angle)}"


# --------------------------------------------------------------------------- #
# Tile processing
# --------------------------------------------------------------------------- #
def process_tile(mrral_path, tid, names, E, mineral_idx, angles):
    """Classify one tile. Returns (per_mineral_gdfs, minang_by_mineral).

    per_mineral_gdfs: {mineral: {angle: GeoDataFrame}}
    minang_by_mineral: {mineral: 1D array of min-angles for pixels whose argmin
                        is that mineral (for distribution diagnostics)}
    """
    cube, transform, crs = read_window(mrral_path)
    argmin, minang, valid = assign_pixels(cube, E)

    out, dist = {}, {}
    for mineral in MINERAL_NAMES:
        cidx = mineral_idx[mineral]
        sel = valid & (argmin == cidx)
        dist[mineral] = minang[sel]
        out[mineral] = {}
        for a in angles:
            mask = mineral_mask(argmin, minang, valid, cidx, a)
            gdf = polygonize_mask(mask, valid, transform, crs,
                                  min_px=MIN_PIXELS, reproject_to=COMMON_CRS)
            if len(gdf) > 0:
                gdf["tile_id"] = tid
                gdf["mineral"] = mineral
                gdf["angle_deg"] = a
                # mean angle per polygon via nearest-pixel sampling is overkill;
                # store the layer threshold as provenance.
                gdf = gdf[["tile_id", "mineral", "angle_deg", "count_px", "geometry"]]
            out[mineral][a] = gdf
    return out, dist


def discover_tiles(tile_dir, tiles):
    found = []
    for tid in tiles:
        hits = sorted(glob.glob(os.path.join(tile_dir, f"{tid}_mrral_*_0327_4.img")))
        if not hits:
            hits = sorted(glob.glob(os.path.join(tile_dir, f"{tid}_mrral_*.img")))
        if not hits:
            print(f"WARNING: no mrral img for {tid} in {tile_dir}")
            continue
        found.append((tid, hits[0]))
    return found


def run(tiles, tile_dir, endmembers_path, out_dir, angles):
    names, E = load_medoid_endmembers(endmembers_path)
    mineral_idx = {m: names.index(m) for m in MINERAL_NAMES}
    n_layers = len(angles)
    os.makedirs(out_dir, exist_ok=True)

    found = discover_tiles(tile_dir, tiles)
    if not found:
        raise SystemExit(f"No tiles found in {tile_dir} for {tiles}")

    # accumulate per mineral per angle across tiles + angle distributions
    accum = {m: {a: [] for a in angles} for m in MINERAL_NAMES}
    dist_accum = {m: [] for m in MINERAL_NAMES}

    for i, (tid, mrral) in enumerate(found, 1):
        print(f"[{i}/{len(found)}] classifying {tid} …")
        per_min, dist = process_tile(mrral, tid, names, E, mineral_idx, angles)
        for m in MINERAL_NAMES:
            dist_accum[m].append(dist[m])
            for a in angles:
                g = per_min[m][a]
                if len(g) > 0:
                    accum[m][a].append(g)

    # write per-mineral gpkgs
    counts = {m: {a: 0 for a in angles} for m in MINERAL_NAMES}
    for m in MINERAL_NAMES:
        out_path = os.path.join(out_dir, f"{m}.gpkg")
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except PermissionError:
                open(out_path, "wb").close()
        layer_colors = {}
        for layer_idx, a in enumerate(angles):
            frames = accum[m][a]
            if not frames:
                continue
            merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                      geometry="geometry", crs=COMMON_CRS)
            lname = _layer_name(a)
            merged.to_file(out_path, layer=lname, driver="GPKG")
            counts[m][a] = len(merged)
            layer_colors[lname] = _layer_color_rgb255(m, layer_idx, n_layers)
        if layer_colors:
            add_qgis_layer_styles(out_path, layer_colors)

    dist_stats = {}
    for m in MINERAL_NAMES:
        allv = np.concatenate(dist_accum[m]) if dist_accum[m] else np.array([])
        allv = allv[np.isfinite(allv)]
        if allv.size:
            dist_stats[m] = {
                "n_pixels": int(allv.size),
                "p10_deg": float(np.percentile(allv, 10)),
                "p50_deg": float(np.percentile(allv, 50)),
            }
        else:
            dist_stats[m] = {"n_pixels": 0, "p10_deg": float("nan"),
                             "p50_deg": float("nan")}
    return counts, dist_stats


def count_table_str(counts, angles, title):
    lines = [f"### {title}", "",
             "| mineral | " + " | ".join(f"ang_{_afmt(a)}" for a in angles) + " |",
             "| --- |" + " --- |" * len(angles)]
    for m in MINERAL_NAMES:
        lines.append("| " + m + " | "
                     + " | ".join(str(counts[m][a]) for a in angles) + " |")
    return "\n".join(lines) + "\n"


def dist_table_str(dist_stats, title):
    lines = [f"### {title}", "",
             "| mineral | n_pixels (argmin=mineral) | p10 angle (deg) | p50 angle (deg) |",
             "| --- | --- | --- | --- |"]
    for m in MINERAL_NAMES:
        s = dist_stats[m]
        lines.append(f"| {m} | {s['n_pixels']} | {s['p10_deg']:.2f} | {s['p50_deg']:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tiered SAM classification of tiles")
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--tile_dir", required=True)
    ap.add_argument("--endmembers",
                    default="reports/label_quantification/endmembers.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--angles", nargs="+", type=float, default=DEFAULT_ANGLES)
    args = ap.parse_args(argv)

    counts, dist_stats = run(args.tiles, args.tile_dir, args.endmembers,
                             args.out_dir, args.angles)

    title = os.path.basename(args.out_dir.rstrip("/")) or "region"
    print("\nPer-mineral x angle polygon counts:")
    hdr = "  " + f"{'mineral':<14}" + "".join(f" {a:>6g}" for a in args.angles)
    print(hdr)
    for m in MINERAL_NAMES:
        print("  " + f"{m:<14}" + "".join(f" {counts[m][a]:>6}" for a in args.angles))

    return counts, dist_stats, title


if __name__ == "__main__":
    main()
