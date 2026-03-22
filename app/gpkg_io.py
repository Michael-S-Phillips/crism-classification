"""Load and update mineral prediction GeoPackages."""
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd

from app.config import COL_CONFIDENCE, COL_NOTE, COL_TIMESTAMP, COL_VERDICT

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']


def load_all_polygons(gpkg_path: str) -> list[dict[str, Any]]:
    """Return flat list of polygon dicts with stable integer poly_id."""
    records = []
    poly_id = 0
    for mineral in MINERALS:
        try:
            gdf = gpd.read_file(gpkg_path, layer=mineral)
        except Exception:
            continue
        for _, row in gdf.iterrows():
            geom = row.geometry.__geo_interface__ if row.geometry else None
            records.append({
                'poly_id':     poly_id,
                'mineral':     mineral,
                'confidence':  int(row.get('confidence', 0)),
                'mean_prob':   float(row.get('mean_prob', 0.0)),
                'count_px':    int(row.get('count_px', 0)),
                'verdict':     row.get(COL_VERDICT, None),
                'verify_conf': row.get(COL_CONFIDENCE, None),
                'verify_note': row.get(COL_NOTE, None),
                'geometry':    geom,
            })
            poly_id += 1
    return records


def ensure_verify_columns(gpkg_path: str) -> None:
    """Add verification columns to every layer if they don't exist yet."""
    for mineral in MINERALS:
        try:
            gdf = gpd.read_file(gpkg_path, layer=mineral)
        except Exception:
            continue
        changed = False
        for col in (COL_VERDICT, COL_CONFIDENCE, COL_NOTE, COL_TIMESTAMP):
            if col not in gdf.columns:
                gdf[col] = None
                changed = True
        if changed:
            gdf.to_file(gpkg_path, layer=mineral, driver='GPKG')


def write_verdict(
    gpkg_path: str,
    poly_id: int,
    all_polys: list[dict],
    verdict: str,
    confidence: str,
    note: str = '',
) -> None:
    """Persist a verdict for a single polygon back to the GeoPackage."""
    meta = all_polys[poly_id]
    mineral = meta['mineral']

    gdf = gpd.read_file(gpkg_path, layer=mineral)

    # Identify per-mineral row index from the global poly_id
    mineral_polys = [p for p in all_polys if p['mineral'] == mineral]
    local_idx = next(
        i for i, p in enumerate(mineral_polys) if p['poly_id'] == poly_id
    )

    gdf.at[local_idx, COL_VERDICT]    = verdict
    gdf.at[local_idx, COL_CONFIDENCE] = confidence
    gdf.at[local_idx, COL_NOTE]       = note
    gdf.at[local_idx, COL_TIMESTAMP]  = datetime.now(timezone.utc).isoformat()

    gdf.to_file(gpkg_path, layer=mineral, driver='GPKG')
