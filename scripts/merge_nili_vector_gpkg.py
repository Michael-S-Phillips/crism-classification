"""
Merge the 4 per-tile, multi-layer Nili Fossae vector GPKGs into a single
GeoPackage with one layer ('minerals') and a `mineral` + `tile_id` column
per polygon. Makes the v3 denoising vector product convenient to load as
a single dataset in QGIS.

Source: data/vector_v3_denoising/*_mineral_map.gpkg
Output: data/vector_v3_denoising/nili_v3_denoising_minerals.gpkg

Per-polygon columns:
  tile_id      str   — e.g. 't1249'
  mineral      str   — 'olivine' | 'lcp' | 'hcp' | 'plagioclase'
  confidence   int   — 1 / 2 / 3 (model tier)
  threshold    float — lower probability bound for this polygon's tier
  mean_prob, std_prob, min_prob, max_prob, median_prob  — zonal statistics
  count_px     int   — pixel count
  geometry     — polygon, reprojected to common Mars 2000 geographic CRS

Usage (no args):
    conda run -n crism python scripts/merge_nili_vector_gpkg.py
"""
from __future__ import annotations

import glob
import os
import sys

import fiona
import geopandas as gpd
import pandas as pd
from pyproj import CRS

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector_v3_denoising')
OUT_PATH = os.path.join(VECTOR_DIR, 'nili_v3_denoising_minerals.gpkg')

# Mars 2000 geographic (degrees lon/lat), matches plot_nili_seamless.py
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)


def tile_id_from_basename(path: str) -> str:
    """`/path/to/t1249_mrral_20n073_0327_4_mineral_map.gpkg` → `t1249`."""
    base = os.path.basename(path)
    # Strip suffix to get 't1249_mrral_20n073_0327_4'
    stem = base.replace('_mineral_map.gpkg', '')
    return stem.split('_mrral_')[0]


def main():
    paths = sorted(glob.glob(os.path.join(VECTOR_DIR, '*_mineral_map.gpkg')))
    paths = [p for p in paths if not p.endswith('nili_v3_denoising_minerals.gpkg')]
    if not paths:
        sys.exit(f'No per-tile GPKGs found under {VECTOR_DIR}')

    print(f'Merging {len(paths)} per-tile GPKG files:')

    frames = []
    for path in paths:
        tid = tile_id_from_basename(path)
        try:
            layers = fiona.listlayers(path)
        except Exception as e:
            print(f'  SKIP {path}: cannot list layers ({e})')
            continue
        for layer in layers:
            try:
                gdf = gpd.read_file(path, layer=layer)
            except Exception as e:
                print(f'  SKIP {path}::{layer}: {e}')
                continue
            if len(gdf) == 0:
                continue
            # Make sure mineral + tile_id columns exist
            if 'mineral' not in gdf.columns:
                gdf['mineral'] = layer
            gdf['tile_id'] = tid
            # Reproject to common CRS
            gdf = gdf.to_crs(COMMON_CRS)
            frames.append(gdf)
            print(f'  {tid}  layer={layer:<12}  rows={len(gdf):>5d}')

    if not frames:
        sys.exit('No polygons collected; nothing to write.')

    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=COMMON_CRS)

    # Column order — put identifiers first for QGIS attribute table readability
    preferred = ['tile_id', 'mineral', 'confidence', 'threshold',
                 'mean_prob', 'std_prob', 'min_prob', 'max_prob', 'median_prob',
                 'count_px']
    cols = [c for c in preferred if c in merged.columns] + \
           [c for c in merged.columns if c not in preferred and c != 'geometry'] + \
           ['geometry']
    merged = merged[cols]

    # Single layer, simple name
    merged.to_file(OUT_PATH, layer='minerals', driver='GPKG')
    print(f'\nWrote {len(merged):,} polygons → {OUT_PATH}')
    print(f'CRS: Mars 2000 geographic (degrees)')
    print(f'Layer: minerals  (single layer; filter by `mineral` and `tile_id`)')

    # Quick summary
    summary = (
        merged.groupby(['tile_id', 'mineral'])
              .size().unstack(fill_value=0)
    )
    print('\nPolygon counts:')
    print(summary.to_string())


if __name__ == '__main__':
    main()
