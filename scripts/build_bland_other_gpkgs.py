"""
Author single-polygon GPKGs for the 8 hand-picked bland tiles.

Each output GPKG ('T<tile_num>.gpkg') has one row with Category="Other (High)"
and a polygon covering the source mrral tile's full extent.

Output dir: /Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/ (existing repository of
labeled-tile GPKGs the build pipeline consumes via find_mrral_pairs).

Idempotent: skips files that read back as valid bland GPKGs; overwrites
invalid/garbage files. Run any time the bland-tile set changes.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import argparse
import os
import sys

import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bland_tile_gpkg import build_bland_gpkg_for_tile


BLAND_TILES = [
    ('t1241', '/Volumes/Mars_GIS/CRISM/MRDR/mc12/t1241_mrral_20n033_0327_4.img'),
    ('t1242', '/Volumes/Mars_GIS/CRISM/MRDR/mc12/t1242_mrral_20n038_0327_4.img'),
    ('t1243', '/Volumes/Mars_GIS/CRISM/MRDR/mc12/t1243_mrral_20n043_0327_4.img'),
    ('t1280', '/Volumes/Mars_GIS/CRISM/MRDR/mc09/t1280_mrral_20n228_0327_4.img'),
    ('t1313', '/Volumes/Mars_GIS/CRISM/MRDR/mc12/t1313_mrral_25n033_0327_4.img'),
    ('t1314', '/Volumes/Mars_GIS/CRISM/MRDR/mc12/t1314_mrral_25n038_0327_4.img'),
    ('t1315', '/Volumes/Mars_GIS/CRISM/MRDR/mc12/t1315_mrral_25n043_0327_4.img'),
    ('t1336', '/Volumes/Mars_GIS/CRISM/MRDR/mc15/t1336_mrral_25n148_0327_4.img'),
]

DEFAULT_OUT_DIR = '/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units'


def _is_valid_bland_gpkg(path: str) -> bool:
    """Validate an existing file is a one-row bland GPKG."""
    try:
        gdf = gpd.read_file(path)
    except Exception:
        return False
    if len(gdf) != 1:
        return False
    if gdf.iloc[0].get('Category') != 'Other (High)':
        return False
    if gdf.iloc[0].get('Mineral ID 1') != 'bland':
        return False
    return True


def write_one_bland_gpkg(mrral_path: str, tile_id: str, out_dir: str) -> str:
    """Write the GPKG for one tile. Returns the output path."""
    out_path = os.path.join(out_dir, f'{tile_id.upper()}.gpkg')

    if os.path.exists(out_path) and _is_valid_bland_gpkg(out_path):
        return out_path

    gdf = build_bland_gpkg_for_tile(mrral_path)
    # Layer name matches the convention of existing T*.gpkg files
    gdf.to_file(out_path, layer=tile_id.upper(), driver='GPKG')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for tid, mrral in BLAND_TILES:
        if not os.path.exists(mrral):
            print(f'  SKIP {tid}: source tile not found at {mrral}')
            continue
        out = write_one_bland_gpkg(mrral, tid, args.out_dir)
        print(f'  {tid} -> {out}')
    print('Done.')


if __name__ == '__main__':
    main()
