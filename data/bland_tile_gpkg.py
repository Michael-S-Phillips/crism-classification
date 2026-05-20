"""
Author a single-polygon GeoPackage for a "bland" mrral tile.

Used by scripts/build_bland_other_gpkgs.py to create the 8 hand-picked
bland-tile labels (Category="Other (High)") that replace the existing
mineral-adjacent "Other" labels in the v3 classifier training set.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import geopandas as gpd
import rasterio
from shapely.geometry import box


def build_bland_gpkg_for_tile(mrral_path: str) -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame representing the bland-tile labeling.

    The polygon covers the full tile extent (bounding box) in the tile's
    native CRS. `extract_mrral_pixels_from_pair` filters nodata at the
    pixel-read step, so it's safe to over-cover here.

    Schema mirrors the existing /mnt/mrdr/categorized_mineral_units/T*.gpkg
    files so the build pipeline ingests it without special-casing.
    """
    with rasterio.open(mrral_path) as src:
        bounds = src.bounds
        crs = src.crs

    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    gdf = gpd.GeoDataFrame(
        {
            'Polygon Number': [0],
            'Color':           ['#aaaaaa'],
            'Number of Points': [None],
            'Denominator':     [None],
            'Template':        [None],
            'Mineral ID 1':    ['bland'],
            'Mineral ID 2':    [None],
            'Mineral ID 3':    [None],
            'Mineral ID 4':    [None],
            'wvl':             [None],
            'Spectrum Mean':   [None],
            'params':          [None],
            'Parameters Mean': [None],
            'Best Denom ID':   [None],
            'Ratio Spectrum':  [None],
            'Category':        ['Other (High)'],
        },
        geometry=[geom],
        crs=crs,
    )
    return gdf
