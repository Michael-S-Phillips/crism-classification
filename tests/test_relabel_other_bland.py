"""Tests for the relabel-other-bland data pipeline change.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.bland_tile_gpkg import build_bland_gpkg_for_tile


def _make_fake_mrral_tile(path, height=20, width=20, n_bands=59, crs_wkt=None):
    """Write a tiny synthetic mrral .img + .hdr pair for testing."""
    if crs_wkt is None:
        # Mars 2000 equirectangular, central meridian 0
        crs_wkt = (
            'PROJCS["MRO Mars Equirectangular [IAU 2000] [0.00N; 0.00E]",'
            'GEOGCS["GCS_Mars_2000",DATUM["D_Mars_2000",'
            'SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
            'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Equirectangular"],PARAMETER["central_meridian",0],'
            'UNIT["metre",1]]'
        )
    transform = from_origin(0, 0, 200, 200)   # 200 m/pixel
    profile = {
        'driver': 'ENVI', 'dtype': 'float32', 'count': n_bands,
        'height': height, 'width': width, 'crs': crs_wkt, 'transform': transform,
    }
    data = np.random.uniform(0.0, 0.3, size=(n_bands, height, width)).astype(np.float32)
    with rasterio.open(path, 'w', **profile) as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)


class TestBlandTileGpkg:
    def test_produces_single_polygon_with_other_high_category(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))

        gdf = build_bland_gpkg_for_tile(str(mrral))

        assert len(gdf) == 1
        assert gdf.iloc[0]['Category'] == 'Other (High)'
        assert gdf.iloc[0]['Mineral ID 1'] == 'bland'

    def test_geometry_covers_tile_extent(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral), height=20, width=20)

        gdf = build_bland_gpkg_for_tile(str(mrral))

        with rasterio.open(mrral) as src:
            expected_bounds = src.bounds
        actual_bounds = gdf.total_bounds
        # Polygon should be the tile bounding box
        assert actual_bounds[0] == pytest.approx(expected_bounds.left)
        assert actual_bounds[1] == pytest.approx(expected_bounds.bottom)
        assert actual_bounds[2] == pytest.approx(expected_bounds.right)
        assert actual_bounds[3] == pytest.approx(expected_bounds.top)

    def test_crs_matches_source_tile(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))

        gdf = build_bland_gpkg_for_tile(str(mrral))

        with rasterio.open(mrral) as src:
            tile_crs = src.crs
        # CRS should round-trip identically (matters for downstream extract step)
        assert gdf.crs.to_wkt() == tile_crs.to_wkt() or gdf.crs == tile_crs
