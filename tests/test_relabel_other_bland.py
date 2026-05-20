"""Tests for the relabel-other-bland data pipeline change.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import importlib.util
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


def _load_module(path):
    spec = importlib.util.spec_from_file_location('m', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestBuildBlandOtherGpkgsScript:
    def test_writes_gpkg_for_one_tile(self, tmp_path, monkeypatch):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))
        out_dir = tmp_path / 'gpkgs'
        out_dir.mkdir()

        script_path = os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'build_bland_other_gpkgs.py')
        m = _load_module(script_path)

        # Call the writer function directly so the test doesn't need to mock argparse
        out_path = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        assert os.path.isfile(out_path)
        gdf = gpd.read_file(out_path)
        assert len(gdf) == 1
        assert gdf.iloc[0]['Category'] == 'Other (High)'
        assert gdf.iloc[0]['Mineral ID 1'] == 'bland'

    def test_idempotent_re_run_validates_existing(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))
        out_dir = tmp_path / 'gpkgs'
        out_dir.mkdir()

        script_path = os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'build_bland_other_gpkgs.py')
        m = _load_module(script_path)

        path_a = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        mtime_a = os.path.getmtime(path_a)

        # Re-run: should detect existing valid file and skip
        path_b = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        mtime_b = os.path.getmtime(path_b)
        assert path_a == path_b
        assert mtime_a == mtime_b   # not overwritten

    def test_overwrites_invalid_existing_file(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))
        out_dir = tmp_path / 'gpkgs'
        out_dir.mkdir()
        bad_path = out_dir / 'T9999.gpkg'
        bad_path.write_text('not a valid gpkg')

        script_path = os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'build_bland_other_gpkgs.py')
        m = _load_module(script_path)
        out_path = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        # Should have overwritten the bad file with a real GPKG
        gdf = gpd.read_file(out_path)
        assert len(gdf) == 1


class TestSubsampleBlandRows:
    def test_caps_each_bland_tile_to_sample_size(self):
        from scripts.build_mrral_dataset import (
            subsample_bland_other_rows, BLAND_TILES_ORDERED,
        )
        # 200K rows per bland tile + some non-bland rows
        rows = []
        for tid in BLAND_TILES_ORDERED:
            for r in range(200):
                rows.append({'tile_id': tid, 'pixel_row': r, 'pixel_col': 0,
                              'other': 1, 'olivine_t1': 0})
        rows.extend([
            {'tile_id': 't0435', 'pixel_row': r, 'pixel_col': 0,
             'other': 0, 'olivine_t1': 1}
            for r in range(50)
        ])
        df = pd.DataFrame(rows)
        out = subsample_bland_other_rows(df, sample_per_tile=100, seed=42)
        # Each bland tile reduced to 100 rows
        for tid in BLAND_TILES_ORDERED:
            n = int((out['tile_id'] == tid).sum())
            assert n == 100, f'{tid}: got {n}, expected 100'
        # Non-bland row count unchanged
        assert int((out['tile_id'] == 't0435').sum()) == 50

    def test_keeps_all_when_tile_has_fewer_than_sample(self):
        from scripts.build_mrral_dataset import subsample_bland_other_rows
        df = pd.DataFrame([
            {'tile_id': 't1241', 'pixel_row': r, 'pixel_col': 0,
             'other': 1, 'olivine_t1': 0}
            for r in range(50)
        ])
        out = subsample_bland_other_rows(df, sample_per_tile=100, seed=42)
        assert int((out['tile_id'] == 't1241').sum()) == 50

    def test_reproducible_seed(self):
        from scripts.build_mrral_dataset import subsample_bland_other_rows
        rows = [
            {'tile_id': 't1241', 'pixel_row': r, 'pixel_col': 0,
             'other': 1, 'olivine_t1': 0}
            for r in range(200)
        ]
        df = pd.DataFrame(rows)
        out_a = subsample_bland_other_rows(df.copy(), sample_per_tile=50, seed=42)
        out_b = subsample_bland_other_rows(df.copy(), sample_per_tile=50, seed=42)
        # Same rows in both runs (compare by pixel_row, since indices may differ)
        assert sorted(out_a['pixel_row'].tolist()) == sorted(out_b['pixel_row'].tolist())


class TestAssignBlandSplits:
    def test_assigns_70_15_15_split(self):
        from scripts.build_mrral_dataset import (
            assign_bland_tile_splits, BLAND_TILES_ORDERED,
        )
        # 10000 rows per bland tile (statistical test — need enough samples)
        rows = []
        for tid in BLAND_TILES_ORDERED:
            for r in range(10000):
                rows.append({'tile_id': tid, 'split': 'train'})
        df = pd.DataFrame(rows)
        out = assign_bland_tile_splits(df, seed=42)
        # Aggregate across all 8 tiles: 80000 rows total
        n = len(out)
        n_train = int((out['split'] == 'train').sum())
        n_val   = int((out['split'] == 'val').sum())
        n_test  = int((out['split'] == 'test').sum())
        # Allow ±2% tolerance per split (large samples)
        assert 0.68 * n <= n_train <= 0.72 * n, n_train / n
        assert 0.13 * n <= n_val   <= 0.17 * n, n_val / n
        assert 0.13 * n <= n_test  <= 0.17 * n, n_test / n

    def test_non_bland_rows_untouched(self):
        from scripts.build_mrral_dataset import assign_bland_tile_splits
        df = pd.DataFrame([
            {'tile_id': 't0435', 'split': 'val'},
            {'tile_id': 't1241', 'split': 'train'},
            {'tile_id': 't0886', 'split': 'test'},
        ])
        out = assign_bland_tile_splits(df, seed=42)
        # t0435 and t0886 are non-bland → keep original 'val'/'test'
        assert out.loc[out['tile_id'] == 't0435', 'split'].iloc[0] == 'val'
        assert out.loc[out['tile_id'] == 't0886', 'split'].iloc[0] == 'test'

    def test_reproducible_seed(self):
        from scripts.build_mrral_dataset import (
            assign_bland_tile_splits, BLAND_TILES_ORDERED,
        )
        rows = [
            {'tile_id': BLAND_TILES_ORDERED[0], 'split': 'train'}
            for _ in range(100)
        ]
        df = pd.DataFrame(rows)
        out_a = assign_bland_tile_splits(df.copy(), seed=42)
        out_b = assign_bland_tile_splits(df.copy(), seed=42)
        assert out_a['split'].tolist() == out_b['split'].tolist()
