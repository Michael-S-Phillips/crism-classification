import numpy as np
import pytest
import os
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import box
from data.extract_pixels import (
    find_tile_pairs, extract_pixels_from_pair, NODATA_VALUE
)

@pytest.fixture
def synthetic_tile(tmp_path):
    """Create a tiny synthetic 10x10 mrrsu raster with 3 bands."""
    img_path = tmp_path / "t0001_mrrsu_test.img"
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    crs = CRS.from_epsg(4326)
    data = np.random.rand(3, 10, 10).astype(np.float32)
    data[:, 0, 0] = NODATA_VALUE  # one nodata pixel at row=0, col=0
    with rasterio.open(
        str(img_path), 'w', driver='GTiff', height=10, width=10,
        count=3, dtype='float32', crs=crs, transform=transform
    ) as dst:
        dst.write(data)
    return str(img_path), data

@pytest.fixture
def synthetic_gpkg(tmp_path):
    """Create a geopackage with two polygons."""
    gdf = gpd.GeoDataFrame({
        'Category': ['lcp (High)', 'Type 1 olivine (Low)'],
        'geometry': [box(1, 1, 4, 4), box(6, 6, 9, 9)]
    }, crs='EPSG:4326')
    gpkg_path = tmp_path / "T0001.gpkg"
    gdf.to_file(str(gpkg_path), driver='GPKG')
    return str(gpkg_path)

def test_nodata_value():
    assert NODATA_VALUE == 65535

def test_extract_pixels_returns_records(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair(
        tile_id='t0001',
        mrrsu_path=img_path,
        gpkg_path=synthetic_gpkg,
        n_bands=3
    )
    assert len(records) > 0

def test_extract_pixels_schema(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    r = records[0]
    assert 'tile_id' in r
    assert 'polygon_id' in r
    assert 'pixel_row' in r
    assert 'pixel_col' in r
    assert 'b0' in r and 'b2' in r
    assert 'olivine_t1' in r
    assert 'confidence_weight' in r
    assert 'confidence_tier' in r

def test_nodata_pixels_excluded(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    pairs = [(r['pixel_row'], r['pixel_col']) for r in records]
    assert (0, 0) not in pairs

def test_confidence_weight_lcp_high(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    lcp_records = [r for r in records if r['lcp'] == 1.0]
    assert len(lcp_records) > 0
    for r in lcp_records:
        assert r['confidence_weight'] == 1.0
        assert r['confidence_tier'] == 'High'

def test_olivine_low_weight(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    olivine_records = [r for r in records if r['olivine_t1'] == 1.0]
    assert len(olivine_records) > 0
    for r in olivine_records:
        assert r['confidence_weight'] == 0.25
        assert r['confidence_tier'] == 'Low'

def test_find_tile_pairs_finds_existing():
    pairs = find_tile_pairs(
        gpkg_dir='/mnt/crism/MRDR/categorized_mineral_units',
        data_root='/mnt/crism/MRDR'
    )
    assert len(pairs) == 38
    t_id, gpkg_path, mrrsu_path = pairs[0]
    assert os.path.exists(gpkg_path)
    assert os.path.exists(mrrsu_path)
    assert t_id.startswith('t0')

def test_find_tile_pairs_sorted():
    pairs = find_tile_pairs('/mnt/crism/MRDR/categorized_mineral_units', '/mnt/crism/MRDR')
    ids = [p[0] for p in pairs]
    assert ids == sorted(ids)

def test_other_polygon_filtering(synthetic_tile, tmp_path):
    """Other polygons not in other_polygon_ids set should be excluded."""
    img_path, _ = synthetic_tile
    # Create gpkg with one 'Other' polygon
    gdf = gpd.GeoDataFrame({
        'Category': ['Other (High)'],
        'geometry': [box(1, 1, 4, 4)]
    }, crs='EPSG:4326')
    gpkg_path = str(tmp_path / "T9999.gpkg")
    gdf.to_file(gpkg_path, driver='GPKG')
    # With empty set, Other polygon should be excluded
    records = extract_pixels_from_pair('t9999', img_path, gpkg_path, n_bands=3, other_polygon_ids=set())
    assert len(records) == 0
    # With the index in the set, it should be included
    records = extract_pixels_from_pair('t9999', img_path, gpkg_path, n_bands=3, other_polygon_ids={0})
    assert len(records) > 0
