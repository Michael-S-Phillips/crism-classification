# tests/test_gpkg_io.py
import pytest
import geopandas as gpd
from shapely.geometry import box
from app.gpkg_io import load_all_polygons, ensure_verify_columns, write_verdict
from app.config import COL_VERDICT, COL_CONFIDENCE

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

@pytest.fixture
def tmp_gpkg(tmp_path):
    path = str(tmp_path / 'test.gpkg')
    for mineral in MINERALS:
        gdf = gpd.GeoDataFrame(
            {'mineral': [mineral, mineral],
             'confidence': [1, 2],
             'mean_prob': [0.4, 0.6],
             'count_px': [10, 20],
             'geometry': [box(0,0,1,1), box(1,0,2,1)]},
            crs='EPSG:4326',
        )
        gdf.to_file(path, layer=mineral, driver='GPKG')
    return path

def test_load_all_polygons_returns_list(tmp_gpkg):
    polys = load_all_polygons(tmp_gpkg)
    assert isinstance(polys, list)
    assert len(polys) == len(MINERALS) * 2

def test_load_all_polygons_has_poly_id(tmp_gpkg):
    polys = load_all_polygons(tmp_gpkg)
    ids = [p['poly_id'] for p in polys]
    assert ids == list(range(len(polys)))

def test_load_all_polygons_geojson_geometry(tmp_gpkg):
    polys = load_all_polygons(tmp_gpkg)
    assert all('geometry' in p for p in polys)

def test_ensure_verify_columns_adds_columns(tmp_gpkg):
    ensure_verify_columns(tmp_gpkg)
    gdf = gpd.read_file(tmp_gpkg, layer='olivine')
    assert COL_VERDICT in gdf.columns
    assert COL_CONFIDENCE in gdf.columns

def test_write_verdict_persists(tmp_gpkg):
    ensure_verify_columns(tmp_gpkg)
    polys = load_all_polygons(tmp_gpkg)
    poly = polys[0]  # poly_id=0, olivine, confidence=1 (first row in layer)
    write_verdict(tmp_gpkg, poly['poly_id'], polys,
                  verdict='correct', confidence='high', note='clear olivine')
    gdf = gpd.read_file(tmp_gpkg, layer=poly['mineral'])
    row = gdf.iloc[0]
    assert row[COL_VERDICT] == 'correct'
    assert row[COL_CONFIDENCE] == 'high'

def test_write_verdict_targets_correct_row(tmp_gpkg):
    """Verify second polygon in the layer is not overwritten."""
    ensure_verify_columns(tmp_gpkg)
    polys = load_all_polygons(tmp_gpkg)
    write_verdict(tmp_gpkg, polys[0]['poly_id'], polys,
                  verdict='correct', confidence='high', note='')
    gdf = gpd.read_file(tmp_gpkg, layer='olivine')
    assert gdf.iloc[1][COL_VERDICT] != 'correct'
