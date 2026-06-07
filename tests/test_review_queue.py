import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from scripts.review.polygon_queue import PolygonItem, PolygonQueue

MARS_2000_WKT = 'PROJCS["Mars 2000 Equirect",GEOGCS["Mars 2000",DATUM["Mars 2000",SPHEROID["Mars 2000",3396190,169.8944472]],PRIMEM["Reference Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'


def _square(x, y, size, tile_id):
    """Build a 1-row GeoDataFrame holding one square polygon."""
    geom = Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
    return gpd.GeoDataFrame(
        {'tile_id': [tile_id], 'mineral': ['hcp'], 'threshold': [0.95]},
        geometry=[geom], crs=MARS_2000_WKT,
    )


def _write_layered_gpkg(path, polys_by_layer):
    """polys_by_layer: dict[layer_name] -> list of (x, y, size, tile_id)."""
    for layer, polys in polys_by_layer.items():
        if not polys:
            continue
        gdfs = [_square(*p) for p in polys]
        merged = pd.concat(gdfs, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=MARS_2000_WKT)
        merged.to_file(path, driver='GPKG', layer=layer)


def test_queue_walks_layers_high_to_low(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.85': [(0, 0, 100, 't0001')],
        'thresh_0.90': [(0, 0, 100, 't0002')],
        'thresh_0.95': [(0, 0, 100, 't0003')],
        'thresh_0.97': [(0, 0, 100, 't0004')],
    })
    q = PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')
    items = list(q)
    assert [i.layer for i in items] == ['thresh_0.97', 'thresh_0.95', 'thresh_0.90', 'thresh_0.85']
    assert [i.tile_id for i in items] == ['t0004', 't0003', 't0002', 't0001']
    # Each item's pred_prob is parsed from the layer name
    assert items[0].pred_prob == pytest.approx(0.97)
    assert items[1].pred_prob == pytest.approx(0.95)


def test_queue_sorts_by_area_within_layer(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.95': [
            (0, 0,   50, 't0001'),   # area 2,500
            (0, 0,  200, 't0002'),   # area 40,000  <- biggest, comes first
            (0, 0,  100, 't0003'),   # area 10,000
        ],
    })
    q = PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')
    items = list(q)
    assert [i.tile_id for i in items] == ['t0002', 't0003', 't0001']
    # area_m2 is computed and exposed
    assert items[0].area_m2 == pytest.approx(40000.0)


def test_polygon_uid_is_stable(tmp_path):
    """polygon_uid must use the file-order index (not the post-sort index)."""
    gpkg = tmp_path / 'hcp.gpkg'
    # File-order: smaller polygon first, bigger second. After area-sort,
    # the bigger one yields first — but its uid must reflect file order, not
    # iteration order.
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.95': [(0, 0, 50, 't0001'), (10, 10, 200, 't0002')],
    })
    items = list(PolygonQueue(gpkg_path=str(gpkg), mineral='hcp'))
    # The bigger polygon (t0002) yields first
    assert items[0].tile_id == 't0002'
    # but its uid carries index 1 (it was written second)
    assert items[0].polygon_uid.endswith('::1')
    # the smaller (file-order-first) polygon yields second with index 0
    assert items[1].polygon_uid.endswith('::0')
    # And the whole sequence is stable across re-instantiation
    uids_run2 = [i.polygon_uid for i in PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')]
    assert [i.polygon_uid for i in items] == uids_run2


def test_queue_skips_decided_polygons(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.95': [(0, 0, 200, 't0001'), (0, 0, 100, 't0002')],
    })
    # The first polygon (t0001 — bigger) is already decided
    first_uid = next(iter(PolygonQueue(gpkg_path=str(gpkg), mineral='hcp'))).polygon_uid
    decisions_csv = tmp_path / 'decisions.csv'
    pd.DataFrame([{'polygon_uid': first_uid, 'decision': 'confirm'}]).to_csv(decisions_csv, index=False)

    q = PolygonQueue(gpkg_path=str(gpkg), mineral='hcp', decisions_csv=str(decisions_csv))
    items = list(q)
    assert len(items) == 1
    assert items[0].tile_id == 't0002'
