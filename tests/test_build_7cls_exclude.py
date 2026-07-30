"""Guard the plagioclase strip-artifact exclusion in build_7cls_dataset.

The 5 excluded ROIs are full-width strip artifacts (spectrally not plag, ~31% of
plag pixels — see memory plag-label-contamination). The drop must be keyed on the
(tile_id, polygon_id) PAIR, not polygon_id alone, or a legitimate polygon that
happens to share an id in another tile would be wrongly removed.
"""
import numpy as np
import pandas as pd

from scripts.build_7cls_dataset import (
    _drop_excluded_polygons, PLAG_EXCLUDE_POLYGONS)


def test_drops_exactly_the_strip_polygons_by_pair():
    rows = [dict(tile_id=t, polygon_id=p, plagioclase=1.0)
            for (t, p) in PLAG_EXCLUDE_POLYGONS]
    rows.append(dict(tile_id='t0638', polygon_id=1, plagioclase=1.0))   # keeper (same tile, other poly)
    rows.append(dict(tile_id='t9999', polygon_id=949, plagioclase=1.0)) # keeper (poly 949 in a DIFFERENT tile)
    out = _drop_excluded_polygons(pd.DataFrame(rows))
    for (t, p) in PLAG_EXCLUDE_POLYGONS:
        assert not ((out['tile_id'] == t) & (out['polygon_id'] == p)).any()
    # pair-keyed: 949 in t9999 survives, and t0638/1 survives
    assert ((out['tile_id'] == 't9999') & (out['polygon_id'] == 949)).any()
    assert ((out['tile_id'] == 't0638') & (out['polygon_id'] == 1)).any()
    assert len(out) == 2


def test_dtype_robust_int64_polygon_id():
    df = pd.DataFrame({'tile_id': ['t0638'],
                       'polygon_id': np.array([949], dtype='int64'),
                       'plagioclase': [1.0]})
    assert len(_drop_excluded_polygons(df)) == 0


def test_noop_when_no_polygon_id_column():
    df = pd.DataFrame({'tile_id': ['t0638'], 'x': [1]})
    assert len(_drop_excluded_polygons(df)) == 1
