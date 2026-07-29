"""Regression guards for the review-app decision-widget state bug.

Root cause (fixed in 648ae1a): the confidence / co-occurring / corrected
inputs were created WITHOUT an explicit ``key=``. Streamlit then derives a
widget's identity from its construction params, which are identical on every
polygon — so a selection (confidence=Low, co-occurring=hcp, ...) silently
STUCK from one polygon to the next, and re-rendering surfaced the previous
polygon's state ("wrong polygon after Previous/Next"). The fix keys each of
the three per-polygon inputs on ``(mineral, polygon_uid)`` so every polygon
starts at its defaults.

Two invariants keep that fix honest:
  1. every per-polygon decision widget carries a key referencing polygon_uid;
  2. polygon_uid is globally unique across the queue, so the per-uid keys
     (and the per-uid cache dict) never collide two distinct polygons.
"""
import ast
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from scripts.review.polygon_queue import PolygonQueue

APP_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scripts', 'review', 'app.py')

# (streamlit function, label substring) for the three inputs that MUST be
# keyed per polygon. The sidebar 'mineral' radio is deliberately excluded — it
# is session-level state, not per-polygon.
DECISION_WIDGETS = [
    ('multiselect', 'also present'),
    ('selectbox', 'if rejected, actually'),
    ('radio', 'confidence'),
]


def decision_widget_key_report(source: str) -> dict:
    """{label substring -> key source text (or None if unkeyed/absent)}.

    Walks the AST for `st.<widget>(...)` calls, matches each to a decision
    widget by its first string-literal argument (the label), and extracts the
    source of its `key=` keyword. Returns None for a widget with no key kwarg
    and omits a widget entirely if the call is missing.
    """
    tree = ast.parse(source)
    report: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == 'st'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        label = str(node.args[0].value)
        for widget, sub in DECISION_WIDGETS:
            if func.attr == widget and sub in label:
                key_kw = next((k for k in node.keywords if k.arg == 'key'), None)
                report[sub] = ast.get_source_segment(source, key_kw.value) if key_kw else None
    return report


def test_decision_widgets_are_keyed_per_polygon():
    """RED on the pre-648ae1a code (keyless radio); GREEN once every decision
    input is keyed on polygon_uid."""
    with open(APP_PY) as fh:
        source = fh.read()
    report = decision_widget_key_report(source)

    for _widget, sub in DECISION_WIDGETS:
        assert sub in report, f'decision widget {sub!r} not found in app.py — did it get renamed?'
        key_src = report[sub]
        assert key_src is not None, f'decision widget {sub!r} has no key= (state will stick across polygons)'
        assert 'polygon_uid' in key_src, (
            f'decision widget {sub!r} key {key_src!r} must reference polygon_uid '
            f'so each polygon gets a distinct widget identity')


def _square(x, y, size, tile_id):
    geom = Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
    return gpd.GeoDataFrame({'tile_id': [tile_id], 'mineral': ['hcp'], 'threshold': [0.95]},
                            geometry=[geom], crs=None)


def test_polygon_uid_globally_unique_across_queue(tmp_path):
    """Distinct polygons — across layers AND across tiles within a layer —
    must yield distinct polygon_uids, or the per-uid widget keys and the
    per-uid cache dict would conflate two different polygons."""
    gpkg = tmp_path / 'hcp.gpkg'
    layers = {
        'thresh_0.90': [(0, 0, 100, 't0001'), (200, 0, 100, 't0002'),
                        (400, 0, 100, 't0001')],   # same tile repeated in a layer
        'thresh_0.95': [(0, 0, 100, 't0001'), (200, 0, 100, 't0002')],
    }
    for layer, polys in layers.items():
        merged = pd.concat([_square(*p) for p in polys], ignore_index=True)
        gpd.GeoDataFrame(merged, geometry='geometry', crs=None).to_file(
            str(gpkg), driver='GPKG', layer=layer)

    uids = [i.polygon_uid for i in PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')]
    assert len(uids) == 5
    assert len(uids) == len(set(uids)), f'duplicate polygon_uid(s): {uids}'
