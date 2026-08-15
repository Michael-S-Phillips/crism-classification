"""Canonical string formatting for probability thresholds.

Single source of truth for turning a float threshold into the token used in
gpkg layer names (`thresh_0.999`) and in the `polygon_uid` layer field that
keys review `decisions.csv`. Shared by:

  - scripts/review/polygon_queue.py          (uid token, `_canonical_layer`)
  - scripts/vectorize_per_mineral_thresholds_nili_6cls.py  (`_fmt_thresh`)
  - scripts/build_review_set_stratified.py   (emitted stratum layer names)

Deliberately dependency-free (stdlib only) so any of those entrypoints can
import it without dragging in geopandas/streamlit.
"""
from __future__ import annotations

__all__ = ['fmt_threshold']


def fmt_threshold(t: float) -> str:
    """Shortest decimal representation of ``t`` that round-trips, 2 dp minimum.

    Two decimals for every value on the classic grid (0.50 / 0.85 / 0.97 /
    0.99 ...), so layer names and the `thresh_0.NN` tokens already recorded in
    review decisions.csv stay byte-identical. Extra decimals ONLY when 2 dp
    would not round-trip -- a 0.995 / 0.999 / 0.9999 ladder would otherwise
    render as `1.00` three times, collapsing three distinct rungs into one
    ambiguous token (both in QGIS and, far worse, in polygon_uid, which keys
    decisions.csv). polygon_queue's `thresh_(?:\\d+_)?(\\d+(?:\\.\\d+)?)`
    already accepts either width.
    """
    for nd in (2, 3, 4):
        s = f'{t:.{nd}f}'
        if abs(float(s) - t) < 1e-12:
            return s
    return f'{t:.6f}'
