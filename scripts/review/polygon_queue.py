"""Polygon queue iterator for the MC13 review app.

Walks threshold layers high→low within a single gpkg, sorts polygons within
a layer by area descending, and skips polygons that already appear in a
decisions.csv ledger (resumability).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import fiona
import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry.base import BaseGeometry

# Mars 2000 sphere — used to compute polygon areas in m² when the source gpkg
# is in geographic degrees (mc13 vector outputs are). 3396190 m matches the
# IAU 2000 spheroid used by the rest of the pipeline.
_MARS_GEOD = pyproj.Geod(a=3396190.0, b=3396190.0)


def _polygon_area_m2(geom: BaseGeometry, crs: Optional[pyproj.CRS]) -> float:
    if crs is not None and crs.is_geographic:
        return abs(_MARS_GEOD.geometry_area_perimeter(geom)[0])
    # Projected CRS (or unknown) — assume coords are in meters.
    return float(geom.area)


@dataclass(frozen=True)
class PolygonItem:
    polygon_uid: str           # f"{tile_id}::{layer}::{index_in_layer}"
    tile_id: str
    layer: str                 # e.g. "thresh_0.95"
    predicted_class: str       # mineral the gpkg represents
    geometry: BaseGeometry
    area_m2: float
    pred_prob: float           # parsed from layer name
    source_gpkg: str           # parent/basename, e.g. "vector_mc13_relabeled/hcp.gpkg"
    source_crs: Optional[str] = None  # WKT of the gpkg CRS (mc13 vectors are in geographic
                                       # degrees; mrral tiles are per-tile equirect meters —
                                       # loader uses this to reproject before rasterizing)


# Matches both the legacy layer name (`thresh_0.99`) and the rank-prefixed form
# (`thresh_01_0.99`) the vectorizer now writes so QGIS stacks layers highest-first.
# The optional `\d+_` group is the sort-rank prefix; the threshold is always the
# trailing float.
_LAYER_RE = re.compile(r'^thresh_(?:\d+_)?(?P<p>\d+(?:\.\d+)?)$')


def _layer_threshold(name: str) -> Optional[float]:
    m = _LAYER_RE.match(name)
    return float(m.group('p')) if m else None


def _canonical_layer(prob: float) -> str:
    """Canonical, uid-stable layer token for a threshold — independent of the
    physical gpkg layer name. Physical layers may carry a rank prefix
    (`thresh_01_0.99`) for QGIS ordering, but polygon_uid always uses this form
    (`thresh_0.99`) so decisions.csv references stay valid across re-vectorizations
    that change only the physical layer names."""
    return f'thresh_{prob:.2f}'


class PolygonQueue:
    """Iterable over PolygonItems for a single mineral gpkg.

    Yields items in: layer threshold high→low, then area descending.
    Skips polygons whose polygon_uid is present in ``decisions_csv``.
    """

    def __init__(
        self,
        gpkg_path: str,
        mineral: str,
        decisions_csv: Optional[str] = None,
    ):
        if not os.path.exists(gpkg_path):
            raise FileNotFoundError(gpkg_path)
        self.gpkg_path = gpkg_path
        self.mineral = mineral
        self._skip_uids: set[str] = set()
        if decisions_csv and os.path.exists(decisions_csv):
            df = pd.read_csv(decisions_csv)
            if 'polygon_uid' in df.columns:
                self._skip_uids = set(df['polygon_uid'].astype(str).tolist())

        layers = [L for L in fiona.listlayers(gpkg_path)
                  if _layer_threshold(L) is not None]
        layers.sort(key=_layer_threshold, reverse=True)
        self._layers = layers
        # canonical uid token -> physical gpkg layer name (identity for legacy
        # gpkgs; maps thresh_0.99 -> thresh_01_0.99 for rank-prefixed ones).
        self._canon_to_physical = {
            _canonical_layer(_layer_threshold(L)): L for L in layers}

        gpkg_parent = os.path.basename(os.path.dirname(os.path.abspath(gpkg_path)))
        gpkg_file = os.path.basename(gpkg_path)
        self._source_gpkg = f'{gpkg_parent}/{gpkg_file}'

    def __iter__(self) -> Iterator[PolygonItem]:
        for layer in self._layers:
            prob = _layer_threshold(layer)
            gdf = gpd.read_file(self.gpkg_path, layer=layer).reset_index(drop=True)
            if gdf.empty:
                continue
            layer_crs = gdf.crs
            layer_crs_wkt = layer_crs.to_wkt() if layer_crs is not None else None
            # Capture the file-order index BEFORE sorting so polygon_uid is
            # stable across runs (fiona/gpd read features in fid order).
            gdf['_original_idx'] = gdf.index
            gdf['_area'] = [_polygon_area_m2(g, layer_crs) for g in gdf.geometry]
            gdf = gdf.sort_values('_area', ascending=False, kind='mergesort')
            canon = _canonical_layer(prob)   # uid token, independent of rank prefix
            for _, row in gdf.iterrows():
                tile_id = str(row.get('tile_id', ''))
                uid = f'{tile_id}::{canon}::{int(row["_original_idx"])}'
                if uid in self._skip_uids:
                    continue
                yield PolygonItem(
                    polygon_uid=uid,
                    tile_id=tile_id,
                    layer=canon,
                    predicted_class=self.mineral,
                    geometry=row.geometry,
                    area_m2=float(row['_area']),
                    pred_prob=prob,
                    source_gpkg=self._source_gpkg,
                    source_crs=layer_crs_wkt,
                )

    def lookup_items(self, polygon_uids: list[str]) -> dict[str, PolygonItem]:
        """Look up PolygonItems by uid regardless of decision-skip state.

        Used to rehydrate the session's Previous-button history from
        decisions.csv on app restart. Reads only the layers referenced by the
        requested uids; unknown uids are silently omitted from the result.
        """
        # uid layer token is canonical (thresh_0.99); map it to the physical gpkg
        # layer (which may be rank-prefixed) before reading.
        wanted_by_layer: dict[str, list[tuple[str, int]]] = {}
        for uid in polygon_uids:
            parts = uid.split('::')
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[2])
            except ValueError:
                continue
            wanted_by_layer.setdefault(parts[1], []).append((uid, idx))

        results: dict[str, PolygonItem] = {}
        for canon, wanted in wanted_by_layer.items():
            physical = self._canon_to_physical.get(canon)
            if physical is None:
                continue
            prob = _layer_threshold(physical)
            gdf = gpd.read_file(self.gpkg_path, layer=physical).reset_index(drop=True)
            if gdf.empty:
                continue
            layer_crs = gdf.crs
            layer_crs_wkt = layer_crs.to_wkt() if layer_crs is not None else None
            for uid, idx in wanted:
                if idx < 0 or idx >= len(gdf):
                    continue
                row = gdf.iloc[idx]
                tile_id = str(row.get('tile_id', ''))
                area = _polygon_area_m2(row.geometry, layer_crs)
                results[uid] = PolygonItem(
                    polygon_uid=uid,
                    tile_id=tile_id,
                    layer=canon,
                    predicted_class=self.mineral,
                    geometry=row.geometry,
                    area_m2=area,
                    pred_prob=prob,
                    source_gpkg=self._source_gpkg,
                    source_crs=layer_crs_wkt,
                )
        return results
