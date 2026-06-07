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
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class PolygonItem:
    polygon_uid: str           # f"{tile_id}::{layer}::{index_in_layer}"
    tile_id: str
    layer: str                 # e.g. "thresh_0.95"
    predicted_class: str       # mineral the gpkg represents
    geometry: BaseGeometry
    area_m2: float
    pred_prob: float           # parsed from layer name
    source_gpkg: str           # basename, e.g. "vector_mc13_relabeled/hcp.gpkg"


_LAYER_RE = re.compile(r'^thresh_(?P<p>\d+\.\d+)$')


def _layer_threshold(name: str) -> Optional[float]:
    m = _LAYER_RE.match(name)
    return float(m.group('p')) if m else None


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

        gpkg_parent = os.path.basename(os.path.dirname(os.path.abspath(gpkg_path)))
        gpkg_file = os.path.basename(gpkg_path)
        self._source_gpkg = f'{gpkg_parent}/{gpkg_file}'

    def __iter__(self) -> Iterator[PolygonItem]:
        for layer in self._layers:
            prob = _layer_threshold(layer)
            gdf = gpd.read_file(self.gpkg_path, layer=layer).reset_index(drop=True)
            if gdf.empty:
                continue
            # Capture the file-order index BEFORE sorting so polygon_uid is
            # stable across runs (fiona/gpd read features in fid order).
            gdf['_original_idx'] = gdf.index
            gdf = gdf.assign(_area=gdf.geometry.area)
            gdf = gdf.sort_values('_area', ascending=False, kind='mergesort')
            for _, row in gdf.iterrows():
                tile_id = str(row.get('tile_id', ''))
                uid = f'{tile_id}::{layer}::{int(row["_original_idx"])}'
                if uid in self._skip_uids:
                    continue
                yield PolygonItem(
                    polygon_uid=uid,
                    tile_id=tile_id,
                    layer=layer,
                    predicted_class=self.mineral,
                    geometry=row.geometry,
                    area_m2=float(row['_area']),
                    pred_prob=prob,
                    source_gpkg=self._source_gpkg,
                )
