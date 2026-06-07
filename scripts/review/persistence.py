"""Decision-log csv + parquet writers for the MC13 review app.

decisions.csv is the source of truth (append-only). Both parquet files are
derived: on each polygon decision the corresponding pixel rows are written.
Re-appending the same polygon_uid replaces the prior rows (idempotent).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd


# Order MUST match data/mrral_pixels.parquet exactly so downstream pipelines
# (patch-cache builder, train.py) can consume the new parquet unchanged.
_LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
_DECISION_COLS = [
    'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
    'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
]


def confirmed_schema_columns() -> list[str]:
    return (
        ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
        + [f'm{i}' for i in range(59)]
        + _LABEL_COLS
        + ['confidence_weight', 'confidence_tier', 'split']
    )


def hard_negatives_schema_columns() -> list[str]:
    return confirmed_schema_columns() + ['negative_of']


class DecisionLog:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    def append(self, record: dict) -> None:
        row = {k: record.get(k, '') for k in _DECISION_COLS}
        row['ts'] = dt.datetime.now(dt.timezone.utc).isoformat()
        # Use file POSITION after open instead of pre-open exists() check, so
        # concurrent appends can't both decide to write a header. 'a' mode
        # seeks to end on open; tell()==0 means we just created the file.
        with open(self.csv_path, 'a', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=_DECISION_COLS)
            if fp.tell() == 0:
                w.writeheader()
            w.writerow(row)

    def uids_seen(self) -> set[str]:
        if not os.path.exists(self.csv_path):
            return set()
        df = pd.read_csv(self.csv_path)
        if 'polygon_uid' not in df.columns:
            return set()
        return set(df['polygon_uid'].astype(str).tolist())

    def most_recent_for(self, polygon_uid: str) -> Optional[dict]:
        """Return the most recent decision row for ``polygon_uid``, or None."""
        if not os.path.exists(self.csv_path):
            return None
        df = pd.read_csv(self.csv_path)
        if 'polygon_uid' not in df.columns:
            return None
        matches = df[df['polygon_uid'].astype(str) == polygon_uid]
        if matches.empty:
            return None
        return matches.iloc[-1].to_dict()


def _polygon_id_int(polygon_uid: str) -> int:
    """Deterministic int64-safe integer from a polygon_uid string.

    Stable across processes (md5 has no PYTHONHASHSEED dependence). 32-bit
    range to keep small + parquet-friendly.
    """
    h = hashlib.md5(polygon_uid.encode('utf-8')).hexdigest()
    return int(h[:8], 16)


def _label_dict_for(label_class: str) -> dict[str, float]:
    out = {c: 0.0 for c in _LABEL_COLS}
    if label_class == 'olivine':
        out['olivine_t1'] = 1.0  # use the more-confident tier slot for new confirmed olivine
    elif label_class in out:
        out[label_class] = 1.0
    return out


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write parquet via .tmp + os.replace so a crash mid-write can't corrupt
    an existing parquet at ``path``."""
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _rows_for_polygon(
    tile_id: str,
    polygon_uid: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spectra: np.ndarray,
    label_dict: dict[str, float],
) -> pd.DataFrame:
    n = spectra.shape[0]
    polygon_id_int = _polygon_id_int(polygon_uid)
    data = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id_int, dtype=np.int64),
        'pixel_row': rows.astype(np.int64),
        'pixel_col': cols.astype(np.int64),
    }
    for i in range(59):
        data[f'm{i}'] = spectra[:, i].astype(np.float64)
    for c in _LABEL_COLS:
        data[c] = np.full(n, label_dict[c], dtype=np.float64)
    data['confidence_weight'] = np.full(n, 1.0, dtype=np.float64)
    data['confidence_tier'] = ['High'] * n
    data['split'] = ['train'] * n
    return pd.DataFrame(data, columns=confirmed_schema_columns())


class ConfirmedPixelsWriter:
    """Buffered parquet writer keyed by polygon_uid (reappend = replace)."""

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        os.makedirs(os.path.dirname(parquet_path) or '.', exist_ok=True)
        self._buf: dict[str, pd.DataFrame] = {}   # polygon_uid -> rows

    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray, label_class: str) -> None:
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra,
                                _label_dict_for(label_class))
        self._buf[polygon_uid] = df

    def flush(self) -> None:
        # Load existing parquet (if any), drop rows for any uids in buffer
        if os.path.exists(self.parquet_path):
            existing = pd.read_parquet(self.parquet_path)
            # Rows in existing whose polygon_id maps to a uid we're rewriting
            buf_polygon_ids = {_polygon_id_int(uid) for uid in self._buf}
            existing = existing[~existing['polygon_id'].isin(buf_polygon_ids)]
        else:
            existing = pd.DataFrame(columns=confirmed_schema_columns())
        all_new = pd.concat(list(self._buf.values()), ignore_index=True) \
                  if self._buf else pd.DataFrame(columns=confirmed_schema_columns())
        out = pd.concat([existing, all_new], ignore_index=True)
        out = out[confirmed_schema_columns()]  # enforce column order
        _atomic_write_parquet(out, self.parquet_path)
        self._buf.clear()

    def drop_polygon(self, polygon_uid: str) -> None:
        """Remove all rows for ``polygon_uid``. No-op if parquet/rows missing."""
        if not os.path.exists(self.parquet_path):
            return
        pid = _polygon_id_int(polygon_uid)
        existing = pd.read_parquet(self.parquet_path)
        kept = existing[existing['polygon_id'] != pid]
        if len(kept) == len(existing):
            return
        _atomic_write_parquet(kept, self.parquet_path)


class HardNegativesWriter:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        os.makedirs(os.path.dirname(parquet_path) or '.', exist_ok=True)
        self._buf: dict[str, pd.DataFrame] = {}

    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray,
                        predicted_class: str,
                        corrected_class: Optional[str]) -> None:
        if corrected_class:
            label = _label_dict_for(corrected_class)
            negative_of = ''
        else:
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = predicted_class
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra, label)
        df['negative_of'] = negative_of
        df = df[hard_negatives_schema_columns()]
        self._buf[polygon_uid] = df

    def flush(self) -> None:
        if os.path.exists(self.parquet_path):
            existing = pd.read_parquet(self.parquet_path)
            buf_polygon_ids = {_polygon_id_int(uid) for uid in self._buf}
            existing = existing[~existing['polygon_id'].isin(buf_polygon_ids)]
        else:
            existing = pd.DataFrame(columns=hard_negatives_schema_columns())
        all_new = pd.concat(list(self._buf.values()), ignore_index=True) \
                  if self._buf else pd.DataFrame(columns=hard_negatives_schema_columns())
        out = pd.concat([existing, all_new], ignore_index=True)
        out = out[hard_negatives_schema_columns()]
        _atomic_write_parquet(out, self.parquet_path)
        self._buf.clear()

    def drop_polygon(self, polygon_uid: str) -> None:
        """Remove all rows for ``polygon_uid``. No-op if parquet/rows missing."""
        if not os.path.exists(self.parquet_path):
            return
        pid = _polygon_id_int(polygon_uid)
        existing = pd.read_parquet(self.parquet_path)
        kept = existing[existing['polygon_id'] != pid]
        if len(kept) == len(existing):
            return
        _atomic_write_parquet(kept, self.parquet_path)
