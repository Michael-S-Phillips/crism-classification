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
_LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']
_DECISION_COLS = [
    'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
    'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
    # Semicolon-separated list of co-occurring mineral classes when a polygon
    # is confirmed as the predicted class but also shows signals from others
    # (e.g. olivine + hcp). Empty for single-class confirms and all rejects.
    'co_occurring_classes',
    # Reviewer confidence (High/Moderate/Low) logged for every decision. Only
    # applied to the parquet weight for confirms and mineral reassignments.
    'confidence',
]

# Reviewer confidence → per-polygon training sample weight. Stamped together
# with a 'Reviewed-<tier>' label that is intentionally OUTSIDE
# data/dataset.py::_TIER_WEIGHTS so _collapse_labels passes the literal weight
# through verbatim (leaving base-parquet High/Moderate/Low weights untouched).
REVIEW_CONFIDENCE_WEIGHTS = {'High': 1.0, 'Moderate': 0.75, 'Low': 0.5}


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

    def _migrate_schema_if_needed(self) -> None:
        """One-time migration: if the existing CSV's header is missing any
        columns from _DECISION_COLS, rewrite the file with the full schema
        (existing rows get empty values for the new columns). Safe to call
        on every append — it's a fast no-op when the schema is already
        current. Required because csv.DictWriter would otherwise write rows
        with more fields than the header advertises, corrupting the file."""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, 'r') as fp:
            header_line = fp.readline().rstrip('\r\n')
        if not header_line:
            return
        existing_cols = header_line.split(',')
        missing = [c for c in _DECISION_COLS if c not in existing_cols]
        if not missing:
            return
        # Read with pandas (tolerates trailing-comma ambiguities), add the
        # missing columns as empty strings, reorder to canonical, write back
        # via atomic rename so a kill mid-rewrite can't truncate the file.
        df = pd.read_csv(self.csv_path)
        for c in missing:
            df[c] = ''
        df = df[_DECISION_COLS]
        tmp = self.csv_path + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.csv_path)

    def append(self, record: dict) -> None:
        self._migrate_schema_if_needed()
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


# UI labels → parquet label column. The parquet schema name 'other' is fixed
# (it matches mrral_pixels.parquet) but the UI exposes the more accurate
# 'bland' / 'dust' names. Both aliases map to the same label column.
_BLAND_ALIASES = ('bland', 'dust', 'dusty', 'other')


def _label_dict_for(label_class: str) -> dict[str, float]:
    return _label_dict_for_many([label_class])


def _label_dict_for_many(label_classes) -> dict[str, float]:
    """Build a multi-label dict: every class in ``label_classes`` is positive,
    everything else is zero. Used for co-occurring mineral confirmations
    (e.g. a polygon that is both olivine and hcp)."""
    out = {c: 0.0 for c in _LABEL_COLS}
    for lc in label_classes or ():
        if lc == 'olivine':
            out['olivine_t1'] = 1.0
        elif lc in _BLAND_ALIASES:
            out['other'] = 1.0
        elif lc in out:
            out[lc] = 1.0
    return out


def _is_mineral_class(label_class: str) -> bool:
    """True if ``label_class`` denotes a positive mineral assignment (vs. a
    non-mineral tag like 'ambiguous' that should be recorded as a negative)."""
    return label_class in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration') \
           or label_class in _BLAND_ALIASES


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
    weight: float = 1.0,
    tier: str = 'High',
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
    data['confidence_weight'] = np.full(n, weight, dtype=np.float64)
    data['confidence_tier'] = [tier] * n
    data['split'] = ['train'] * n
    return pd.DataFrame(data, columns=confirmed_schema_columns())


def _polygon_filename(polygon_uid: str) -> str:
    """Stable filesystem-safe filename derived from polygon_uid.

    Uses the same deterministic md5-based int as polygon_id in the row data,
    so the same uid always maps to the same file. 8-char hex is filesystem-
    safe and short."""
    return f'p_{_polygon_id_int(polygon_uid):08x}.parquet'


def _maybe_migrate_legacy_single_file(directory_path: str) -> None:
    """If a legacy single-file parquet exists at ``<directory_path>.parquet``,
    move it INTO the directory as ``legacy.parquet`` so the dataset-style
    reads still see its rows. Idempotent: no-op once migrated.

    The legacy file was created by the prior read-modify-write flush
    pattern and could be multi-GB (2.43 GB observed in one MC11 review
    session); the read-side wedge for parquet-into-pandas was the OOM
    source. The new per-polygon-file pattern fixes that by never reading
    the accumulated history during a write."""
    legacy_single = directory_path + '.parquet'
    legacy_target = os.path.join(directory_path, 'legacy.parquet')
    if not os.path.exists(legacy_single):
        return
    if os.path.exists(legacy_target):
        return
    os.makedirs(directory_path, exist_ok=True)
    # Plain rename — no data copy, instant even for 2+ GB files.
    os.rename(legacy_single, legacy_target)


class ConfirmedPixelsWriter:
    """Append-only parquet dataset keyed by polygon_uid.

    Each ``append_polygon`` call writes ONE small parquet file under
    ``<output_dir>/p_<hash>.parquet`` via atomic rename. There is no
    read-modify-write — flush() is a no-op preserved for API compatibility.
    Re-appending the same polygon_uid overwrites its file. Drop is
    ``os.remove(...)``. Downstream readers consume the dataset via
    ``pd.read_parquet('<output_dir>/')`` (pyarrow union of all files).

    The legacy single-file path (``<output_dir>.parquet``) is auto-migrated
    into the directory as ``legacy.parquet`` on first instantiation so
    historical data is preserved.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        _maybe_migrate_legacy_single_file(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # parquet_path retained as an alias so any external code reading
        # this attribute still works; points at the directory now.
        self.parquet_path = output_dir

    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray, label_class: str,
                        extra_classes: Optional[list] = None,
                        confidence: str = 'High') -> None:
        """Write rows for ``polygon_uid`` with positive labels for
        ``label_class`` and every class in ``extra_classes`` (co-occurring
        minerals), stamped with the reviewer ``confidence`` weight/tier.
        ``confidence`` must be a key in ``REVIEW_CONFIDENCE_WEIGHTS``
        (High/Moderate/Low); an unknown value raises KeyError."""
        all_classes = [label_class] + list(extra_classes or [])
        weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra,
                                _label_dict_for_many(all_classes),
                                weight=weight, tier=f'Reviewed-{confidence}')
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        _atomic_write_parquet(df, path)

    def flush(self) -> None:
        # No-op: append_polygon writes atomically per call. Kept for API
        # compatibility with the prior buffered-flush interface.
        return

    def drop_polygon(self, polygon_uid: str) -> None:
        """Remove the per-polygon file for ``polygon_uid`` if present."""
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        if os.path.exists(path):
            os.remove(path)


class HardNegativesWriter:
    """Same per-polygon-file pattern as ConfirmedPixelsWriter, plus the
    ``negative_of`` column that distinguishes ¬predicted-class rejections
    from explicit corrected-class rejections."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        _maybe_migrate_legacy_single_file(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.parquet_path = output_dir

    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray,
                        predicted_class: str,
                        corrected_class: Optional[str],
                        confidence: str = 'High') -> None:
        """Write reject rows for ``polygon_uid``. ``confidence`` is applied
        (weight + 'Reviewed-<tier>') ONLY to mineral reassignments; pure-negative
        and non-mineral-tag rejects keep fixed weight=1.0 / tier='High'.
        ``confidence`` must be a key in ``REVIEW_CONFIDENCE_WEIGHTS``; an unknown
        value raises KeyError."""
        # Three cases for the reject:
        #  - no corrected_class:    "not {predicted_class}" with no positive label
        #  - mineral corrected:     positive label for the corrected class,
        #                           stamped with the reviewer confidence weight
        #  - non-mineral tag (e.g. 'ambiguous'): all-zero labels, negative_of=tag
        weight, tier = 1.0, 'High'
        if not corrected_class:
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = predicted_class
        elif _is_mineral_class(corrected_class):
            label = _label_dict_for(corrected_class)
            negative_of = ''
            weight = REVIEW_CONFIDENCE_WEIGHTS[confidence]
            tier = f'Reviewed-{confidence}'
        else:
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = corrected_class
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra, label,
                                weight=weight, tier=tier)
        df['negative_of'] = negative_of
        df = df[hard_negatives_schema_columns()]
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        _atomic_write_parquet(df, path)

    def flush(self) -> None:
        return

    def drop_polygon(self, polygon_uid: str) -> None:
        path = os.path.join(self.output_dir, _polygon_filename(polygon_uid))
        if os.path.exists(path):
            os.remove(path)
