"""Real-data verification of the unit-balanced split rollout.

Runs every labeled-source loader from ``build_7cls_dataset`` against the real
local parquet + review dirs, then reports, for each source AND for the
concatenated union:
  - per-class positive-pixel counts per split
  - per-class achieved split fractions

Assertions (union):
  - every class with >0 pixels holds >=5% of its pixels in val AND in test.

Cross-source leakage check (reported, non-fatal):
  - sources are split independently, so a same-class polygon can straddle
    splits across sources. Over the UNION, for each class, count val-polygon /
    train-polygon centroid pairs within LINK_DEG (cos-lat scaled, 360-wrap).
    Reported per class (NOT asserted — a known design question).

Writes reports/split_rebalance_check.md. Rerun after future builds.

Usage:
    conda run -n crism python scripts/verify_split_rebalance.py
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ)
sys.path.insert(0, _HERE)

import build_7cls_dataset as b7  # noqa: E402
from split_units import (  # noqa: E402
    LINK_DEG, MIN_HOLDOUT_FRAC, NOMINAL_WH, POS_THRESH, SPLIT_ORDER,
    achieved_fractions, tile_center_deg,
)

# Columns reported / asserted on (matches build's summary frac_cols).
REPORT_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
               'bland', 'alteration', 'junk']

CONFIRMED_DIRS = b7.DEFAULT_CONFIRMED_DIRS
HN_DIRS = b7.DEFAULT_HN_DIRS
BASE_PARQUET = b7.DEFAULT_BASE_PARQUET
N_BLAND = b7.N_BLAND_PER_SOURCE


# Only these columns matter for split verification; dropping the ~70 band
# columns (m0..) keeps the union concat + leakage scan inside 15 GB RAM.
_KEEP = (['tile_id', 'polygon_id', 'pixel_row', 'pixel_col', 'split',
          'confidence_weight', 'confidence_tier', 'other'] + REPORT_COLS)


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in dict.fromkeys(_KEEP) if c in df.columns]
    return df[cols].copy()


def _present(df: pd.DataFrame) -> list[str]:
    return [c for c in REPORT_COLS if c in df.columns]


def _count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class positive-pixel counts per split (rows=class, cols=splits)."""
    cols = _present(df)
    data = {}
    for s in SPLIT_ORDER:
        sub = df[df['split'] == s]
        data[s] = [int((sub[c] > POS_THRESH).sum()) for c in cols]
    tab = pd.DataFrame(data, index=cols)
    tab['total'] = tab[list(SPLIT_ORDER)].sum(axis=1)
    return tab


def _frac_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = _present(df)
    return achieved_fractions(df, df['split'], cols)


def _md_table(df: pd.DataFrame, float_fmt: bool = False) -> str:
    cols = list(df.columns)
    head = '| class | ' + ' | '.join(str(c) for c in cols) + ' |'
    sep = '| --- | ' + ' | '.join('---' for _ in cols) + ' |'
    rows = [head, sep]
    for idx, row in df.iterrows():
        if float_fmt:
            vals = ' | '.join(f'{v:.3f}' for v in row)
        else:
            vals = ' | '.join(f'{int(v):,}' for v in row)
        rows.append(f'| {idx} | {vals} |')
    return '\n'.join(rows)


def _polygon_centroids_split(df: pd.DataFrame) -> pd.DataFrame:
    """One row per polygon: centroid lat/lon + its split.

    Assumes a polygon (tile_id, polygon_id) is a single split within one
    source frame (true by construction — whole units go to one split).
    """
    g = (df.groupby(['tile_id', 'polygon_id'], sort=False)
           .agg(mean_row=('pixel_row', 'mean'),
                mean_col=('pixel_col', 'mean'),
                split=('split', 'first'))
           .reset_index())
    latlon = g['tile_id'].map(tile_center_deg)
    t_lat = latlon.map(lambda x: x[0]).to_numpy(dtype=float)
    t_lon = latlon.map(lambda x: x[1]).to_numpy(dtype=float)
    g['lat'] = t_lat - 5.0 * ((g['mean_row'].to_numpy() / NOMINAL_WH) - 0.5)
    g['lon'] = t_lon + 5.0 * ((g['mean_col'].to_numpy() / NOMINAL_WH) - 0.5)
    return g


def _leakage_pairs(val_lat, val_lon, tr_lat, tr_lon, link_deg=LINK_DEG) -> int:
    """Count val/train centroid pairs within link_deg (cos-lat scaled, wrap)."""
    if len(val_lat) == 0 or len(tr_lat) == 0:
        return 0
    link2 = link_deg * link_deg
    n_pairs = 0
    for i in range(len(val_lat)):
        dlat = val_lat[i] - tr_lat
        dlon = (val_lon[i] - tr_lon + 180.0) % 360.0 - 180.0
        mlat = np.radians((val_lat[i] + tr_lat) / 2.0)
        dscaled = dlon * np.cos(mlat)
        n_pairs += int(np.count_nonzero(dlat * dlat + dscaled * dscaled <= link2))
    return n_pairs


def _is_band(col: str) -> bool:
    return len(col) > 1 and col[0] == 'm' and col[1:].isdigit()


def _patch_slim_hn_reader() -> None:
    """Monkeypatch b7._read_hn_tag to project out the ~70 band columns at read
    time. Split logic never touches band values, and reading them for the 2.8 GB
    hard-negative pool (re-read per source) OOMs a 15 GB workstation. The build
    on HPC (64 GB) reads full frames; this projection is verification-only."""
    import pyarrow.compute as _pc
    import pyarrow.dataset as _pads

    def slim_read(hn_dirs, tag):
        if tag is None:
            def expr_for(sch):
                f = _pc.field('negative_of')
                return f.is_null() | (f == '')
        else:
            def expr_for(sch):
                return _pc.field('negative_of') == tag
        parts = []
        for hn_dir in b7._as_dirs(hn_dirs):
            if not os.path.exists(hn_dir):
                print(f'  WARNING: hn_dir missing, skipping: {hn_dir}')
                continue
            dset = _pads.dataset(hn_dir, format='parquet')
            cols = [c for c in dset.schema.names if not _is_band(c)]
            table = dset.scanner(columns=cols,
                                 filter=expr_for(dset.schema)).to_table()
            parts.append(table.to_pandas())
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    b7._read_hn_tag = slim_read


def main() -> int:
    _patch_slim_hn_reader()
    print('=== building sources with the real local data ===')
    base_full = b7._build_base(BASE_PARQUET, N_BLAND)
    base = _slim(base_full)  # template for confirmed loader (slim => slim output)
    del base_full

    print('\n--- confirmed mineral positives ---')
    confirmed = b7.load_confirmed_mineral_positives(CONFIRMED_DIRS, base)
    if confirmed is not None:
        confirmed = _slim(confirmed)

    print('\n--- reassigned minerals ---')
    reassigned = _slim(b7.load_reassigned_minerals(HN_DIRS))

    print('\n--- mc13 blands ---')
    mc13_bland = _slim(b7.load_bland_review(HN_DIRS, 'mc13_blands', mc13=True,
                                            seed_offset=10, n_bland=N_BLAND))
    print('\n--- mc11 blands ---')
    mc11_bland = _slim(b7.load_bland_review(HN_DIRS, 'mc11_blands', mc13=False,
                                            seed_offset=20, n_bland=N_BLAND))

    print('\n--- junk (ambiguous) ---')
    junk_df = _slim(b7.load_junk_ambiguous(HN_DIRS))

    print('\n--- alteration ---')
    alt_df = _slim(b7.load_alteration_mc11(HN_DIRS))

    sources: list[tuple[str, pd.DataFrame]] = [('base', base)]
    for name, frag in [('confirmed', confirmed), ('reassigned', reassigned),
                       ('mc13_bland', mc13_bland), ('mc11_bland', mc11_bland),
                       ('junk', junk_df), ('alteration', alt_df)]:
        if frag is not None and len(frag):
            sources.append((name, frag))

    # ── union ────────────────────────────────────────────────────────────────
    all_cols: list[str] = list(base.columns)
    for extra in ('bland', 'junk', 'alteration'):
        if extra not in all_cols:
            all_cols.append(extra)
    aligned = []
    for name, frag in sources:
        f = frag.copy()
        for c in all_cols:
            if c not in f.columns:
                f[c] = (np.float32(0.0)
                        if c not in ('tile_id', 'split', 'confidence_tier',
                                     'negative_of', 'polygon_id') else '')
        f['__source'] = name
        aligned.append(f[all_cols + ['__source']])
    union = pd.concat(aligned, ignore_index=True)

    # ── assertions on the union ───────────────────────────────────────────────
    union_frac = _frac_table(union)
    union_counts = _count_table(union)
    failures = []
    for c in union_frac.index:
        if int(union_counts.loc[c, 'total']) == 0:
            continue
        for hold in ('val', 'test'):
            fr = float(union_frac.loc[c, hold])
            if fr < MIN_HOLDOUT_FRAC:
                failures.append(f'{c} {hold} fraction {fr:.3f} < {MIN_HOLDOUT_FRAC}')

    # ── cross-source leakage over the union ───────────────────────────────────
    print('\n=== cross-source leakage check ===')
    leakage: dict[str, int] = {}
    leak_detail: dict[str, tuple[int, int]] = {}
    for c in _present(union):
        # tag each polygon by source so cross-source polygons stay distinct
        pos = union[union[c] > POS_THRESH]
        if pos.empty:
            leakage[c] = 0
            leak_detail[c] = (0, 0)
            continue
        cents = []
        for name, frag in sources:
            fsub = frag[frag[c] > POS_THRESH] if c in frag.columns else frag.iloc[:0]
            if fsub.empty:
                continue
            g = _polygon_centroids_split(fsub)
            cents.append(g)
        if not cents:
            leakage[c] = 0
            leak_detail[c] = (0, 0)
            continue
        allc = pd.concat(cents, ignore_index=True)
        val = allc[allc['split'] == 'val']
        tr = allc[allc['split'] == 'train']
        leakage[c] = _leakage_pairs(val['lat'].to_numpy(), val['lon'].to_numpy(),
                                    tr['lat'].to_numpy(), tr['lon'].to_numpy())
        leak_detail[c] = (len(val), len(tr))
        print(f'  {c:>14}: {leakage[c]:>6} val/train pairs within {LINK_DEG} '
              f'({len(val)} val polys, {len(tr)} train polys)')

    # ── write report ──────────────────────────────────────────────────────────
    ts = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines: list[str] = []
    lines.append('# Split rebalance verification')
    lines.append('')
    lines.append(f'_Generated: {ts} by `scripts/verify_split_rebalance.py`_')
    lines.append('')
    lines.append(f'- Base parquet: `{BASE_PARQUET}`')
    lines.append(f'- Confirmed dirs: {", ".join("`%s`" % d for d in CONFIRMED_DIRS)}')
    lines.append(f'- Hard-negative dirs: {", ".join("`%s`" % d for d in HN_DIRS)}')
    lines.append(f'- LINK_DEG = {LINK_DEG}, MIN_HOLDOUT_FRAC = {MIN_HOLDOUT_FRAC}, '
                 f'targets = 70/15/15')
    lines.append('')

    status = 'PASS' if not failures else 'FAIL'
    lines.append(f'## Union min-holdout assertion: **{status}**')
    lines.append('')
    if failures:
        for f in failures:
            lines.append(f'- FAIL: {f}')
    else:
        lines.append('Every class with >0 pixels holds >=5% of its pixels in '
                     'both val and test.')
    lines.append('')

    lines.append('## Union — positive-pixel counts per split')
    lines.append('')
    lines.append(_md_table(union_counts))
    lines.append('')
    lines.append('## Union — achieved split fractions')
    lines.append('')
    lines.append(_md_table(union_frac, float_fmt=True))
    lines.append('')

    lines.append('## Cross-source leakage (val/train same-class polygon pairs '
                 f'within {LINK_DEG} deg)')
    lines.append('')
    lines.append('Sources are split independently, so a same-class polygon in '
                 'one source can land in val while a nearby same-class polygon '
                 'in another source lands in train. Reported, not asserted.')
    lines.append('')
    leak_df = pd.DataFrame(
        {'leaking_pairs': [leakage[c] for c in _present(union)],
         'val_polys': [leak_detail[c][0] for c in _present(union)],
         'train_polys': [leak_detail[c][1] for c in _present(union)]},
        index=_present(union))
    lines.append(_md_table(leak_df))
    lines.append('')
    total_leak = sum(leakage.values())
    lines.append(f'**Total leaking val/train pairs across all classes: {total_leak}**')
    lines.append('')

    lines.append('## Per-source detail')
    lines.append('')
    for name, frag in sources:
        lines.append(f'### {name} ({len(frag):,} rows)')
        lines.append('')
        lines.append('Counts:')
        lines.append('')
        lines.append(_md_table(_count_table(frag)))
        lines.append('')
        lines.append('Fractions:')
        lines.append('')
        lines.append(_md_table(_frac_table(frag), float_fmt=True))
        lines.append('')

    report_path = os.path.join(_PROJ, 'reports', 'split_rebalance_check.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nWrote {report_path}')

    print(f'\nUnion min-holdout assertion: {status}')
    if failures:
        for f in failures:
            print(f'  {f}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
