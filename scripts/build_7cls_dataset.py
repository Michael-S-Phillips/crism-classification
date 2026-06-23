"""
Build the 7-class training parquet for the v3-bland 7-class run.

Design decisions:
  - Plagioclase: restored from Argyre/Hellas gpkg (splits from base parquet).
    MTRDR synth rows injected additionally at train/val time via
    --synth_train_cache/parquet and --synth_val_cache/parquet.
  - Alteration: MC11 review only (65 polygons, 103.9k pixels). No Argyre/
    Hellas/Nili gpkg alteration. Polygon-level 70/15/15 holdout so
    val_AP_alteration is measured on clean, same-distribution held-out pixels.
  - Bland (was "other"): three sources, each capped at N_BLAND_PER_SOURCE rows,
    each assigned independent 70/15/15 splits:
      1. Bland tiles (8 tiles, ~900k → subsampled)
      2. MC13 review blands (rejected polygons from MC13 review session)
      3. MC11 review blands (rejected polygons from MC11 review session)
  - Junk (new class): ambiguous hard_negatives (negative_of='ambiguous', 34k).
    confidence_weight=2.0, 70/15/15 tile-level split.
  - MC13 confirmed mineral positives: tile-level 70/15/15 splits (20 tiles).

Classes (7):  olivine | lcp | hcp | plagioclase | bland | alteration | junk

Output:  data/mrral_pixels_7cls.parquet

Usage (local, for quick schema verification — full build on HPC):
    conda run -n crism python scripts/build_7cls_dataset.py --dry_run
    conda run -n crism python scripts/build_7cls_dataset.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BASE_PARQUET  = os.path.join(PROJ, 'data', 'mrral_pixels.parquet')
DEFAULT_CONFIRMED_DIR = os.path.join(PROJ, 'data', 'mc13_review', 'confirmed_pixels')
DEFAULT_HN_DIR        = os.path.join(PROJ, 'data', 'mc13_review', 'hard_negatives')
DEFAULT_OUT           = os.path.join(PROJ, 'data', 'mrral_pixels_7cls.parquet')

N_BLAND_PER_SOURCE = 300_000   # rows per bland source (bland tiles, mc13, mc11)
MAX_PX_PER_POLYGON = 20_000    # per-polygon cap applied to review blands
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
REVIEW_WEIGHT = 2.0
SEED = 42

# MC13 tiles: t1028..t1396 (rows where tile_id matches this range)
_MC13_TILE_NUMS = set(range(1028, 1397))


def _is_mc13(tid: str) -> bool:
    try:
        return int(tid.lstrip('t')) in _MC13_TILE_NUMS
    except ValueError:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

_MAX_BLAND_RAW: int | None = None   # set by --max_bland_raw (debug/dry-run only)

import pyarrow.dataset as _pads


def _read_hn_tag(hn_dir: str, tag: str | None) -> pd.DataFrame:
    """Read hard_negatives rows by negative_of tag via predicate pushdown."""
    if tag is None:
        expr = pc.field('negative_of').is_null() | (pc.field('negative_of') == '')
    else:
        expr = pc.field('negative_of') == tag
    if _MAX_BLAND_RAW is not None:
        # Use scanner.head() so we never materialise the full file set.
        ds = _pads.dataset(hn_dir, format='parquet')
        table = ds.scanner(filter=expr).head(_MAX_BLAND_RAW)
    else:
        table = pq.read_table(hn_dir, filters=expr)
    return table.to_pandas()


def _per_polygon_cap(df: pd.DataFrame, max_per: int, seed: int) -> pd.DataFrame:
    """Subsample each (tile_id, polygon_id) group to at most max_per rows."""
    if df.empty:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(['tile_id', 'polygon_id'], sort=False):
        if len(g) <= max_per:
            parts.append(g)
        else:
            idx = rng.choice(len(g), size=max_per, replace=False)
            parts.append(g.iloc[idx])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def _assign_tile_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """70/15/15 split by tile_id so val/test have distinct spatial footprints."""
    tids = np.sort(df['tile_id'].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tids))
    n_hold = max(1, int(round(len(tids) * SPLIT_FRACS['val'])))
    test_tids = set(tids[perm[:n_hold]])
    val_tids  = set(tids[perm[n_hold:2 * n_hold]])
    out = df.copy()
    out['split'] = np.where(
        out['tile_id'].isin(test_tids), 'test',
        np.where(out['tile_id'].isin(val_tids), 'val', 'train'))
    return out


def _assign_polygon_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """70/15/15 split by (tile_id, polygon_id) so val/test hold out whole polygons."""
    polys = np.sort(df.apply(
        lambda r: f'{r.tile_id}_{r.polygon_id}', axis=1).unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(polys))
    n_hold = max(1, int(round(len(polys) * SPLIT_FRACS['val'])))
    test_polys = set(polys[perm[:n_hold]])
    val_polys  = set(polys[perm[n_hold:2 * n_hold]])
    key = df.apply(lambda r: f'{r.tile_id}_{r.polygon_id}', axis=1)
    out = df.copy()
    out['split'] = np.where(
        key.isin(test_polys), 'test',
        np.where(key.isin(val_polys), 'val', 'train'))
    return out


def _stamp_7cls_cols(df: pd.DataFrame,
                     bland: float = 0.0,
                     junk: float = 0.0,
                     alteration: float | None = None,
                     zero_plag: bool = True) -> pd.DataFrame:
    """Add / reset the 7-class-specific columns on df in-place copy.

    zero_plag=False preserves existing plagioclase values (for gpkg mineral rows
    where Argyre/Hellas plag should be kept).
    """
    out = df.copy()
    out['bland'] = np.float32(bland)
    out['junk']  = np.float32(junk)
    if zero_plag:
        out['plagioclase'] = np.float32(0.0)
    elif 'plagioclase' not in out.columns:
        out['plagioclase'] = np.float32(0.0)
    if alteration is not None:
        out['alteration'] = np.float32(alteration)
    elif 'alteration' not in out.columns:
        out['alteration'] = np.float32(0.0)
    # keep 'other' mirroring bland for backward compat with 5/6-class pipelines
    out['other'] = out['bland']
    return out


# ── Source loaders ────────────────────────────────────────────────────────────

def _build_base(path: str, n_bland_target: int) -> pd.DataFrame:
    print(f'Loading base parquet: {path}')
    df = pd.read_parquet(path)
    print(f'  {len(df):,} rows, columns: {list(df.columns)[:8]} …')

    bland_mask = df.get('other', pd.Series(0.0, index=df.index)) > 0
    print(f'  bland tile rows (other=1): {int(bland_mask.sum()):,}')

    # ── non-bland rows: preserve gpkg plag (Argyre/Hellas); keep existing splits ──
    non_bland = df[~bland_mask].copy()
    non_bland = _stamp_7cls_cols(non_bland, bland=0.0, junk=0.0, alteration=None,
                                  zero_plag=False)

    # ── bland tile rows: subsample to n_bland_target ──
    bland_df = df[bland_mask].copy()
    bland_df = _subsample(bland_df, n_bland_target, SEED)
    bland_df = _stamp_7cls_cols(bland_df, bland=1.0, junk=0.0, alteration=0.0)
    print(f'  bland tiles after subsample: {len(bland_df):,} '
          f'(target {n_bland_target:,})')

    out = pd.concat([non_bland, bland_df], ignore_index=True)
    print(f'  base after modification: {len(out):,} rows')
    return out


def load_confirmed_mineral_positives(confirmed_dir: str,
                                      template: pd.DataFrame) -> pd.DataFrame | None:
    if not os.path.exists(confirmed_dir):
        print(f'  no confirmed_pixels dir found, skipping')
        return None
    files = [f for f in os.listdir(confirmed_dir) if f.endswith('.parquet')]
    if not files:
        return None
    print(f'Loading confirmed mineral positives from {confirmed_dir} '
          f'({len(files)} files)')
    parts = [pd.read_parquet(os.path.join(confirmed_dir, f)) for f in files]
    df = pd.concat(parts, ignore_index=True)
    print(f'  {len(df):,} confirmed rows')

    # Per-polygon cap: confirmed positives are hyper-concentrated (olivine 73%
    # in top-5 polygons, largest single polygon 127.8k px) and would otherwise
    # teach the model a few memorized tiles. Cap each polygon like the bland
    # review sources do — same remediation that fixed the ft_with_review
    # regression. Applied before split assignment so per-class balance reflects
    # the capped data.
    df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 300)
    print(f'  {len(df):,} after {MAX_PX_PER_POLYGON:,}/polygon cap')

    # stamp 7cls cols (plag in confirmed pixels is always 0 — no real plag in MC13)
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=0.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    df = _assign_tile_splits(df, SEED + 300)
    splits = df['split'].value_counts().to_dict()
    print(f'  confirmed minerals: tile-level splits {splits}')

    # align columns to template
    for c in template.columns:
        if c not in df.columns:
            df[c] = np.float32(0.0) if c not in ('tile_id', 'split', 'confidence_tier') else ''
    return df[template.columns.tolist()]


def load_bland_review(hn_dir: str, source_label: str,
                       mc13: bool, seed_offset: int,
                       n_bland: int = N_BLAND_PER_SOURCE) -> pd.DataFrame:
    """Load corrected (bland) hard_negatives for one spatial source."""
    bland_raw = _read_hn_tag(hn_dir, tag=None)
    # filter to the requested spatial region
    if mc13:
        mask = bland_raw['tile_id'].apply(_is_mc13)
    else:
        mask = ~bland_raw['tile_id'].apply(_is_mc13)
    df = bland_raw[mask].copy()
    print(f'  {source_label}: {len(df):,} raw rows '
          f'({df["tile_id"].nunique()} tiles, '
          f'{df.groupby(["tile_id","polygon_id"]).ngroups} polygons)')

    df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + seed_offset)
    print(f'  {source_label}: {len(df):,} after {MAX_PX_PER_POLYGON:,}/polygon cap')

    df = _subsample(df, n_bland, SEED + seed_offset + 1)
    print(f'  {source_label}: {len(df):,} after subsample to {n_bland:,}')

    if df.empty:
        print(f'  {source_label}: 0 rows after filtering — skipping')
        return df
    df = _assign_tile_splits(df, SEED + seed_offset + 2)
    splits = df['split'].value_counts().to_dict()
    print(f'  {source_label}: splits {splits}')

    df = _stamp_7cls_cols(df, bland=1.0, junk=0.0, alteration=0.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    return df


def load_junk_ambiguous(hn_dir: str) -> pd.DataFrame:
    """Load ambiguous rows as junk=1."""
    df = _read_hn_tag(hn_dir, tag='ambiguous')
    print(f'  junk (ambiguous): {len(df):,} rows '
          f'({df["tile_id"].nunique()} tiles)')

    df = _assign_tile_splits(df, SEED + 100)
    splits = df['split'].value_counts().to_dict()
    print(f'  junk splits: {splits}')

    df = _stamp_7cls_cols(df, bland=0.0, junk=1.0, alteration=0.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    return df


def load_alteration_mc11(hn_dir: str) -> pd.DataFrame:
    """Load mc11 alteration review rows (negative_of='alteration'), polygon holdout."""
    df = _read_hn_tag(hn_dir, tag='alteration')
    print(f'  mc11 alteration: {len(df):,} rows '
          f'({df["tile_id"].nunique()} tiles, '
          f'{df["polygon_id"].nunique()} polygons)')

    df = _assign_polygon_splits(df, SEED + 200)
    splits = df['split'].value_counts().to_dict()
    print(f'  alteration splits: {splits}')

    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=1.0)
    df['confidence_weight'] = np.float32(REVIEW_WEIGHT)
    df['confidence_tier']   = 'Reviewed'
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base_parquet',  default=DEFAULT_BASE_PARQUET)
    ap.add_argument('--confirmed_dir', default=DEFAULT_CONFIRMED_DIR)
    ap.add_argument('--hn_dir',        default=DEFAULT_HN_DIR)
    ap.add_argument('--out',           default=DEFAULT_OUT)
    ap.add_argument('--n_bland',       type=int, default=N_BLAND_PER_SOURCE,
                    help=f'Rows per bland source (default {N_BLAND_PER_SOURCE:,})')
    ap.add_argument('--dry_run',       action='store_true',
                    help='Print counts only; do not write the output file')
    ap.add_argument('--max_bland_raw', type=int, default=None,
                    help='Cap raw bland rows per tag read (debug/dry-run only; '
                         'do NOT use for real builds)')
    args = ap.parse_args()

    if args.max_bland_raw is not None:
        global _MAX_BLAND_RAW  # noqa: PLW0603
        _MAX_BLAND_RAW = args.max_bland_raw

    n_bland = args.n_bland

    # ── 1. Base parquet (gpkgs + bland tiles, gpkg plag restored) ────────────
    base = _build_base(args.base_parquet, n_bland)

    # ── 2. MC13 confirmed mineral positives ───────────────────────────────────
    print('\nLoading confirmed mineral positives …')
    confirmed = load_confirmed_mineral_positives(args.confirmed_dir, base)

    # ── 3. Bland review sources ───────────────────────────────────────────────
    print('\nLoading bland review sources …')
    mc13_bland = load_bland_review(args.hn_dir, 'mc13_blands', mc13=True,  seed_offset=10, n_bland=n_bland)
    mc11_bland = load_bland_review(args.hn_dir, 'mc11_blands', mc13=False, seed_offset=20, n_bland=n_bland)

    # ── 4. Junk (ambiguous) ───────────────────────────────────────────────────
    print('\nLoading junk (ambiguous) source …')
    junk_df = load_junk_ambiguous(args.hn_dir)

    # ── 5. Alteration (mc11 review only) ─────────────────────────────────────
    print('\nLoading mc11 alteration review …')
    alt_df = load_alteration_mc11(args.hn_dir)

    # ── 6. Align schemas and concatenate ─────────────────────────────────────
    all_cols = base.columns.tolist()
    # 7-class specific columns must be present in every fragment
    for extra in ('bland', 'junk', 'alteration'):
        if extra not in all_cols:
            all_cols.append(extra)

    fragments = [base]
    for label, frag in [('confirmed', confirmed),
                         ('mc13_bland', mc13_bland),
                         ('mc11_bland', mc11_bland),
                         ('junk', junk_df),
                         ('alteration', alt_df)]:
        if frag is None or len(frag) == 0:
            continue
        for c in all_cols:
            if c not in frag.columns:
                frag = frag.copy()
                frag[c] = np.float32(0.0) if c not in (
                    'tile_id', 'split', 'confidence_tier', 'negative_of') else ''
        fragments.append(frag[all_cols])
        print(f'  {label}: {len(frag):,} rows added')

    out = pd.concat(fragments, ignore_index=True)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print('\n=== 7-class dataset summary ===')
    label_cols = ['olivine_t1', 'lcp', 'hcp', 'plagioclase', 'bland',
                  'alteration', 'junk']
    for split in ('train', 'val', 'test'):
        sub = out[out['split'] == split]
        print(f'\n{split}: {len(sub):,} rows')
        for c in label_cols:
            if c in sub.columns:
                n = int((sub[c] > 0.5).sum())
                print(f'  {c:>14}: {n:>9,}')
        if 'confidence_tier' in sub.columns:
            print(f'  tiers: {sub["confidence_tier"].value_counts().to_dict()}')

    print(f'\nTotal: {len(out):,} rows')
    print(f'Output: {args.out}')

    if args.dry_run:
        print('\n[dry_run] Not writing output.')
        return

    # ── 8. Write ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_parquet(args.out, index=False)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'\nWrote {args.out} ({size_mb:.0f} MB)')


if __name__ == '__main__':
    main()
