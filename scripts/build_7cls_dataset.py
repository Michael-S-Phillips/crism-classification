"""
Build the 7-class training parquet for the v3-bland 7-class run.

Design decisions:
  - Splits: EVERY labeled source is split by the unit-aware, pixel-balanced
    splitter (scripts/split_units.py). Polygons within 0.25 deg (cos-lat
    scaled, 360-wraparound) are clustered into geographic units, and whole
    units are assigned to a single split (70/15/15 targets on per-class
    *pixel* fractions, with a >=5% val/test min-holdout guard). This kills the
    adjacent-tile leakage the old inherited/tile-level/polygon-level splits
    carried (same mapped unit's pixels landing in both train and val). The
    base parquet's inherited splits are OVERRIDDEN by this splitter. Each
    source loader still splits independently with its own seed offset, but
    those per-source splits are PROVISIONAL diagnostics only: at concat time
    main() jointly re-splits the combined frame with one
    assign_unit_balanced_splits pass over all 8 classes (_joint_resplit),
    so nearby same-class polygons from DIFFERENT sources (e.g. the three
    alteration sources) land in the same split instead of straddling
    train/val.
  - Plagioclase: restored from Argyre/Hellas gpkg (base-parquet mineral rows,
    plag preserved via zero_plag=False) then re-split by the unit splitter.
    MTRDR synth rows injected additionally at train/val time via
    --synth_train_cache/parquet and --synth_val_cache/parquet.
  - Alteration: review tags (negative_of='alteration') + alteration confirms /
    co-occurring alteration on mineral rows. Unit-balanced holdout on the
    'alteration' column so val_AP_alteration is measured on clean,
    same-distribution held-out pixels.
  - Bland (was "other"): four sources, each capped/subsampled then unit-split
    on the 'other' column:
      1. Bland tiles (subsampled to N_BLAND_PER_SOURCE)
      2. MC13 review blands (rejected polygons from MC13 review session)
      3. MC11 review blands (rejected polygons from MC11 review session)
  - Junk (new class): ambiguous hard_negatives (negative_of='ambiguous'). Now
    per-polygon capped (MAX_PX_PER_POLYGON) like every other review loader,
    then unit-split on the 'junk' column. Per-polygon reviewer confidence
    weight preserved.
  - Confirmed mineral positives + reject->mineral reassignments: per-polygon
    capped, then unit-split on the mineral balance columns. Per-polygon
    confidence weights preserved.
  - Review sessions are ADDITIVE: original MC13 session + confidence-graded v3
    session (MC13 + MC11), concatenated across confirmed_pixels / hard_negatives
    dirs.
  - ndviz relabel session (--ndviz_dir, data/ndviz_relabels/hard_negatives):
    interactive N-D visualizer relabels ingested with a PIXEL-LEVEL supersede —
    every ndviz pixel is anti-joined out of the combined frame (all decision
    types), then ndviz's own positives (mineral/bland reassignments, alteration,
    junk) are appended before the joint re-split. Absent dir = clean no-op.

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from split_units import assign_unit_balanced_splits, achieved_fractions

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BASE_PARQUET  = os.path.join(PROJ, 'data', 'mrral_pixels.parquet')
# Review sessions are additive: the original MC13 session plus the
# confidence-graded v3 session (MC13 + MC11, mixed in one dir).
DEFAULT_CONFIRMED_DIRS = [
    os.path.join(PROJ, 'data', 'mc13_review', 'confirmed_pixels'),
    os.path.join(PROJ, 'data', 'mc13_review_7cls_v3', 'confirmed_pixels'),
]
DEFAULT_HN_DIRS = [
    os.path.join(PROJ, 'data', 'mc13_review', 'hard_negatives'),
    os.path.join(PROJ, 'data', 'mc13_review_7cls_v3', 'hard_negatives'),
]
# The N-D visualizer relabel session (review-format rows: full 59-band spectra,
# real pixel coords, confidence_weight/tier, negative_of stamp). Handled
# SEPARATELY from DEFAULT_HN_DIRS so its pixel-level supersede (anti-join) can
# exclude it from the other sources. Absent dir -> clean no-op.
DEFAULT_NDVIZ_DIR     = os.path.join(PROJ, 'data', 'ndviz_relabels', 'hard_negatives')
DEFAULT_OUT           = os.path.join(PROJ, 'data', 'mrral_pixels_7cls.parquet')

N_BLAND_PER_SOURCE = 300_000   # rows per bland source (bland tiles, mc13, mc11)
MAX_PX_PER_POLYGON = 20_000    # per-polygon cap applied to review blands
SEED = 42

# Mineral classes balanced by the unit-aware splitter across every labeled
# source (olivine tier-1/tier-2 kept separate; alteration included). 'bland'
# and 'junk' are volume classes balanced on their own single-class column.
BALANCE_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                'alteration']

# All 8 classes, for the JOINT re-split main() applies over the concatenated
# frame. The union frame carries every label column, so the volume classes
# (bland, junk) join the mineral balance columns; multi-label rows (e.g.
# mineral + co-occurring alteration) are handled naturally by the greedy
# scorer — each positive class contributes to its unit's class-pixel vector.
JOINT_BALANCE_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                      'bland', 'alteration', 'junk']

# MC13 tiles: t1028..t1396 (rows where tile_id matches this range)
_MC13_TILE_NUMS = set(range(1028, 1397))

# Mineral label columns used to distinguish mineral reassignments from bland
# reassignments inside the negative_of='' hard-negative pool.
_REASSIGN_MINERAL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase']


def _is_mc13(tid: str) -> bool:
    try:
        return int(tid.lstrip('t')) in _MC13_TILE_NUMS
    except ValueError:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

_MAX_BLAND_RAW: int | None = None   # set by --max_bland_raw (debug/dry-run only)

import pyarrow.dataset as _pads


def _as_dirs(dirs: str | list[str]) -> list[str]:
    """Accept a single dir or a list of dirs (old + new review sessions)."""
    return [dirs] if isinstance(dirs, str) else list(dirs)


def _session_of(path: str) -> str:
    """Classify a review dir as the ungraded legacy session or the graded v3.

    Legacy rows are stamped confidence_tier='High', identical to hand-labeled
    High rows, so tier cannot identify the session — the path must.
    """
    return 'v3' if '_7cls_v3' in os.path.normpath(path) else 'legacy'


def _read_hn_tag(hn_dirs: str | list[str], tag: str | None) -> pd.DataFrame:
    """Read hard_negatives rows by negative_of tag via predicate pushdown.
    Accepts one dir or several (multi-session review data); schemas may
    differ across sessions (e.g. old files lack the alteration column) —
    pd.concat unifies with NaN, which downstream fillna paths handle."""
    if tag is None:
        expr = pc.field('negative_of').is_null() | (pc.field('negative_of') == '')
    else:
        expr = pc.field('negative_of') == tag
    parts = []
    for hn_dir in _as_dirs(hn_dirs):
        if not os.path.exists(hn_dir):
            print(f'  WARNING: hn_dir missing, skipping: {hn_dir}')
            continue
        if _MAX_BLAND_RAW is not None:
            # Use scanner.head() so we never materialise the full file set.
            ds = _pads.dataset(hn_dir, format='parquet')
            table = ds.scanner(filter=expr).head(_MAX_BLAND_RAW)
        else:
            table = pq.read_table(hn_dir, filters=expr)
        frag = table.to_pandas()
        frag['review_session'] = _session_of(hn_dir)
        parts.append(frag)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


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


def _fill_confidence_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure confidence_weight/confidence_tier are present and NaN-free.

    Files written before confidence support lack these columns; in a mixed
    directory pd.concat leaves NaN in the rows from those files. Both cases
    default to weight=1.0 / tier='High' (which collapse to weight 1.0
    downstream)."""
    if 'confidence_weight' not in df.columns:
        df['confidence_weight'] = np.float32(1.0)
    else:
        df['confidence_weight'] = df['confidence_weight'].fillna(np.float32(1.0))
    if 'confidence_tier' not in df.columns:
        df['confidence_tier'] = 'High'
    else:
        df['confidence_tier'] = df['confidence_tier'].fillna('High')
    return df


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
    else:
        # alteration=None preserves existing values (e.g. alteration confirms
        # and co-occurring alteration). Legacy files lack the column, so a
        # mixed-dir concat leaves NaN — fill those rows with 0.
        out['alteration'] = out['alteration'].fillna(np.float32(0.0))
    # keep 'other' mirroring bland for backward compat with 5/6-class pipelines
    out['other'] = out['bland']
    return out


# ── Source loaders ────────────────────────────────────────────────────────────

_HAND_MAFIC_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp']


def _apply_hand_mineral_policy(non_bland: pd.DataFrame,
                               hand_minerals: str) -> pd.DataFrame:
    """Apply the hand-label policy to the non-bland (gpkg) rows.

    The user's hand-drawn mineral labels are the noisier population; this flag
    lets a build train on review-derived mineral labels only. Plagioclase is
    100% hand-labeled (review found no plag), so it is the sole hand mineral
    that survives the restrictive policies.

      'all'       -> unchanged (default; byte-identical to prior behavior).
      'plag_only' -> keep only rows with plagioclase>0.5; on those rows zero the
                     mafic columns (olivine_t1/t2, lcp, hcp) so no hand mafic
                     label leaks via a multi-label gpkg row; drop everything
                     else.
      'none'      -> drop ALL non-bland gpkg rows.

    Only operates on the non-bland frame; bland tiles are handled by the caller
    and are never touched here.
    """
    if hand_minerals == 'all':
        return non_bland
    if hand_minerals == 'none':
        print(f'  hand_minerals=none: dropping all {len(non_bland):,} non-bland '
              f'gpkg rows')
        return non_bland.iloc[:0].copy()
    if hand_minerals == 'plag_only':
        plag_mask = non_bland['plagioclase'] > 0.5
        kept = non_bland[plag_mask].copy()
        n_dropped = len(non_bland) - len(kept)
        # Zero the mafic columns on kept rows so multi-label (plag+mafic) gpkg
        # rows cannot leak a hand mafic label.
        mafic_leaks = int((kept[_HAND_MAFIC_COLS] > 0.5).any(axis=1).sum())
        for c in _HAND_MAFIC_COLS:
            kept[c] = np.float32(0.0)
        print(f'  hand_minerals=plag_only: kept {len(kept):,} plag rows '
              f'(zeroed mafic on {mafic_leaks:,} multi-label rows), '
              f'dropped {n_dropped:,} non-plag non-bland rows')
        return kept
    raise ValueError(f'unknown hand_minerals policy: {hand_minerals!r}')


# Plagioclase label artifacts (audited 2026-07-30): 5 full-width "strip" ROIs that
# are spectrally featureless (BD1300<=0, no 1.3um feldspar band) — filled boxes
# spanning the whole CRISM strip, NOT plag. Together they are ~31% of plag training
# pixels (109,860 px) and teach the model that dust/bland spectra = plagioclase.
# Dropped at the base-parquet read so every downstream build (7cls, pyx, reviewonly)
# is clean. See memory plag-label-contamination + reports/plag_roi_audit.csv.
PLAG_EXCLUDE_POLYGONS: set[tuple[str, int]] = {
    ('t0638', 949), ('t0636', 823), ('t0566', 958),
    ('t0636', 769), ('t0638', 852),
}


def _drop_excluded_polygons(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows belonging to known-bad (tile_id, polygon_id) label artifacts."""
    if not PLAG_EXCLUDE_POLYGONS or 'polygon_id' not in df.columns:
        return df
    mask = pd.Series(False, index=df.index)
    tid = df['tile_id'].astype(str)
    for t, p in PLAG_EXCLUDE_POLYGONS:
        mask |= (tid == t) & (df['polygon_id'] == p)
    n = int(mask.sum())
    if n:
        print(f'  dropped {n:,} rows from {len(PLAG_EXCLUDE_POLYGONS)} excluded '
              f'plag-artifact polygons (strip ROIs — plag-label-contamination)')
    return df[~mask].reset_index(drop=True)


def _build_base(path: str, n_bland_target: int,
                hand_minerals: str = 'all') -> pd.DataFrame:
    print(f'Loading base parquet: {path}')
    df = pd.read_parquet(path)
    df = _drop_excluded_polygons(df)
    print(f'  {len(df):,} rows, columns: {list(df.columns)[:8]} …')

    bland_mask = df.get('other', pd.Series(0.0, index=df.index)) > 0
    print(f'  bland tile rows (other=1): {int(bland_mask.sum()):,}')

    # ── non-bland rows: preserve gpkg plag (Argyre/Hellas) ──
    non_bland = df[~bland_mask].copy()
    non_bland = _stamp_7cls_cols(non_bland, bland=0.0, junk=0.0, alteration=None,
                                  zero_plag=False)
    # Hand-label policy: 'all' is a no-op; 'plag_only'/'none' restrict which
    # hand mineral labels survive (see _apply_hand_mineral_policy).
    non_bland = _apply_hand_mineral_policy(non_bland, hand_minerals)
    # OVERRIDE the inherited (base-parquet) splits with the unit-aware splitter:
    # adjacent-tile polygons mapping the same unit are clustered and held out
    # together, killing the interleave leakage the inherited splits carried.
    if len(non_bland):
        non_bland['split'] = assign_unit_balanced_splits(non_bland, BALANCE_COLS, SEED)
        print('  non-bland gpkg rows: unit-balanced achieved val/test fractions:')
        print(achieved_fractions(non_bland, non_bland['split'], BALANCE_COLS)
              .to_string())

    # ── bland tile rows: subsample to n_bland_target ──
    bland_df = df[bland_mask].copy()
    bland_df = _subsample(bland_df, n_bland_target, SEED)
    bland_df = _stamp_7cls_cols(bland_df, bland=1.0, junk=0.0, alteration=0.0)
    if len(bland_df):
        bland_df['split'] = assign_unit_balanced_splits(bland_df, ['other'], SEED + 1)
    print(f'  bland tiles after subsample: {len(bland_df):,} '
          f'(target {n_bland_target:,})')
    if len(bland_df):
        print('  bland tiles: unit-balanced achieved fractions:')
        print(achieved_fractions(bland_df, bland_df['split'], ['other'])
              .to_string())

    out = pd.concat([non_bland, bland_df], ignore_index=True)
    print(f'  base after modification: {len(out):,} rows')
    return out


def load_confirmed_mineral_positives(confirmed_dirs: str | list[str],
                                      template: pd.DataFrame) -> pd.DataFrame | None:
    parts = []
    for confirmed_dir in _as_dirs(confirmed_dirs):
        if not os.path.exists(confirmed_dir):
            print(f'  no confirmed_pixels dir at {confirmed_dir}, skipping')
            continue
        files = [f for f in os.listdir(confirmed_dir) if f.endswith('.parquet')]
        if not files:
            continue
        print(f'Loading confirmed mineral positives from {confirmed_dir} '
              f'({len(files)} files)')
        session = _session_of(confirmed_dir)
        for f in files:
            frag = pd.read_parquet(os.path.join(confirmed_dir, f))
            frag['review_session'] = session
            parts.append(frag)
    if not parts:
        return None
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

    # stamp 7cls cols. alteration=None preserves the parquet's alteration
    # labels (alteration confirms + co-occurring alteration) — a 0.0 stamp
    # here used to wipe them into all-zero-label rows.
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=None)
    # Preserve the per-polygon reviewer confidence weight/tier; fill legacy/
    # mixed-schema rows that lack these columns with default 1.0/'High'.
    df = _fill_confidence_defaults(df)
    df['split'] = assign_unit_balanced_splits(df, BALANCE_COLS, SEED + 300)
    splits = df['split'].value_counts().to_dict()
    print(f'  confirmed minerals: unit-balanced splits {splits}')

    # align columns to template
    for c in template.columns:
        if c not in df.columns:
            df[c] = np.float32(0.0) if c not in ('tile_id', 'split', 'confidence_tier') else ''
    keep = template.columns.tolist()
    if 'review_session' in df.columns and 'review_session' not in keep:
        keep.append('review_session')
    return df[keep]


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

    # Mineral reassignments (negative_of='' with a mineral label=1.0) share this
    # pool; they belong in load_reassigned_minerals, not the bland pool.
    df = df[~(df[_REASSIGN_MINERAL_COLS] > 0).any(axis=1)].copy()
    print(f'  {source_label}: {len(df):,} rows after stripping mineral reassignments')

    df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + seed_offset)
    print(f'  {source_label}: {len(df):,} after {MAX_PX_PER_POLYGON:,}/polygon cap')

    df = _subsample(df, n_bland, SEED + seed_offset + 1)
    print(f'  {source_label}: {len(df):,} after subsample to {n_bland:,}')

    if df.empty:
        print(f'  {source_label}: 0 rows after filtering — skipping')
        return df
    df = _stamp_7cls_cols(df, bland=1.0, junk=0.0, alteration=0.0)
    df['split'] = assign_unit_balanced_splits(df, ['other'], SEED + seed_offset + 2)
    splits = df['split'].value_counts().to_dict()
    print(f'  {source_label}: unit-balanced splits {splits}')

    df = _fill_confidence_defaults(df)
    return df


def load_reassigned_minerals(hn_dir: str) -> pd.DataFrame:
    """Reject→mineral reassignments live in hard_negatives with negative_of=''
    and a mineral label = 1.0. Ingest them as weighted mineral positives
    (preserving the reviewer confidence weight/tier), capped per polygon and
    tile-split — NOT as bland (the prior behaviour, which mistrained them)."""
    pool = _read_hn_tag(hn_dir, tag=None)
    if pool.empty:
        return pool
    mineral_mask = (pool[_REASSIGN_MINERAL_COLS] > 0).any(axis=1)
    df = pool[mineral_mask].copy()
    print(f'  reassigned minerals: {len(df):,} rows '
          f'({df["tile_id"].nunique()} tiles, '
          f'{df.groupby(["tile_id","polygon_id"]).ngroups} polygons)')
    if df.empty:
        return df
    df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 400)
    # Preserve the parquet's confidence weight/tier (do not zero plagioclase —
    # a reject→plagioclase reassignment is a real plag positive; alteration=None
    # keeps co-occurring alteration labels). Stamp BEFORE the split so the
    # 'alteration' balance column exists (legacy hn files lack it).
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=None,
                          zero_plag=False)
    df['split'] = assign_unit_balanced_splits(df, BALANCE_COLS, SEED + 400)
    splits = df['split'].value_counts().to_dict()
    print(f'  reassigned minerals: unit-balanced splits {splits}')
    df = _fill_confidence_defaults(df)
    return df


def load_junk_ambiguous(hn_dir: str) -> pd.DataFrame:
    """Load ambiguous rows as junk=1."""
    df = _read_hn_tag(hn_dir, tag='ambiguous')
    print(f'  junk (ambiguous): {len(df):,} rows '
          f'({df["tile_id"].nunique()} tiles)')

    df = _per_polygon_cap(df, MAX_PX_PER_POLYGON, SEED + 100)
    print(f'  junk (ambiguous): {len(df):,} after {MAX_PX_PER_POLYGON:,}/polygon cap')

    # Stamp BEFORE the split so the 'junk' balance column exists; all rows are
    # junk, so balancing on it is exactly total-pixel balancing.
    df = _stamp_7cls_cols(df, bland=0.0, junk=1.0, alteration=0.0)
    df['split'] = assign_unit_balanced_splits(df, ['junk'], SEED + 100)
    splits = df['split'].value_counts().to_dict()
    print(f'  junk unit-balanced splits: {splits}')

    df = _fill_confidence_defaults(df)
    return df


def load_alteration_mc11(hn_dir: str) -> pd.DataFrame:
    """Load mc11 alteration review rows (negative_of='alteration'), polygon holdout."""
    df = _read_hn_tag(hn_dir, tag='alteration')
    print(f'  mc11 alteration: {len(df):,} rows '
          f'({df["tile_id"].nunique()} tiles, '
          f'{df["polygon_id"].nunique()} polygons)')

    # Stamp BEFORE the split so 'alteration'=1.0 exists for balancing.
    df = _stamp_7cls_cols(df, bland=0.0, junk=0.0, alteration=1.0)
    df['split'] = assign_unit_balanced_splits(df, ['alteration'], SEED + 200)
    splits = df['split'].value_counts().to_dict()
    print(f'  alteration unit-balanced splits: {splits}')

    df = _fill_confidence_defaults(df)
    return df


def load_ndviz_relabels(ndviz_dir: str):
    """Load the N-D visualizer relabel session (review-format rows).

    Returns (positives_fragment, suppression_keys). Mirrors the assembler's
    ndviz handling (scripts/label_quant/assemble_labeled_spectra.py): every
    ndviz pixel supersedes lower-precedence sources at the PIXEL level.

    - Absent / empty dir -> (empty DataFrame, empty MultiIndex) — clean no-op.
    - suppression_keys: unique (tile_id, pixel_row, pixel_col) over ALL rows,
      regardless of decision type (int32 coords to match the corpus frames for
      the anti-join). Discards contribute suppression only.
    - positives (negative_of semantics written by the app):
        '' + any mineral>0.5      -> mineral reassignment
                                     (_stamp_7cls_cols zero_plag=False so a
                                     reject->plagioclase stays plag; alteration
                                     preserved via alteration=None).
        '' + other>0.5, no mineral -> bland stamp.
        'alteration'               -> alteration stamp.
        'ambiguous'                -> junk stamp.
        anything else (orig class) -> discard: suppression only, NO positive.
      Per-polygon capped like every other review loader; confidence weight/tier
      preserved. No split assignment here — main's joint re-split covers it.
    """
    empty_keys = pd.MultiIndex.from_arrays([[], [], []])
    if not ndviz_dir or not os.path.exists(ndviz_dir):
        return pd.DataFrame(), empty_keys
    files = [f for f in os.listdir(ndviz_dir) if f.endswith('.parquet')]
    if not files:
        return pd.DataFrame(), empty_keys
    raw = pd.read_parquet(ndviz_dir)
    if raw.empty:
        return pd.DataFrame(), empty_keys

    # suppression keys over EVERY row (all decision types), int32 coords.
    keys = raw[['tile_id', 'pixel_row', 'pixel_col']].copy()
    keys['pixel_row'] = keys['pixel_row'].astype(np.int32)
    keys['pixel_col'] = keys['pixel_col'].astype(np.int32)
    suppression_keys = pd.MultiIndex.from_frame(keys).unique()

    neg = (raw['negative_of'].fillna('').astype(str) if 'negative_of' in raw.columns
           else pd.Series([''] * len(raw), index=raw.index))
    mineral_hit = (raw[_REASSIGN_MINERAL_COLS] > 0.5).any(axis=1)
    other = (raw['other'] if 'other' in raw.columns
             else pd.Series(0.0, index=raw.index))

    parts = []
    reassign_min = raw.loc[(neg == '') & mineral_hit]
    if len(reassign_min):
        parts.append(_stamp_7cls_cols(reassign_min, bland=0.0, junk=0.0,
                                      alteration=None, zero_plag=False))
    reassign_bland = raw.loc[(neg == '') & (~mineral_hit) & (other > 0.5)]
    if len(reassign_bland):
        parts.append(_stamp_7cls_cols(reassign_bland, bland=1.0, junk=0.0,
                                      alteration=0.0))
    alt = raw.loc[neg == 'alteration']
    if len(alt):
        parts.append(_stamp_7cls_cols(alt, bland=0.0, junk=0.0, alteration=1.0))
    amb = raw.loc[neg == 'ambiguous']
    if len(amb):
        parts.append(_stamp_7cls_cols(amb, bland=0.0, junk=1.0, alteration=0.0))
    # discards (neg not in {'', 'alteration', 'ambiguous'}) add no positive row.

    if not parts:
        return pd.DataFrame(), suppression_keys
    positives = pd.concat(parts, ignore_index=True)
    positives = _fill_confidence_defaults(positives)
    positives = _per_polygon_cap(positives, MAX_PX_PER_POLYGON, SEED + 500)
    return positives, suppression_keys


def _apply_ndviz(out: pd.DataFrame, ndviz_dir: str) -> pd.DataFrame:
    """Apply the ndviz relabel session's pixel-level supersede to the combined
    frame: anti-join every ndviz pixel out of ``out`` (all decision types), then
    append ndviz's own positive rows. Absent dir -> ``out`` unchanged (no-op).

    Must run AFTER the fragments concat and BEFORE the joint re-split so the
    ndviz positives get their split assigned in the joint pass. The append
    happens after the anti-join, so ndviz positives are never suppressed by
    their own keys. MTRDR synth rows are injected at train time (not in this
    parquet), so the anti-join cannot touch them.
    """
    positives, suppression_keys = load_ndviz_relabels(ndviz_dir)
    if len(suppression_keys) == 0 and (positives is None or positives.empty):
        print('  ndviz: no relabel session (no-op)')
        return out

    if len(suppression_keys) > 0:
        n_before = len(out)
        key = pd.MultiIndex.from_arrays([
            out['tile_id'].to_numpy(),
            out['pixel_row'].to_numpy().astype(np.int32),
            out['pixel_col'].to_numpy().astype(np.int32),
        ])
        out = out.loc[~key.isin(suppression_keys)].reset_index(drop=True)
        print(f'  ndviz suppression: dropped {n_before - len(out):,} rows '
              f'({len(suppression_keys):,} superseded pixels)')

    if positives is not None and not positives.empty:
        for c in out.columns:
            if c not in positives.columns:
                positives = positives.copy()
                positives[c] = np.float32(0.0) if c not in (
                    'tile_id', 'split', 'confidence_tier', 'negative_of') else ''
        out = pd.concat([out, positives[out.columns.tolist()]],
                        ignore_index=True)
        print(f'  ndviz positives added: {len(positives):,} rows')
    return out


def _joint_resplit(out: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Jointly re-split the concatenated (all-sources) frame, in place.

    Each source loader assigns splits independently, so nearby same-class
    polygons from different sources (e.g. base-gpkg alteration vs confirmed
    co-occurring alteration vs dedicated mc11 review tags) can straddle
    train/val — cross-source leakage the per-source splitters cannot see.
    One unit-balanced pass over the union clusters polygons across sources
    into shared geographic units and overrides 'split' for every row. Only
    'split' is touched; label columns (incl. 'other' mirroring 'bland') are
    left as-is. Returns the same frame for convenience.
    """
    out['split'] = assign_unit_balanced_splits(out, JOINT_BALANCE_COLS, seed)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base_parquet',  default=DEFAULT_BASE_PARQUET)
    ap.add_argument('--confirmed_dir', nargs='+', default=DEFAULT_CONFIRMED_DIRS,
                    help='One or more confirmed_pixels dirs (review sessions '
                         'are additive)')
    ap.add_argument('--hn_dir',        nargs='+', default=DEFAULT_HN_DIRS,
                    help='One or more hard_negatives dirs')
    ap.add_argument('--ndviz_dir',     default=DEFAULT_NDVIZ_DIR,
                    help='N-D visualizer relabel session (review-format rows). '
                         'Pixel-level supersede: every ndviz pixel replaces the '
                         'lower-precedence sources, then ndviz positives are '
                         'appended. Absent dir = no-op.')
    ap.add_argument('--out',           default=DEFAULT_OUT)
    ap.add_argument('--hand_minerals', choices=['all', 'plag_only', 'none'],
                    default='all',
                    help="Hand-drawn (gpkg) mineral label policy. 'all' "
                         "(default): keep every hand mineral label. 'plag_only': "
                         "keep only hand plagioclase (mafic cols zeroed on kept "
                         "rows), drop hand olivine/lcp/hcp. 'none': drop all "
                         "non-bland gpkg rows. Bland tiles unaffected in all "
                         "cases.")
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
    base = _build_base(args.base_parquet, n_bland, hand_minerals=args.hand_minerals)

    # ── 2. MC13 confirmed mineral positives ───────────────────────────────────
    print('\nLoading confirmed mineral positives …')
    confirmed = load_confirmed_mineral_positives(args.confirmed_dir, base)

    # ── 3. Bland review sources ───────────────────────────────────────────────
    print('\nLoading bland review sources …')
    mc13_bland = load_bland_review(args.hn_dir, 'mc13_blands', mc13=True,  seed_offset=10, n_bland=n_bland)
    mc11_bland = load_bland_review(args.hn_dir, 'mc11_blands', mc13=False, seed_offset=20, n_bland=n_bland)

    # ── 3b. Reassigned minerals (reject→mineral) ─────────────────────────────
    print('\nLoading reassigned mineral positives …')
    reassigned = load_reassigned_minerals(args.hn_dir)

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
                         ('reassigned', reassigned),
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

    # ── 6a. ndviz relabel session: pixel-level supersede ─────────────────────
    # Anti-join every ndviz pixel out of the combined frame (all decision
    # types), then append ndviz's own positive rows. Runs BEFORE the joint
    # re-split so the appended positives get a split in that pass. Absent dir =
    # no-op. (Loaded here — right before use — so its counts print in context.)
    print('\nApplying ndviz relabel session (pixel-level supersede) …')
    out = _apply_ndviz(out, args.ndviz_dir)

    # ── 6b. Joint re-split across sources ────────────────────────────────────
    # The per-source split assignments above are provisional diagnostics only
    # (their per-source fraction prints remain useful); this single
    # unit-balanced pass over the combined frame overrides 'split' so nearby
    # same-class polygons from different sources share one unit and one split
    # (fixes cross-source alteration leakage).
    out = _joint_resplit(out)
    print(f'\nJoint re-split over combined frame ({len(out):,} rows) applied.')

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print('\n=== 7-class dataset summary ===')
    frac_cols = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                 'bland', 'alteration', 'junk']
    label_cols = [c for c in frac_cols if c != 'olivine_t2']
    for split in ('train', 'val', 'test'):
        sub = out[out['split'] == split]
        print(f'\n{split}: {len(sub):,} rows')
        for c in label_cols:
            if c in sub.columns:
                n = int((sub[c] > 0.5).sum())
                print(f'  {c:>14}: {n:>9,}')
        if 'confidence_tier' in sub.columns:
            print(f'  tiers: {sub["confidence_tier"].value_counts().to_dict()}')

    # per-class × split positive-pixel table + achieved fractions over the
    # combined frame (the honest, unit-balanced holdout view).
    present = [c for c in frac_cols if c in out.columns]
    counts = pd.DataFrame(
        {s: [(out.loc[out['split'] == s, c] > 0.5).sum() for c in present]
         for s in ('train', 'val', 'test')},
        index=present)
    print('\nPer-class × split positive-pixel counts:')
    print(counts.to_string())
    print('\nAchieved per-class split fractions (combined frame):')
    print(achieved_fractions(out, out['split'], present).to_string())

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
