"""Merge mined dust hard negatives into the 7-class training parquet.

Runs on HPC, where mrral_pixels_7cls_handcore.parquet lives. Reads that file's
schema rather than assuming it, labels every mined pixel bland, and delegates
split assignment to split_units.assign_unit_balanced_splits over the CONCATENATED
frame -- so a mined negative near a val unit is absorbed into that unit and
follows its split. Writes a NEW parquet; the input stays an input.

Three things about this seam are load-bearing and are each covered by a
regression test in tests/test_merge_hard_negatives.py:

  * SPECTRA COLUMN NAMES DIFFER between the two files. The miner writes
    `band_00..band_58` (self-describing, on purpose); the real target parquet
    names its spectra `m0..m58` and has ZERO `band_*` columns. The mapping is
    derived from BOTH schemas (`spectra_columns_of`) and is an error, never a
    silent zero-fill, when it cannot be established.
  * POLYGON_ID IS AN INTEGER COLUMN (int64 in data/mrral_pixels.parquet), so
    synthetic ids are integers offset above the base's max, not `dustneg_<n>`
    strings -- those raise ArrowInvalid at to_parquet.
  * SYNTHETIC POLYGONS ARE PER TILE, not per pixel. See
    `synthetic_polygon_ids` for the measured reason.

Spec: docs/superpowers/specs/2026-08-17-dust-hard-negatives-design.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.split_units import assign_unit_balanced_splits  # noqa: E402

BLAND_CANDIDATES = ('bland', 'other')
MINERAL_COLS = ('olivine', 'olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                'plagioclase', 'alteration', 'junk')
PIXEL_KEY = ('tile_id', 'pixel_row', 'pixel_col')

# The two spectra-column naming conventions in this repo. `m<N>` is what the
# real training parquets use (data/mrral_pixels.parquet -> the 7-class handcore
# build); `band_<NN>` is what the miner and some proxy frames write. Detection,
# never substitution: a schema that matches neither is an error.
SPECTRA_PATTERNS = (re.compile(r'^m(\d+)$'), re.compile(r'^band_(\d+)$'))

# The split seed the comparator parquet was built with
# (scripts/build_7cls_dataset.py:98, SEED = 42). Merging with a different seed
# re-shuffles the BASE units' train/val/test assignment for no reason, so the
# hard-negative arm would differ from its comparator in two variables instead
# of one.
DEFAULT_SEED = 42


def bland_column_of(columns) -> str:
    for c in BLAND_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(
        f'no bland column: tried {BLAND_CANDIDATES}, parquet has {list(columns)}')


def spectra_columns_of(columns) -> dict[int, str]:
    """Map band index -> spectra column name, derived from a schema.

    The band COUNT and the band NAMES both come from the data, never from a
    hard-coded 59: the target parquet is the authority on how many channels it
    carries and what they are called. Raises rather than returning an empty
    map, because the caller's only alternative to a real mapping is writing
    zeros into a training set.
    """
    matched = []
    for pat in SPECTRA_PATTERNS:
        hits = {int(m.group(1)): c for c in columns
                for m in (pat.fullmatch(str(c)),) if m}
        if hits:
            matched.append((pat.pattern, hits))
    if not matched:
        raise ValueError(
            'cannot identify the spectra columns: none of '
            f'{[p.pattern for p in SPECTRA_PATTERNS]} matched. Columns: '
            f'{list(columns)}')
    if len(matched) > 1:
        raise ValueError(
            'ambiguous spectra columns: the schema matches more than one '
            f'naming convention {[p for p, _ in matched]}. Columns: '
            f'{list(columns)}')
    hits = matched[0][1]
    idx = sorted(hits)
    if idx != list(range(len(idx))):
        raise ValueError(
            f'spectra columns are not a contiguous 0..N-1 run: got indices '
            f'{idx[:5]}..{idx[-5:]} ({len(idx)} columns)')
    return hits


def spectra_map(neg_columns, target_columns) -> dict[str, str]:
    """target spectra column -> the mined frame's column holding that band.

    This is the seam the mining and merging halves meet at. The miner's output
    stays self-describing (`band_NN`); the translation to whatever the target
    parquet calls its channels happens HERE, once, by index.
    """
    tgt = spectra_columns_of(target_columns)
    src = spectra_columns_of(neg_columns)
    missing = sorted(set(tgt) - set(src))
    if missing:
        raise ValueError(
            f'the mined frame has {len(src)} spectra columns but the target '
            f'parquet needs {len(tgt)}; missing band indices {missing[:10]}'
            f'{"..." if len(missing) > 10 else ""}. Mined columns are named '
            f'{src[min(src)]!r}.., target {tgt[min(tgt)]!r}..')
    return {tgt[i]: src[i] for i in tgt}


def bland_confidence_of(base: pd.DataFrame, bland_col: str) -> tuple[str, float]:
    """The (confidence_tier, confidence_weight) this target parquet's own bland
    rows use, so mined dust negatives read at train time as an ordinary bland
    row instead of an invented tier.

    Design spec (2026-08-17-dust-hard-negatives-design.md, lines 114-115):
    "confidence_weight / confidence_tier: match whatever the existing bland
    rows use, read from the target parquet. Do not invent a tier." A tier
    that matches no data/dataset.py WEIGHT_SCHEMES key falls back to the
    stamped confidence_weight, which is harmless only by coincidence under
    the scheme active today ('level', where high == the fallback == 1.0) and
    silently under-weights mined rows relative to real bland rows under any
    scheme that treats 'high' differently (e.g. 'hand_up').

    Picked by majority vote over bland_col > 0 rows: the tier borne by the
    most bland rows already in the file, then -- restricted to that tier --
    the most common confidence_weight. Ties are broken by sorting candidates
    (alphabetically for the tier, numerically ascending for the weight) so
    the choice is deterministic rather than dependent on row order.

    Raises ValueError rather than guessing when the base parquet has no
    confidence_tier column, or no rows with bland_col > 0 -- there is nothing
    to "match" in that case, and a made-up default is exactly the bug this
    function exists to avoid.
    """
    if 'confidence_tier' not in base.columns:
        raise ValueError(
            "base parquet has no 'confidence_tier' column; cannot match mined "
            "negatives to existing bland rows' confidence tier/weight")
    bland_rows = base[base[bland_col] > 0]
    if bland_rows.empty:
        raise ValueError(
            f'base parquet has no rows with {bland_col!r} > 0; cannot infer '
            'the confidence tier/weight mined negatives should carry -- '
            'refusing to invent one')

    tier_counts = Counter(bland_rows['confidence_tier'])
    top = max(tier_counts.values())
    tier = sorted(t for t, c in tier_counts.items() if c == top)[0]

    weight = 1.0
    if 'confidence_weight' in base.columns:
        same_tier = bland_rows.loc[bland_rows['confidence_tier'] == tier,
                                    'confidence_weight']
        same_tier = pd.to_numeric(same_tier, errors='coerce').dropna()
        if not same_tier.empty:
            weight_counts = Counter(same_tier)
            wtop = max(weight_counts.values())
            weight = sorted(w for w, c in weight_counts.items() if c == wtop)[0]
    return tier, float(weight)


def next_polygon_id(base: pd.DataFrame) -> int:
    """First synthetic polygon_id: one above the base parquet's largest.

    `polygon_id` is int64 in the real parquet (verified against
    data/mrral_pixels.parquet), so synthetic ids must be integers -- writing
    `dustneg_<n>` strings makes the merged frame's to_parquet raise
    ArrowInvalid("Could not convert 'dustneg_0' with type str: tried to
    convert to int64"). Offsetting above the max also guarantees no synthetic
    polygon collides with a real one, which would merge mined dust into a
    hand-labeled polygon's unit.
    """
    if 'polygon_id' not in base.columns or base.empty:
        return 0
    ids = pd.to_numeric(base['polygon_id'], errors='coerce').dropna()
    return int(ids.max()) + 1 if not ids.empty else 0


def synthetic_polygon_ids(neg_df: pd.DataFrame, start_id: int) -> np.ndarray:
    """One synthetic polygon PER TILE, numbered from `start_id`.

    Why per tile and not per pixel (which is what the spec's "one per thinned
    cluster" naively reduces to, since thinning leaves single pixels):

    `polygon_units` runs SINGLE-LINKAGE clustering at 0.25 deg over polygon
    centroids. A per-pixel synthetic polygon has the pixel itself as its
    centroid, and the miner's `--min_sep 5` leaves mined pixels ~0.017 deg
    apart -- 15x inside the linkage radius -- so they chain. Chaining is
    transitive, so the chain does not stop at the tile edge: it walks the
    whole tile (5 deg) and then hops to the adjacent tile's mined pixels,
    swallowing every labeled polygon it passes within 0.25 deg on the way.
    Measured on 3 adjacent tiles x 900 mined px + one real labeled polygon:
    ONE unit of 3,500 rows, which assign_unit_balanced_splits then handed
    wholesale to `val`. At 175 tiles that is the entire mined set plus every
    nearby hand label in a single split -- val/test destroyed.

    Per tile, the centroid is the mean of that tile's mined pixels, i.e. near
    the tile center. Tile centers are 5 deg apart, 20x the linkage radius, so
    per-tile centroids cannot chain the corpus together. Two adjacent tiles
    whose mined pixels both hug their shared edge can still merge -- and
    should: that is genuinely contiguous terrain, and it merges 2 units, not
    N. Mined pixels near a hand-labeled polygon still get absorbed into that
    polygon's unit and follow its split, which is the leakage guard the spec
    asks for.

    Determinism: ids are handed out in sorted tile order, so the same mined
    parquet always produces the same ids regardless of row order.
    """
    if 'tile_id' not in neg_df.columns:
        raise ValueError(
            "mined frame has no 'tile_id' column; cannot group mined pixels "
            'into per-tile synthetic polygons')
    tiles = neg_df['tile_id'].astype(str).to_numpy()
    order = {t: start_id + i for i, t in enumerate(sorted(pd.unique(tiles)))}
    return np.fromiter((order[t] for t in tiles), dtype=np.int64,
                       count=len(tiles))


def drop_pixels_present_in_labels(neg: pd.DataFrame, base: pd.DataFrame
                                  ) -> tuple[pd.DataFrame, int]:
    """Anti-join: drop mined pixels that the labeled parquet already carries.

    Design spec lines 86-88, "Exclusion -- labels": never contradict a hand
    label. The MINER cannot enforce this -- it runs locally, where the training
    parquet does not exist -- so the merge must. 34 of the 83 labeled tiles are
    also mined tiles, so without this a hand-labeled lcp pixel arrives a second
    time as a `bland` negative and the two rows fight each other in the loss.

    Returns (kept, n_dropped). Keyed on the physical
    (tile_id, pixel_row, pixel_col), with coordinates coerced to a common
    integer dtype so an int32/int64 mismatch between the two files cannot make
    the anti-join silently match nothing.
    """
    missing = [c for c in PIXEL_KEY
               if c not in neg.columns or c not in base.columns]
    if missing:
        raise ValueError(
            f'cannot anti-join mined pixels against the hand labels: column(s) '
            f'{missing} absent from the mined frame or the labeled parquet')

    def _key(df):
        return pd.MultiIndex.from_arrays([
            df['tile_id'].astype(str).to_numpy(),
            pd.to_numeric(df['pixel_row']).to_numpy().astype(np.int64),
            pd.to_numeric(df['pixel_col']).to_numpy().astype(np.int64),
        ])

    hit = _key(neg).isin(_key(base))
    return neg.loc[~hit].reset_index(drop=True), int(hit.sum())


def build_negative_rows(neg_df, target_columns, bland_col, start_id: int,
                         confidence_tier: str, confidence_weight: float):
    """Mined pixels as rows matching `target_columns` exactly, labelled bland.

    `confidence_tier`/`confidence_weight` must come from `bland_confidence_of`
    (or an equivalent read of the target parquet's own bland rows) -- see that
    function's docstring for why hard-coding either is a bug.

    `start_id` is the first synthetic polygon_id; pass `next_polygon_id(base)`.
    """
    n = len(neg_df)
    target_columns = list(target_columns)
    # Raises when the target's spectra columns cannot be identified: writing
    # zeros there would put ~10^5 all-zero "spectra" into a training set,
    # labelled bland, and nothing downstream would notice.
    smap = spectra_map(neg_df.columns, target_columns)
    poly = synthetic_polygon_ids(neg_df, start_id)

    out = pd.DataFrame(index=range(n))
    for col in target_columns:
        if col in smap:
            out[col] = neg_df[smap[col]].to_numpy()
        elif col in PIXEL_KEY:
            out[col] = neg_df[col].to_numpy() if col in neg_df.columns else 0.0
        elif col == 'polygon_id':
            out[col] = poly
        elif col in BLAND_CANDIDATES:
            # Both when the schema carries both: the 7-class build keeps
            # 'other' mirroring 'bland' (build_7cls_dataset._stamp_7cls_cols),
            # so a mined row with bland=1, other=0 would be the only
            # inconsistent bland row in the file.
            out[col] = np.ones(n, dtype=np.float32)
        elif col in MINERAL_COLS:
            out[col] = np.zeros(n, dtype=np.float32)
        elif col == 'split':
            out[col] = pd.Series([pd.NA] * n, dtype='object')
        elif col == 'confidence_weight':
            out[col] = np.full(n, confidence_weight, dtype=np.float32)
        elif col == 'confidence_tier':
            out[col] = confidence_tier
        else:
            out[col] = np.zeros(n, dtype=np.float32)
    return out[target_columns]


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (split out so tests can assert on defaults)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', required=True, help='training parquet (input, untouched)')
    ap.add_argument('--negatives', required=True, help='parquet from mine_dust_hard_negatives')
    ap.add_argument('--out', required=True, help='NEW parquet to write')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED,
                    help='split seed (default: %(default)s, matching '
                         'build_7cls_dataset.SEED, which built the comparator '
                         'parquet; a different seed re-shuffles the BASE units)')
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    base = pd.read_parquet(args.labels)
    neg = pd.read_parquet(args.negatives)
    bland_col = bland_column_of(base.columns)
    conf_tier, conf_weight = bland_confidence_of(base, bland_col)
    print(f'base {len(base):,} rows; bland column is {bland_col!r}; '
          f'confidence_tier={conf_tier!r} confidence_weight={conf_weight}; '
          f'{len(neg):,} mined negatives')

    # Never contradict a hand label (spec lines 86-88).
    neg, n_dropped = drop_pixels_present_in_labels(neg, base)
    print(f'anti-join vs hand labels: dropped {n_dropped:,} mined pixels already '
          f'in the labeled parquet; {len(neg):,} negatives remain')
    if neg.empty:
        raise SystemExit('every mined pixel is already hand-labeled; nothing to merge')

    start_id = next_polygon_id(base)
    rows = build_negative_rows(neg, base.columns, bland_col, start_id=start_id,
                                confidence_tier=conf_tier,
                                confidence_weight=conf_weight)
    n_poly = rows['polygon_id'].nunique() if 'polygon_id' in rows.columns else 0
    print(f'synthetic polygons: {n_poly:,} (one per mined tile), '
          f'polygon_id from {start_id:,}')
    merged = pd.concat([base, rows], ignore_index=True)

    label_cols = [c for c in base.columns
                  if c in MINERAL_COLS or c == bland_col]
    merged['split'] = assign_unit_balanced_splits(merged, label_cols, seed=args.seed)
    print('split distribution after reassignment:')
    print(merged['split'].value_counts())
    print('mined-negative split distribution:')
    print(merged.iloc[len(base):]['split'].value_counts())

    merged.to_parquet(args.out, index=False)
    print(f'wrote {args.out}: {len(merged):,} rows')


if __name__ == '__main__':
    main()
