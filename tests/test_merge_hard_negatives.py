"""Merging mined negatives into the training parquet: schema and labels.

The fixture schema here MIRRORS THE REAL TARGET PARQUET
(`data/mrral_pixels.parquet`, from which the 7-class handcore build derives):
spectra live in `m0..m58`, and `polygon_id` is an INTEGER column. An earlier
version of this file invented a `band_00..band_58` base schema, which made the
merge's `band_*` pass-through look correct while, against the real parquet,
every `m*` column fell to the catch-all and was written as 0.0 -- ~10^5
all-zero spectra labelled bland. Do not "simplify" the fixture back.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.merge_hard_negatives import (
    bland_column_of, bland_confidence_of, build_negative_rows)
from scripts.split_units import polygon_units

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Real schemas ─────────────────────────────────────────────────────────────
# 7-class handcore: base parquet columns + the 7-class stamps (bland/junk).
# Verified against data/mrral_pixels.parquet (73 cols, polygon_id int64,
# spectra m0..m58) plus scripts/build_7cls_dataset.py::_stamp_7cls_cols.
SEVEN = (['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
         + [f'm{i}' for i in range(59)]
         + ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other',
            'confidence_weight', 'confidence_tier', 'split', 'alteration',
            'bland', 'junk'])
# The older 5/6-class local proxy: no 'bland'/'junk'; the bland class is 'other'.
FIVE = [c for c in SEVEN if c not in ('bland', 'junk')]

MINERALS_IN_FIXTURE = ('olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                       'alteration', 'junk')

# Real tile ids (present in data/tile_centers.csv) from the mined MC quadrants.
# t1082/t1083/t1084 are ADJACENT: centers 5 deg apart in longitude.
T = ('t1082', 't1083', 't1084')

# The miner's own output columns (scripts/mine_dust_hard_negatives.py).
MINED_MRRSU = ('R770', 'RBR', 'RPEAK1', 'OLINDEX3', 'BD1300', 'LCPINDEX2',
               'HCPINDEX2', 'BD1900_2', 'BD2210_2', 'D2300')


def _neg(n=3, tile='t1082', rows=None, cols=None):
    """A miner-shaped frame: band_00..band_58 + the audit mrrsu columns."""
    d = {'tile_id': [tile] * n,
         'pixel_row': np.arange(n) if rows is None else np.asarray(rows),
         'pixel_col': np.arange(n) if cols is None else np.asarray(cols)}
    for b in range(59):
        d[f'band_{b:02d}'] = np.full(n, 0.1 + 0.001 * b, dtype=np.float32)
    for name in MINED_MRRSU:
        d[name] = np.full(n, 6.0)
    return pd.DataFrame(d)


def _base(bland_tiers, bland_weights=None, bland_col='bland',
          columns=None, other_tier='High', other_weight=1.0):
    """A minimal base parquet: some bland rows with the given
    (tier[, weight]) values, plus one non-bland row so `bland_col > 0`
    filtering is exercised rather than trivially satisfied by every row."""
    if columns is None:
        columns = SEVEN
    n_bland = len(bland_tiers)
    n = n_bland + 1
    if bland_weights is None:
        bland_weights = [1.0] * n_bland
    d = {}
    for col in columns:
        if col == bland_col or (bland_col == 'bland' and col == 'other'):
            d[col] = [1.0] * n_bland + [0.0]
        elif col in MINERALS_IN_FIXTURE:
            d[col] = [0.0] * n_bland + [1.0]
        elif col == 'confidence_tier':
            d[col] = list(bland_tiers) + [other_tier]
        elif col == 'confidence_weight':
            d[col] = list(bland_weights) + [other_weight]
        elif col == 'tile_id':
            d[col] = ['t1082'] * n
        elif col == 'split':
            d[col] = ['train'] * n
        elif re.fullmatch(r'm\d+', col):
            d[col] = np.full(n, 0.2, dtype=np.float64)
        else:
            d[col] = list(range(n))
    return pd.DataFrame(d)


def test_bland_column_is_detected_in_both_vocabularies():
    """The 7-class build calls it 'bland'; older parquets call it 'other'.
    Hard-coding either silently mislabels every mined pixel."""
    assert bland_column_of(SEVEN) == 'bland'
    assert bland_column_of(FIVE) == 'other'


def test_missing_bland_column_raises():
    with pytest.raises(ValueError, match='bland'):
        bland_column_of(['tile_id', 'olivine', 'lcp'])


def test_rows_are_labelled_bland_and_nothing_else():
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert (out['bland'] == 1).all()
    for c in MINERALS_IN_FIXTURE:
        assert (out[c] == 0).all(), f'{c} must be 0 on a dust negative'


def test_other_mirrors_bland_when_the_schema_carries_both():
    """The 7-class build keeps 'other' mirroring 'bland' for backward compat
    (build_7cls_dataset._stamp_7cls_cols). A mined row with bland=1 and
    other=0 is internally inconsistent with every real bland row in the file."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert (out['other'] == 1).all()


def test_output_columns_match_the_target_schema_exactly():
    """A column order or set mismatch makes the concat produce NaN columns that
    train silently as zeros."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert list(out.columns) == SEVEN


# ── Finding 2: the miner's band_NN must reach the target's m<N> ──────────────


def test_target_spectra_columns_receive_the_mined_bands():
    """THE SEAM. The miner writes band_00..band_58; the target parquet's
    spectra columns are m0..m58 and it has ZERO band_* columns. A pass-through
    keyed on 'band_' leaves every m* to the catch-all, writing 0.0 -- ~10^5
    all-zero spectra labelled bland, which train silently."""
    neg = _neg(2)
    neg['band_07'] = [0.31, 0.42]
    neg['band_58'] = [0.11, 0.12]
    out = build_negative_rows(neg, SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['m7'].tolist() == pytest.approx([0.31, 0.42])
    assert out['m58'].tolist() == pytest.approx([0.11, 0.12])
    spectra = [f'm{i}' for i in range(59)]
    assert (out[spectra].to_numpy() != 0).all(), 'no mined spectrum may be all-zero'


def test_spectra_width_comes_from_the_target_schema_not_a_hardcoded_59():
    """Derive the mapping from the schema. A hardcoded 59 breaks the day the
    band subset changes, and breaks silently (zeros), not loudly."""
    narrow = [c for c in SEVEN if not re.fullmatch(r'm\d+', c)]
    narrow = narrow[:4] + [f'm{i}' for i in range(30)] + narrow[4:]
    out = build_negative_rows(_neg(2), narrow, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert list(out.columns) == narrow
    assert (out[[f'm{i}' for i in range(30)]].to_numpy() != 0).all()


def test_unidentifiable_target_spectra_columns_fail_loudly():
    """Failing loudly beats writing 10^5 zero spectra into a training set."""
    blind = [c for c in SEVEN if not re.fullmatch(r'm\d+', c)]
    with pytest.raises(ValueError, match='spectra'):
        build_negative_rows(_neg(2), blind, 'bland', start_id=0,
                             confidence_tier='High', confidence_weight=1.0)


def test_a_band_target_schema_still_works():
    """Some proxy parquets really do use band_NN. Detection, not substitution."""
    banded = ([c for c in SEVEN if not re.fullmatch(r'm\d+', c)][:4]
              + [f'band_{i:02d}' for i in range(59)]
              + [c for c in SEVEN if not re.fullmatch(r'm\d+', c)][4:])
    neg = _neg(2)
    neg['band_07'] = [0.31, 0.42]
    out = build_negative_rows(neg, banded, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['band_07'].tolist() == pytest.approx([0.31, 0.42])


# ── Finding 1: polygon_id must be an integer, above the base's max ───────────


def test_polygon_ids_are_integers_not_strings():
    """polygon_id is int64 in the real parquet. `dustneg_<n>` strings make the
    merged frame's to_parquet raise
    ArrowInvalid: Could not convert 'dustneg_0' ... to int64."""
    out = build_negative_rows(_neg(4, tile='t1082'), SEVEN, 'bland',
                               start_id=100, confidence_tier='High',
                               confidence_weight=1.0)
    assert pd.api.types.is_integer_dtype(out['polygon_id']), \
        f'polygon_id dtype is {out["polygon_id"].dtype}, must be integer'
    assert out['polygon_id'].min() >= 100


def test_merged_frame_round_trips_through_parquet(tmp_path):
    """The exact production crash, in miniature: concat base + mined rows and
    write it, the way main() does."""
    base = _base(bland_tiers=['High'] * 3)
    rows = build_negative_rows(_neg(4), base.columns, 'bland', start_id=1000,
                                confidence_tier='High', confidence_weight=1.0)
    merged = pd.concat([base, rows], ignore_index=True)
    merged['split'] = 'train'
    merged.to_parquet(tmp_path / 'merged.parquet', index=False)
    back = pd.read_parquet(tmp_path / 'merged.parquet')
    assert pd.api.types.is_integer_dtype(back['polygon_id'])


# ── Finding 3: synthetic polygons must not collapse the split geometry ──────


def _mined_grid(tiles=T, stride=50, start=25, n_side=30):
    """A realistic thinned mining result: a grid spanning each 1500x1500 tile.
    stride 50 px ~ 0.167 deg < the 0.25 deg single-linkage radius, so per-pixel
    polygons chain across the whole tile and then across tile boundaries."""
    coords = [start + stride * i for i in range(n_side)]
    frames = []
    for t in tiles:
        rr, cc = np.meshgrid(coords, coords, indexing='ij')
        frames.append(_neg(rr.size, tile=t, rows=rr.ravel(), cols=cc.ravel()))
    return pd.concat(frames, ignore_index=True)


def test_mined_pixels_do_not_collapse_into_a_single_split_unit():
    """MEASURED COLLAPSE. One synthetic polygon PER PIXEL makes polygon_units'
    single-linkage chain every mined pixel (and every labeled polygon within
    0.25 deg) into ONE unit, which assign_unit_balanced_splits then hands
    wholesale to a single split -- destroying val/test."""
    neg = _mined_grid()
    base = _base(bland_tiers=['High'] * 3)
    rows = build_negative_rows(neg, base.columns, 'bland', start_id=9000,
                                confidence_tier='High', confidence_weight=1.0)
    merged = pd.concat([base, rows], ignore_index=True)
    units = polygon_units(merged)
    mined_units = units.iloc[len(base):]
    sizes = units.value_counts()
    largest = sizes.iloc[0] / len(merged)
    assert mined_units.nunique() >= len(T), (
        f'mined pixels occupy {mined_units.nunique()} unit(s) across {len(T)} '
        f'tiles; whole-corpus chaining collapsed them')
    assert largest < 0.6, (
        f'largest unit holds {largest:.0%} of all rows -- one split would take '
        f'the entire mined set')


def test_one_synthetic_polygon_per_tile():
    """The grouping choice: mined pixels are grouped per TILE. Tile centers are
    5 deg apart, 20x the 0.25 deg linkage radius, so per-tile centroids cannot
    chain the corpus together the way per-pixel centroids do."""
    neg = _mined_grid()
    out = build_negative_rows(neg, SEVEN, 'bland', start_id=500,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['polygon_id'].nunique() == len(T)
    per_tile = out.groupby(neg['tile_id'].to_numpy())['polygon_id'].nunique()
    assert (per_tile == 1).all(), 'a tile must map to exactly one synthetic polygon'
    # distinct tiles must not share an id, or two tiles merge into one polygon
    assert out.groupby(out['polygon_id'])['tile_id'].nunique().eq(1).all()


def test_split_is_not_assigned_here():
    """Splits come from assign_unit_balanced_splits over the CONCATENATED frame.
    Writing 'train' here would put dust from val terrain into train."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['split'].isna().all()


# --- confidence_tier / confidence_weight must match the base parquet's own
# bland rows (design spec lines 114-115), not an invented value. ---


def test_bland_confidence_of_matches_the_only_tier_present():
    base = _base(bland_tiers=['High', 'High', 'High'], bland_weights=[1.0, 1.0, 1.0])
    tier, weight = bland_confidence_of(base, 'bland')
    assert tier == 'High'
    assert weight == pytest.approx(1.0)


def test_bland_confidence_of_picks_majority_tier_when_mixed():
    """Real bland rows in a target parquet are not guaranteed to share one
    tier. The majority tier is the sensible representative; picking the wrong
    one would still be an invented value, just a differently-invented one."""
    base = _base(bland_tiers=['Moderate', 'High', 'High', 'High'],
                 bland_weights=[0.85, 1.0, 1.0, 1.0])
    tier, weight = bland_confidence_of(base, 'bland')
    assert tier == 'High'
    assert weight == pytest.approx(1.0)


def test_bland_confidence_of_breaks_ties_deterministically():
    """2 vs 2 tie between 'High' and 'Low' -- alphabetically 'High' sorts
    first. The specific rule matters less than that it never depends on row
    order (i.e. is not "whichever tier happened to appear first")."""
    base_a = _base(bland_tiers=['High', 'High', 'Low', 'Low'],
                   bland_weights=[1.0, 1.0, 0.70, 0.70])
    base_b = _base(bland_tiers=['Low', 'Low', 'High', 'High'],
                   bland_weights=[0.70, 0.70, 1.0, 1.0])
    assert bland_confidence_of(base_a, 'bland') == bland_confidence_of(base_b, 'bland')
    tier, _ = bland_confidence_of(base_a, 'bland')
    assert tier == 'High'


def test_bland_confidence_of_raises_when_base_has_no_bland_rows():
    base = _base(bland_tiers=[])
    with pytest.raises(ValueError, match='bland'):
        bland_confidence_of(base, 'bland')


def test_bland_confidence_of_raises_when_no_confidence_tier_column():
    base = _base(bland_tiers=['High'])
    base = base.drop(columns=['confidence_tier'])
    with pytest.raises(ValueError, match='confidence_tier'):
        bland_confidence_of(base, 'bland')


def test_build_negative_rows_uses_the_supplied_confidence_not_a_fixed_value():
    """The core regression: mined rows must carry whatever
    (confidence_tier, confidence_weight) the caller determined from the base
    parquet's bland rows -- not a value invented inside build_negative_rows.
    Exercised at two different (tier, weight) pairs so a hard-coded constant
    of either kind cannot pass both."""
    out_a = build_negative_rows(_neg(3), SEVEN, 'bland', start_id=0,
                                 confidence_tier='Moderate', confidence_weight=0.85)
    assert (out_a['confidence_tier'] == 'Moderate').all()
    assert out_a['confidence_weight'].to_numpy() == pytest.approx([0.85, 0.85, 0.85])

    out_b = build_negative_rows(_neg(2), SEVEN, 'bland', start_id=0,
                                 confidence_tier='reviewed-legacy', confidence_weight=1.5)
    assert (out_b['confidence_tier'] == 'reviewed-legacy').all()
    assert out_b['confidence_weight'].to_numpy() == pytest.approx([1.5, 1.5])


def test_merged_negatives_end_to_end_adopt_base_bland_confidence():
    """Full base -> bland_confidence_of -> build_negative_rows chain, the way
    main() calls it: mined rows should be indistinguishable in confidence
    from the base parquet's real bland rows."""
    base = _base(bland_tiers=['Low', 'Low', 'Low'], bland_weights=[0.70, 0.70, 0.70],
                 other_tier='High', other_weight=1.0)
    bland_col = bland_column_of(base.columns)
    tier, weight = bland_confidence_of(base, bland_col)
    rows = build_negative_rows(_neg(3), base.columns, bland_col, start_id=0,
                                confidence_tier=tier, confidence_weight=weight)
    assert (rows['confidence_tier'] == 'Low').all()
    assert rows['confidence_weight'].to_numpy() == pytest.approx([0.70, 0.70, 0.70])


# ── Finding 4: never contradict a hand label (anti-join), end to end ─────────

N_OVERLAP = 120        # mined pixels that are ALSO hand-labeled lcp in t1083
N_LABELED_EXTRA = 280  # hand-labeled pixels the miner never touched
N_BASE_BLAND = 400


def _realistic_base(mined: pd.DataFrame) -> pd.DataFrame:
    """A base parquet with a real hand-labeled lcp polygon in t1083 that
    OVERLAPS the mined grid, plus a bland polygon in t1082.

    34 of the 83 labeled tiles are also mined tiles, so this overlap is the
    realistic case, not a contrived one: without an anti-join a hand-labeled
    lcp pixel arrives a second time as a bland negative."""
    t1083 = mined[mined['tile_id'] == 't1083'].reset_index(drop=True)
    overlap = t1083.iloc[:N_OVERLAP]
    rows = []
    # hand-labeled lcp: the overlapping pixels + pixels the miner never saw
    lcp_rows = pd.DataFrame({
        'tile_id': ['t1083'] * (N_OVERLAP + N_LABELED_EXTRA),
        'polygon_id': [7001] * (N_OVERLAP + N_LABELED_EXTRA),
        'pixel_row': list(overlap['pixel_row']) + [700 + i for i in range(N_LABELED_EXTRA)],
        'pixel_col': list(overlap['pixel_col']) + [701] * N_LABELED_EXTRA,
    })
    rows.append((lcp_rows, 'lcp'))
    bland_rows = pd.DataFrame({
        'tile_id': ['t1082'] * N_BASE_BLAND,
        'polygon_id': [7002] * N_BASE_BLAND,
        'pixel_row': [1000 + i for i in range(N_BASE_BLAND)],
        'pixel_col': [1001] * N_BASE_BLAND,
    })
    rows.append((bland_rows, 'bland'))

    frames = []
    for frag, cls in rows:
        n = len(frag)
        d = {}
        for col in SEVEN:
            if col in frag.columns:
                d[col] = frag[col].to_numpy()
            elif col in ('bland', 'other'):
                d[col] = np.full(n, 1.0 if cls == 'bland' else 0.0)
            elif col in MINERALS_IN_FIXTURE:
                d[col] = np.full(n, 1.0 if col == cls else 0.0)
            elif col == 'confidence_tier':
                d[col] = ['High'] * n
            elif col == 'confidence_weight':
                d[col] = np.full(n, 1.0)
            elif col == 'split':
                d[col] = ['train'] * n
            else:
                d[col] = np.full(n, 0.2)
        frames.append(pd.DataFrame(d))
    out = pd.concat(frames, ignore_index=True)
    out['polygon_id'] = out['polygon_id'].astype(np.int64)
    out['pixel_row'] = out['pixel_row'].astype(np.int64)
    out['pixel_col'] = out['pixel_col'].astype(np.int64)
    return out


@pytest.fixture(scope='module')
def merged_end_to_end(tmp_path_factory):
    """Run scripts/merge_hard_negatives.py for real, on realistic fixtures."""
    d = tmp_path_factory.mktemp('merge')
    mined = _mined_grid()
    base = _realistic_base(mined)
    labels_p, neg_p, out_p = d / 'base.parquet', d / 'neg.parquet', d / 'out.parquet'
    base.to_parquet(labels_p, index=False)
    mined.to_parquet(neg_p, index=False)
    r = subprocess.run(
        [sys.executable, 'scripts/merge_hard_negatives.py',
         '--labels', str(labels_p), '--negatives', str(neg_p), '--out', str(out_p)],
        cwd=PROJ, capture_output=True, text=True)
    assert r.returncode == 0, f'merge failed:\nSTDOUT\n{r.stdout}\nSTDERR\n{r.stderr}'
    return pd.read_parquet(out_p), base, mined, r.stdout


def test_hand_labeled_pixels_are_anti_joined_out(merged_end_to_end):
    """design.md:86-88 -- drop any mined (tile_id, pixel_row, pixel_col)
    already present in the labeled parquet. The miner cannot do it (no parquet
    locally), so the merge must."""
    merged, base, mined, _ = merged_end_to_end
    assert len(merged) == len(base) + len(mined) - N_OVERLAP, (
        f'{len(merged)} rows; expected {len(base)} + {len(mined)} - {N_OVERLAP} '
        f'(the mined pixels that are already hand-labeled)')
    key = ['tile_id', 'pixel_row', 'pixel_col']
    labeled_keys = set(map(tuple, base[key].to_numpy()))
    mined_only = merged[merged['polygon_id'] > int(base['polygon_id'].max())]
    assert not any(tuple(r) in labeled_keys for r in mined_only[key].to_numpy()), \
        'a hand-labeled pixel arrived a second time as a bland negative'


def test_end_to_end_polygon_ids_are_integers_above_the_base_max(merged_end_to_end):
    merged, base, _, _ = merged_end_to_end
    assert pd.api.types.is_integer_dtype(merged['polygon_id'])
    n_base = len(base)
    assert merged['polygon_id'].iloc[n_base:].min() > int(base['polygon_id'].max())


def test_end_to_end_mined_spectra_are_not_zero(merged_end_to_end):
    merged, base, _, _ = merged_end_to_end
    spectra = [f'm{i}' for i in range(59)]
    mined_rows = merged.iloc[len(base):]
    assert (mined_rows[spectra].to_numpy() != 0).all(), \
        'mined spectra written as zeros -- the band_NN -> m<N> seam is broken'


def test_end_to_end_splits_are_not_all_one_split(merged_end_to_end):
    """The collapse symptom at the level that matters: if every mined row lands
    in one split, val/test are destroyed."""
    merged, base, _, _ = merged_end_to_end
    mined_rows = merged.iloc[len(base):]
    assert mined_rows['split'].notna().all()
    assert merged['split'].nunique() >= 2


# ── Finding 6: the split seed must match the comparator ─────────────────────


def test_seed_defaults_to_42_like_the_comparator_build():
    """build_7cls_dataset.py:98 SEED = 42 built the comparator parquet. A
    different seed re-shuffles the base units' assignment for no reason."""
    from scripts.merge_hard_negatives import _build_parser
    assert _build_parser().get_default('seed') == 42


# ── Finding 5: the retrain arm's cache must be buildable ────────────────────

_CACHE_SLURM = os.path.join(PROJ, 'scripts', 'hpc_build_dualcr_labeled_cache.slurm')
_VAR_RE = re.compile(
    r'^(WORK_DIR|DATA_ROOT|DATA_DIR|PYTHON|PARQUET|RAW_CACHE|DUAL_CACHE|SPLITS)=')


def _slurm_vars(env_overrides=None):
    """Evaluate the cache job's variable-definition block in a real shell."""
    with open(_CACHE_SLURM) as fh:
        defs = [ln for ln in fh if _VAR_RE.match(ln)]
    script = 'set -u\n' + ''.join(defs) + \
        '\nprintf "%s\\n%s\\n%s\\n" "$PARQUET" "$RAW_CACHE" "$DUAL_CACHE"\n'
    env = dict(os.environ)
    env.update(env_overrides or {})
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().split('\n')


def test_cache_job_defaults_are_byte_identical_to_todays_values():
    """Every existing caller must behave exactly as before."""
    parquet, raw, dual = _slurm_vars()
    d = '/xdisk/sbyrne/phillipsm/CRISM_MRDR/crism_classification/data'
    assert parquet == f'{d}/mrral_pixels_7cls_handcore.parquet'
    assert raw == f'{d}/patch_cache_handcore'
    assert dual == f'{d}/patch_cache_handcore_dualcr'


def test_cache_job_paths_are_overridable_for_the_hardneg_arm():
    """hpc_finetune_dualcr_hardneg.slurm needs
    patch_cache_handcore_dualcr_hardneg built from the merged parquet. With
    the paths hardcoded, that cache can never be produced and the retrain
    aborts at its prerequisite check."""
    parquet, raw, dual = _slurm_vars({
        'PARQUET': '/x/mrral_pixels_7cls_handcore_hardneg.parquet',
        'RAW_CACHE': '/x/patch_cache_handcore_hardneg',
        'DUAL_CACHE': '/x/patch_cache_handcore_dualcr_hardneg',
    })
    assert parquet == '/x/mrral_pixels_7cls_handcore_hardneg.parquet'
    assert raw == '/x/patch_cache_handcore_hardneg'
    assert dual == '/x/patch_cache_handcore_dualcr_hardneg'
