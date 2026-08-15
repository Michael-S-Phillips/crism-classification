"""Build a rank-stratified polygon review set for calibrating the 6-class pyx model.

WHY STRATIFIED BY RANK
----------------------
Measured over 24 mc_deploy_pyx tiles the model's outputs are ranks, not
probabilities: `pyx` scores >= 0.5 on 81% of all valid pixels, while
`alteration` fires on 0.36% but saturates near 1.0 when it does. Sampling by
absolute threshold, or proportionally to population, would put nearly every
sampled polygon in the lowest band and tell us nothing about where the real
decision boundary sits.

Equal-N-per-stratum sampling instead lets the completed review estimate
PRECISION PER STRATUM. That vector of (stratum, n_reviewed, n_confirmed) pairs
is the input to a post-hoc calibration fit (isotonic / Platt) that maps the
model's ranks back onto probabilities. **The product of this review set is a
calibration curve, not just a polygon list** — which is why:

  - strata are log-spaced in (1 - p), so the top rungs (0.999, 0.9999) get
    their own estimates instead of being swallowed by a single ">0.99" bin;
  - each emitted gpkg layer IS a stratum, named `thresh_<lo>`, so the `layer`
    field the review app already writes into decisions.csv identifies the
    stratum with no extra bookkeeping;
  - a deliberate share of every cell is spectrally BLAND ground. Precision
    cannot be measured from positives alone, and the stated failure mode is
    pyx/olivine false-firing on flat spectra. See `--bland_share`.

SOURCE
------
Polygons come from the existing deployment vectorization at
`reports/mc_deploy_pyx/<mc>/<mineral>.gpkg`, which already carries the 8-rung
ladder [0.50 0.85 0.97 0.99 0.995 0.999 0.9995 0.9999]. The six stratum lower
bounds are all rungs of that ladder, so nothing needs re-vectorizing.

A polygon's stratum is decided by its own `mean_prob`, and its candidates are
drawn ONLY from the source layer whose threshold equals the stratum's lower
bound. That pairing means the polygon's geometry is exactly the connected
component at p >= lo, and its score lies in [lo, hi). It also makes exact
duplicate geometry across strata impossible: identical geometry in the next
layer up would require every pixel >= hi, which forces mean_prob >= hi and
excludes it from this stratum.

OUTPUT
------
Per-mineral GeoPackages the existing review app consumes unchanged:

    <out_dir>/olivine.gpkg      layers thresh_01_0.9999 ... thresh_06_0.50
    <out_dir>/pyx.gpkg
    <out_dir>/plagioclase.gpkg
    <out_dir>/alteration.gpkg
    <out_dir>/manifest.csv       one row per emitted polygon
    <out_dir>/allocation.csv     class x stratum table

Point the review app's "gpkg dir" at <out_dir> and give it a FRESH output dir
so its decisions.csv only contains this set.

Usage:
    conda run -n crism python scripts/build_review_set_stratified.py --dry_run
    conda run -n crism python scripts/build_review_set_stratified.py \
        --budget 600 --seed 20260815

Then in the review app: gpkg dir = data/vector_review_set_stratified_pyx,
output dir = a FRESH data/*review*/ so its decisions.csv holds only this set.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Iterable, Optional

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from data.continuum_removal import continuum_removed, good_band_mask_59
from scripts.threshold_names import fmt_threshold

N_BANDS = 59

# Log-spaced in (1 - p): the gaps 0.5 -> 0.15 -> 0.03 -> 0.01 -> 0.001 -> 0.0001
# each shrink by roughly a decade, so a saturated model's top decile is not one
# undifferentiated bin.  Half-open [lo, hi) everywhere except the last, which
# closes at 1.0 so a polygon at exactly p == 1.0 has a home.
STRATA: list[tuple[float, float]] = [
    (0.5, 0.85),
    (0.85, 0.97),
    (0.97, 0.99),
    (0.99, 0.999),
    (0.999, 0.9999),
    (0.9999, 1.0),
]

# The four mineral classes. bland/junk are the model's reject channels — they
# are not review targets, they are what the minerals get confused WITH.
MINERALS = ['olivine', 'pyx', 'plagioclase', 'alteration']

DEFAULT_CHARTS = ['mc11', 'mc13', 'mc26']
DEFAULT_SRC_DIR = os.path.join(PROJ, 'reports', 'mc_deploy_pyx')
# Named `vector_*` so the review app's gpkg-dir dropdown (_gpkg_dir_choices
# globs data/vector_*) discovers it without the "other — type a path" escape.
DEFAULT_OUT_DIR = os.path.join(PROJ, 'data', 'vector_review_set_stratified_pyx')
# Every review ledger the app has ever written lives at data/<something>/decisions.csv.
DEFAULT_DECISIONS_GLOB = os.path.join(PROJ, 'data', '*', 'decisions.csv')

BAND_COLS = [f'band_{b:02d}' for b in range(N_BANDS)]
# Column order the vectorizer writes; the review app + downstream tooling read
# these by name, so the emitted gpkg reproduces them exactly (extra provenance
# columns are appended after, never interleaved).
VECTORIZER_COLS = ['tile_id', 'mineral', 'threshold', 'count_px', 'mean_prob'] + BAND_COLS
EXTRA_COLS = ['mc', 'stratum', 'stratum_lo', 'stratum_hi',
              'band_depth', 'bland_pool', 'source_uid', 'review_uid']


# --------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------

def stratum_index(p: float) -> Optional[int]:
    """Index of the stratum containing score ``p``, or None if below 0.5/NaN.

    Half-open [lo, hi); the final stratum is closed at 1.0. A score of exactly
    0.99 therefore lands in [0.99, 0.999) -- NOT in [0.97, 0.99).
    """
    if p is None or not np.isfinite(p):
        return None
    for i, (lo, hi) in enumerate(STRATA):
        if i == len(STRATA) - 1:
            if lo <= p <= hi:
                return i
        elif lo <= p < hi:
            return i
    return None


def stratum_layer_name(i: int, rank_prefixed: bool = True) -> str:
    """Physical gpkg layer name for stratum ``i``.

    Rank-prefixed (`thresh_01_0.9999`) so QGIS stacks strictest-first; the
    review app canonicalises this to `thresh_0.9999` for polygon_uid via
    polygon_queue._canonical_layer, which uses the same shortest-round-trip
    formatter as fmt_threshold here. At two decimals `0.999` and `0.9999`
    would both render `1.00` and collide -- see scripts/threshold_names.py.
    """
    lo = STRATA[i][0]
    if not rank_prefixed:
        return f'thresh_{fmt_threshold(lo)}'
    rank = len(STRATA) - i          # stratum 5 (top) -> rank 01
    return f'thresh_{rank:02d}_{fmt_threshold(lo)}'


# --------------------------------------------------------------------------
# spectral flatness ("bland adjacency")
# --------------------------------------------------------------------------

def mean_band_depth(band_means: np.ndarray) -> np.ndarray:
    """Mean hull-continuum-removed band depth of polygon mean spectra.

    ``band_means``: (N, 59) the `band_00..band_58` columns the vectorizer
    stores — each polygon's mean mrral reflectance. Returns (N,) in [0, 1]:
    0.0 is perfectly flat (no absorptions at all), larger is deeper.

    Chosen over the model's own `bland` channel for two reasons: (1) it is
    already in the gpkg, so no 183-tile npz re-read is needed, and (2) using a
    model output to select the negatives that measure that same model's
    precision is circular — a spectrum the model wrongly thinks is
    mineral-bearing is exactly the one it would NOT flag as bland. The hull CR
    is the same one the review app's "continuum removed" checkbox draws, so
    what the selector calls flat is what the reviewer sees as flat.
    """
    band_means = np.asarray(band_means, dtype=np.float32)
    if band_means.ndim != 2 or band_means.shape[1] != N_BANDS:
        raise ValueError(f'expected (N, {N_BANDS}), got {band_means.shape}')
    cr = _hull_cr_excluding_bad_bands(band_means)
    good = good_band_mask_59()
    return (1.0 - cr[:, good].mean(axis=1)).astype(np.float64)


def _hull_cr_excluding_bad_bands(band_means: np.ndarray) -> np.ndarray:
    """Hull CR with band 0 kept OUT OF THE FIT.

    An upper hull is anchored by its extremes, so one artefact band sets the
    continuum for the whole spectrum. Band 0 (410.1 nm) carries the blue-edge
    artefact, and the vectorizer CLIPS it to CLIP_MAX = 0.5 rather than
    discarding it (the model pipeline instead marks such pixels INVALID, so it
    never sees them). Real deployed polygons therefore pair band_00 = 0.5000
    with band_01 = 0.0403, and fitting through that inflated the deepest
    apparent band from 0.043 to 0.416 -- a 10x error that would have made
    genuinely flat spectra look mineral-bearing and corrupted the bland-adjacent
    pool this function exists to select.

    Reuses the spectrum-viewer's port rather than duplicating the maths: it is
    numpy-only, and tests/test_plugin_cr_parity.py pins it against
    data/continuum_removal.py, so the two cannot drift apart silently.
    """
    import importlib.util
    port_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'qgis_plugins', 'crism_spectrum_viewer', 'crism_cr.py')
    spec = importlib.util.spec_from_file_location('_crism_cr_port', port_path)
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    return np.asarray(port.hull_cr(band_means,
                                   extra_exclude=port.bad_band_mask()))


# --------------------------------------------------------------------------
# adaptive budget allocation
# --------------------------------------------------------------------------

def equal_fill(capacity: dict, budget: int, priority_tiers: list) -> dict:
    """Fill each key toward an equal target, then redistribute the remainder.

    ``capacity[k]`` is how many candidates key k actually has. Every key first
    gets ``min(budget // n_keys, capacity[k])``. Whatever that leaves unspent —
    because some keys are empty or short — is redistributed: ``priority_tiers``
    is a list of key groups, and a tier is round-robined to exhaustion (of the
    leftover, or of that tier's remaining capacity) before the next tier gets
    anything. Keys not named in any tier are appended as a final tier, so no
    key is unreachable.

    Guarantees ``sum(result) == min(budget, sum(capacity))``.
    """
    keys = list(capacity)
    if not keys or budget <= 0:
        return {k: 0 for k in keys}
    target = budget // len(keys)
    alloc = {k: min(target, max(0, int(capacity[k]))) for k in keys}
    leftover = budget - sum(alloc.values())

    tiers = [[k for k in tier if k in capacity] for tier in priority_tiers]
    named = {k for tier in tiers for k in tier}
    tiers.append([k for k in keys if k not in named])

    for tier in tiers:
        if leftover <= 0:
            break
        while leftover > 0:
            progressed = False
            for k in tier:
                if leftover == 0:
                    break
                if alloc[k] < capacity[k]:
                    alloc[k] += 1
                    leftover -= 1
                    progressed = True
            if not progressed:      # tier saturated — spill to the next one
                break
    return alloc


def allocate_cells(capacity: dict, budget: int) -> dict:
    """Allocate ``budget`` across (mineral, stratum) cells.

    Redistribution goes TOP STRATUM FIRST, spilling downward only when a
    stratum's cells are saturated. That is where the decision boundary of a
    saturated model actually sits: plagioclase having nothing above 0.97 should
    buy extra olivine/pyx polygons at 0.999+, not shrink the set and not pad
    the [0.5, 0.85) bin that is already the easiest to estimate.
    """
    tiers = [[(m, s) for m in MINERALS] for s in range(len(STRATA) - 1, -1, -1)]
    return equal_fill(capacity, budget, tiers)


# --------------------------------------------------------------------------
# within-cell sampling
# --------------------------------------------------------------------------

def bland_pool_mask(cell: pd.DataFrame, bland_pool_frac: float) -> pd.Series:
    """Boolean Series (aligned to ``cell.index``) marking the flattest polygons.

    The bland-adjacent pool is the lowest ``bland_pool_frac`` of the cell by
    mean band depth. Deliberately RELATIVE to the cell rather than an absolute
    depth cut: at 0.9999 there may be no truly featureless polygons left, and a
    fixed cut would then silently return an empty pool for exactly the stratum
    whose precision matters most. Ranking is done on a cand_key-sorted view so
    ties resolve deterministically.
    """
    if cell.empty:
        return pd.Series([], dtype=bool, index=cell.index)
    c = cell.sort_values('cand_key', kind='mergesort')
    order = np.argsort(c['band_depth'].to_numpy(), kind='mergesort')
    n_pool = max(1, int(round(bland_pool_frac * len(c))))
    flags = np.zeros(len(c), dtype=bool)
    flags[order[:n_pool]] = True
    return pd.Series(flags, index=c.index).reindex(cell.index)


def _take(rng: np.random.Generator, pool: pd.DataFrame, n: int,
          chart_col: str = 'mc') -> list:
    """Sample ``n`` cand_keys from ``pool``, spread across charts.

    Charts get an equal target with the same redistribute-the-remainder rule as
    the class x stratum allocation, so a set is never all-Nili just because
    mc13 has the most polygons.
    """
    if n <= 0 or pool.empty:
        return []
    n = min(n, len(pool))
    charts = sorted(pool[chart_col].unique())
    cap = {c: int((pool[chart_col] == c).sum()) for c in charts}
    # Rotate the chart priority per call so the +1 remainders don't always land
    # on the alphabetically-first chart.
    rot = int(rng.integers(0, len(charts)))
    per_chart = equal_fill(cap, n, [charts[rot:] + charts[:rot]])
    picked: list = []
    for c in charts:
        k = per_chart[c]
        if k <= 0:
            continue
        uids = pool.loc[pool[chart_col] == c, 'cand_key'].to_numpy()
        picked.extend(rng.choice(uids, size=k, replace=False).tolist())
    return picked


def sample_cell(cell: pd.DataFrame, n: int, rng: np.random.Generator,
                bland_share: float = 0.35,
                bland_pool_frac: float = 0.30) -> list:
    """Pick ``n`` cand_keys from one (mineral, stratum) cell.

    ``bland_pool_frac`` of the cell — the flattest polygons by
    ``band_depth`` — is the bland-adjacent pool; ``bland_share`` of the sample
    is drawn from it, the rest from everything else. Both are guaranteed, and
    either side backfills from the other if it runs short, so the cell always
    yields min(n, len(cell)) polygons.

    Rows are ordered by cand_key before drawing so the result depends only on
    the seed, not on gpkg read order.
    """
    if n <= 0 or cell.empty:
        return []
    n = min(n, len(cell))
    cell = cell.sort_values('cand_key', kind='mergesort').reset_index(drop=True)
    cell = cell.assign(bland_pool=bland_pool_mask(cell, bland_pool_frac))

    n_bland = min(int(round(bland_share * n)), int(cell['bland_pool'].sum()))
    picked = _take(rng, cell[cell['bland_pool']], n_bland)
    picked += _take(rng, cell[~cell['bland_pool']], n - len(picked))
    if len(picked) < n:     # non-bland side was short — backfill from the pool
        remaining = cell[cell['bland_pool'] & ~cell['cand_key'].isin(picked)]
        picked += _take(rng, remaining, n - len(picked))
    return picked


# --------------------------------------------------------------------------
# candidate loading
# --------------------------------------------------------------------------

def source_uid(tile_id: str, stratum: int, index_in_layer: int) -> str:
    """polygon_uid the review app WOULD assign this polygon in its source gpkg.

    Mirrors polygon_queue: `{tile_id}::{canonical_layer}::{index_in_layer}`
    where index_in_layer is the file-order row index within the source layer.
    Used only to skip polygons already present in a decisions.csv.
    """
    return f'{tile_id}::thresh_{fmt_threshold(STRATA[stratum][0])}::{int(index_in_layer)}'


def _physical_layer_for(layers: Iterable[str], lo: float) -> Optional[str]:
    for name in layers:
        if not name.startswith('thresh_'):
            continue
        try:
            val = float(name.split('_')[-1])
        except ValueError:
            continue
        if abs(val - lo) < 1e-12:
            return name
    return None


def reviewed_uids(patterns: list[str]) -> set:
    """Union of polygon_uid over every decisions.csv matched by ``patterns``."""
    seen: set = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            try:
                df = pd.read_csv(path)
            except Exception as exc:                      # pragma: no cover
                print(f'  WARNING: unreadable decisions csv {path}: {exc}')
                continue
            if 'polygon_uid' in df.columns:
                seen |= set(df['polygon_uid'].astype(str))
    return seen


def load_candidates(src_dir: str, charts: list[str], minerals: list[str],
                    skip_uids: set, verbose: bool = True) -> pd.DataFrame:
    """Attribute-only pass over the source gpkgs -> one row per candidate.

    Geometry is deliberately NOT read here: 565k polygons x 59 bands is already
    heavy, and only the sampled few hundred need shapes. `load_selected_geoms`
    re-reads just the layers that were sampled from.
    """
    frames = []
    n_skipped = 0
    for mineral in minerals:
        for mc in charts:
            path = os.path.join(src_dir, mc, f'{mineral}.gpkg')
            if not os.path.exists(path):
                if verbose:
                    print(f'  missing {path} — skipping')
                continue
            layers = fiona.listlayers(path)
            for si, (lo, hi) in enumerate(STRATA):
                phys = _physical_layer_for(layers, lo)
                if phys is None:
                    continue
                df = gpd.read_file(path, layer=phys, ignore_geometry=True)
                if df.empty:
                    continue
                df = df.reset_index(drop=True)
                mp = df['mean_prob'].to_numpy(dtype=np.float64)
                keep = np.array([stratum_index(p) == si for p in mp])
                if not keep.any():
                    continue
                sub = df.loc[keep].copy()
                sub['index_in_layer'] = np.flatnonzero(keep)
                sub['mc'] = mc
                sub['stratum'] = si
                sub['stratum_lo'] = lo
                sub['stratum_hi'] = hi
                sub['physical_layer'] = phys
                sub['source_gpkg'] = path
                sub['source_uid'] = [
                    source_uid(t, si, i)
                    for t, i in zip(sub['tile_id'].astype(str), sub['index_in_layer'])
                ]
                before = len(sub)
                sub = sub[~sub['source_uid'].isin(skip_uids)]
                n_skipped += before - len(sub)
                if sub.empty:
                    continue
                # source_uid is the uid the review app would assign INSIDE one
                # mineral's gpkg, so the same string can name a different
                # polygon in another mineral. cand_key disambiguates for all
                # internal bookkeeping; source_uid is used only for the
                # already-reviewed check.
                sub['cand_key'] = mineral + '::' + sub['source_uid']
                sub['band_depth'] = mean_band_depth(sub[BAND_COLS].to_numpy())
                frames.append(sub)
            if verbose:
                print(f'  {mc}/{mineral}: '
                      f'{sum(len(f) for f in frames):,} candidates so far')
    if not frames:
        out = pd.DataFrame(columns=VECTORIZER_COLS + [
            'index_in_layer', 'mc', 'stratum', 'stratum_lo', 'stratum_hi',
            'physical_layer', 'source_gpkg', 'source_uid', 'cand_key',
            'band_depth'])
        out.attrs['n_skipped_reviewed'] = 0
        return out
    out = pd.concat(frames, ignore_index=True)
    out.attrs['n_skipped_reviewed'] = n_skipped
    return out


def load_selected_geoms(selected: pd.DataFrame) -> gpd.GeoDataFrame:
    """Re-read the sampled layers WITH geometry and attach shapes by position.

    Verifies that the geometry read sees the same tile_id/mean_prob at each
    recorded position as the attribute-only pass did; a mismatch would mean the
    two reads disagree about row order, which would silently attach the wrong
    polygon to a uid.
    """
    pieces = []
    for (path, phys), grp in selected.groupby(['source_gpkg', 'physical_layer'],
                                              sort=True):
        gdf = gpd.read_file(path, layer=phys).reset_index(drop=True)
        idx = grp['index_in_layer'].to_numpy()
        got = gdf.iloc[idx]
        exp_tid = grp['tile_id'].astype(str).to_numpy()
        if not np.array_equal(got['tile_id'].astype(str).to_numpy(), exp_tid):
            raise RuntimeError(
                f'row order drift between attribute and geometry reads of '
                f'{path}::{phys} — refusing to emit mismatched polygons')
        if not np.allclose(got['mean_prob'].to_numpy(),
                           grp['mean_prob'].to_numpy(), atol=1e-6):
            raise RuntimeError(
                f'mean_prob mismatch on re-read of {path}::{phys}')
        piece = grp.copy()
        piece['geometry'] = got.geometry.to_numpy()
        pieces.append(gpd.GeoDataFrame(piece, geometry='geometry', crs=gdf.crs))
    if not pieces:
        return gpd.GeoDataFrame(columns=VECTORIZER_COLS + ['geometry'],
                                geometry='geometry')
    crs = pieces[0].crs
    out = pd.concat(pieces, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry='geometry', crs=crs)


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def write_review_set(gdf: gpd.GeoDataFrame, out_dir: str,
                     verbose: bool = True) -> pd.DataFrame:
    """Write per-mineral gpkgs with one layer per stratum. Returns the manifest.

    The manifest carries `review_uid` — the polygon_uid PolygonQueue will
    generate for each emitted polygon — so a completed decisions.csv joins back
    to (mineral, stratum, band_depth) on one key.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []
    for mineral in MINERALS:
        sub = gdf[gdf['mineral'] == mineral]
        if sub.empty:
            continue
        path = os.path.join(out_dir, f'{mineral}.gpkg')
        if os.path.exists(path):
            os.remove(path)
        for si in sorted(sub['stratum'].unique(), reverse=True):
            cell = sub[sub['stratum'] == si].copy()
            # PolygonQueue indexes by file-order position; sort by tile then
            # source_uid so the emitted order (and therefore review_uid) is a
            # deterministic function of the selection, not of concat order.
            cell = cell.sort_values(['tile_id', 'source_uid'],
                                    kind='mergesort').reset_index(drop=True)
            canon = f'thresh_{fmt_threshold(STRATA[si][0])}'
            cell['review_uid'] = [
                f'{t}::{canon}::{i}'
                for i, t in enumerate(cell['tile_id'].astype(str))
            ]
            cell = cell[VECTORIZER_COLS + EXTRA_COLS + ['geometry']]
            layer = stratum_layer_name(si)
            cell.to_file(path, layer=layer, driver='GPKG')
            manifest_rows.append(pd.DataFrame(cell.drop(columns='geometry')).assign(
                gpkg=os.path.basename(path), layer=layer))
            if verbose:
                print(f'  {os.path.basename(path)}::{layer}: {len(cell)} polygons')
    if not manifest_rows:
        return pd.DataFrame()
    manifest = pd.concat(manifest_rows, ignore_index=True)
    manifest.to_csv(os.path.join(out_dir, 'manifest.csv'), index=False)
    return manifest


def allocation_table(df: pd.DataFrame, value: str = 'n') -> pd.DataFrame:
    """mineral x stratum pivot, strata as readable interval labels."""
    labels = {i: f'[{lo:g},{hi:g})' for i, (lo, hi) in enumerate(STRATA)}
    labels[len(STRATA) - 1] = f'[{STRATA[-1][0]:g},1.0]'
    tab = df.pivot_table(index='mineral', columns='stratum', values=value,
                         aggfunc='sum', fill_value=0)
    tab = tab.reindex([m for m in MINERALS if m in tab.index])
    tab.columns = [labels[c] for c in tab.columns]
    tab['TOTAL'] = tab.sum(axis=1)
    tab.loc['TOTAL'] = tab.sum(axis=0)
    return tab.astype(int)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build(src_dir: str, out_dir: str, charts: list[str], budget: int, seed: int,
          decisions_globs: list[str], bland_share: float, bland_pool_frac: float,
          dry_run: bool = False, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)

    skip = reviewed_uids(decisions_globs)
    if verbose:
        print(f'Loaded {len(skip):,} already-reviewed polygon_uids from '
              f'{len(decisions_globs)} pattern(s)')

    cands = load_candidates(src_dir, charts, MINERALS, skip, verbose=verbose)
    n_excluded = int(cands.attrs.get('n_skipped_reviewed', 0))
    if verbose:
        print(f'{len(cands):,} candidate polygons '
              f'({n_excluded:,} excluded as already reviewed)')
    if cands.empty:
        return {'manifest': pd.DataFrame(), 'allocation': pd.DataFrame(),
                'n_excluded_reviewed': n_excluded, 'candidates': cands}

    capacity = {(m, s): 0 for m in MINERALS for s in range(len(STRATA))}
    for (m, s), grp in cands.groupby(['mineral', 'stratum']):
        if (m, s) in capacity:
            capacity[(m, s)] = len(grp)
    alloc = allocate_cells(capacity, budget)

    picks: list[str] = []
    for (m, s), n in alloc.items():
        if n <= 0:
            continue
        cell = cands[(cands['mineral'] == m) & (cands['stratum'] == s)]
        picks.extend(sample_cell(cell, n, rng, bland_share=bland_share,
                                 bland_pool_frac=bland_pool_frac))

    # bland_pool is recorded per polygon using the SAME rule sample_cell used,
    # so the manifest's bland share is the real one, not a re-derived estimate.
    cands = cands.copy()
    cands['bland_pool'] = False
    for (m, s), grp in cands.groupby(['mineral', 'stratum']):
        cands.loc[grp.index, 'bland_pool'] = bland_pool_mask(grp, bland_pool_frac)
    selected = cands[cands['cand_key'].isin(picks)].copy()

    alloc_df = pd.DataFrame(
        [{'mineral': m, 'stratum': s, 'n': n, 'available': capacity[(m, s)]}
         for (m, s), n in alloc.items()])
    table = allocation_table(alloc_df)

    if verbose:
        print('\nAvailable candidates (class x stratum):')
        print(allocation_table(alloc_df, value='available').to_string())
        print('\nFINAL ALLOCATION (class x stratum):')
        print(table.to_string())

    if dry_run:
        return {'manifest': pd.DataFrame(), 'allocation': table,
                'allocation_long': alloc_df, 'n_excluded_reviewed': n_excluded,
                'candidates': cands, 'selected': selected}

    geoms = load_selected_geoms(selected)
    manifest = write_review_set(geoms, out_dir, verbose=verbose)
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'allocation.csv'))
    return {'manifest': manifest, 'allocation': table, 'allocation_long': alloc_df,
            'n_excluded_reviewed': n_excluded, 'candidates': cands,
            'selected': selected}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src_dir', default=DEFAULT_SRC_DIR)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--charts', nargs='+', default=DEFAULT_CHARTS)
    ap.add_argument('--budget', type=int, default=600,
                    help='Total polygons across all class x stratum cells.')
    ap.add_argument('--seed', type=int, default=20260815)
    ap.add_argument('--decisions', nargs='+', default=[DEFAULT_DECISIONS_GLOB],
                    help='Glob(s) for existing decisions.csv ledgers whose '
                         'polygon_uids must be excluded.')
    ap.add_argument('--bland_share', type=float, default=0.35,
                    help='Fraction of each cell drawn from the flattest '
                         'polygons (bland-adjacent negatives).')
    ap.add_argument('--bland_pool_frac', type=float, default=0.30,
                    help='Fraction of each cell, by ascending mean band '
                         'depth, that counts as the bland-adjacent pool.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print the allocation table without writing gpkgs.')
    args = ap.parse_args(argv)

    res = build(src_dir=args.src_dir, out_dir=args.out_dir, charts=args.charts,
                budget=args.budget, seed=args.seed,
                decisions_globs=args.decisions, bland_share=args.bland_share,
                bland_pool_frac=args.bland_pool_frac, dry_run=args.dry_run)
    if not args.dry_run:
        man = res['manifest']
        print(f'\nWrote {len(man)} polygons to {args.out_dir}')
        if not man.empty:
            print(f"bland-adjacent share: "
                  f"{man['bland_pool'].mean():.0%}")
            print('charts: ' + ', '.join(
                f'{k}={v}' for k, v in man['mc'].value_counts().sort_index().items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
