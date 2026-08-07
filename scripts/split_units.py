"""Unit-aware, pixel-balanced train/val/test splitter for CRISM labeled pixels.

Kills adjacent-tile unit leakage: polygons that map the same geologic unit
across neighboring tiles are clustered into a single "unit", and whole units
are assigned to a split. Assignment greedily balances per-class *pixel*
fractions to 70/15/15, with a min-holdout guard so every class keeps >=5% of
its pixels in val and test.

Pure pandas/numpy. No rasterio, no network. Tile centers come from a committed
lookup (`data/tile_centers.csv`) generated once from tile filenames.

Public API:
    tile_center_deg(tile_id) -> (lat, lon)
    polygon_units(df, link_deg=LINK_DEG) -> pd.Series
    assign_unit_balanced_splits(df, label_cols, seed, link_deg=LINK_DEG) -> pd.Series
    achieved_fractions(df, splits, label_cols) -> pd.DataFrame

`df` must have columns: tile_id, polygon_id, pixel_row, pixel_col, and the
label columns (float; positive = value > 0.5).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
LINK_DEG = 0.25
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
MIN_HOLDOUT_FRAC = 0.05
SPLIT_ORDER = ('train', 'val', 'test')
# A val/test split may exceed its target share of a class by at most this
# much when accepting a new unit; units that would overshoot go elsewhere
# (train is never capped, so assignment always succeeds). Without the cap, a
# giant unit (e.g. 30% of a class) arriving while the holdout deficits are
# still full gets dumped into val, overshooting the 15% target to ~30% with
# no way back (observed on the Task F joint union: alteration 0.58/0.29/0.13).
HOLDOUT_OVERSHOOT_TOL = 0.02

NOMINAL_WH = 1500.0  # nominal tile width/height in pixels for centroid approx
POS_THRESH = 0.5     # label positive if value > POS_THRESH

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TILE_CENTERS_CSV = os.path.join(_PROJ, 'data', 'tile_centers.csv')

# module-level cache: tile_id -> (lat, lon)
_TILE_CENTERS: dict[str, tuple[float, float]] | None = None


def _load_tile_centers() -> dict[str, tuple[float, float]]:
    global _TILE_CENTERS
    if _TILE_CENTERS is None:
        if not os.path.exists(_TILE_CENTERS_CSV):
            raise FileNotFoundError(
                f'tile centers lookup missing: {_TILE_CENTERS_CSV}. '
                'Regenerate by globbing /Volumes/Mars_GIS/CRISM/MRDR/mc*/t*_mrral_*.img filenames.')
        cc = pd.read_csv(_TILE_CENTERS_CSV)
        _TILE_CENTERS = {
            str(r.tile_id): (float(r.lat), float(r.lon))
            for r in cc.itertuples(index=False)
        }
    return _TILE_CENTERS


def tile_center_deg(tile_id: str) -> tuple[float, float]:
    """Return (lat, lon) center of a tile from the committed lookup.

    Filename coords (e.g. t1444_mrral_30n328) are the tile's lower-left/
    reference corner; +2.5 deg each gives the center. Raises KeyError for
    unknown tiles.
    """
    centers = _load_tile_centers()
    try:
        return centers[str(tile_id)]
    except KeyError:
        raise KeyError(
            f'unknown tile_id {tile_id!r}: not in {_TILE_CENTERS_CSV}. '
            'If this tile is real, regenerate the lookup from tile filenames.')


# ── Polygon centroids & units ────────────────────────────────────────────────

def _polygon_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (tile_id, polygon_id): mean pixel + geographic centroid."""
    g = (df.groupby(['tile_id', 'polygon_id'], sort=True)
           .agg(mean_row=('pixel_row', 'mean'),
                mean_col=('pixel_col', 'mean'))
           .reset_index())
    latlon = g['tile_id'].map(tile_center_deg)
    t_lat = latlon.map(lambda x: x[0]).to_numpy(dtype=float)
    t_lon = latlon.map(lambda x: x[1]).to_numpy(dtype=float)
    g['lat'] = t_lat - 5.0 * ((g['mean_row'].to_numpy() / NOMINAL_WH) - 0.5)
    g['lon'] = t_lon + 5.0 * ((g['mean_col'].to_numpy() / NOMINAL_WH) - 0.5)
    return g


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _pixel_share_unions(df: pd.DataFrame, cents: pd.DataFrame, uf: _UnionFind) -> None:
    """Union polygon indices (rows of `cents`, i.e. positions in the
    (tile_id, polygon_id) group table) whose ORIGINAL rows in `df` share a
    literal (tile_id, pixel_row, pixel_col) physical pixel.

    This closes the nested threshold-ladder leak: re-reviewed/re-thresholded
    polygons that share pixel footprint but whose mean centroids land more
    than `link_deg` apart (because one polygon's OTHER pixels are far away)
    must still end up in the same split-assignment unit. Applied in ADDITION
    to (not instead of) the centroid-distance linkage in `_link_components`,
    using the same union-find so both criteria merge transitively.

    Confined to the duplicated-pixel subset of `df` for speed: on the real
    corpus only ~16% of pixels are touched by >1 polygon_id, so filtering via
    `duplicated(keep=False)` first avoids a full-corpus groupby.
    """
    key_cols = ['tile_id', 'pixel_row', 'pixel_col']
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return
    idx_map = {
        (t, p): i for i, (t, p) in enumerate(zip(cents['tile_id'], cents['polygon_id']))
    }
    sub = df.loc[dup_mask, key_cols + ['polygon_id']]
    cidx = np.fromiter(
        (idx_map[(t, p)] for t, p in zip(sub['tile_id'], sub['polygon_id'])),
        dtype=int, count=len(sub))
    sub = sub.assign(_cidx=cidx)
    for _, grp in sub.groupby(key_cols, sort=False)['_cidx']:
        uniq = pd.unique(grp.to_numpy())
        if len(uniq) > 1:
            first = int(uniq[0])
            for other in uniq[1:]:
                uf.union(first, int(other))


def _link_components(lat: np.ndarray, lon: np.ndarray, link_deg: float,
                      uf: _UnionFind | None = None) -> np.ndarray:
    """Single-linkage connected components; returns integer component id per point.

    Distance: sqrt(dlat^2 + (dlon_wrapped * cos(mean_lat))^2) in degrees.
    Grid-bucketed by link_deg cells so only nearby polygons are compared.

    If `uf` is provided, unions are added to it in place (allowing callers to
    layer additional union criteria, e.g. literal pixel sharing, into the same
    union-find before/after this pass) and the caller's `uf` is used for the
    final component computation instead of a fresh one.
    """
    n = len(lat)
    if uf is None:
        uf = _UnionFind(n)
    if n <= 1:
        return np.zeros(n, dtype=int)

    cell = link_deg
    # bucket by (lat_cell, lon_cell); lon wraps at 360
    buckets: dict[tuple[int, int], list[int]] = {}
    lat_cell = np.floor(lat / cell).astype(int)
    lon_cell = np.floor((lon % 360.0) / cell).astype(int)
    n_lon_cells = int(np.ceil(360.0 / cell))
    for i in range(n):
        buckets.setdefault((int(lat_cell[i]), int(lon_cell[i])), []).append(i)

    link2 = link_deg * link_deg
    for i in range(n):
        lci, loi = int(lat_cell[i]), int(lon_cell[i])
        # Longitude is scaled by cos(lat) in the distance, so at latitude a pair
        # within link_deg scaled-degrees can differ in RAW longitude by up to
        # link_deg / cos(lat) -- spanning more than one lon-cell. Widen the
        # lon-cell scan by K = ceil(1 / cos(lat)) cells (clamped), evaluating
        # cos at the worst-case (highest-|lat|) edge of the row.
        edge_lat = min(abs(lat[i]) + cell, 89.0)
        K = int(np.ceil(1.0 / np.cos(np.radians(edge_lat))))
        # Clamped to 8: under-covers above |lat|~=83deg (cos(lat) < 1/8), which
        # would need K>8 lon-cells to guarantee the scan catches same-unit
        # pairs. Dormant for the current archive (tiles span -62.5..67.5deg),
        # but polar tiles would need a larger clamp or scaled-space bucketing.
        K = max(1, min(K, 8))
        for dla in (-1, 0, 1):
            for dlo in range(-K, K + 1):
                key = (lci + dla, (loi + dlo) % n_lon_cells)
                for j in buckets.get(key, ()):
                    if j <= i:
                        continue
                    dlat = lat[i] - lat[j]
                    dlon = (lon[i] - lon[j] + 180.0) % 360.0 - 180.0
                    mlat = np.radians((lat[i] + lat[j]) / 2.0)
                    dscaled = dlon * np.cos(mlat)
                    if dlat * dlat + dscaled * dscaled <= link2:
                        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)], dtype=int)
    # normalize to compact 0..k-1 ids, order by first appearance
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def polygon_units(df: pd.DataFrame, link_deg: float = LINK_DEG) -> pd.Series:
    """Unit id per row (indexed like df).

    Polygon centroid = tile center + 5*((mean_col/1500)-.5) lon,
    -(5*((mean_row/1500)-.5)) lat; single-linkage components at link_deg,
    UNIONED WITH any two polygons that share >=1 literal
    (tile_id, pixel_row, pixel_col) pixel (regardless of centroid distance) --
    this is what guarantees nested threshold-ladder re-review polygons that
    share physical pixels always end up in the same unit, and therefore the
    same train/val/test split (see reviewonly_leak_diagnosis.md).
    """
    cents = _polygon_centroids(df)
    uf = _UnionFind(len(cents))
    _pixel_share_unions(df, cents, uf)
    comp = _link_components(cents['lat'].to_numpy(), cents['lon'].to_numpy(), link_deg, uf=uf)
    cents['unit'] = comp
    key = df[['tile_id', 'polygon_id']].merge(
        cents[['tile_id', 'polygon_id', 'unit']],
        on=['tile_id', 'polygon_id'], how='left')
    out = pd.Series(key['unit'].to_numpy(), index=df.index, name='unit')
    return out


# ── Greedy pixel-balanced assignment ─────────────────────────────────────────

def _positive_counts(df: pd.DataFrame, label_cols: list[str]) -> np.ndarray:
    """Boolean positive matrix (n_rows x n_classes)."""
    missing = set(label_cols) - set(df.columns)
    if missing:
        raise KeyError(
            f'label columns not in df: {sorted(missing)}. '
            f'df has columns: {list(df.columns)}')
    return (df[label_cols].to_numpy(dtype=float) > POS_THRESH)


def assign_unit_balanced_splits(df: pd.DataFrame, label_cols, seed: int,
                                link_deg: float = LINK_DEG) -> pd.Series:
    """Assign whole geographic units to train/val/test, balancing per-class pixels.

    Greedy: units by total pixel count descending; each unit -> the split with the
    largest weighted per-class deficit vs SPLIT_FRACS targets; deterministic
    tie-break by split order. A val/test split is skipped for a unit when
    accepting it would push any of the unit's classes past its target share
    + HOLDOUT_OVERSHOOT_TOL (giant units otherwise overshoot the holdouts
    irrecoverably; train is never capped). Then a min-holdout guard forces the
    smallest train donor unit into val/test while a class's val/test fraction
    < MIN_HOLDOUT_FRAC.

    Caveat: with too few units per class, the balance targets can be
    unreachable and the min-holdout guard can produce degenerate splits (e.g.
    a class backed by a single unit ends up 100% in one split). Callers should
    check `achieved_fractions` rather than assume SPLIT_FRACS was hit.

    Returns a pd.Series of 'train'/'val'/'test' indexed like df.
    """
    label_cols = list(label_cols)
    units = polygon_units(df, link_deg).to_numpy()
    pos = _positive_counts(df, label_cols)  # (n, C)
    n_classes = len(label_cols)

    # Single grouped pass to get row positions per unit (O(rows) instead of
    # O(rows x units) from re-scanning `units == u` for every unit id).
    unit_row_indices = pd.Series(np.arange(len(df))).groupby(units).indices
    uniq_units = np.fromiter(unit_row_indices.keys(), dtype=units.dtype)
    # per-unit total pixels and per-class positive pixel counts
    unit_total: dict[int, int] = {}
    unit_class: dict[int, np.ndarray] = {}
    for u in uniq_units:
        idx = unit_row_indices[u]
        unit_total[u] = int(len(idx))
        unit_class[u] = pos[idx].sum(axis=0).astype(float)

    # class totals & per-split targets
    class_total = pos.sum(axis=0).astype(float)  # (C,)
    targets = {s: SPLIT_FRACS[s] * class_total for s in SPLIT_ORDER}  # each (C,)

    # deterministic order: total px desc, seeded random tiebreak, then unit id
    rng = np.random.default_rng(seed)
    tiebreak = {u: rng.random() for u in uniq_units}
    order = sorted(uniq_units, key=lambda u: (-unit_total[u], tiebreak[u], int(u)))

    current = {s: np.zeros(n_classes, dtype=float) for s in SPLIT_ORDER}
    assign: dict[int, str] = {}

    eps = 1e-9
    # Loop-invariant: targets[s] doesn't change per-unit, so precompute the
    # "safe" (non-zero) denominator and its validity mask once instead of
    # rebuilding them on every (unit, split) pair inside the greedy loop.
    tgt_valid = {s: targets[s] > eps for s in SPLIT_ORDER}
    safe_tgt = {s: np.where(tgt_valid[s], targets[s], 1.0) for s in SPLIT_ORDER}
    safe_total = np.where(class_total > eps, class_total, 1.0)
    caps = {s: (SPLIT_FRACS[s] + HOLDOUT_OVERSHOOT_TOL) * safe_total
            for s in SPLIT_ORDER if s != 'train'}
    for u in order:
        uc = unit_class[u]  # (C,)
        best_split = None
        best_score = None
        for s in SPLIT_ORDER:
            if s != 'train':
                # holdout overshoot cap: skip val/test if any class carried by
                # this unit would exceed target share + HOLDOUT_OVERSHOOT_TOL.
                over = (uc > 0) & tgt_valid[s] & (current[s] + uc > caps[s])
                if np.any(over):
                    continue
            deficit = np.where(tgt_valid[s], (targets[s] - current[s]) / safe_tgt[s], 0.0)
            # only classes present in this unit contribute, weighted by its px of c
            score = float(np.sum(deficit * uc))
            if best_score is None or score > best_score + eps:
                best_score = score
                best_split = s
            # ties: keep earliest split-order (already the first encountered)
        assign[u] = best_split
        current[best_split] += uc

    # ── min-holdout guard ────────────────────────────────────────────────────
    def frac_in(split_name: str, c: int) -> float:
        tot = class_total[c]
        if tot <= 0:
            return 1.0  # absent class: nothing to hold out
        return current[split_name][c] / tot

    for holdout in ('val', 'test'):
        for c in range(n_classes):
            if class_total[c] <= 0:
                continue
            while frac_in(holdout, c) < MIN_HOLDOUT_FRAC:
                # smallest train donor unit that contains class c
                donors = [u for u in uniq_units
                          if assign[u] == 'train' and unit_class[u][c] > 0]
                if not donors:
                    break
                donor = min(donors, key=lambda u: (unit_total[u], tiebreak[u], int(u)))
                uc = unit_class[donor]
                current['train'] -= uc
                current[holdout] += uc
                assign[donor] = holdout

    # ── materialize per-row split ────────────────────────────────────────────
    split_arr = np.empty(len(df), dtype=object)
    for u in uniq_units:
        split_arr[unit_row_indices[u]] = assign[u]
    return pd.Series(split_arr, index=df.index, name='split')


def achieved_fractions(df: pd.DataFrame, splits, label_cols) -> pd.DataFrame:
    """Per-class fraction of positive pixels in each split.

    Rows = label_cols, columns = train/val/test. Zero class total -> 0 fractions.
    """
    label_cols = list(label_cols)
    if not isinstance(splits, pd.Series):
        splits = pd.Series(np.asarray(splits), index=df.index)
    pos = _positive_counts(df, label_cols)  # (n, C)
    sp = splits.to_numpy()
    out = pd.DataFrame(0.0, index=label_cols, columns=list(SPLIT_ORDER))
    totals = pos.sum(axis=0).astype(float)
    for si, s in enumerate(SPLIT_ORDER):
        mask = sp == s
        counts = pos[mask].sum(axis=0).astype(float)
        for ci, c in enumerate(label_cols):
            out.loc[c, s] = counts[ci] / totals[ci] if totals[ci] > 0 else 0.0
    return out


# ── Pixel-leak guard (opt-in; NOT called automatically by assign_unit_balanced_splits) ──

def find_pixel_split_leaks(df: pd.DataFrame, splits) -> pd.DataFrame:
    """Return the (tile_id, pixel_row, pixel_col) keys whose rows span more
    than one split, with the distinct splits they hit.

    This is a defense-in-depth check, not a correctness guarantee on its own:
    with a correct `polygon_units()` (pixel-sharing polygons unioned into one
    unit) this should always come back empty. It is a groupby over the full
    frame, so it is deliberately NOT invoked inside `assign_unit_balanced_splits`
    on every call (that would slow large builds); call it explicitly where it's
    cheap -- e.g. once after the final joint re-split in a build script, or in
    tests -- via `assert_no_pixel_split_leak` below.

    Returns an empty DataFrame (columns: tile_id, pixel_row, pixel_col, splits)
    when there is no leak.
    """
    if not isinstance(splits, pd.Series):
        splits = pd.Series(np.asarray(splits), index=df.index)
    tmp = df[['tile_id', 'pixel_row', 'pixel_col']].copy()
    tmp['split'] = splits.to_numpy()
    g = tmp.groupby(['tile_id', 'pixel_row', 'pixel_col'])['split'].agg(
        lambda s: tuple(sorted(set(s))))
    leaked = g[g.map(len) > 1]
    return leaked.reset_index().rename(columns={'split': 'splits'})


def assert_no_pixel_split_leak(df: pd.DataFrame, splits) -> None:
    """Raise AssertionError if any physical pixel spans more than one split.

    See `find_pixel_split_leaks` for cost/usage notes -- call this explicitly
    (e.g. once after a build's final split assignment), not inside the greedy
    assignment loop.
    """
    leaks = find_pixel_split_leaks(df, splits)
    if len(leaks):
        sample = leaks.head(10).to_dict('records')
        raise AssertionError(
            f'{len(leaks)} physical (tile_id, pixel_row, pixel_col) pixels span '
            f'more than one split (expected 0). Sample: {sample}')
