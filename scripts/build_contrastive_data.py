"""Harvest the three patch pools for contrastive plag-vs-olivine refinement.

Three pools (each ``(N, 7, 7, 59)`` float32, clipped to ``[0, 0.5]``):

* **hard_negatives** — pixels inside polygons of
  ``data/vector_mc13_relabeled/plagioclase.gpkg`` (default layer
  ``thresh_0.92``; lowest layer present in this gpkg). These are MC13 tiles
  the current classifier confidently labels plag but which the user has
  flagged as spectrally olivine — the bias we want the contrastive loss to
  remove.
* **positives** — pixels inside polygons of
  ``/mnt/mrdr/categorized_mineral_units/T*.gpkg`` whose ``Category`` is
  ``plagioclase (High)`` or ``plagioclase (Moderate)``.
* **soft_negatives** — pixels inside polygons of the same labeled gpkgs whose
  ``Category`` is ``Type 1 olivine (High)`` or ``Type 2 olivine (High)``.

For every polygon we:

1. Rasterize the geometry into the paired mrral tile's pixel grid.
2. Iterate over in-polygon pixels and clip the 7x7 reflectance window for
   each. The first 59 bands of the cube are used.
3. Drop the patch if the center pixel is NODATA (``65535``).
4. Replace remaining NODATA / non-finite values with 0 and clip to
   ``[0, 0.5]`` to match the existing patch caches.

Outputs to ``{output_dir}/{pool}/patches.npy`` and ``meta.parquet``.

Usage:

  conda run -n crism python scripts/build_contrastive_data.py \\
      --mc13_plag_gpkg data/vector_mc13_relabeled/plagioclase.gpkg \\
      --mc13_threshold_layer thresh_0.92 \\
      --labeled_gpkg_dir /mnt/mrdr/categorized_mineral_units \\
      --output_dir data/contrastive

For a fast smoke test:

  conda run -n crism python scripts/build_contrastive_data.py \\
      --debug_limit_polygons 5 --output_dir /tmp/contrastive_smoke
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------------- constants
NODATA = 65535.0
CLIP_MAX = 0.5
PATCH_SIZE = 7
N_BANDS = 59
PAD = PATCH_SIZE // 2
DEFAULT_TILE_GLOBS = (
    '/mnt/mrdr/mc*/{tile}_mrral_*.img',
    '/mnt/mrdr/mc*/{tile}_mrral*.img',
    '/mnt/mrdr/{tile}_mrral*.img',
)

PLAG_CATS = {
    'plagioclase (high)',
    'plagioclase (moderate)',
}
OLIVINE_HIGH_CATS = {
    'type 1 olivine (high)',
    'type 2 olivine (high)',
}

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- helpers
def find_mrral_for_tile(tile_id: str, tile_dir: Optional[str] = None) -> Optional[str]:
    """Locate the mrral cube for ``tile_id`` (case-insensitive 't1234')."""
    tile = tile_id.lower()
    if tile_dir is not None:
        matches = sorted(glob.glob(os.path.join(tile_dir, f'{tile}_mrral*.img')))
        if matches:
            return matches[0]
    for gtmpl in DEFAULT_TILE_GLOBS:
        matches = sorted(glob.glob(gtmpl.format(tile=tile)))
        if matches:
            return matches[0]
    return None


def tile_id_from_t_gpkg(gpkg_path: str) -> str:
    """``/mnt/.../T0433.gpkg`` -> ``'t0433'``."""
    return os.path.basename(gpkg_path).replace('.gpkg', '').lower()


def extract_patch_at_pixel(
    src: rasterio.DatasetReader,
    row: int,
    col: int,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    """Read a 7x7x59 patch centered at (row, col) from an open mrral cube.

    Returns ``None`` if the center pixel is NODATA / non-finite or out of bounds.
    Layout: (P, P, N_BANDS) float32, clipped to [0, CLIP_MAX], NODATA pixels
    in the neighbourhood replaced by 0.
    """
    if row < 0 or col < 0 or row >= height or col >= width:
        return None
    r0 = max(0, row - PAD); r1 = min(height, row + PAD + 1)
    c0 = max(0, col - PAD); c1 = min(width, col + PAD + 1)
    window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
    chunk = src.read(list(range(1, N_BANDS + 1)), window=window).astype(np.float32)
    cr = row - r0
    cc_local = col - c0
    if cr >= chunk.shape[1] or cc_local >= chunk.shape[2]:
        return None
    center_spec = chunk[:, cr, cc_local]
    if np.any(center_spec >= NODATA) or np.any(~np.isfinite(center_spec)):
        return None
    patch = np.zeros((N_BANDS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    dr0 = max(0, PAD - row)
    dc0 = max(0, PAD - col)
    ph, pw = chunk.shape[1], chunk.shape[2]
    patch[:, dr0:dr0 + ph, dc0:dc0 + pw] = chunk
    bad = (patch >= NODATA) | ~np.isfinite(patch)
    patch[bad] = 0.0
    patch = np.clip(patch, 0.0, CLIP_MAX)
    return patch.transpose(1, 2, 0).copy()


def extract_patches_from_geom(
    src: rasterio.DatasetReader,
    geom,
    transform,
    height: int,
    width: int,
    max_per_polygon: int = 200,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, int, int]]:
    """Rasterize ``geom`` onto ``(height, width)`` and emit per-pixel 7x7x59 patches.

    Returns list of ``(patch, row, col)``. The center-pixel NODATA check is
    enforced. Polygons whose pixel count is below ``PATCH_SIZE**2`` are skipped.
    """
    try:
        mask = rasterize([(geom, 1)], out_shape=(height, width),
                         transform=transform, fill=0, dtype='uint8').astype(bool)
    except Exception as e:                                              # pragma: no cover
        logger.warning(f'rasterize failed: {e}')
        return []
    rr, cc = np.where(mask)
    n_pixels = len(rr)
    if n_pixels < PATCH_SIZE * PATCH_SIZE:
        return []

    rng = np.random.default_rng(seed)
    if max_per_polygon and n_pixels > max_per_polygon:
        keep = rng.choice(n_pixels, size=max_per_polygon, replace=False)
        rr = rr[keep]
        cc = cc[keep]

    out: List[Tuple[np.ndarray, int, int]] = []
    for r, c in zip(rr, cc):
        r0 = max(0, r - PAD); r1 = min(height, r + PAD + 1)
        c0 = max(0, c - PAD); c1 = min(width, c + PAD + 1)
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        chunk = src.read(list(range(1, N_BANDS + 1)), window=window).astype(np.float32)

        # Check center pixel BEFORE clipping (NODATA detection)
        # Center position within the chunk:
        cr = r - r0
        cc_local = c - c0
        if cr >= chunk.shape[1] or cc_local >= chunk.shape[2]:
            continue
        center_spec = chunk[:, cr, cc_local]
        if np.any(center_spec >= NODATA) or np.any(~np.isfinite(center_spec)):
            continue

        patch = np.zeros((N_BANDS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        dr0 = max(0, PAD - r)
        dc0 = max(0, PAD - c)
        ph, pw = chunk.shape[1], chunk.shape[2]
        patch[:, dr0:dr0 + ph, dc0:dc0 + pw] = chunk
        # Replace NODATA / non-finite in the *neighbourhood* with 0
        bad = (patch >= NODATA) | ~np.isfinite(patch)
        patch[bad] = 0.0
        patch = np.clip(patch, 0.0, CLIP_MAX)
        # (N_BANDS, P, P) -> (P, P, N_BANDS) to match the cached layout
        patch = patch.transpose(1, 2, 0).copy()
        out.append((patch, int(r), int(c)))
    return out


# ----------------------------------------------------------------- pools
@dataclass
class PoolRecord:
    patches: List[np.ndarray]
    rows: List[dict]

    def __init__(self):
        self.patches = []
        self.rows = []

    def add(self, patch: np.ndarray, row: int, col: int, **kwargs):
        self.patches.append(patch)
        rec = {'pixel_row': int(row), 'pixel_col': int(col)}
        rec.update(kwargs)
        self.rows.append(rec)

    def extend(self, other: 'PoolRecord'):
        """Append patches + metadata from another PoolRecord in place."""
        self.patches.extend(other.patches)
        self.rows.extend(other.rows)

    def write(self, out_dir: str, name: str) -> Tuple[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        if not self.patches:
            arr = np.zeros((0, PATCH_SIZE, PATCH_SIZE, N_BANDS), dtype=np.float32)
            df = pd.DataFrame(columns=['tile_id', 'pixel_row', 'pixel_col',
                                       'source_polygon', 'source_gpkg'])
        else:
            arr = np.stack(self.patches, axis=0).astype(np.float32)
            df = pd.DataFrame.from_records(self.rows)
        npy = os.path.join(out_dir, 'patches.npy')
        pq = os.path.join(out_dir, 'meta.parquet')
        np.save(npy, arr)
        df.to_parquet(pq, index=False)
        logger.info(f'  [{name}] wrote {len(arr)} patches to {out_dir}')
        return npy, pq


# ----------------------------------------------------------------- harvests
def harvest_hard_negatives(
    gpkg_path: str,
    layer: str,
    tile_dir: Optional[str],
    max_per_polygon: int,
    debug_limit_polygons: Optional[int] = None,
    seed: Optional[int] = None,
) -> PoolRecord:
    """MC13-classifier-confident plag polygons -> hard negatives."""
    import geopandas as gpd
    pool = PoolRecord()
    g = gpd.read_file(gpkg_path, layer=layer)
    if 'tile_id' not in g.columns:
        raise ValueError(
            f"{gpkg_path} layer={layer!r} missing 'tile_id' column; "
            f"have {list(g.columns)[:5]}..."
        )
    logger.info(f'mc13 hard-negative source: {len(g)} polygons in {layer}')
    n_kept = 0
    if debug_limit_polygons:
        g = g.head(debug_limit_polygons)
    # Group by tile for fewer rasterio opens
    for tid, tile_group in g.groupby('tile_id'):
        tile_id = str(tid).lower()
        mrral_path = find_mrral_for_tile(tile_id, tile_dir=tile_dir)
        if not mrral_path:
            logger.warning(f'  [{tile_id}] no mrral cube found, skipping')
            continue
        with rasterio.open(mrral_path) as src:
            tile_crs = src.crs
            transform = src.transform
            H, W = src.height, src.width
            group_proj = tile_group.to_crs(tile_crs) if tile_group.crs != tile_crs else tile_group
            for _, row in group_proj.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                patches = extract_patches_from_geom(
                    src, geom, transform, H, W,
                    max_per_polygon=max_per_polygon, seed=seed,
                )
                for patch, r, c in patches:
                    pool.add(patch, r, c,
                             tile_id=tile_id,
                             source_polygon=str(row.get('tile_id', tile_id)),
                             source_gpkg=os.path.basename(gpkg_path))
                n_kept += len(patches)
        logger.info(f'  [{tile_id}] running total hard negatives: {n_kept}')
    return pool


def harvest_labeled_pool(
    labeled_gpkg_dir: str,
    target_cats: Iterable[str],
    tile_dir: Optional[str],
    max_per_polygon: int,
    debug_limit_polygons: Optional[int] = None,
    seed: Optional[int] = None,
) -> PoolRecord:
    """Walk T*.gpkg files, keep polygons whose Category matches ``target_cats``."""
    import geopandas as gpd
    pool = PoolRecord()
    target_cats = {c.lower() for c in target_cats}
    gpkgs = sorted(glob.glob(os.path.join(labeled_gpkg_dir, 'T*.gpkg')))
    n_polys_total = 0
    polys_seen = 0
    for gp in gpkgs:
        try:
            g = gpd.read_file(gp)
        except Exception as e:                                          # pragma: no cover
            logger.warning(f'  failed to open {gp}: {e}')
            continue
        if 'Category' not in g.columns:
            continue
        mask = g['Category'].astype(str).str.lower().isin(target_cats)
        subset = g[mask]
        if len(subset) == 0:
            continue
        tile_id = tile_id_from_t_gpkg(gp)
        mrral_path = find_mrral_for_tile(tile_id, tile_dir=tile_dir)
        if not mrral_path:
            logger.warning(f'  [{tile_id}] no mrral cube found, skipping ({len(subset)} polys lost)')
            continue
        with rasterio.open(mrral_path) as src:
            tile_crs = src.crs
            transform = src.transform
            H, W = src.height, src.width
            subset_proj = subset.to_crs(tile_crs) if subset.crs != tile_crs else subset
            for _, row in subset_proj.iterrows():
                if debug_limit_polygons and polys_seen >= debug_limit_polygons:
                    break
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                patches = extract_patches_from_geom(
                    src, geom, transform, H, W,
                    max_per_polygon=max_per_polygon, seed=seed,
                )
                for patch, r, c in patches:
                    pool.add(patch, r, c,
                             tile_id=tile_id,
                             source_polygon=str(row.get('Polygon Number', '?')),
                             source_gpkg=os.path.basename(gp))
                n_polys_total += 1
                polys_seen += 1
        if debug_limit_polygons and polys_seen >= debug_limit_polygons:
            break
    logger.info(f'  harvested {len(pool.patches)} patches from {n_polys_total} polygons')
    return pool


def harvest_sam_parquet_hard_negatives(
    parquet_paths: List[str],
    tile_dir: Optional[str],
    max_per_tile: Optional[int] = None,
    seed: Optional[int] = None,
) -> PoolRecord:
    """Per-pixel hard negatives from SAM-flagged classifier-plag pixels.

    Each parquet must have columns: ``row``, ``col``, ``tile_id`` (and
    optionally ``plag_prob``, ``sam_angle_plag``, ``sam_angle_olivine``,
    ``mode``). Pixels are extracted as 7x7x59 patches from the mrral cube
    matching ``tile_id``; rows whose center pixel is NODATA / out of bounds
    are skipped silently. If a tile has more than ``max_per_tile`` rows we
    subsample uniformly.
    """
    pool = PoolRecord()
    rng = np.random.default_rng(seed)
    for pq_path in parquet_paths:
        if not os.path.exists(pq_path):
            logger.warning(f'  parquet not found, skipping: {pq_path}')
            continue
        df = pd.read_parquet(pq_path)
        for col in ('row', 'col', 'tile_id'):
            if col not in df.columns:
                raise ValueError(
                    f"{pq_path} missing required column {col!r}; have {list(df.columns)}"
                )
        logger.info(f'  {pq_path}: {len(df):,} candidate hard-negative pixels')
        added_this_file = 0
        for tid, grp in df.groupby('tile_id'):
            tile_id = str(tid).lower()
            mrral_path = find_mrral_for_tile(tile_id, tile_dir=tile_dir)
            if not mrral_path:
                logger.warning(f'    [{tile_id}] no mrral cube found, skipping')
                continue
            sub = grp.reset_index(drop=True)
            if max_per_tile is not None and len(sub) > max_per_tile:
                keep = rng.choice(len(sub), size=max_per_tile, replace=False)
                sub = sub.iloc[keep].reset_index(drop=True)
            with rasterio.open(mrral_path) as src:
                H, W = src.height, src.width
                for _, prow in sub.iterrows():
                    patch = extract_patch_at_pixel(src, int(prow['row']),
                                                   int(prow['col']), H, W)
                    if patch is None:
                        continue
                    pool.add(patch, int(prow['row']), int(prow['col']),
                             tile_id=tile_id,
                             source_polygon=f"sam:{os.path.basename(pq_path)}",
                             source_gpkg=os.path.basename(pq_path),
                             plag_prob=float(prow.get('plag_prob', np.nan)),
                             sam_angle_plag=float(prow.get('sam_angle_plag', np.nan)))
                    added_this_file += 1
            logger.info(f'    [{tile_id}] cumulative from {os.path.basename(pq_path)}: '
                        f'{added_this_file}')
    return pool


# ----------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc13_plag_gpkg',
                    default='data/vector_mc13_relabeled/plagioclase.gpkg')
    ap.add_argument('--mc13_threshold_layer', default='thresh_0.92',
                    help='Layer to read from the MC13 plag gpkg. The gpkg ships '
                         'thresh_0.92, 0.94, 0.96, 0.97 — use the lowest to '
                         'maximise hard-negative count.')
    ap.add_argument('--labeled_gpkg_dir', default='/mnt/mrdr/categorized_mineral_units')
    ap.add_argument('--output_dir', default='data/contrastive')
    ap.add_argument('--tile_dir', default=None,
                    help='Restrict mrral-cube search to this directory '
                         '(e.g. /mnt/mrdr/mc13/ for MC13-only).')
    ap.add_argument('--patch_size', type=int, default=PATCH_SIZE)
    ap.add_argument('--max_per_polygon', type=int, default=200)
    ap.add_argument('--sam_hard_negative_parquets', nargs='*', default=None,
                    help='Optional SAM-flagged hard-negative pixel parquets to '
                         'append to the MC13 polygon-based hard-negative pool. '
                         'Each parquet must have columns row, col, tile_id '
                         '(typical source: sam_analysis/outputs/argyre/t*_hard_'
                         'negatives_mrdr.parquet).')
    ap.add_argument('--sam_max_pixels_per_tile', type=int, default=None,
                    help='Cap on pixels-per-tile taken from each SAM parquet '
                         '(uniform random subsample). None = take everything.')
    ap.add_argument('--debug_limit_polygons', type=int, default=None,
                    help='Process at most N polygons per pool — for fast CPU smoke tests.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip_hard', action='store_true')
    ap.add_argument('--skip_positives', action='store_true')
    ap.add_argument('--skip_soft', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if args.patch_size != PATCH_SIZE:
        raise NotImplementedError(
            f'patch_size={args.patch_size} not supported; this script is fixed to 7x7'
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_hard:
        logger.info('=== hard_negatives (MC13 classifier-plag polygons) ===')
        pool = harvest_hard_negatives(
            args.mc13_plag_gpkg, args.mc13_threshold_layer,
            tile_dir=args.tile_dir,
            max_per_polygon=args.max_per_polygon,
            debug_limit_polygons=args.debug_limit_polygons,
            seed=args.seed,
        )
        n_mc13 = len(pool.patches)
        if args.sam_hard_negative_parquets:
            logger.info('=== hard_negatives (SAM-flagged pixel parquets) ===')
            sam_pool = harvest_sam_parquet_hard_negatives(
                args.sam_hard_negative_parquets,
                tile_dir=args.tile_dir,
                max_per_tile=args.sam_max_pixels_per_tile,
                seed=args.seed,
            )
            pool.extend(sam_pool)
            logger.info(f'merged: {n_mc13} mc13 + {len(sam_pool.patches)} sam = '
                        f'{len(pool.patches)} total hard negatives')
        pool.write(os.path.join(args.output_dir, 'hard_negatives'), 'hard_negatives')

    if not args.skip_positives:
        logger.info('=== positives ===')
        pool = harvest_labeled_pool(
            args.labeled_gpkg_dir,
            target_cats=PLAG_CATS,
            tile_dir=args.tile_dir,
            max_per_polygon=args.max_per_polygon,
            debug_limit_polygons=args.debug_limit_polygons,
            seed=args.seed,
        )
        pool.write(os.path.join(args.output_dir, 'positives'), 'positives')

    if not args.skip_soft:
        logger.info('=== soft_negatives ===')
        pool = harvest_labeled_pool(
            args.labeled_gpkg_dir,
            target_cats=OLIVINE_HIGH_CATS,
            tile_dir=args.tile_dir,
            max_per_polygon=args.max_per_polygon,
            debug_limit_polygons=args.debug_limit_polygons,
            seed=args.seed,
        )
        pool.write(os.path.join(args.output_dir, 'soft_negatives'), 'soft_negatives')

    logger.info(f'done. patches under {args.output_dir}/{{hard_negatives,positives,soft_negatives}}/')


if __name__ == '__main__':
    main()
