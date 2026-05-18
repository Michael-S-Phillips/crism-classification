"""
Build the global pre-training patch cache.

Reads all mrral tiles under `data_root` (config), samples ~2834 valid patches
per tile, writes 50 sharded `.npy` files of 100k patches each (~58 GB total)
to `--output`, plus a `shard_index.json` recording build provenance.

Usage (HPC):
    python scripts/build_global_patch_cache.py \\
        --output /xdisk/sbyrne/phillipsm/crism_patch_cache/ \\
        --workers 16 --seed 42

Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Tuple

import numpy as np
import rasterio
import rasterio.windows

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.cached_patch_dataset import compute_valid_centers


N_BANDS = 59          # mrral bands 1–59 (matches data/global_patch_dataset.py)
CLIP_MAX = 0.5
MIN_VALID_FRAC = 0.8
NODATA_VALUE = 65535
PATCH_SIZE = 7

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def extract_patches_from_tile(
    hdr_path: str,
    n_target: int,
    patch_size: int = PATCH_SIZE,
    min_valid_frac: float = MIN_VALID_FRAC,
    clip_max: float = CLIP_MAX,
    nodata_value: float = NODATA_VALUE,
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """Sample up to n_target valid patches from one mrral tile.

    Returns:
        patches: (n, patch_size, patch_size, n_bands) float32 where n ≤ n_target
        n_skipped_short: max(0, n_target - n) — how many fewer than requested
    """
    img_path = hdr_path.replace('.hdr', '.img')

    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        # Load band 1 to compute nodata mask (proxy for which positions are valid).
        band1 = src.read(1).astype(np.float32)
        nodata = (band1 == nodata_value) | ~np.isfinite(band1)

        valid_centers = compute_valid_centers(nodata, patch_size, min_valid_frac)
        valid_rs, valid_cs = np.where(valid_centers)

        n_valid = len(valid_rs)
        rng = np.random.default_rng(seed)
        if n_valid == 0:
            return np.zeros((0, patch_size, patch_size, N_BANDS), dtype=np.float32), n_target
        n_take = min(n_target, n_valid)
        choice = rng.choice(n_valid, size=n_take, replace=False)
        sampled_rs = valid_rs[choice]
        sampled_cs = valid_cs[choice]

        half = patch_size // 2
        out = np.zeros((n_take, patch_size, patch_size, N_BANDS), dtype=np.float32)
        for i in range(n_take):
            r, c = int(sampled_rs[i]), int(sampled_cs[i])
            window = rasterio.windows.Window(c - half, r - half, patch_size, patch_size)
            patch = src.read(list(range(1, N_BANDS + 1)), window=window).astype(np.float32)
            # (N_BANDS, P, P) -> (P, P, N_BANDS)
            patch = patch.transpose(1, 2, 0)
            # Replace nodata with 0.0 and clip.
            mask = (patch == nodata_value) | ~np.isfinite(patch)
            patch[mask] = 0.0
            np.clip(patch, 0.0, clip_max, out=patch)
            out[i] = patch

    n_skipped_short = max(0, n_target - n_take)
    return out, n_skipped_short


def _discover_tiles(data_root: str) -> list:
    """Find all mrral tile HDR paths under data_root."""
    patterns = [
        os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
        os.path.join(data_root, 't*mrral*.hdr'),
    ]
    hdrs: list = []
    for p in patterns:
        hdrs = sorted(glob.glob(p))
        if hdrs:
            return hdrs
    return hdrs


def _worker(args_tuple):
    """Pool worker entry point — unpacks args, calls extract_patches_from_tile.

    Defined at module level (not nested) so it's picklable by mp.Pool.
    """
    (hdr_path, n_target, patch_size, min_valid_frac, clip_max, nodata_value, seed) = args_tuple
    try:
        patches, n_skipped_short = extract_patches_from_tile(
            hdr_path=hdr_path,
            n_target=n_target,
            patch_size=patch_size,
            min_valid_frac=min_valid_frac,
            clip_max=clip_max,
            nodata_value=nodata_value,
            seed=seed,
        )
        return (hdr_path, patches, n_skipped_short, None)
    except Exception as e:
        return (hdr_path, None, 0, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for shards + shard_index.json')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root of mrral tiles. Defaults to cfg["data_root"].')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patches_per_tile_target', type=int, default=2834)
    parser.add_argument('--patches_per_shard', type=int, default=100_000)
    args = parser.parse_args()

    # Resolve data_root from --data_root or config.
    data_root = args.data_root
    if data_root is None:
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config,
        )
        from config_loader import load_config
        cfg = load_config(cfg_path)
        data_root = cfg.get('data_root', '/mnt/crism/MRDR')

    os.makedirs(args.output, exist_ok=True)
    log.info(f"Output: {args.output}")
    log.info(f"Workers: {args.workers}, seed: {args.seed}")

    hdrs = _discover_tiles(data_root)
    if not hdrs:
        raise FileNotFoundError(f"No mrral tiles under {data_root}")
    log.info(f"Found {len(hdrs)} tiles in {data_root}")

    # Build the work list — one tuple per tile, with a deterministic per-tile seed.
    work = [
        (
            hdrs[i], args.patches_per_tile_target, PATCH_SIZE,
            MIN_VALID_FRAC, CLIP_MAX, NODATA_VALUE,
            args.seed * 1000003 + i,
        )
        for i in range(len(hdrs))
    ]

    # Run via Pool.imap_unordered. Buffer results into the main process.
    buffer: list = []
    shard_id = 0
    tiles_used = 0
    tiles_skipped: list = []
    shard_records: list = []
    total_skipped_short = 0
    t_start = time.time()

    def flush_shard(buf_arrays: list, shard_id: int) -> dict:
        """Write up to patches_per_shard items from buf_arrays to one shard file."""
        n_total = sum(len(a) for a in buf_arrays)
        take = min(args.patches_per_shard, n_total)
        # Concatenate enough arrays to cover `take`.
        out = np.zeros((take, PATCH_SIZE, PATCH_SIZE, N_BANDS), dtype=np.float32)
        idx = 0
        consumed = 0
        for i, a in enumerate(buf_arrays):
            if idx >= take:
                break
            n_a = len(a)
            need = take - idx
            if n_a <= need:
                out[idx:idx + n_a] = a
                idx += n_a
                consumed = i + 1
            else:
                out[idx:idx + need] = a[:need]
                # Leave the rest of `a` in the buffer for the next shard.
                buf_arrays[i] = a[need:]
                idx += need
                consumed = i
                break

        # Drop the fully-consumed arrays from the buffer.
        del buf_arrays[:consumed]

        # Optional: shuffle within-shard to interleave tile sources.
        # Use a shard-local RNG seeded from the global seed for reproducibility.
        rng = np.random.default_rng(args.seed * 7919 + shard_id)
        perm = rng.permutation(len(out))
        out = out[perm]

        path = os.path.join(args.output, f'global_patches_{shard_id:03d}.npy')
        np.save(path, out)
        log.info(f"Wrote {path} ({len(out)} patches)")
        return {'id': shard_id, 'n_patches': len(out), 'path': path}

    with mp.Pool(args.workers) as pool:
        for hdr_path, patches, n_skipped_short, err in pool.imap_unordered(_worker, work):
            if err is not None:
                log.warning(f"Tile {hdr_path} failed: {err}")
                tiles_skipped.append(os.path.basename(hdr_path))
                continue
            tiles_used += 1
            total_skipped_short += n_skipped_short
            if patches is not None and len(patches) > 0:
                buffer.append(patches)
            # Flush full shards as buffer accumulates.
            while sum(len(a) for a in buffer) >= args.patches_per_shard:
                shard_records.append(flush_shard(buffer, shard_id))
                shard_id += 1

    # Flush any remainder as the final (possibly partial) shard.
    if any(len(a) > 0 for a in buffer):
        shard_records.append(flush_shard(buffer, shard_id))
        shard_id += 1

    total_build_time = time.time() - t_start

    # Write shard_index.json
    index = {
        'n_shards': len(shard_records),
        'patches_per_shard': args.patches_per_shard,
        'patch_size': PATCH_SIZE,
        'n_bands': N_BANDS,
        'min_valid_frac': MIN_VALID_FRAC,
        'clip_max': CLIP_MAX,
        'nodata_value': NODATA_VALUE,
        'seed': args.seed,
        'patches_per_tile_target': args.patches_per_tile_target,
        'tiles_used': tiles_used,
        'tiles_skipped': tiles_skipped,
        'total_skipped_short': total_skipped_short,
        'shards': shard_records,
        'total_build_time_s': total_build_time,
    }
    with open(os.path.join(args.output, 'shard_index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    log.info(f"Built {len(shard_records)} shards in {total_build_time:.0f}s")


if __name__ == '__main__':
    main()
