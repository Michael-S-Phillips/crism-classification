"""
CRISMCachedPatchDataset — mmap-backed reader for the pre-built global patch cache.

Replaces CRISMGlobalPatchDataset (deleted). Reads 7×7×59 float32 patches from
sharded `.npy` files produced by `scripts/build_global_patch_cache.py`. Per-patch
zero-mean / unit-variance normalization happens on read.

Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md
"""
from __future__ import annotations

import glob
import os
from typing import Iterator, Optional

import numpy as np
import torch
from scipy.signal import convolve2d
from torch.utils.data import IterableDataset


def compute_valid_centers(
    nodata: np.ndarray,
    patch_size: int = 7,
    min_valid_frac: float = 0.8,
) -> np.ndarray:
    """Return a (H, W) bool array: True where a patch_size×patch_size patch
    centered at (r, c) has ≥ min_valid_frac valid pixels.

    Centers within `patch_size // 2` of any edge are False (the patch would
    fall off the tile).

    Vectorized via scipy.signal.convolve2d on the nodata mask — one pass over
    the tile regardless of how many candidate centers are evaluated.
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    H, W = nodata.shape
    half = patch_size // 2
    n_pix = patch_size * patch_size

    # Count of nodata pixels in each patch_size×patch_size window, centered.
    kernel = np.ones((patch_size, patch_size), dtype=np.int32)
    nodata_counts = convolve2d(
        nodata.astype(np.int32), kernel, mode='same', boundary='fill', fillvalue=1,
    )
    valid_counts = n_pix - nodata_counts
    valid_frac = valid_counts / n_pix

    valid = valid_frac >= min_valid_frac

    # Force False at edges where a patch would fall off.
    valid[:half, :] = False
    valid[H - half:, :] = False
    valid[:, :half] = False
    valid[:, W - half:] = False
    return valid


class CRISMCachedPatchDataset(IterableDataset):
    """Yields (patch_size, patch_size, n_bands) float32 tensors from a pre-built shard cache.

    Each shard is mmap-loaded on demand; per-shard handles are held for the
    worker's lifetime. Per-patch normalization (zero-mean / unit-variance over
    all values in the patch) is applied on read.
    """

    def __init__(
        self,
        shard_dir: str,
        normalize: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.shard_dir = shard_dir
        self.normalize = normalize
        self.shuffle = shuffle
        self.seed = seed
        self.shards = sorted(glob.glob(os.path.join(shard_dir, 'global_patches_*.npy')))
        if not self.shards:
            raise FileNotFoundError(f"No shards in {shard_dir}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        # Per-worker RNG so seeded runs are reproducible AND workers diverge.
        rng_seed = self.seed
        if rng_seed is not None and worker_info is not None:
            rng_seed = rng_seed + worker_info.id
        rng = np.random.default_rng(rng_seed)

        shard_order = list(range(len(shards)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for si in shard_order:
            arr = np.load(shards[si], mmap_mode='r')  # (N_si, P, P, B)
            indices = np.arange(len(arr))
            if self.shuffle:
                rng.shuffle(indices)
            for i in indices:
                patch = np.asarray(arr[i], dtype=np.float32).copy()  # detach from mmap
                if self.normalize:
                    mean = float(patch.mean())
                    std = float(patch.std()) + 1e-8
                    patch = (patch - mean) / std
                yield torch.from_numpy(patch)
