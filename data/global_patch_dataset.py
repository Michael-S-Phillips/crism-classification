"""
CRISMGlobalPatchDataset — streams random 7×7 spatial patches from all mrral tiles.

Uses IterableDataset with per-worker tile sharding. No pre-extraction needed;
training can begin immediately. Each worker randomly samples tiles (weighted by
area) and random valid patch centers within each tile.

Usage:
    import glob
    from data.global_patch_dataset import CRISMGlobalPatchDataset

    hdr_files = sorted(glob.glob('/mnt/crism/MRDR/mc*/t*mrral*.hdr'))
    ds = CRISMGlobalPatchDataset(hdr_files)
    loader = DataLoader(ds, batch_size=512, num_workers=8, pin_memory=True)
"""
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset

try:
    import rasterio
    import rasterio.windows
except ImportError:
    raise ImportError("rasterio is required: conda install -c conda-forge rasterio")

NODATA = 65535.0
N_BANDS = 59          # mrral bands 1–59 (410–2457 nm); ignore bands 60–72
CLIP_MAX = 0.5        # reflectance clip — covers P99 with headroom
MIN_VALID_FRAC = 0.8  # fraction of patch pixels that must be non-NaN/nodata


class CRISMGlobalPatchDataset(IterableDataset):
    """
    Infinite stream of (patch_size, patch_size, N_BANDS) float32 tensors.

    Two-level random sampling:
      1. Sample a tile with probability proportional to its pixel area.
      2. Sample a random valid patch center (no NaN-heavy patches) within that tile.

    Workers each receive a shard of the tile list; file handles are cached
    per-worker (no shared state needed between workers).
    """

    def __init__(
        self,
        hdr_paths: list,
        patch_size: int = 7,
        min_valid_frac: float = MIN_VALID_FRAC,
        clip_max: float = CLIP_MAX,
        max_retries: int = 20,
        normalize: bool = True,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.hdr_paths = list(hdr_paths)
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.min_valid_frac = min_valid_frac
        self.clip_max = clip_max
        self.max_retries = max_retries
        self.normalize = normalize
        # Uniform tile weights — all mrral tiles are similar size (~1636x1340)
        # Avoids opening all 1764 files at init time just to compute areas.
        n = len(self.hdr_paths)
        self._weights = np.ones(n, dtype=np.float64) / n if n > 0 else np.array([])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Shard: this worker handles every num_workers-th tile
            indices = list(range(worker_info.id, len(self.hdr_paths), worker_info.num_workers))
        else:
            indices = list(range(len(self.hdr_paths)))

        if not indices:
            return

        hdr_paths = [self.hdr_paths[i] for i in indices]
        weights = self._weights[indices]
        total = weights.sum()
        if total == 0:
            return
        weights = weights / total

        rng = np.random.default_rng()
        handles: dict = {}  # hdr_path -> rasterio.DatasetReader

        while True:
            # Sample a tile
            tile_idx = int(rng.choice(len(hdr_paths), p=weights))
            hdr = hdr_paths[tile_idx]
            img_path = hdr.replace('.hdr', '.img')

            # Open/cache file handle
            if hdr not in handles:
                try:
                    handles[hdr] = rasterio.open(img_path)
                except Exception:
                    continue
            src = handles[hdr]
            H, W = src.height, src.width
            if H < self.patch_size or W < self.patch_size:
                continue

            # Sample a valid patch center (with retries)
            for _ in range(self.max_retries):
                r = int(rng.integers(self.half, H - self.half))
                c = int(rng.integers(self.half, W - self.half))

                window = rasterio.windows.Window(
                    c - self.half, r - self.half,
                    self.patch_size, self.patch_size,
                )
                try:
                    # Read bands 1–59 (rasterio is 1-indexed)
                    patch = src.read(list(range(1, N_BANDS + 1)), window=window)
                    patch = patch.astype(np.float32)  # (59, 7, 7)
                except Exception:
                    continue

                # Validity check: fraction of pixels where all bands are valid
                nodata_mask = (patch == NODATA) | ~np.isfinite(patch)
                any_nodata = nodata_mask.any(axis=0)  # (7, 7) — True if any band is bad
                valid_frac = float(1.0 - any_nodata.mean())
                if valid_frac < self.min_valid_frac:
                    continue

                # Replace nodata/NaN with 0.0 (these positions are masked anyway)
                patch[nodata_mask] = 0.0

                # Clip to physical reflectance range
                patch = np.clip(patch, 0.0, self.clip_max)

                # Per-patch spectral normalization: zero mean, unit variance
                # computed from valid pixels only; nodata positions reset to 0.0
                if self.normalize:
                    valid_pixels = ~any_nodata  # (7, 7)
                    if valid_pixels.any():
                        valid_vals = patch[:, valid_pixels]  # (59, n_valid)
                        mu = float(valid_vals.mean())
                        sigma = float(valid_vals.std())
                        if sigma > 1e-6:
                            patch = (patch - mu) / sigma
                    patch[nodata_mask] = 0.0  # re-zero nodata after normalization

                # (59, 7, 7) → (7, 7, 59) — spatial-first for transformer
                patch = patch.transpose(1, 2, 0)

                yield torch.from_numpy(patch.copy())
                break
