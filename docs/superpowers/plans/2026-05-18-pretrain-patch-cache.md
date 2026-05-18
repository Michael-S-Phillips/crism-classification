# Pre-training Patch Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the streaming `CRISMGlobalPatchDataset` with a pre-built mmap-backed cache (5M patches, 50 sharded `.npy` files on `/xdisk`), eliminating the ~109 ms/patch BSQ read amplification that's making pre-training 14× slower than budget.

**Architecture:** New `CRISMCachedPatchDataset` (`data/cached_patch_dataset.py`) reads from `np.load(mmap_mode='r')` shards. New build script (`scripts/build_global_patch_cache.py`) uses `multiprocessing.Pool.imap_unordered` to extract valid patches per tile in parallel, with a main coordinator buffering and flushing 100K-patch shards. Both pretrain scripts swap datasets and add `persistent_workers=True`.

**Tech Stack:** Python 3.11, PyTorch (`IterableDataset`), NumPy (mmap), rasterio (one-time read at build), scipy (vectorized validity mask via `convolve2d`), multiprocessing.Pool, pytest, SLURM for HPC.

**Spec:** `docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md`

**Reference patterns:**
- `data/global_patch_dataset.py` — the streaming dataset being deleted (use for spec parity: same clip, same normalization formula)
- `scripts/cache_mrral_patches.py` — existing labeled-cache builder (config plumbing, glob patterns)
- `scripts/hpc_build_cache.slurm` — existing build SLURM template (env activation, account, partition)
- `scripts/pretrain_spatial_mae_spend.py` — current consumer of streaming dataset (to be updated)

---

## File Structure

**New files:**
- `data/cached_patch_dataset.py` — `CRISMCachedPatchDataset` class
- `scripts/build_global_patch_cache.py` — cache builder driver
- `scripts/hpc_build_global_cache.slurm` — HPC build job
- `tests/test_cached_patch_dataset.py` — dataset tests
- `tests/test_build_global_patch_cache.py` — builder integration test

**Modified files:**
- `scripts/pretrain_spatial_mae_denoising.py` — swap dataset import + DataLoader args
- `scripts/pretrain_spatial_mae_spend.py` — swap dataset import + DataLoader args
- `scripts/hpc_pretrain_denoising.slurm` — add `global_patch_cache_dir` to generated config
- `scripts/hpc_pretrain_spend.slurm` — add `global_patch_cache_dir` to generated config
- `data/dataset.py` — fix doc comment that references the deleted `CRISMGlobalPatchDataset`

**Deleted files:**
- `data/global_patch_dataset.py`
- `tests/test_global_patch_dataset.py`

---

## Task 1: Validity-mask helper

A pure function that takes a 2D nodata mask and returns the per-center fraction of valid pixels for each `patch_size × patch_size` window. Used by the builder to enumerate eligible patch centers without per-patch rejection.

**Files:**
- Create: `data/cached_patch_dataset.py`
- Test: `tests/test_cached_patch_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cached_patch_dataset.py` with this content:

```python
"""Tests for CRISMCachedPatchDataset + helpers."""
from __future__ import annotations

import numpy as np
import pytest

from data.cached_patch_dataset import compute_valid_centers


class TestComputeValidCenters:
    """A 2D nodata mask -> a (H, W) bool array of valid center positions."""

    def test_all_valid_returns_inner_grid_true(self):
        # 10x10 all-valid (no nodata anywhere) → every center where the
        # full 7x7 patch fits should be True.
        nodata = np.zeros((10, 10), dtype=bool)
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        assert valid.shape == (10, 10)
        # Centers in [3, 6] (inclusive) along each axis have a valid 7x7 patch.
        assert valid[3:7, 3:7].all()
        # Edge centers can't fit a 7x7 patch → must be False.
        assert not valid[:3, :].any()
        assert not valid[7:, :].any()
        assert not valid[:, :3].any()
        assert not valid[:, 7:].any()

    def test_all_nodata_returns_all_false(self):
        nodata = np.ones((10, 10), dtype=bool)
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        assert valid.shape == (10, 10)
        assert not valid.any()

    def test_threshold_boundary(self):
        # 7x7 = 49 pixels. min_valid_frac=0.8 → need ≥ 40 valid (39.2 → ≥40).
        # Build a tile where center (3, 3) has exactly 10 nodata pixels in its
        # 7x7 patch — valid frac = 39/49 ≈ 0.796 < 0.8 → must be False.
        nodata = np.zeros((10, 10), dtype=bool)
        # Set 10 nodata in the top-left of the 7x7 patch around center (3, 3)
        nodata[0, 0:10] = True            # row 0: 10 nodata
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        # Center (3, 3) sees rows 0-6: 7 nodata in row 0 → 42 valid ≥ 40 → True
        assert valid[3, 3]
        # Now set more nodata until we're below threshold
        nodata[1, 0:10] = True            # rows 0-1: 14 nodata
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        # Center (3, 3) sees rows 0-6: 14 nodata in rows 0-1 → 35 valid < 40 → False
        assert not valid[3, 3]

    def test_patch_size_must_be_odd(self):
        nodata = np.zeros((10, 10), dtype=bool)
        with pytest.raises(AssertionError):
            compute_valid_centers(nodata, patch_size=6, min_valid_frac=0.8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_cached_patch_dataset.py -v`
Expected: 4 FAILs with `ModuleNotFoundError: No module named 'data.cached_patch_dataset'`

- [ ] **Step 3: Implement the helper**

Create `data/cached_patch_dataset.py` with this content:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_cached_patch_dataset.py::TestComputeValidCenters -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add data/cached_patch_dataset.py tests/test_cached_patch_dataset.py && git commit -m "$(cat <<'EOF'
feat(cache): add compute_valid_centers helper

Vectorized via scipy.signal.convolve2d on the nodata mask. Returns a
boolean array marking which (r, c) positions have a valid patch_size
window centered on them. Used by the cache builder to enumerate
eligible patch centers without per-patch rejection.

Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `CRISMCachedPatchDataset` class

**Files:**
- Modify: `data/cached_patch_dataset.py` (add the class)
- Test: `tests/test_cached_patch_dataset.py` (add a class-level test fixture and basic tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cached_patch_dataset.py`:

```python
import torch
from data.cached_patch_dataset import CRISMCachedPatchDataset


def _make_fake_cache(tmp_path, n_shards=3, n_per_shard=20, n_bands=59, patch_size=7,
                     fill_value=0.25, seed=0):
    """Write a tiny set of shards for tests."""
    rng = np.random.default_rng(seed)
    for s in range(n_shards):
        shape = (n_per_shard, patch_size, patch_size, n_bands)
        # Use small random values in [0, 0.5] to simulate clipped I/F.
        arr = rng.uniform(0.0, 0.5, size=shape).astype(np.float32)
        np.save(tmp_path / f'global_patches_{s:03d}.npy', arr)
    return str(tmp_path)


class TestCachedDatasetInit:
    def test_discovers_shard_files(self, tmp_path):
        shard_dir = _make_fake_cache(tmp_path, n_shards=4)
        ds = CRISMCachedPatchDataset(shard_dir)
        assert len(ds.shards) == 4
        assert all(p.endswith('.npy') for p in ds.shards)

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CRISMCachedPatchDataset(str(tmp_path))

    def test_raises_on_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            CRISMCachedPatchDataset('/tmp/definitely-not-a-cache-dir-12345')


class TestCachedDatasetIter:
    def test_yields_correct_shape_and_dtype(self, tmp_path):
        shard_dir = _make_fake_cache(tmp_path, n_shards=2, n_per_shard=10)
        ds = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=False)
        patches = list(ds)
        assert len(patches) == 20
        for p in patches:
            assert isinstance(p, torch.Tensor)
            assert p.shape == (7, 7, 59)
            assert p.dtype == torch.float32

    def test_normalize_false_preserves_clip_range(self, tmp_path):
        shard_dir = _make_fake_cache(tmp_path, n_shards=1, n_per_shard=10)
        ds = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=False)
        for p in ds:
            assert (p >= 0).all()
            assert (p <= 0.5).all()

    def test_normalize_true_gives_zero_mean_unit_std(self, tmp_path):
        shard_dir = _make_fake_cache(tmp_path, n_shards=1, n_per_shard=50)
        ds = CRISMCachedPatchDataset(shard_dir, normalize=True, shuffle=False)
        for p in ds:
            assert abs(p.mean().item()) < 1e-4
            assert 0.5 < p.std().item() < 1.5

    def test_same_seed_deterministic(self, tmp_path):
        shard_dir = _make_fake_cache(tmp_path, n_shards=2, n_per_shard=10)
        ds_a = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=True, seed=42)
        ds_b = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=True, seed=42)
        patches_a = list(ds_a)
        patches_b = list(ds_b)
        for a, b in zip(patches_a, patches_b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_different_seeds_differ(self, tmp_path):
        shard_dir = _make_fake_cache(tmp_path, n_shards=2, n_per_shard=10)
        ds_a = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=True, seed=0)
        ds_b = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=True, seed=1)
        patches_a = list(ds_a)
        patches_b = list(ds_b)
        # Different seeds → different first patch (overwhelmingly probable for
        # a 20-patch cache with 50% chance of identical first index)
        all_match = all(torch.equal(a, b) for a, b in zip(patches_a, patches_b))
        assert not all_match
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_cached_patch_dataset.py::TestCachedDatasetInit tests/test_cached_patch_dataset.py::TestCachedDatasetIter -v`
Expected: FAILs with `ImportError: cannot import name 'CRISMCachedPatchDataset'`

- [ ] **Step 3: Implement the class**

Append to `data/cached_patch_dataset.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_cached_patch_dataset.py -v`
Expected: 12 PASS (4 helper from Task 1 + 3 init + 5 iter behaviors from Task 2).

If `TestCachedDatasetIter::test_different_seeds_differ` fails flakily, the assertion is wrong for very small shards. With `n_per_shard=10` and 2 shards, there are many possible orderings — flakiness is unlikely but not impossible. If it fails reproducibly, expand to `n_per_shard=20` and re-run.

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add data/cached_patch_dataset.py tests/test_cached_patch_dataset.py && git commit -m "$(cat <<'EOF'
feat(cache): add CRISMCachedPatchDataset class

IterableDataset that yields (7, 7, 59) float32 tensors from a directory
of np.load(mmap_mode='r') shard files. Supports per-patch normalization,
per-epoch shuffle with optional seed, and worker sharding via
torch.utils.data.get_worker_info.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Worker sharding and mmap reuse tests

**Files:**
- Test: `tests/test_cached_patch_dataset.py` (add multi-worker tests)

- [ ] **Step 1: Write the failing tests**

First, add `from torch.utils.data import DataLoader` to the top of `tests/test_cached_patch_dataset.py`, alongside the existing imports (do NOT place it mid-file with the new test class — keep all imports at the top).

Then append to `tests/test_cached_patch_dataset.py`:

```python
class TestCachedDatasetWorkers:
    def test_worker_sharding_disjoint_and_complete(self, tmp_path):
        """With num_workers=N, the union of all yielded patches equals
        the full cache, with no patch yielded twice within an epoch."""
        shard_dir = _make_fake_cache(tmp_path, n_shards=6, n_per_shard=10, seed=0)
        ds = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=False, seed=0)
        loader = DataLoader(ds, batch_size=4, num_workers=3, drop_last=False)
        seen = []
        for batch in loader:
            # batch: (B, 7, 7, 59)
            for i in range(batch.shape[0]):
                seen.append(batch[i].numpy().tobytes())
        # Total = 6 shards × 10 patches = 60
        assert len(seen) == 60
        # All unique
        assert len(set(seen)) == 60

    def test_mmap_reuse_one_load_per_shard_per_worker(self, tmp_path, monkeypatch):
        """np.load is called once per shard per worker, not once per patch."""
        shard_dir = _make_fake_cache(tmp_path, n_shards=3, n_per_shard=20)
        original_load = np.load
        call_count = {'n': 0}

        def counting_load(*args, **kwargs):
            call_count['n'] += 1
            return original_load(*args, **kwargs)

        monkeypatch.setattr(np, 'load', counting_load)

        ds = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=False)
        # num_workers=0 → main process; expect exactly 3 np.load calls (one per shard)
        # for a single full epoch.
        patches = list(ds)
        assert len(patches) == 60
        assert call_count['n'] == 3, f"expected 3 np.load calls, got {call_count['n']}"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_cached_patch_dataset.py::TestCachedDatasetWorkers -v`
Expected: 2 PASS. If the worker-sharding test fails on a system with no multiprocessing fork support, set `multiprocessing_context='fork'` in the DataLoader or skip with `pytest.skipif`. Test on the local box where DataLoader fork works.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add tests/test_cached_patch_dataset.py && git commit -m "$(cat <<'EOF'
test(cache): worker sharding disjoint coverage + mmap reuse

Two integration-level tests for CRISMCachedPatchDataset that exercise
the DataLoader multi-worker contract (no duplicates across workers in
one epoch, all patches accounted for) and confirm np.load is called
once per shard per worker (not once per patch).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Per-tile patch-extraction function (builder helper)

A function that takes one tile's HDR path and returns valid patches as a numpy array. Used by the builder's Pool workers.

**Files:**
- Create: `scripts/build_global_patch_cache.py` (new file)
- Test: `tests/test_build_global_patch_cache.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_global_patch_cache.py`:

```python
"""Tests for scripts/build_global_patch_cache.py."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def _make_fake_tile(path_prefix: str, H: int = 50, W: int = 50, n_bands: int = 59,
                    nodata_value: float = 65535, nodata_frac: float = 0.0,
                    seed: int = 0):
    """Write a minimal ENVI/.img+.hdr pair we can read with rasterio."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 0.3, size=(n_bands, H, W)).astype(np.float32)
    if nodata_frac > 0:
        nodata_mask = rng.uniform(0, 1, size=(H, W)) < nodata_frac
        data[:, nodata_mask] = nodata_value

    img_path = path_prefix + '.img'
    hdr_path = path_prefix + '.hdr'

    profile = {
        'driver': 'ENVI',
        'dtype': 'float32',
        'count': n_bands,
        'height': H,
        'width': W,
        'nodata': nodata_value,
        'interleave': 'bsq',
    }
    with rasterio.open(img_path, 'w', **profile) as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)
    return hdr_path, img_path


def test_extract_patches_from_tile_returns_correct_shape(tmp_path):
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50)
    patches, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr,
        n_target=30,
        patch_size=7,
        min_valid_frac=0.8,
        clip_max=0.5,
        nodata_value=65535,
        seed=42,
    )
    assert patches.shape == (30, 7, 7, 59)
    assert patches.dtype == np.float32
    assert n_skipped_short == 0


def test_extract_patches_clipped_to_clip_max(tmp_path):
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50)
    patches, _ = extract_patches_from_tile(
        hdr_path=hdr, n_target=10, patch_size=7,
        min_valid_frac=0.8, clip_max=0.5, nodata_value=65535, seed=0,
    )
    assert (patches >= 0).all()
    assert (patches <= 0.5).all()


def test_extract_skips_short_when_fewer_valid_centers(tmp_path):
    """A tile with nearly all nodata returns fewer than n_target patches."""
    from build_global_patch_cache import extract_patches_from_tile

    # 80% nodata → most centers fail min_valid_frac=0.8
    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_b'), H=30, W=30, nodata_frac=0.8)
    patches, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr, n_target=100, patch_size=7,
        min_valid_frac=0.8, clip_max=0.5, nodata_value=65535, seed=0,
    )
    # Should return fewer patches than requested
    assert patches.shape[0] < 100
    assert n_skipped_short == 100 - patches.shape[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_build_global_patch_cache.py -v`
Expected: 3 FAILs with `ModuleNotFoundError: No module named 'build_global_patch_cache'`

- [ ] **Step 3: Implement the builder helper**

Create `scripts/build_global_patch_cache.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_build_global_patch_cache.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/build_global_patch_cache.py tests/test_build_global_patch_cache.py && git commit -m "$(cat <<'EOF'
feat(cache): add per-tile patch extraction helper for builder

extract_patches_from_tile opens one mrral tile, computes the valid-center
mask via compute_valid_centers, samples n_target centers uniformly without
replacement, and returns the patches as a (n, P, P, 59) float32 array.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Builder main coordinator

The CLI driver that discovers tiles, dispatches work via `mp.Pool.imap_unordered`, buffers shards, and writes the index.

**Files:**
- Modify: `scripts/build_global_patch_cache.py` (add `main()` + supporting)
- Test: `tests/test_build_global_patch_cache.py` (add end-to-end synthetic test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_global_patch_cache.py`:

```python
import subprocess


def _make_synthetic_data_root(tmp_path, n_tiles: int = 3, H: int = 50, W: int = 50):
    """Create n_tiles fake mrral tiles in mc<NN>/ subdirectories."""
    root = tmp_path / 'fake_mrdr'
    root.mkdir()
    mc = root / 'mc02'
    mc.mkdir()
    hdrs = []
    for i in range(n_tiles):
        prefix = mc / f't{i:04d}_mrral_00n000_0327_4'
        hdr, _img = _make_fake_tile(str(prefix), H=H, W=W, seed=i)
        hdrs.append(hdr)
    return str(root), hdrs


def test_main_builds_a_complete_cache_end_to_end(tmp_path):
    """Run the builder against a 3-tile synthetic data root, confirm shards
    and shard_index.json are written correctly."""
    data_root, _hdrs = _make_synthetic_data_root(tmp_path, n_tiles=3, H=60, W=60)
    output = tmp_path / 'patch_cache'

    # Invoke the builder as a subprocess so we exercise the CLI entry point.
    # Small targets: 30 patches/tile × 3 tiles = 90 total, shards of 40 → 3 shards.
    result = subprocess.run(
        [
            sys.executable, '-u',
            os.path.join(os.path.dirname(__file__), '..', 'scripts', 'build_global_patch_cache.py'),
            '--output', str(output),
            '--data_root', data_root,
            '--workers', '2',
            '--seed', '42',
            '--patches_per_tile_target', '30',
            '--patches_per_shard', '40',
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"builder failed: {result.stderr}"

    # Expect 3 shards: 40 + 40 + 10
    shard_files = sorted(glob.glob(str(output / 'global_patches_*.npy')))
    assert len(shard_files) == 3, f"expected 3 shards, got {shard_files}"

    sizes = [np.load(f, mmap_mode='r').shape[0] for f in shard_files]
    assert sum(sizes) == 90
    assert sizes[0] == 40
    assert sizes[1] == 40
    assert sizes[2] == 10

    # Shard index well-formed
    index_path = output / 'shard_index.json'
    assert index_path.exists()
    with open(index_path) as f:
        idx = json.load(f)
    assert idx['n_shards'] == 3
    assert idx['patches_per_shard'] == 40
    assert idx['patch_size'] == 7
    assert idx['n_bands'] == 59
    assert idx['min_valid_frac'] == 0.8
    assert idx['clip_max'] == 0.5
    assert idx['seed'] == 42
    assert idx['tiles_used'] == 3
    assert isinstance(idx['shards'], list) and len(idx['shards']) == 3
```

Note: we need `glob` and `np` imported in this test file already — they are (from earlier tests).

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_build_global_patch_cache.py::test_main_builds_a_complete_cache_end_to_end -v`
Expected: FAIL with non-zero return code from the subprocess (since `main()` doesn't exist yet).

- [ ] **Step 3: Implement the main coordinator**

Append to `scripts/build_global_patch_cache.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_build_global_patch_cache.py -v`
Expected: 4 PASS (3 helper + 1 end-to-end)

The end-to-end test may take 5–15 s. If it times out, the synthetic-tile fixture may be too large; reduce H/W.

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/build_global_patch_cache.py tests/test_build_global_patch_cache.py && git commit -m "$(cat <<'EOF'
feat(cache): add main coordinator for global patch cache builder

Uses mp.Pool.imap_unordered with one task per tile. Main process
buffers patches across tiles, writes 100k-patch shards as the buffer
fills, and finalizes a partial last shard if needed. Writes
shard_index.json with provenance fields per the spec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: HPC SLURM build job

**Files:**
- Create: `scripts/hpc_build_global_cache.slurm`

- [ ] **Step 1: Write the SLURM file**

Create `scripts/hpc_build_global_cache.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=build_global_patch_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/build_global_cache_%j.log
#SBATCH --error=logs/build_global_cache_%j.log

# Build the global pre-training patch cache.
# CPU-only — patch extraction is I/O-bound (reading BSQ mrral tiles).
# ~5.5 hours expected with 16 workers, 24 hr SLURM budget for safety.
# Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python
OUTPUT=/xdisk/sbyrne/phillipsm/crism_patch_cache

cd "$WORK_DIR"

# Make sure config.local.yaml points at the data on /xdisk.
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${WORK_DIR}/checkpoints
checkpoints_dir: ${WORK_DIR}/checkpoints
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
global_patch_cache_dir: ${OUTPUT}
EOF
fi

mkdir -p logs "$OUTPUT"

echo "=== Global patch cache build start: $(date) ==="

${PYTHON} -u scripts/build_global_patch_cache.py \
    --output "$OUTPUT" \
    --workers 16 \
    --seed 42 \
    --patches_per_tile_target 2834 \
    --patches_per_shard 100000

echo "=== Global patch cache build end: $(date) ==="
```

- [ ] **Step 2: Verify the file looks well-formed**

Run: `head -1 /mnt/mrdr/crism_classification/scripts/hpc_build_global_cache.slurm` — expect `#!/bin/bash`
Run: `grep -c '^#SBATCH' /mnt/mrdr/crism_classification/scripts/hpc_build_global_cache.slurm` — expect 10

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/hpc_build_global_cache.slurm && git commit -m "$(cat <<'EOF'
feat(cache): add HPC SLURM job for global patch cache build

CPU-only job: 16 cpus-per-task, 64 GB RAM, 24 hr budget. Calls
build_global_patch_cache.py with production args (2834 patches/tile,
100k patches/shard, seed=42). Output directory matches the
global_patch_cache_dir in config.local.yaml.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update pretraining scripts to use the new dataset

**Files:**
- Modify: `scripts/pretrain_spatial_mae_denoising.py`
- Modify: `scripts/pretrain_spatial_mae_spend.py`

- [ ] **Step 1: Update `pretrain_spatial_mae_denoising.py`**

Find the existing data-loading block (around lines 68–93 in the current file):

```python
    # ── Data ──────────────────────────────────────────────────────────────
    data_root = cfg.get('data_root', '/mnt/crism/MRDR')
    globs_to_try = [
        os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
        os.path.join(data_root, 't*mrral*.hdr'),
    ]
    hdr_files = []
    for g in globs_to_try:
        hdr_files = sorted(glob.glob(g))
        if hdr_files:
            break
    if not hdr_files:
        raise FileNotFoundError(
            f"No mrral HDR files found. Tried:\n" + "\n".join(f"  {g}" for g in globs_to_try)
        )
    log.info(f"Found {len(hdr_files)} mrral tiles")

    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(hdr_files, patch_size=7, min_valid_frac=0.8)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
```

Replace with:

```python
    # ── Data ──────────────────────────────────────────────────────────────
    shard_dir = cfg.get('global_patch_cache_dir')
    if not shard_dir:
        raise KeyError("config.local.yaml must define global_patch_cache_dir")
    log.info(f"Global patch cache: {shard_dir}")

    from data.cached_patch_dataset import CRISMCachedPatchDataset
    ds = CRISMCachedPatchDataset(shard_dir=shard_dir, normalize=True, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
```

If `glob` is no longer used elsewhere in the file, remove `import glob` from the top.

- [ ] **Step 2: Verify `--help` still works on denoising script**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python scripts/pretrain_spatial_mae_denoising.py --help 2>&1 | tail -5`
Expected: argparse help text printed without ImportError.

- [ ] **Step 3: Apply the identical change to `pretrain_spatial_mae_spend.py`**

Find the equivalent block (around lines 67–93) in `scripts/pretrain_spatial_mae_spend.py`. The lines to remove are the same. The lines to insert are the same:

```python
    # ── Data ──────────────────────────────────────────────────────────────
    shard_dir = cfg.get('global_patch_cache_dir')
    if not shard_dir:
        raise KeyError("config.local.yaml must define global_patch_cache_dir")
    log.info(f"Global patch cache: {shard_dir}")

    from data.cached_patch_dataset import CRISMCachedPatchDataset
    ds = CRISMCachedPatchDataset(shard_dir=shard_dir, normalize=True, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
```

Remove `import glob` from the top if unused after the change.

- [ ] **Step 4: Verify `--help` still works on SPEND script**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python scripts/pretrain_spatial_mae_spend.py --help 2>&1 | tail -5`
Expected: argparse help text printed.

- [ ] **Step 5: Run all model tests to confirm no regressions**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_spend_spatial_mae.py tests/test_denoising_spatial_mae.py tests/test_spatial_mae.py -v`
Expected: all PASS (these tests don't touch the dataset code; they should still pass).

- [ ] **Step 6: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/pretrain_spatial_mae_denoising.py scripts/pretrain_spatial_mae_spend.py && git commit -m "$(cat <<'EOF'
feat(cache): wire pretraining scripts to use CRISMCachedPatchDataset

Both pretrain scripts now read from a pre-built mmap shard cache
referenced by cfg['global_patch_cache_dir']. The streaming dataset
import and the tile-globbing block are removed. DataLoader gets
persistent_workers=True so the per-worker mmap state survives
across epochs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update pretraining SLURM scripts to add the new config key

**Files:**
- Modify: `scripts/hpc_pretrain_denoising.slurm`
- Modify: `scripts/hpc_pretrain_spend.slurm`

- [ ] **Step 1: Edit `hpc_pretrain_denoising.slurm`**

Find the heredoc that generates `config.local.yaml`:

```bash
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${WORK_DIR}/checkpoints
checkpoints_dir: ${WORK_DIR}/checkpoints
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi
```

Add a new line for `global_patch_cache_dir`. Replace with:

```bash
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${WORK_DIR}/checkpoints
checkpoints_dir: ${WORK_DIR}/checkpoints
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
global_patch_cache_dir: /xdisk/sbyrne/phillipsm/crism_patch_cache
EOF
fi
```

- [ ] **Step 2: Apply the same edit to `hpc_pretrain_spend.slurm`**

Same change: add the `global_patch_cache_dir: ...` line inside the heredoc, after `patch_cache_dir: ...`.

- [ ] **Step 3: Sanity check**

Run: `grep -A1 'global_patch_cache_dir' /mnt/mrdr/crism_classification/scripts/hpc_pretrain_denoising.slurm /mnt/mrdr/crism_classification/scripts/hpc_pretrain_spend.slurm`
Expected: both files show the line `global_patch_cache_dir: /xdisk/sbyrne/phillipsm/crism_patch_cache`.

- [ ] **Step 4: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/hpc_pretrain_denoising.slurm scripts/hpc_pretrain_spend.slurm && git commit -m "$(cat <<'EOF'
feat(cache): point pretraining SLURM configs at the global cache

Both pretraining SLURM scripts now emit a global_patch_cache_dir line
in the auto-generated config.local.yaml. patch_cache_dir is preserved
for the supervised classifier pipeline (no collision).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Delete the streaming dataset and fix stale references

**Files:**
- Delete: `data/global_patch_dataset.py`
- Delete: `tests/test_global_patch_dataset.py`
- Modify: `data/dataset.py` (fix stale doc reference)

- [ ] **Step 1: Find and read the stale doc reference**

Run: `grep -n 'CRISMGlobalPatchDataset' /mnt/mrdr/crism_classification/data/dataset.py`
Expected: a single hit around line 300 in a docstring.

Read the surrounding context to understand what the comment is saying. The reference is in a docstring describing normalization parity between `CRISMGlobalPatchDataset` and a sibling class.

- [ ] **Step 2: Update the comment in `data/dataset.py`**

Open `data/dataset.py`, find the line that contains `CRISMGlobalPatchDataset` (around line 300), and rewrite the surrounding sentence to refer to the equivalent normalization scheme without naming the deleted class. The replacement should be a minimal edit — preserve every other word in the docstring. Example:

Before:
```
    Applies the same normalization as CRISMGlobalPatchDataset (clip to [0, 0.5]).
```

After:
```
    Applies the same clipping (to [0, 0.5]) as the pretraining patch cache builder.
```

- [ ] **Step 3: Delete the two files**

Run:
```bash
cd /mnt/mrdr/crism_classification && rm data/global_patch_dataset.py tests/test_global_patch_dataset.py
```

- [ ] **Step 4: Verify nothing else imports `CRISMGlobalPatchDataset` or `data.global_patch_dataset`**

Run: `grep -rn 'CRISMGlobalPatchDataset\|data\.global_patch_dataset\|from data\.global_patch_dataset\|import.*global_patch_dataset' /mnt/mrdr/crism_classification/ --include='*.py'`
Expected: zero hits. If any hit appears, investigate and update.

- [ ] **Step 5: Run the full test suite for the touched modules**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_cached_patch_dataset.py tests/test_build_global_patch_cache.py tests/test_spend_spatial_mae.py tests/test_denoising_spatial_mae.py tests/test_spatial_mae.py tests/test_dataset.py -v 2>&1 | tail -20`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add -A data/dataset.py data/global_patch_dataset.py tests/test_global_patch_dataset.py && git commit -m "$(cat <<'EOF'
chore(cache): remove streaming CRISMGlobalPatchDataset

The new CRISMCachedPatchDataset replaces it entirely. The supervised
classifier pipeline doesn't import it; pretraining scripts have been
updated. Fix the one stale docstring reference in data/dataset.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

`git add -A` is appropriate here because we have both modifications and deletions in one logical change. If the deletes don't stage cleanly, fall back to:
```bash
cd /mnt/mrdr/crism_classification && git rm data/global_patch_dataset.py tests/test_global_patch_dataset.py && git add data/dataset.py && git commit -m "..."
```

---

## Task 10: Final integration check

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full repo test suite**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/ -q 2>&1 | tail -20`
Expected: all tests PASS (or only pre-existing skips). No new failures introduced by this plan.

If a previously-passing test now fails, investigate before declaring done. Typical suspects:
- A test that imported `data.global_patch_dataset` and wasn't caught in Task 9's grep
- A test that referenced `cfg['patch_cache_dir']` expecting a specific path

- [ ] **Step 2: Verify the pretraining scripts can still parse their CLI**

Run:
```bash
cd /mnt/mrdr/crism_classification && \
  conda run -n crism python scripts/pretrain_spatial_mae_denoising.py --help > /dev/null && \
  conda run -n crism python scripts/pretrain_spatial_mae_spend.py --help > /dev/null && \
  echo "pretrain scripts parse OK"
```
Expected: `pretrain scripts parse OK`

- [ ] **Step 3: Verify the build script CLI**

Run:
```bash
cd /mnt/mrdr/crism_classification && \
  conda run -n crism python scripts/build_global_patch_cache.py --help > /dev/null && \
  echo "build script parses OK"
```
Expected: `build script parses OK`

- [ ] **Step 4: Final commit (only if there are outstanding fixes from steps 1–3)**

If steps 1–3 surfaced anything that needed fixing, commit those fixes with a `chore: ...` message. If everything is clean, no commit needed.

---

## Out of scope (do not implement in this plan)

The spec is explicit; mirroring it here:

- Cache for the labeled-patch supervised dataset — already exists, unrelated.
- Augmentation pipeline — pretraining doesn't use augmentation today.
- Multi-band reading optimizations on the streaming path — streaming dataset is deleted.
- Conversion of mrral tiles to BIP/BIL interleave — only the cache is needed.
- Backward-compat shims for `CRISMGlobalPatchDataset` imports — none exist outside the deleted code.
- Cache versioning / stale-cache detection — `shard_index.json` records build params, but no code consumes them for validation in this plan.
- Resume support for the cache builder — full rebuild is the recovery path.
- Parallel sharded reads during training (multiple shards mmap'd per worker simultaneously) — single-shard-at-a-time is already fast enough per the projection.
- Submitting the build job on HPC, then resubmitting pretraining — that's a manual operation step for the user once this plan lands. The plan delivers the code; the user runs `sbatch` on the new SLURM script.
