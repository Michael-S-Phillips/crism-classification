"""Tests for CRISMCachedPatchDataset + helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.cached_patch_dataset import compute_valid_centers
from data.cached_patch_dataset import CRISMCachedPatchDataset


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
        shard_dir = _make_fake_cache(tmp_path, n_shards=2, n_per_shard=20)
        ds_a = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=True, seed=0)
        ds_b = CRISMCachedPatchDataset(shard_dir, normalize=False, shuffle=True, seed=1)
        patches_a = list(ds_a)
        patches_b = list(ds_b)
        # Different seeds → different first patch (overwhelmingly probable for
        # a 40-patch cache with 50% chance of identical first index)
        all_match = all(torch.equal(a, b) for a, b in zip(patches_a, patches_b))
        assert not all_match


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
