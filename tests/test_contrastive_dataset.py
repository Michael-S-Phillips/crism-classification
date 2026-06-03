"""Tests for ContrastiveTripletDataset."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.contrastive_dataset import ContrastiveTripletDataset


def _make_pool(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 0.5, size=(n, 7, 7, 59)).astype(np.float32)


def test_shapes_in_memory():
    pos = _make_pool(10, seed=1)
    hard = _make_pool(20, seed=2)
    soft = _make_pool(30, seed=3)
    ds = ContrastiveTripletDataset(pos, hard, soft,
                                   n_hard_per_batch=4, n_soft_per_batch=6, seed=0)
    assert len(ds) == 10
    a, p, h, s = ds[0]
    assert a.shape == (7, 7, 59) and a.dtype == torch.float32
    assert p.shape == (7, 7, 59)
    assert h.shape == (4, 7, 7, 59)
    assert s.shape == (6, 7, 7, 59)


def test_anchor_neq_positive_when_pool_large():
    pos = _make_pool(50)
    hard = _make_pool(5)
    soft = _make_pool(5)
    ds = ContrastiveTripletDataset(pos, hard, soft,
                                   n_hard_per_batch=1, n_soft_per_batch=1, seed=42)
    # Sample many anchors; the positive should never be the SAME tensor as anchor
    for idx in range(len(pos)):
        a, p, _, _ = ds[idx]
        assert not torch.equal(a, p), f'anchor==positive at idx {idx}'


def test_reproducible_with_seed():
    pos = _make_pool(8, seed=1)
    hard = _make_pool(15, seed=2)
    soft = _make_pool(15, seed=3)
    ds1 = ContrastiveTripletDataset(pos, hard, soft,
                                    n_hard_per_batch=3, n_soft_per_batch=3, seed=7)
    ds2 = ContrastiveTripletDataset(pos, hard, soft,
                                    n_hard_per_batch=3, n_soft_per_batch=3, seed=7)
    a1, p1, h1, s1 = ds1[0]
    a2, p2, h2, s2 = ds2[0]
    assert torch.equal(a1, a2)
    assert torch.equal(p1, p2)
    assert torch.equal(h1, h2)
    assert torch.equal(s1, s2)


def test_single_positive_pool_returns_same_anchor():
    pos = _make_pool(1)
    hard = _make_pool(5)
    soft = _make_pool(5)
    ds = ContrastiveTripletDataset(pos, hard, soft,
                                   n_hard_per_batch=2, n_soft_per_batch=2, seed=0)
    a, p, _, _ = ds[0]
    assert torch.equal(a, p), 'single-positive pool: positive should equal anchor (only option)'


def test_load_from_npy_path(tmp_path):
    pos_path = tmp_path / 'pos.npy'
    hard_path = tmp_path / 'hard.npy'
    soft_path = tmp_path / 'soft.npy'
    np.save(pos_path, _make_pool(5, seed=1))
    np.save(hard_path, _make_pool(7, seed=2))
    np.save(soft_path, _make_pool(7, seed=3))
    ds = ContrastiveTripletDataset(str(pos_path), str(hard_path), str(soft_path),
                                   n_hard_per_batch=2, n_soft_per_batch=2)
    assert len(ds) == 5
    a, p, h, s = ds[0]
    assert a.shape == (7, 7, 59)
    assert h.shape == (2, 7, 7, 59)


def test_empty_pool_rejected():
    pos = _make_pool(3)
    empty = np.zeros((0, 7, 7, 59), dtype=np.float32)
    full = _make_pool(3)
    with pytest.raises(ValueError):
        ContrastiveTripletDataset(empty, full, full)
    with pytest.raises(ValueError):
        ContrastiveTripletDataset(full, empty, full)
    with pytest.raises(ValueError):
        ContrastiveTripletDataset(full, full, empty)


def test_bad_shape_rejected():
    bad = np.zeros((3, 5, 5, 59), dtype=np.float32)
    # wrong band dim
    bad2 = np.zeros((3, 7, 7, 30), dtype=np.float32)
    pos = _make_pool(3)
    # bad2 has wrong last-dim -> _load_patches raises
    with pytest.raises(ValueError):
        ContrastiveTripletDataset(bad2, pos, pos)


def test_dataloader_collates_cleanly():
    pos = _make_pool(20, seed=1)
    hard = _make_pool(10, seed=2)
    soft = _make_pool(10, seed=3)
    ds = ContrastiveTripletDataset(pos, hard, soft,
                                   n_hard_per_batch=3, n_soft_per_batch=3, seed=0)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    a, p, h, s = batch
    assert a.shape == (4, 7, 7, 59)
    assert p.shape == (4, 7, 7, 59)
    assert h.shape == (4, 3, 7, 7, 59)
    assert s.shape == (4, 3, 7, 7, 59)
