"""Task 5: CR denoising-MAE pretrain smoke tests.

`scripts/pretrain_spatial_mae_denoising.py --continuum_removed` continuum-removes
each cached patch BEFORE the denoising corruption, so the MAE reconstructs in CR
space. CR is applied on-read by CRISMCachedPatchDataset(continuum_removed=True)
ahead of the per-patch normalization.

Two checks:
  1. The cache reader's CR-on-read output matches data.continuum_removal on the
     raw patch (the reconstruction target is in CR space).
  2. One CPU pretrain step with --continuum_removed on a tiny synthetic cache
     runs to completion and the logged denoising loss is finite.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.continuum_removal import continuum_removed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_shard(cache_dir: str, n: int = 32, seed: int = 0) -> np.ndarray:
    """Write one global_patches_000.npy shard of raw reflectance patches."""
    rng = np.random.default_rng(seed)
    patches = rng.uniform(0.0, 0.5, size=(n, 7, 7, 59)).astype(np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, 'global_patches_000.npy'), patches)
    return patches


def test_cached_dataset_cr_on_read_matches_module(tmp_path):
    """CRISMCachedPatchDataset(continuum_removed=True, normalize=False) yields CR
    patches identical to data.continuum_removal.continuum_removed(raw_patch)."""
    from data.cached_patch_dataset import CRISMCachedPatchDataset

    cache_dir = str(tmp_path / 'cache')
    patches = _make_shard(cache_dir, n=16, seed=3)

    ds = CRISMCachedPatchDataset(
        shard_dir=cache_dir, normalize=False, shuffle=False,
        continuum_removed=True)
    got = np.stack([t.numpy() for t in ds])

    assert got.shape == (16, 7, 7, 59)
    # Shuffle off → same order as the shard.
    for i in range(len(patches)):
        np.testing.assert_allclose(
            got[i], continuum_removed(patches[i]), rtol=0, atol=1e-6)
    # CR is a valid representation: finite, never > 1.0001.
    assert np.all(np.isfinite(got))
    assert got.max() <= 1.0001

    # Raw mode (CR off) is byte-identical to the stored patches.
    ds_raw = CRISMCachedPatchDataset(
        shard_dir=cache_dir, normalize=False, shuffle=False)
    got_raw = np.stack([t.numpy() for t in ds_raw])
    np.testing.assert_array_equal(got_raw, patches)


def test_pretrain_cr_one_step_finite_loss(tmp_path):
    """One epoch of the pretrain script with --continuum_removed runs on CPU and
    logs a finite denoising loss."""
    cache_dir = str(tmp_path / 'cache')
    ckpt_dir = str(tmp_path / 'ckpt')
    _make_shard(cache_dir, n=32, seed=1)

    cfg_path = tmp_path / 'config_smoke.yaml'
    cfg_path.write_text(
        f'global_patch_cache_dir: {cache_dir}\n'
        f'checkpoints_dir: {ckpt_dir}\n'
    )

    cmd = [
        sys.executable, os.path.join(_ROOT, 'scripts',
                                     'pretrain_spatial_mae_denoising.py'),
        '--config', str(cfg_path),
        '--continuum_removed',
        '--epochs', '1', '--warmup', '1',
        '--batch_size', '8', '--patches_per_epoch', '16',
        '--num_workers', '0',
        '--embed_dim', '32', '--n_heads', '2', '--n_layers', '2',
        '--decoder_dim', '16', '--decoder_layers', '1', '--mask_ratio', '0.5',
        '--no_wandb', '--run_name', 'cr_smoke',
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    res = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True)

    assert res.returncode == 0, f'pretrain failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}'
    m = re.findall(r'denoising_loss=([0-9.eE+-]+)', res.stderr + res.stdout)
    assert m, f'no denoising_loss logged:\nSTDERR:\n{res.stderr}'
    loss = float(m[-1])
    assert np.isfinite(loss) and loss >= 0.0, f'non-finite loss {loss}'
