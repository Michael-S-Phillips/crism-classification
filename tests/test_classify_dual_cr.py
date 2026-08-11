"""Task 7: --dual_cr on scripts/classify_tile_supervised.py.

Two separate 59s live in this script and they must not be conflated:
  * the mrral SOURCE tile always has 59 bands — --dual_cr does not change the file
  * the MODEL's input width becomes 118 under --dual_cr, because dual_continuum()
    widens each spectrum

Overloading one constant either reads 118 bands from a 59-band file (loud) or
builds a 59-channel model for 118-channel input. A wrong-width map is the single
worst failure mode in this task, so the checkpoint's own first-layer weight is
treated as authoritative.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.continuum_removal import dual_continuum


def _synthetic_tile(H=12, W=13, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 0.5, size=(H, W, 59)).astype(np.float32)


def test_source_and_model_band_counts_are_separate_constants():
    """N_SRC_BANDS (tile read) must stay 59; MODEL_N_BANDS is what --dual_cr moves."""
    import scripts.classify_tile_supervised as cls
    assert cls.N_SRC_BANDS == 59
    assert cls.MODEL_N_BANDS == 59, 'module default must be the 59-band path'
    src = open(os.path.join(ROOT, 'scripts', 'classify_tile_supervised.py')).read()
    assert 'range(1, N_SRC_BANDS + 1)' in src, (
        'the source-tile read must use N_SRC_BANDS, never the model width')


# ── checkpoint channel guard ─────────────────────────────────────────────────

def test_ckpt_channel_guard_accepts_a_matching_checkpoint():
    from scripts.classify_tile_supervised import assert_ckpt_channels
    state = {'encoder.band_embed.weight': torch.zeros(256, 59)}
    assert_ckpt_channels(state, dual_cr=False)      # must not raise
    state118 = {'encoder.band_embed.weight': torch.zeros(256, 118)}
    assert_ckpt_channels(state118, dual_cr=True)    # must not raise


def test_ckpt_channel_guard_rejects_118ckpt_without_dual_cr():
    """A 118-channel checkpoint run without --dual_cr must abort, not map."""
    from scripts.classify_tile_supervised import assert_ckpt_channels
    state = {'encoder.band_embed.weight': torch.zeros(256, 118)}
    with pytest.raises(SystemExit) as e:
        assert_ckpt_channels(state, dual_cr=False)
    assert '118' in str(e.value) and 'dual_cr' in str(e.value)


def test_ckpt_channel_guard_rejects_59ckpt_with_dual_cr():
    """The free reverse guard: existing 59-band checkpoints cannot be run dual."""
    from scripts.classify_tile_supervised import assert_ckpt_channels
    state = {'encoder.band_embed.weight': torch.zeros(128, 59)}
    with pytest.raises(SystemExit) as e:
        assert_ckpt_channels(state, dual_cr=True)
    assert '59' in str(e.value)


def test_ckpt_channel_guard_fails_loudly_on_a_missing_key():
    """No key → refuse. Skipping the check is what produces a silently wrong map.

    Verified 2026-08-11 across checkpoints/*.pt: all 139 SpatialSpectral
    classifier checkpoints carry encoder.band_embed.weight at width 59. The 15
    that lack it (svit_*, vit_*) are other model families that already fail
    strict load_state_dict into SpatialSpectralClassifier, so refusing here
    breaks no working invocation.
    """
    from scripts.classify_tile_supervised import assert_ckpt_channels
    with pytest.raises(SystemExit) as e:
        assert_ckpt_channels({'head.weight': torch.zeros(7, 128)}, dual_cr=False)
    assert 'band_embed' in str(e.value)


def test_real_checkpoint_key_and_shape():
    """The guard's key/axis assumption, checked against a real checkpoint.

    encoder.band_embed is nn.Linear(n_bands, embed_dim), so weight is
    (embed_dim, n_bands) and shape[-1] IS the channel count.
    """
    p = os.path.join(ROOT, 'checkpoints', 'ft_7cls_handcore_level_best.pt')
    if not os.path.exists(p):
        pytest.skip('ft_7cls_handcore_level_best.pt not on this machine')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    state = ck['model_state'] if 'model_state' in ck else ck
    w = state['encoder.band_embed.weight']
    assert tuple(w.shape) == (256, 59), tuple(w.shape)
    from scripts.classify_tile_supervised import assert_ckpt_channels
    assert_ckpt_channels(state, dual_cr=False)


# ── inference transform ──────────────────────────────────────────────────────

def test_run_supervised_dual_cr_feeds_dual_continuum():
    """dual_cr must build the 118-channel cube with dual_continuum(), and the
    patches the model sees must equal dual_continuum of the padded raw tile.

    The whole-tile call and a per-patch call agree because the lstsq linear fit is
    row-independent, so this is not an approximation.
    """
    import scripts.classify_tile_supervised as cls
    from models.spatial_spectral_transformer import SpatialSpectralClassifier

    tile = _synthetic_tile(H=10, W=11, seed=4)
    H, W, _ = tile.shape
    seen = []

    class Spy(SpatialSpectralClassifier):
        def forward(self, x, *a, **kw):
            seen.append(x.detach().cpu().numpy().copy())
            return super().forward(x, *a, **kw)

    model = Spy(n_bands=118, patch_size=cls.PATCH_SIZE, n_classes=cls.N_CLASSES,
                embed_dim=16, n_heads=2, n_layers=1)
    model.eval()

    probs = cls.run_supervised(tile, model, torch.device('cpu'), batch_size=64,
                               continuum_removed=True, dual_cr=True)
    assert probs.shape == (H * W, cls.N_CLASSES)
    assert np.all(np.isfinite(probs))
    assert seen and seen[0].shape[-1] == 118, (
        f'model was fed {seen[0].shape[-1] if seen else "nothing"} channels, not 118')

    # Parity against the reference transform, patch by patch.
    P, PAD = cls.PATCH_SIZE, cls.PAD
    ref_cube = dual_continuum(
        np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant'))
    batch = np.concatenate(seen, axis=0)
    for k in range(min(20, batch.shape[0])):
        r, c = k // W, k % W
        np.testing.assert_allclose(batch[k], ref_cube[r:r + P, c:c + P, :],
                                   rtol=0, atol=1e-6)


def test_run_supervised_59band_path_unchanged():
    """The hull-only CR path must be bit-identical to before: 59 channels in."""
    import scripts.classify_tile_supervised as cls
    from data.continuum_removal import continuum_removed as hull_cr
    from models.spatial_spectral_transformer import SpatialSpectralClassifier

    tile = _synthetic_tile(H=10, W=10, seed=2)
    seen = []

    class Spy(SpatialSpectralClassifier):
        def forward(self, x, *a, **kw):
            seen.append(x.detach().cpu().numpy().copy())
            return super().forward(x, *a, **kw)

    model = Spy(n_bands=59, patch_size=cls.PATCH_SIZE, n_classes=cls.N_CLASSES,
                embed_dim=16, n_heads=2, n_layers=1)
    model.eval()
    cls.run_supervised(tile, model, torch.device('cpu'), batch_size=64,
                       continuum_removed=True)
    assert seen[0].shape[-1] == 59
    P, PAD = cls.PATCH_SIZE, cls.PAD
    ref = hull_cr(np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant'))
    W = tile.shape[1]
    batch = np.concatenate(seen, axis=0)
    for k in range(min(20, batch.shape[0])):
        r, c = k // W, k % W
        np.testing.assert_allclose(batch[k], ref[r:r + P, c:c + P, :],
                                   rtol=0, atol=1e-6)


def test_dual_cr_requires_continuum_removed_at_the_cli():
    import subprocess
    p = subprocess.run(
        [sys.executable, 'scripts/classify_tile_supervised.py',
         '--tile', '/nonexistent.img', '--dual_cr'],
        cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0
    assert '--dual_cr requires --continuum_removed' in p.stderr, p.stderr
