"""Tests for ContrastiveEncoder."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.contrastive_encoder import ContrastiveEncoder
from models.spatial_spectral_transformer import SpatialSpectralTransformer


def _small_encoder(**kw):
    """Cheap variant for fast tests."""
    defaults = dict(n_bands=59, patch_size=7, embed_dim=32,
                    n_heads=4, n_layers=2, proj_dim=16)
    defaults.update(kw)
    return ContrastiveEncoder(**defaults)


def test_forward_shape_and_normalized():
    enc = _small_encoder()
    x = torch.randn(4, 7, 7, 59)
    z = enc(x)
    assert z.shape == (4, 16)
    # L2-normalised → norms are 1
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encode_returns_center_token_embedding():
    enc = _small_encoder()
    enc.eval()                                   # disable dropout for determinism
    x = torch.randn(4, 7, 7, 59)
    with torch.no_grad():
        h = enc.encode(x)
        out = enc.encoder(x)
    assert h.shape == (4, 32)
    expected = out[:, enc._center_idx]
    assert torch.allclose(h, expected)


def test_center_idx_position():
    """Center index for a 7x7 patch with CLS prepended must be 49//2 + 1 = 25."""
    enc = _small_encoder(patch_size=7)
    assert enc._center_idx == 25


def test_load_encoder_state_dict_warm_start():
    src = SpatialSpectralTransformer(n_bands=59, patch_size=7,
                                     embed_dim=32, n_heads=4, n_layers=2)
    enc = _small_encoder()
    missing, unexpected = enc.load_encoder_state_dict(src.state_dict())
    assert missing == [] and unexpected == [], (
        f'expected exact match; missing={missing} unexpected={unexpected}')
    # After loading, encode() output should match the source's center-token output
    enc.eval(); src.eval()
    x = torch.randn(2, 7, 7, 59)
    with torch.no_grad():
        from_src = src(x)[:, enc._center_idx]
        from_enc = enc.encode(x)
    assert torch.allclose(from_src, from_enc, atol=1e-5)


def test_gradients_flow_through_proj_and_encoder():
    enc = _small_encoder()
    x = torch.randn(4, 7, 7, 59)
    z = enc(x)
    z.sum().backward()
    # Projection head parameters should have gradients
    for p in enc.proj.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    # At least one encoder parameter should also receive gradients
    enc_grads = [p.grad for p in enc.encoder.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in enc_grads), 'no encoder gradient'


def test_projection_dim_configurable():
    for d in (32, 64, 128):
        enc = _small_encoder(proj_dim=d)
        z = enc(torch.randn(2, 7, 7, 59))
        assert z.shape == (2, d)


def test_eval_mode_deterministic():
    enc = _small_encoder(dropout=0.5)
    enc.eval()
    x = torch.randn(2, 7, 7, 59)
    with torch.no_grad():
        a = enc(x)
        b = enc(x)
    assert torch.allclose(a, b)
