"""Tests for the signal/noise decomposition encoder."""
import pytest
import torch

from models.decomp_spatial_vit import DecompSpVit


@pytest.fixture
def model():
    return DecompSpVit(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.0,
        T_min=0.3, T_max=1.0,
    )


def test_forward_shapes(model):
    B = 4
    x = torch.randn(B, 7, 7, 59)
    out = model(x)

    # Documented forward return tuple: (logits, s_hat, T_hat, b_hat, eps_hat, x_hat)
    logits, s_hat, T_hat, b_hat, eps_hat, x_hat = out
    assert logits.shape == (B, 5)
    assert s_hat.shape == (B, 49, 59)
    assert T_hat.shape == (B, 59)
    assert b_hat.shape == (B, 59)
    assert eps_hat.shape == (B, 49, 59)
    assert x_hat.shape == (B, 49, 59)


def test_T_hat_is_bounded(model):
    B = 4
    x = torch.randn(B, 7, 7, 59) * 5.0   # noisy input
    _, _, T_hat, _, _, _ = model(x)
    # T_hat is sigmoid-scaled to [T_min, T_max]; both bounds are inclusive
    assert torch.all(T_hat >= 0.3 - 1e-6)
    assert torch.all(T_hat <= 1.0 + 1e-6)


def test_reconstruction_equation(model):
    """x_hat MUST equal T_hat[:,None,:] * s_hat + b_hat[:,None,:] + eps_hat."""
    B = 2
    x = torch.randn(B, 7, 7, 59)
    _, s_hat, T_hat, b_hat, eps_hat, x_hat = model(x)
    expected = T_hat[:, None, :] * s_hat + b_hat[:, None, :] + eps_hat
    torch.testing.assert_close(x_hat, expected, rtol=1e-5, atol=1e-5)


def test_classifier_reads_center_pixel_embedding(model):
    """logits at batch i must depend on encoder output at center-pixel token of batch i."""
    B = 3
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59)
    logits = model(x)[0]
    assert logits.shape == (B, 5)
    # Perturbing the center pixel should change logits more than perturbing a corner
    x2_center = x.clone(); x2_center[:, 3, 3, :] += 1.0
    x2_corner = x.clone(); x2_corner[:, 0, 0, :] += 1.0
    delta_center = (model(x2_center)[0] - logits).abs().sum().item()
    delta_corner = (model(x2_corner)[0] - logits).abs().sum().item()
    assert delta_center > delta_corner, (
        f"Classifier should be more sensitive to center pixel: "
        f"delta_center={delta_center:.4f}, delta_corner={delta_corner:.4f}"
    )


def test_load_mae_encoder_checkpoint():
    """Encoder state from a SpatialSpectralMAE checkpoint should load cleanly."""
    import os
    ckpt_path = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/checkpoints/spatial_mae_128d_6l_best.pt'
    if not os.path.exists(ckpt_path):
        pytest.skip(f"MAE checkpoint not available at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = DecompSpVit(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.1,
    )
    missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
    # No unexpected keys (encoder state matches the encoder submodule)
    assert unexpected == [], f"Unexpected keys when loading MAE encoder: {unexpected}"
    # Missing keys are allowed (the new heads aren't in the MAE checkpoint), but the
    # encoder's core weights should all be present
    assert not any(k.startswith('encoder.encoder') for k in missing), \
        f"Core encoder weights missing: {[k for k in missing if k.startswith('encoder.encoder')]}"


def test_param_groups_split_encoder_and_heads(model):
    """get_param_groups should return distinct groups for encoder vs new heads."""
    groups = model.get_param_groups(head_lr=1e-3, encoder_lr=1e-5)
    assert len(groups) == 2
    encoder_lr = groups[0]['lr']; head_lr = groups[1]['lr']
    assert encoder_lr == 1e-5
    assert head_lr == 1e-3
    # Every encoder param should be in the encoder group, no overlap
    encoder_param_ids = {id(p) for p in groups[0]['params']}
    head_param_ids = {id(p) for p in groups[1]['params']}
    assert encoder_param_ids.isdisjoint(head_param_ids)
    # Total params should match the model
    total = sum(p.numel() for p in model.parameters())
    grouped = sum(p.numel() for g in groups for p in g['params'])
    assert grouped == total
