"""Tests for the denoising spatial-spectral MAE."""
import pytest
import torch

from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE


@pytest.fixture
def model():
    return DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2,
        mask_ratio=0.75,
        sigma_gauss=0.0087, sigma_spike=0.0058, sigma_column=0.0049,
    )


def test_forward_returns_loss_recon_mask(model):
    B = 4
    x = torch.randn(B, 7, 7, 59)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    loss, recon, mask = out
    assert loss.ndim == 0
    assert recon.shape == (B, 49, 59)
    assert mask.shape == (B, 49)
    assert mask.dtype == torch.bool


def test_loss_is_finite_and_positive(model):
    model.train()
    B = 8
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59) * 0.1
    loss, _, _ = model(x)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_loss_is_all_positions_not_masked_only(model):
    """The denoising MAE's loss is computed over all 49 positions of the patch.
    With σ=0 the recon target is exactly the encoder input, so the returned
    loss should equal the all-position MSE of recon vs input.
    """
    model.eval()
    model.noise_aug.sigma_gauss = 0.0
    model.noise_aug.sigma_spike = 0.0
    model.noise_aug.sigma_column = 0.0
    B = 4
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59) * 0.1
    loss, recon, mask = model(x)
    x_flat = x.reshape(B, 49, 59)
    expected_loss = ((recon - x_flat) ** 2).mean()
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)


def test_mask_ratio_respected(model):
    """The mask must hide approximately the configured fraction of tokens."""
    model.train()
    B = 50
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59)
    _, _, mask = model(x)
    fraction_masked = mask.float().mean().item()
    expected = int(49 * 0.75) / 49
    assert abs(fraction_masked - expected) < 0.01, f"got {fraction_masked}, expected ≈ {expected}"


def test_noise_aug_called_in_train_mode(model):
    """In train mode, the encoder sees corrupted input → different recon than σ=0 run."""
    model.train()
    torch.manual_seed(0)
    x = torch.randn(3, 7, 7, 59) * 0.1
    _, recon_with_noise, _ = model(x)

    model.noise_aug.sigma_gauss = 0.0
    model.noise_aug.sigma_spike = 0.0
    model.noise_aug.sigma_column = 0.0
    torch.manual_seed(0)
    _, recon_no_noise, _ = model(x)
    assert not torch.allclose(recon_with_noise, recon_no_noise, rtol=1e-3)


def test_encoder_state_dict_loads_into_classifier(model):
    """Pre-trained denoising MAE encoder must load into SpatialSpectralClassifier."""
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    classifier = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6,
    )
    encoder_state = model.encoder_state_dict()
    missing, unexpected = classifier.load_encoder_state_dict(encoder_state)
    assert unexpected == []
    assert not any(k.startswith('encoder.encoder') for k in missing), \
        f"core encoder weights missing: {[k for k in missing if k.startswith('encoder.encoder')]}"
