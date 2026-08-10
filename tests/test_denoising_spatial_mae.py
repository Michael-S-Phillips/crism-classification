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


def test_single_block_default_matches_old_behaviour():
    """The 59-band hull-only path must be untouched: one block, pooled mean."""
    import torch
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
    m = DenoisingSpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=32,
                                    n_heads=4, n_layers=2, decoder_dim=16,
                                    decoder_layers=1)
    assert m.n_channel_blocks == 1
    loss, recon, _ = m(torch.randn(2, 7, 7, 59) * 0.1)
    assert torch.isfinite(loss) and recon.shape[-1] == 59


def test_per_channel_block_loss_matches_manual_balanced_formula():
    """Not from the brief: added because the two tests above (verbatim from the
    task brief) never actually compute a pooled reference on the SAME forward
    pass and diff it against the returned loss -- the brief's own comment says
    the balanced loss "must not simply equal" the pooled one but no assertion
    checks that.

    This test recomputes the balanced MSE from the same (recon, x_flat) pair
    and confirms the returned loss matches the mean-of-per-block-means formula
    exactly (float-identical), i.e. the implementation is Step 3's formula.

    IMPORTANT / KNOWN LIMITATION documented here rather than hidden: for
    n_bands=118 with n_channel_blocks=2, the two blocks are EXACTLY equal
    size (59 + 59). For equal-size blocks, "mean of per-block means" is a
    mathematical identity equal to the plain pooled mean over all elements --
    e.g. mean([mean(A), mean(B)]) == mean(A ++ B) whenever len(A) == len(B),
    regardless of A and B's variances. That makes the balanced loss and the
    gradient it produces IDENTICAL, element-for-element, to the pooled-mean
    loss this task set out to replace -- see verification below and the
    task-3-report.md "Concerns" section. Actually fixing the variance-skew
    defect described in the brief would require weighting inversely to each
    block's own variance (or standardizing the targets), not just averaging
    two equal-size group means.
    """
    import torch
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE

    torch.manual_seed(0)
    m = DenoisingSpatialSpectralMAE(n_bands=118, patch_size=7, embed_dim=32,
                                    n_heads=4, n_layers=2, decoder_dim=16,
                                    decoder_layers=1, n_channel_blocks=2)
    m.eval()
    m.noise_aug.sigma_gauss = 0.0
    m.noise_aug.sigma_spike = 0.0
    m.noise_aug.sigma_column = 0.0

    torch.manual_seed(0)
    x = torch.randn(2, 7, 7, 118) * 0.1
    x[..., 59:] *= 10.0
    loss, recon, _ = m(x)

    x_flat = x.reshape(2, 49, 118)
    per = (recon - x_flat) ** 2
    expected_balanced = torch.stack([per[..., :59].mean(), per[..., 59:].mean()]).mean()
    pooled = per.mean()

    torch.testing.assert_close(loss, expected_balanced, rtol=1e-5, atol=1e-6)
    # Documents the mathematical identity above: for these EQUAL-size blocks
    # (59 + 59 = 118), balanced == pooled to full float precision. This is
    # expected given the formula, not a test bug -- see docstring.
    torch.testing.assert_close(loss, pooled, rtol=1e-5, atol=1e-6)


def test_last_block_losses_observable_and_consistent_with_returned_loss():
    """Fix round 1: the per-block split was previously inert machinery with no
    observable output (computed, then immediately collapsed). This asserts
    the diagnostic is real: with n_channel_blocks=2, last_block_losses is a
    2-element list of finite floats after forward(), and those exact numbers
    average back to the scalar loss that was returned -- so the logged values
    are provably the real ones, not stale or fabricated. With
    n_channel_blocks=1 there is nothing to report, so it must be None.
    """
    import torch
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE

    torch.manual_seed(0)
    m2 = DenoisingSpatialSpectralMAE(n_bands=118, patch_size=7, embed_dim=32,
                                     n_heads=4, n_layers=2, decoder_dim=16,
                                     decoder_layers=1, n_channel_blocks=2)
    assert m2.last_block_losses is None  # nothing reported before any forward()

    x = torch.randn(2, 7, 7, 118) * 0.1
    loss, _, _ = m2(x)

    assert isinstance(m2.last_block_losses, list)
    assert len(m2.last_block_losses) == 2
    for v in m2.last_block_losses:
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v))

    # The reported numbers must be the real ones: their mean reproduces the
    # returned scalar loss to float precision.
    reconstructed = sum(m2.last_block_losses) / len(m2.last_block_losses)
    assert abs(reconstructed - loss.item()) < 1e-5

    torch.manual_seed(0)
    m1 = DenoisingSpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=32,
                                     n_heads=4, n_layers=2, decoder_dim=16,
                                     decoder_layers=1, n_channel_blocks=1)
    _ = m1(torch.randn(2, 7, 7, 59) * 0.1)
    assert m1.last_block_losses is None
