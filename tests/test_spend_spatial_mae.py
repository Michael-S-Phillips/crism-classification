"""Tests for the SPEND-style spatial-spectral MAE."""
import pytest
import torch

from models.spend_spatial_mae import compute_spectral_mask_ratio, SpendSpatialSpectralMAE


class TestSpectralMaskSchedule:
    """Anneal schedule for spectral_mask_ratio over the training run."""

    def test_returns_base_before_anneal_start(self):
        assert compute_spectral_mask_ratio(
            epoch=100, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.5

    def test_returns_base_just_before_anneal_start(self):
        assert compute_spectral_mask_ratio(
            epoch=160, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.5

    def test_returns_base_at_anneal_start(self):
        # At epoch == anneal_start_epoch the formula still evaluates to base
        # (anneal_end - epoch) / (anneal_end - anneal_start) = 20/20 = 1
        assert compute_spectral_mask_ratio(
            epoch=161, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.5

    def test_linear_interpolation_mid_range(self):
        # epoch 170 is 11/20 of the way to anneal_end → ratio = 0.5 * 11/20
        assert compute_spectral_mask_ratio(
            epoch=170, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == pytest.approx(0.275)

    def test_near_end_of_anneal(self):
        # epoch 180 is 1/20 of the way out → ratio = 0.5 * 1/20 = 0.025
        assert compute_spectral_mask_ratio(
            epoch=180, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == pytest.approx(0.025)

    def test_returns_zero_at_anneal_end(self):
        assert compute_spectral_mask_ratio(
            epoch=181, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.0

    def test_returns_zero_after_anneal_end(self):
        assert compute_spectral_mask_ratio(
            epoch=200, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.0


@pytest.fixture
def model():
    return SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2,
        mask_ratio=0.75,
        spectral_mask_ratio=0.5,
    )


class TestSkeletonAndAttributes:
    def test_instantiates_with_expected_attributes(self, model):
        assert model.n_bands == 59
        assert model.n_tokens == 49
        assert model.mask_ratio == 0.75
        assert model.spectral_mask_ratio == 0.5

    def test_inherits_encoder_state_dict_method(self, model):
        # Inherited from SpatialSpectralMAE; must still work for downstream loading
        state = model.encoder_state_dict()
        assert any(k.startswith('band_embed') for k in state)
        assert any(k.startswith('encoder.') for k in state)


class TestBandPartition:
    """The random per-batch band partition splits 59 bands into input/target."""

    def test_shape_and_dtype(self, model):
        target_mask = model._partition_bands(device=torch.device('cpu'))
        assert target_mask.shape == (59,)
        assert target_mask.dtype == torch.bool

    def test_target_count_at_ratio_half(self, model):
        # With ratio 0.5 and 59 bands: round(59 * 0.5) = 30 → 30 target bands.
        model.spectral_mask_ratio = 0.5
        target_mask = model._partition_bands(device=torch.device('cpu'))
        assert int(target_mask.sum().item()) == 30

    def test_target_count_at_ratio_zero(self, model):
        model.spectral_mask_ratio = 0.0
        target_mask = model._partition_bands(device=torch.device('cpu'))
        assert int(target_mask.sum().item()) == 0

    def test_unbiased_over_many_samples(self, model):
        """Every band index appears in the target-half across enough samples."""
        model.spectral_mask_ratio = 0.5
        counts = torch.zeros(59, dtype=torch.long)
        torch.manual_seed(0)
        for _ in range(1000):
            counts += model._partition_bands(device=torch.device('cpu')).long()
        # Each band should appear in target-half ~500 times.
        # Generous bound: every band hits target-half in ≥ 10 of 1000 samples.
        assert counts.min().item() >= 10, f"min count={counts.min().item()} — partition is biased"

    def test_samples_differ_across_calls(self, model):
        model.spectral_mask_ratio = 0.5
        torch.manual_seed(0)
        m1 = model._partition_bands(device=torch.device('cpu'))
        m2 = model._partition_bands(device=torch.device('cpu'))
        assert not torch.equal(m1, m2), "Two consecutive partitions should differ"


class TestForwardPass:
    """SPEND forward pass: shape, loss localization, masking interaction."""

    def test_forward_returns_loss_recon_mask(self, model):
        model.train()
        B = 4
        x = torch.randn(B, 7, 7, 59) * 0.1
        out = model(x)
        assert isinstance(out, tuple) and len(out) == 3
        loss, recon, mask = out
        assert loss.ndim == 0
        assert recon.shape == (B, 49, 59)
        assert mask.shape == (B, 49)
        assert mask.dtype == torch.bool

    def test_loss_is_finite_and_positive(self, model):
        model.train()
        B = 8
        torch.manual_seed(0)
        x = torch.randn(B, 7, 7, 59) * 0.1
        loss, _, _ = model(x)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_loss_is_target_band_only_at_ratio_half(self, model):
        """At spectral_mask_ratio=0.5, the returned loss equals MSE only on
        the target-band positions of the reconstruction (not on input bands)."""
        model.eval()
        model.spectral_mask_ratio = 0.5
        torch.manual_seed(42)
        B = 4
        x = torch.randn(B, 7, 7, 59) * 0.1

        # Patch _partition_bands to return a deterministic mask so we can
        # reconstruct the expected loss after the call.
        chosen_targets = torch.zeros(59, dtype=torch.bool)
        chosen_targets[torch.arange(0, 59, 2)] = True  # even indices = target
        model._partition_bands = lambda device: chosen_targets.to(device)

        loss, recon, _ = model(x)
        x_flat = x.reshape(B, 49, 59)
        expected_loss = (
            (recon[:, :, chosen_targets] - x_flat[:, :, chosen_targets]) ** 2
        ).mean()
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)

    def test_mask_ratio_75_percent_spatial_tokens_hidden(self, model):
        """Spatial masking is preserved from the parent class: ~75% hidden."""
        model.train()
        B = 50
        torch.manual_seed(0)
        x = torch.randn(B, 7, 7, 59)
        _, _, mask = model(x)
        fraction_masked = mask.float().mean().item()
        expected = int(49 * 0.75) / 49
        assert abs(fraction_masked - expected) < 0.01, (
            f"got {fraction_masked}, expected ≈ {expected}"
        )

    def test_encoder_sees_zeroed_target_bands(self, model):
        """Two forward passes that differ only at target-band positions
        should produce identical encoder visible-token outputs, because
        target bands are zeroed before encoding."""
        model.eval()
        model.spectral_mask_ratio = 0.5
        chosen_targets = torch.zeros(59, dtype=torch.bool)
        chosen_targets[torch.arange(0, 59, 2)] = True
        model._partition_bands = lambda device: chosen_targets.to(device)

        torch.manual_seed(0)
        x_a = torch.randn(2, 7, 7, 59) * 0.1
        x_b = x_a.clone()
        # Perturb target bands only
        x_b[..., chosen_targets] += 5.0

        # Same spatial mask so we compare apples to apples.
        torch.manual_seed(123)
        _, recon_a, _ = model(x_a)
        torch.manual_seed(123)
        _, recon_b, _ = model(x_b)

        # Encoder input is band-masked → identical at every position →
        # recon should be identical for both inputs.
        torch.testing.assert_close(recon_a, recon_b, rtol=1e-5, atol=1e-5)
