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
