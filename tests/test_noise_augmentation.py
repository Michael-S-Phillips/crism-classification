"""Tests for the CRISM noise augmentation module."""
import pytest
import torch

from models.noise_augmentation import CrismNoiseAugmentation


@pytest.fixture
def aug():
    return CrismNoiseAugmentation(
        sigma_gauss=0.0087,
        sigma_spike=0.0058,
        sigma_column=0.0049,
        spike_center_band=15,
        spike_fwhm_bands=3,
        spike_band_range=(13, 17),
        n_bands=59,
        patch_size=7,
    )


def test_forward_shape_preserved(aug):
    aug.train()
    x = torch.randn(8, 7, 7, 59)
    out = aug(x)
    assert out.shape == x.shape


def test_eval_mode_disables_corruption(aug):
    aug.eval()
    x = torch.randn(4, 7, 7, 59)
    out = aug(x)
    torch.testing.assert_close(out, x)


def test_train_mode_changes_output(aug):
    aug.train()
    torch.manual_seed(0)
    x = torch.randn(4, 7, 7, 59) * 0.1
    out = aug(x)
    assert not torch.allclose(out, x), "training mode must produce a different output from input"


def test_empirical_gaussian_sigma(aug):
    """Empirical std of (corrupted - clean), averaged over many patches, should
    approximate the configured σ_gauss when other corruptions are off."""
    aug_only_gauss = CrismNoiseAugmentation(
        sigma_gauss=0.01, sigma_spike=0.0, sigma_column=0.0,
        n_bands=59, patch_size=7,
    )
    aug_only_gauss.train()
    torch.manual_seed(42)
    x = torch.zeros(2000, 7, 7, 59)
    out = aug_only_gauss(x)
    empirical_sigma = (out - x).std().item()
    assert 0.009 < empirical_sigma < 0.011, f"empirical σ = {empirical_sigma}"


def test_empirical_spike_only(aug):
    """Spike-only augmentation: only bands inside the spike range should be nonzero."""
    aug_only_spike = CrismNoiseAugmentation(
        sigma_gauss=0.0, sigma_spike=0.01, sigma_column=0.0,
        spike_center_band=15, spike_fwhm_bands=3, spike_band_range=(13, 17),
        n_bands=59, patch_size=7,
    )
    aug_only_spike.train()
    torch.manual_seed(0)
    x = torch.zeros(1000, 7, 7, 59)
    out = aug_only_spike(x)
    delta = out - x
    outside_max = delta[:, :, :, :13].abs().max().item()
    assert outside_max < 1e-6, f"corruption leaked outside spike band range: max={outside_max}"
    above_max = delta[:, :, :, 18:].abs().max().item()
    assert above_max < 1e-6, f"corruption leaked above spike band range: max={above_max}"
    inside_std = delta[:, :, :, 13:18].std().item()
    assert inside_std > 1e-5, f"no spike content inside range: std={inside_std}"


def test_empirical_column_only(aug):
    """Column-only: within a patch, all 7 rows of a given column should be perturbed identically."""
    aug_only_column = CrismNoiseAugmentation(
        sigma_gauss=0.0, sigma_spike=0.0, sigma_column=0.01,
        n_bands=59, patch_size=7,
    )
    aug_only_column.train()
    torch.manual_seed(0)
    x = torch.zeros(50, 7, 7, 59)
    out = aug_only_column(x)
    delta = out - x
    for i in range(5):
        for c in range(7):
            col = delta[i, :, c, :]
            row0 = col[0]
            for r in range(1, 7):
                torch.testing.assert_close(col[r], row0, rtol=1e-5, atol=1e-6)


def test_all_components_combine_additively(aug):
    """When all three σ values are set, outside the spike range the empirical std
    is approximately sqrt(σ_gauss² + σ_column²)."""
    aug.train()
    torch.manual_seed(0)
    x = torch.zeros(1000, 7, 7, 59)
    out = aug(x)
    delta = out - x
    outside_std = delta[:, :, :, 0:13].std().item()
    expected_outside = (0.0087 ** 2 + 0.0049 ** 2) ** 0.5
    assert 0.8 * expected_outside < outside_std < 1.2 * expected_outside, \
        f"outside-spike std = {outside_std}, expected ≈ {expected_outside}"
