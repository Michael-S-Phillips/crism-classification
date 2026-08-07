import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_spectral_augmentation_preserves_shape():
    from training.augmentations import SpectralAugmentation
    aug = SpectralAugmentation(noise_std=0.005, band_dropout=0.15, shift_std=0.01)
    aug.train()
    x = torch.ones(59)
    out = aug(x)
    assert out.shape == (59,), f"Shape changed: {out.shape}"


def test_spectral_augmentation_applies_noise():
    from training.augmentations import SpectralAugmentation
    torch.manual_seed(42)
    aug = SpectralAugmentation(noise_std=0.1, band_dropout=0.0, shift_std=0.0)
    aug.train()
    x = torch.zeros(59)
    out = aug(x)
    assert not torch.allclose(out, x), "Noise should modify the spectrum"


def test_spectral_augmentation_band_dropout():
    from training.augmentations import SpectralAugmentation
    torch.manual_seed(0)
    aug = SpectralAugmentation(noise_std=0.0, band_dropout=0.5, shift_std=0.0)
    aug.train()
    x = torch.ones(59)
    out = aug(x)
    n_zeros = (out == 0).sum().item()
    assert n_zeros > 0, "Band dropout should zero some bands"
    assert n_zeros < 59, "Band dropout should not zero all bands"


def test_no_augmentation_in_eval_mode():
    from training.augmentations import SpectralAugmentation
    aug = SpectralAugmentation(noise_std=1.0, band_dropout=0.5, shift_std=1.0)
    aug.eval()
    x = torch.ones(59)
    out = aug(x)
    assert torch.allclose(out, x), "No augmentation should be applied in eval mode"
