"""
Spectral augmentation transforms for mrral hyperspectral data.
Applied only in training mode (aug.train()); identity in eval mode.
"""
import torch
import torch.nn as nn


class SpectralAugmentation(nn.Module):
    """
    Stochastic spectral augmentation for 1D reflectance spectra.

    Three independent transforms applied in sequence:
    1. Gaussian noise:   x += N(0, noise_std)
    2. Band dropout:     randomly zero out each band with probability band_dropout
    3. Spectral shift:   x += N(0, shift_std) (same offset all bands)

    In eval mode, all transforms are disabled (returns input unchanged).
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        band_dropout: float = 0.10,
        shift_std: float = 0.005,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.band_dropout = band_dropout
        self.shift_std = shift_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_bands,) or (batch, n_bands)"""
        if not self.training:
            return x
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.band_dropout > 0:
            mask = torch.bernoulli(
                torch.full(x.shape, 1 - self.band_dropout, device=x.device)
            )
            x = x * mask
        if self.shift_std > 0:
            shift_shape = x.shape[:-1] + (1,) if x.dim() > 1 else (1,)
            x = x + torch.randn(shift_shape, device=x.device) * self.shift_std
        return x
