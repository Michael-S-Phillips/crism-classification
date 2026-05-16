"""CRISM-physics-motivated noise augmentation for denoising MAE pre-training.

Three corruption components applied additively to a clean patch:
  1. ε_gauss   — per-pixel, per-band, independent 𝒩(0, σ_gauss²)
  2. ε_spike   — band-localized perturbation centered at the 1 µm detector seam.
                 One scalar magnitude per patch (𝒩(0, σ_spike²)); the spike
                 profile is a Gaussian bump in band space centered at
                 spike_center_band, zeroed outside spike_band_range.
  3. ε_column  — one bias per (column, band) drawn from 𝒩(0, σ_column²),
                 broadcast across all rows of that column.

σ values are estimated from the labeled-polygon parquet — see the spec.

The module is a no-op in eval mode, so the same model can be used for
inference without corruption.
"""
from typing import Tuple

import torch
import torch.nn as nn


def _spike_profile(
    n_bands: int,
    center: int,
    fwhm_bands: float,
    band_range: Tuple[int, int],
) -> torch.Tensor:
    """A 1-D Gaussian bump in band space, peak 1.0 at `center`, zeroed outside band_range."""
    sigma = fwhm_bands / 2.355
    bands = torch.arange(n_bands, dtype=torch.float32)
    profile = torch.exp(-0.5 * ((bands - center) / sigma) ** 2)
    lo, hi = band_range
    mask = (bands < lo) | (bands > hi)
    profile[mask] = 0.0
    return profile


class CrismNoiseAugmentation(nn.Module):
    """Apply CRISM-physics-motivated corruption to a clean patch.

    Forward: (B, patch_size, patch_size, n_bands) → same shape with noise added.
    No-op in eval mode.
    """

    def __init__(
        self,
        sigma_gauss: float = 0.0087,
        sigma_spike: float = 0.0058,
        sigma_column: float = 0.0049,
        spike_center_band: int = 15,
        spike_fwhm_bands: float = 3.0,
        spike_band_range: Tuple[int, int] = (13, 17),
        n_bands: int = 59,
        patch_size: int = 7,
    ):
        super().__init__()
        self.sigma_gauss = sigma_gauss
        self.sigma_spike = sigma_spike
        self.sigma_column = sigma_column
        self.n_bands = n_bands
        self.patch_size = patch_size

        profile = _spike_profile(n_bands, spike_center_band, spike_fwhm_bands, spike_band_range)
        self.register_buffer('_spike_profile', profile, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # 1. Gaussian per-pixel, per-band
        if self.sigma_gauss > 0:
            eps = torch.randn_like(x) * self.sigma_gauss
        else:
            eps = torch.zeros_like(x)

        # 2. 1 µm spike — one magnitude per patch, weighted by profile
        if self.sigma_spike > 0:
            mag = torch.randn(B, device=device, dtype=dtype) * self.sigma_spike
            spike = mag.unsqueeze(-1) * self._spike_profile.to(device=device, dtype=dtype)
            eps = eps + spike.view(B, 1, 1, self.n_bands)

        # 3. Column bias — per-(column, band), broadcast over rows.
        if self.sigma_column > 0:
            col_bias = torch.randn(
                B, 1, self.patch_size, self.n_bands, device=device, dtype=dtype,
            ) * self.sigma_column
            eps = eps + col_bias

        return x + eps
