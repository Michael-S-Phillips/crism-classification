"""CRISM-physics-motivated noise augmentation for denoising MAE pre-training.

Three corruption components applied additively to a clean patch:
  1. ε_gauss   — per-pixel, per-band, independent 𝒩(0, σ_gauss²)
  2. ε_spike   — band-localized perturbation centered at the 1 µm detector seam.
                 One scalar magnitude per patch (𝒩(0, σ_spike²)); the spike
                 profile is a Gaussian bump in band space centered at
                 spike_center_band, zeroed outside spike_band_range.
  3. ε_column  — one bias per (column, band) drawn from 𝒩(0, σ_column²),
                 broadcast across all rows of that column.

σ values are estimated from the labeled-polygon parquet — see the spec. They
are ABSOLUTE scalars, calibrated against 59-band hull-continuum-removed
(hull-CR) data, whose global std is ~0.0705 (`data/mrral_cr_scales.json`,
`hull_std`). That makes them representation-dependent: sigma_gauss=0.0087 is
really "12.3% of hull-CR's own std", not an absolute physical constant. Fed
the standardised 118-channel dual-continuum representation (hull-CR ⊕
linear-CR, each block divided by its own global std so both sit at std ≈
1.0 — see `data.continuum_removal.dual_continuum`), these same absolute
sigmas would land at ~0.9% of unit data std: a ~14x weakening of the
denoising corruption relative to what they were tuned for. To preserve the
original *relative* noise level, this module scales all three sigmas by
`noise_scale = 1 / CR_SCALES['hull_std']` whenever it detects the
118-channel dual representation (see `dual_continuum` below); for the
original 59-band representation `noise_scale` is exactly 1.0, i.e. the
sigmas are unchanged bit-for-bit.

The module is a no-op in eval mode, so the same model can be used for
inference without corruption.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from data import continuum_removal as _cr


def _single_bump(
    n_bands: int,
    center: float,
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


def _spike_profile(
    n_bands: int,
    center: int,
    fwhm_bands: float,
    band_range: Tuple[int, int],
    dual_continuum: bool = False,
    block_size: int = 59,
) -> torch.Tensor:
    """A 1-D Gaussian bump (or, in dual mode, two mirrored bumps) in band space.

    Single-block (default): one bump, peak 1.0 at `center`, zeroed outside
    `band_range`. Dual mode (118-channel hull-CR ⊕ linear-CR): the 1 µm
    detector seam appears at the same relative position in BOTH blocks, so
    a second, identical bump is added at `center + block_size` over
    `(band_range[0] + block_size, band_range[1] + block_size)` — i.e. the
    linear block's own copy of the seam. Without this, only the hull block
    (bands 0..block_size-1) would get a spike and the linear block's
    corresponding seam bands would be silently left uncorrupted.
    """
    profile = _single_bump(n_bands, center, fwhm_bands, band_range)
    if dual_continuum:
        lo, hi = band_range
        mirrored = _single_bump(
            n_bands, center + block_size, fwhm_bands,
            (lo + block_size, hi + block_size),
        )
        profile = torch.maximum(profile, mirrored)
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
        dual_continuum: Optional[bool] = None,
    ):
        super().__init__()

        # How the module learns it is in dual-continuum (118-channel) mode:
        # `dual_continuum` defaults to None, which auto-detects from
        # `n_bands == 118`. Every existing caller (DenoisingSpatialSpectralMAE,
        # train_contrastive.py) already threads its own `n_bands` through to
        # this constructor with no other change, so the fix "just works" for
        # 118-channel runs with zero plumbing elsewhere and the 59-band
        # default is untouched. The parameter is still exposed (rather than
        # hardcoding the `== 118` check inline) so a caller with some other,
        # unrelated 118-band representation can explicitly opt out
        # (`dual_continuum=False`) instead of silently mis-firing.
        if dual_continuum is None:
            dual_continuum = (n_bands == 118)
        self.dual_continuum = dual_continuum

        # Preserve the corruption-to-signal ratio the sigmas were tuned at.
        # Both continuum-removal blocks are standardised to std ≈ 1.0 by
        # `data.continuum_removal.dual_continuum`, so a single scalar
        # multiplier (1 / hull_std) restores the original relative noise
        # level for the whole 118-channel tensor. The two blocks' measured
        # stds differ slightly (0.9936 hull vs 0.9655 linear on real data),
        # so this single scalar leaves the linear block ~3% "hotter" in
        # relative terms than the hull block. That residual is far below the
        # precision of the sigma estimates themselves (which came from a
        # 59-band-only calibration in the first place) — this is a
        # deliberate simplification, not an oversight. Do NOT "fix" it by
        # bookkeeping separate scales per block.
        noise_scale = (1.0 / _cr.CR_SCALES['hull_std']) if dual_continuum else 1.0

        self.sigma_gauss = sigma_gauss * noise_scale
        self.sigma_spike = sigma_spike * noise_scale
        self.sigma_column = sigma_column * noise_scale
        self.n_bands = n_bands
        self.patch_size = patch_size

        profile = _spike_profile(
            n_bands, spike_center_band, spike_fwhm_bands, spike_band_range,
            dual_continuum=dual_continuum,
        )
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
