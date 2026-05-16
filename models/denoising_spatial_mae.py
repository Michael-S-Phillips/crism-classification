"""
Denoising variant of SpatialSpectralMAE.

Differs from the parent class in three ways:
  1. Input is corrupted by CrismNoiseAugmentation before encoding
  2. Reconstruction target is x_clean (not the encoder input)
  3. Loss is averaged over all 49 positions (not masked-only)

Encoder state dict is structurally identical to the parent's, so the
resulting checkpoint loads into SpatialSpectralClassifier / DecompSpVit /
DecompSpVitAdv unchanged via load_encoder_state_dict.

Spec: docs/superpowers/specs/2026-05-16-denoising-mae-design.md
"""
from typing import Tuple

import torch

from models.noise_augmentation import CrismNoiseAugmentation
from models.spatial_mae import SpatialSpectralMAE


class DenoisingSpatialSpectralMAE(SpatialSpectralMAE):
    """SpatialSpectralMAE + denoising objective.

    The architecture (encoder + decoder + projections + mask token) is inherited
    from SpatialSpectralMAE unchanged. Only the forward pass is overridden to
    insert the noise augmentation and rewrite the loss aggregation.
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        decoder_dim: int = 64,
        decoder_layers: int = 2,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
        sigma_gauss: float = 0.0087,
        sigma_spike: float = 0.0058,
        sigma_column: float = 0.0049,
        spike_center_band: int = 15,
        spike_fwhm_bands: float = 3.0,
        spike_band_range: Tuple[int, int] = (13, 17),
    ):
        super().__init__(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            decoder_dim=decoder_dim, decoder_layers=decoder_layers,
            mask_ratio=mask_ratio, dropout=dropout,
        )
        self.noise_aug = CrismNoiseAugmentation(
            sigma_gauss=sigma_gauss,
            sigma_spike=sigma_spike,
            sigma_column=sigma_column,
            spike_center_band=spike_center_band,
            spike_fwhm_bands=spike_fwhm_bands,
            spike_band_range=spike_band_range,
            n_bands=n_bands,
            patch_size=patch_size,
        )

    def forward(self, x_clean: torch.Tensor):
        """Returns (loss, recon, mask).

        loss:  scalar MSE on all 49 positions of recon vs x_clean
        recon: (B, n_tokens, n_bands) — reconstructed spectra at every position
        mask:  (B, n_tokens) bool — True = was masked at the encoder
        """
        x_corrupted = self.noise_aug(x_clean)

        B = x_clean.shape[0]
        device = x_clean.device
        N = self.n_tokens

        visible_ids, masked_ids, mask = self._mask_tokens(B, device)

        enc_out = self.encoder.encode_visible(x_corrupted, visible_ids)
        enc_proj = self.enc_to_dec(enc_out[:, 1:])

        decoder_tokens = self.mask_token.expand(B, N, -1).clone()
        scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        decoder_tokens.scatter_(1, scatter_idx, enc_proj)
        pos_ids = torch.arange(1, N + 1, device=device)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed(pos_ids)
        decoded = self.decoder(decoder_tokens)
        recon = self.reconstruction_head(decoded)

        x_flat = x_clean.reshape(B, N, self.n_bands)
        loss = ((recon - x_flat) ** 2).mean()

        return loss, recon, mask
