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
        n_channel_blocks: int = 1,
    ):
        super().__init__(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            decoder_dim=decoder_dim, decoder_layers=decoder_layers,
            mask_ratio=mask_ratio, dropout=dropout,
        )
        assert n_bands % n_channel_blocks == 0, (
            f"n_bands ({n_bands}) must be divisible by n_channel_blocks "
            f"({n_channel_blocks})"
        )
        self.n_channel_blocks = n_channel_blocks
        # Per-block losses from the most recent forward() call, as plain
        # floats (detached, CPU-side -- never part of the autograd graph).
        # None when n_channel_blocks == 1 (nothing to report).
        #
        # Why this exists: for equal-sized blocks, averaging per-block means
        # is mathematically identical to the plain pooled mean (see
        # task-3-report.md "Critical finding") -- the branch is inert with
        # respect to the loss VALUE and its gradient. Its only remaining
        # purpose is diagnostic observability: if the two block losses ever
        # diverge sharply (e.g. one block's cache was written un-standardised,
        # or data/mrral_cr_scales.json has gone stale relative to the
        # transform actually applied upstream), that divergence is visible
        # here even though it does not change what the optimizer sees.
        self.last_block_losses: "list[float] | None" = None
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
        if self.n_channel_blocks == 1:
            loss = ((recon - x_flat) ** 2).mean()
            self.last_block_losses = None
        else:
            # Per-block MSE, then averaged. A single pooled mean would weight the
            # objective by each block's variance -- with hull-CR at std 0.0705
            # and linear-CR at 0.1726 (2.45x), the pretrain would spend itself on
            # the linear block, which is the raw-space MAE failure mode relocated.
            #
            # NOTE: for equal-sized blocks this averaging is mathematically
            # identical to the plain pooled mean (see task-3-report.md), so it
            # changes neither the loss value nor its gradient. The per-block
            # split is kept only as a diagnostic (self.last_block_losses,
            # logged by the pretrain script) -- divergence between block
            # losses can reveal a cache written un-standardised or a stale
            # data/mrral_cr_scales.json, even though it doesn't rebalance
            # the objective itself.
            per = (recon - x_flat) ** 2
            bs = self.n_bands // self.n_channel_blocks
            block_losses = [per[..., i * bs:(i + 1) * bs].mean()
                            for i in range(self.n_channel_blocks)]
            loss = torch.stack(block_losses).mean()
            self.last_block_losses = [bl.detach().cpu().item() for bl in block_losses]

        return loss, recon, mask
