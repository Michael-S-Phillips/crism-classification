"""
SPEND-style spatial-spectral MAE (v4).

Pretraining objective: Noise2Noise between two random spectral-band views
of the same CRISM patch. Adjacent bands image the same surface but have
independent detector-noise realizations, so predicting one view from the
other forces the model to learn the underlying clean spectrum without any
synthetic-noise assumptions.

Spec: docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md
"""
from __future__ import annotations

import torch

from models.spatial_mae import SpatialSpectralMAE


def compute_spectral_mask_ratio(
    epoch: int,
    anneal_start_epoch: int = 161,
    anneal_end_epoch: int = 181,
    base: float = 0.5,
) -> float:
    """Schedule for the fraction of bands zeroed during SPEND pretraining.

    Three phases:
      - epoch < anneal_start_epoch: returns `base` (SPEND phase A)
      - anneal_start_epoch <= epoch < anneal_end_epoch: linearly decreases
        from `base` toward 0 (SPEND anneal phase B)
      - epoch >= anneal_end_epoch: returns 0.0 (plain MAE phase C)
    """
    if epoch < anneal_start_epoch:
        return base
    if epoch >= anneal_end_epoch:
        return 0.0
    return base * (anneal_end_epoch - epoch) / (anneal_end_epoch - anneal_start_epoch)


class SpendSpatialSpectralMAE(SpatialSpectralMAE):
    """SpatialSpectralMAE + SPEND spectral-partition Noise2Noise objective.

    The architecture (encoder + decoder + projections + mask token) is
    inherited from SpatialSpectralMAE unchanged. The forward pass is
    overridden to (1) sample a random per-batch band partition, (2) zero
    target-half bands in the encoder input, and (3) compute MSE loss only on
    the target-half bands of the reconstruction.
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
        spectral_mask_ratio: float = 0.5,
    ):
        super().__init__(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            decoder_dim=decoder_dim, decoder_layers=decoder_layers,
            mask_ratio=mask_ratio, dropout=dropout,
        )
        # Mutable attribute; the training loop updates it each epoch.
        self.spectral_mask_ratio: float = spectral_mask_ratio

    def _partition_bands(self, device: torch.device) -> torch.Tensor:
        """Sample one random band partition for this batch.

        Returns a boolean mask `target_mask: bool[n_bands]` where True
        indicates a target-half band (encoder input zeroes these out;
        the loss is evaluated on these).

        Per-batch (not per-sample) partition: all samples in the batch
        share the same target-mask.
        """
        n_target = round(self.n_bands * self.spectral_mask_ratio)
        target_mask = torch.zeros(self.n_bands, dtype=torch.bool, device=device)
        if n_target == 0:
            return target_mask
        target_idx = torch.randperm(self.n_bands, device=device)[:n_target]
        target_mask[target_idx] = True
        return target_mask

    def forward(self, x_clean: torch.Tensor):
        """Returns (loss, recon, mask).

        loss:  scalar MSE on target-band positions of recon vs x_clean,
               across all 49 spatial positions. If spectral_mask_ratio == 0,
               the loss degenerates to MSE on all bands at all positions
               (equivalent to v3's all-position MAE loss).
        recon: (B, n_tokens, n_bands) — reconstructed spectra at every position
        mask:  (B, n_tokens) bool — True = was spatially masked at the encoder
        """
        B = x_clean.shape[0]
        device = x_clean.device
        N = self.n_tokens

        # 1. Sample one band partition for the whole batch.
        target_mask = self._partition_bands(device)  # (n_bands,) bool
        input_band_mask = ~target_mask                # bands the encoder sees

        # 2. Zero out target bands at every pixel in the encoder input.
        x_in = x_clean * input_band_mask.view(1, 1, 1, self.n_bands).to(x_clean.dtype)

        # 3. Standard spatial masking + encoder pass (parent-class machinery).
        visible_ids, masked_ids, mask = self._mask_tokens(B, device)
        enc_out = self.encoder.encode_visible(x_in, visible_ids)
        enc_proj = self.enc_to_dec(enc_out[:, 1:])

        decoder_tokens = self.mask_token.expand(B, N, -1).clone()
        scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        decoder_tokens.scatter_(1, scatter_idx, enc_proj)
        pos_ids = torch.arange(1, N + 1, device=device)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed(pos_ids)
        decoded = self.decoder(decoder_tokens)
        recon = self.reconstruction_head(decoded)

        # 4. SPEND loss: MSE on target bands only (or all bands when ratio=0).
        x_flat = x_clean.reshape(B, N, self.n_bands)
        if target_mask.any():
            loss = ((recon[:, :, target_mask] - x_flat[:, :, target_mask]) ** 2).mean()
        else:
            # ratio==0 → no target bands → fall back to all-band MSE (phase C).
            loss = ((recon - x_flat) ** 2).mean()

        return loss, recon, mask
