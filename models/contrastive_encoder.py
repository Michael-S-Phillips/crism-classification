"""ContrastiveEncoder — SpatialSpectralTransformer + L2-normalised projection.

For contrastive (InfoNCE) fine-tuning of the encoder used by the downstream
classifier. The projection head is discarded at eval time; ``encode(x)``
returns the center-token embedding, which is what
``SpatialSpectralClassifier`` consumes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spatial_spectral_transformer import SpatialSpectralTransformer


class ContrastiveEncoder(nn.Module):
    """Wraps a SpatialSpectralTransformer with a 2-layer projection head.

    Parameters
    ----------
    n_bands, patch_size, embed_dim, n_heads, n_layers, dropout
        Forwarded straight to ``SpatialSpectralTransformer``.
    proj_dim
        Dimensionality of the L2-normalised projection used by InfoNCE.
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands,
            patch_size=patch_size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        # CLS sits at slot 0; spatial token (n_tokens//2) is the center pixel.
        self._center_idx = self.encoder.n_tokens // 2 + 1
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

    # ------------------------------------------------------------------ API
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the center-token embedding ``(B, embed_dim)``.

        Used by downstream linear probe / fine-tune; matches the slot
        ``SpatialSpectralClassifier`` reads from.
        """
        out = self.encoder(x)                  # (B, n_tokens+1, embed_dim)
        return out[:, self._center_idx]        # (B, embed_dim)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """L2-normalised projection of an already-encoded embedding."""
        z = self.proj(h)
        return F.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the L2-normalised projected embedding ``(B, proj_dim)``."""
        return self.project(self.encode(x))

    # ------------------------------------------------------------------ ckpt
    def load_encoder_state_dict(self, state: dict):
        """Warm-start the underlying SpatialSpectralTransformer."""
        return self.encoder.load_encoder_state_dict(state)
