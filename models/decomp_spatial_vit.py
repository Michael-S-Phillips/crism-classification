"""
Signal/noise decomposition encoder for CRISM patches.

Decomposes input I/F into:
    x  ≈  T(λ) · s + b(λ) + ε

where s is per-pixel surface reflectance (the signal), T and b are
per-patch multiplicative and additive atmospheric terms, and ε is the
per-pixel stochastic residual. The classifier reads the shared encoder's
center-pixel embedding; the reconstruction objective pressures the
embedding to represent surface mineralogy.

Spec: docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md
"""
from typing import Tuple

import torch
import torch.nn as nn

from models.spatial_spectral_transformer import SpatialSpectralTransformer


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Two-layer MLP with GELU activation."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class DecompSpVit(nn.Module):
    """
    Decomposition-aware classifier over CRISM patches.

    Forward returns: (logits, s_hat, T_hat, b_hat, eps_hat, x_hat)
        logits:  (B, n_classes)         — classifier output, sigmoid not applied
        s_hat:   (B, n_tokens, n_bands) — per-pixel surface reflectance
        T_hat:   (B, n_bands)           — per-patch multiplicative correction in [T_min, T_max]
        b_hat:   (B, n_bands)           — per-patch additive offset (unconstrained)
        eps_hat: (B, n_tokens, n_bands) — per-pixel residual
        x_hat:   (B, n_tokens, n_bands) — reconstruction = T·s + b + eps
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        head_hidden: int = 256,
        T_min: float = 0.3,
        T_max: float = 1.0,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_tokens = patch_size * patch_size
        self.embed_dim = embed_dim
        self.T_min = T_min
        self.T_max = T_max

        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )

        # Heads
        self.signal_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)
        self.residual_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)
        # Atmosphere head outputs 2*n_bands — first n_bands → T_hat (sigmoid-scaled),
        # second n_bands → b_hat (unconstrained).
        self.atmosphere_head = _mlp(embed_dim, head_hidden, 2 * n_bands, dropout=dropout)
        self.class_head = nn.Linear(embed_dim, n_classes)

        # CLS token is slot 0; center-pixel token in the grid is at flat index
        # n_tokens//2 in the spatial layout (e.g., (3,3) for 7×7 = 24), then +1
        # for the CLS offset → 25.
        self._center_idx = self.n_tokens // 2 + 1

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, patch_size, patch_size, n_bands)
        z = self.encoder(x)                      # (B, n_tokens+1, embed_dim)
        cls = z[:, 0]                            # (B, embed_dim)
        tokens = z[:, 1:]                        # (B, n_tokens, embed_dim)

        s_hat = self.signal_decoder(tokens)      # (B, n_tokens, n_bands)
        eps_hat = self.residual_decoder(tokens)  # (B, n_tokens, n_bands)

        Tb = self.atmosphere_head(cls)           # (B, 2*n_bands)
        T_raw, b_hat = Tb[:, :self.n_bands], Tb[:, self.n_bands:]
        # Sigmoid-scale T to [T_min, T_max]
        T_hat = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(T_raw)

        # Broadcast per-patch T_hat and b_hat across the n_tokens dimension
        x_hat = T_hat.unsqueeze(1) * s_hat + b_hat.unsqueeze(1) + eps_hat

        center_token = z[:, self._center_idx]    # (B, embed_dim)
        logits = self.class_head(center_token)   # (B, n_classes)

        return logits, s_hat, T_hat, b_hat, eps_hat, x_hat

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """Param groups for differential LR fine-tuning.
        Groups: encoder (slow) and heads (fast). Returned in that order so the
        index matches the training loop's lr-scheduling convention.
        """
        encoder_params = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        """Load encoder weights from a SpatialSpectralMAE checkpoint."""
        return self.encoder.load_encoder_state_dict(state)
