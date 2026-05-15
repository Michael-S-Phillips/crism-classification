"""
Adversarial signal/noise decomposition encoder for CRISM patches (v2).

Additive decomposition: x ≈ s + n. Disentanglement is enforced by
adversarial decorrelation — a gradient-reversal layer + discriminator
push the noise embedding to be class-uninformative; the reconstruction
loss closes the additive identity; the classifier reads only the signal
embedding.

Spec: docs/superpowers/specs/2026-05-15-adversarial-decomposition-design.md
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from models.spatial_spectral_transformer import SpatialSpectralTransformer


class GradientReversalLayer(Function):
    """Identity in forward; multiplies upstream gradient by `-lambda_adv` in backward.

    Standard DANN trick (Ganin & Lempitsky 2015). lambda_adv is passed as a
    runtime argument so it can be scheduled per-epoch from the training loop.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse sign and scale.
        return grad_output.neg() * ctx.lambda_adv, None


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class DecompSpVitAdv(nn.Module):
    """
    Adversarial signal/noise decomposition classifier.

    Forward returns: (logits, s_hat, n_hat, x_hat, disc_logits, s_emb_center, n_emb_center)

    Args:
      lambda_adv:  scalar weight on the gradient-reversed adversarial path.
                   Mutable from outside via `model.lambda_adv = value` so a
                   training-loop scheduler can update it per epoch.
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
        disc_hidden: int = 64,
        lambda_adv: float = 1.0,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_tokens = patch_size * patch_size
        self.embed_dim = embed_dim
        self.lambda_adv = lambda_adv

        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )

        # Two lightweight projections off each token → signal / noise embeddings
        self.signal_projection = nn.Linear(embed_dim, embed_dim)
        self.noise_projection = nn.Linear(embed_dim, embed_dim)

        # Per-token decoders → per-pixel reflectance & residual
        self.signal_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)
        self.noise_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)

        # Classifier reads center-pixel signal embedding
        self.classifier = nn.Linear(embed_dim, n_classes)

        # Discriminator reads center-pixel noise embedding via GRL
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, disc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(disc_hidden, n_classes),
        )

        self._center_idx = self.n_tokens // 2 + 1   # +1 for CLS in the encoder output

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        z = self.encoder(x)                              # (B, 50, 128)
        tokens = z[:, 1:]                                # (B, 49, 128)

        s_emb = self.signal_projection(tokens)           # (B, 49, 128)
        n_emb = self.noise_projection(tokens)            # (B, 49, 128)

        s_hat = self.signal_decoder(s_emb)               # (B, 49, 59)
        n_hat = self.noise_decoder(n_emb)                # (B, 49, 59)
        x_hat = s_hat + n_hat                            # additive recon

        # In post-CLS-strip `tokens`, the center spatial pixel is at index
        # _center_idx - 1 (since _center_idx is the slot in the CLS-prepended
        # sequence z).
        center_s_emb = s_emb[:, self._center_idx - 1]    # (B, 128)
        center_n_emb = n_emb[:, self._center_idx - 1]    # (B, 128)

        logits = self.classifier(center_s_emb)           # (B, n_classes)

        # GRL: forward identity, backward flips and scales the encoder's
        # gradient signal by lambda_adv.
        n_emb_grl = GradientReversalLayer.apply(center_n_emb, self.lambda_adv)
        disc_logits = self.discriminator(n_emb_grl)      # (B, n_classes)

        return logits, s_hat, n_hat, x_hat, disc_logits, center_s_emb, center_n_emb

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        encoder_params = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        return self.encoder.load_encoder_state_dict(state)
