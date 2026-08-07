"""
Lightweight Vision Transformer for spectral-spatial classification.
Treats each pixel in the patch as a token with spectral embedding.
"""
import math
import torch
import torch.nn as nn


class SpectralViT(nn.Module):
    """
    ViT over a (patch_size x patch_size) spatial grid of spectral tokens.
    Each pixel's n_bands values are embedded to embed_dim.
    A learned CLS token aggregates spatial context for classification.

    Input: (batch, n_bands, patch_size, patch_size)
    Output: (batch, n_classes) logits
    """

    def __init__(
        self,
        n_bands: int = 60,
        n_classes: int = 6,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_tokens = patch_size * patch_size

        # Spectral embedding: project n_bands to embed_dim per pixel
        self.spectral_embed = nn.Linear(n_bands, embed_dim)

        # Learned CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) where C=n_bands, H=W=patch_size
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        # Spectral embedding: (B, H*W, embed_dim)
        x = self.spectral_embed(x)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, H*W+1, embed_dim)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token output
        return self.head(x)
