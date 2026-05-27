# models/spatial_spectral_classifier_aux.py
"""Mineral classifier with a late-fusion auxiliary head for smoothed mrrsu params.

Identical to SpatialSpectralClassifier except the center-token embedding is
concatenated with a small MLP embedding of an auxiliary feature vector
([mean_7x7 RPEAK1, mean_7x7 BD1300]) before the classification head. The encoder
is unchanged and loads from any SpatialSpectralMAE checkpoint.

Spec: docs/superpowers/specs/2026-05-27-mrrsu-aux-injection-design.md
"""
import torch
import torch.nn as nn

from models.spatial_spectral_transformer import SpatialSpectralTransformer


class SpatialSpectralClassifierAux(nn.Module):
    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        aux_dim: int = 2,
        aux_hidden: int = 16,
    ):
        super().__init__()
        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, aux_hidden),
        )
        self.head = nn.Linear(embed_dim + aux_hidden, n_classes)
        self._center_idx = self.encoder.n_tokens // 2 + 1  # +1 for CLS

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # x: (B, patch, patch, n_bands); aux: (B, aux_dim)
        out = self.encoder(x)                       # (B, n_tokens+1, embed_dim)
        center = out[:, self._center_idx]           # (B, embed_dim)
        aux_emb = self.aux_mlp(aux)                  # (B, aux_hidden)
        return self.head(torch.cat([center, aux_emb], dim=-1))

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        head_params = list(self.aux_mlp.parameters()) + list(self.head.parameters())
        head_param_ids = {id(p) for p in head_params}
        encoder_params = [p for p in self.parameters() if id(p) not in head_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        return self.encoder.load_encoder_state_dict(state)
