"""
SpectralHybridClassifier: combines mrral spectral encoder with mrrsu branch.

Processes a 119-dim input [mrral_59 | mrrsu_60] by routing mrral bands through
a SpectralTransformer encoder (MAE-pretrained) and mrrsu summary parameters
through a shallow MLP, then concatenating and classifying.

This architecture leverages:
  - MAE-pretrained spectral representation (mrral 59 bands)
  - Domain-scientist mineral indices (mrrsu summary params: OLINDEX3, BD1300, etc.)
"""
import torch
import torch.nn as nn

from models.spectral_transformer import SpectralTransformer


class SpectralHybridClassifier(nn.Module):
    """
    Hybrid classifier for CRISM mineral classification.

    Input:  (batch, n_mrral + n_mrrsu)  — first n_mrral dims are mrral, rest mrrsu
    Output: (batch, n_classes) — raw logits

    Args:
        n_mrral:      Number of mrral spectral bands (default: 59)
        n_mrrsu:      Number of mrrsu summary parameter bands (default: 60)
        n_classes:    Number of output classes (default: 5)
        embed_dim:    SpectralTransformer embedding dimension (default: 128)
        n_heads:      Number of attention heads (default: 4)
        n_layers:     Number of transformer encoder layers (default: 4)
        dropout:      Dropout rate applied in both branches (default: 0.1)
        mrrsu_hidden: Hidden dimension of the mrrsu MLP branch (default: 64)
    """

    def __init__(
        self,
        n_mrral: int = 59,
        n_mrrsu: int = 60,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        mrrsu_hidden: int = 64,
    ):
        super().__init__()
        self.n_mrral = n_mrral
        self.n_mrrsu = n_mrrsu

        # Spectral encoder — use SpectralTransformer with Identity head
        # so forward() returns the raw CLS embedding (embed_dim,) not logits.
        self.encoder = SpectralTransformer(
            n_bands=n_mrral, n_classes=embed_dim,
            embed_dim=embed_dim, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )
        self.encoder.head = nn.Identity()

        # mrrsu branch: lightweight MLP
        self.mrrsu_branch = nn.Sequential(
            nn.Linear(n_mrrsu, mrrsu_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mrrsu_hidden, mrrsu_hidden),
            nn.GELU(),
        )

        # Joint classification head
        self.head = nn.Linear(embed_dim + mrrsu_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_mrral + n_mrrsu) — layout matches CRISMCombinedDataset
        """
        mrral = x[:, :self.n_mrral]           # (batch, 59)
        mrrsu = x[:, self.n_mrral:]           # (batch, 60)
        cls_embed = self.encoder(mrral)       # (batch, embed_dim)
        mrrsu_feat = self.mrrsu_branch(mrrsu) # (batch, mrrsu_hidden)
        combined = torch.cat([cls_embed, mrrsu_feat], dim=1)
        return self.head(combined)            # (batch, n_classes)

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """
        Differential LR: pretrained encoder gets slow LR, mrrsu branch and
        classification head get fast LR.

        Returns:
            [{'params': encoder_params, 'lr': encoder_lr},
             {'params': new_params,     'lr': head_lr}]
        """
        encoder_param_ids = {id(p) for p in self.encoder.parameters()}
        encoder_params = list(self.encoder.parameters())
        new_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': new_params,     'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        """
        Load SpectralTransformer encoder weights from a MAE checkpoint.
        Delegates to the inner encoder's load_encoder_state_dict method.

        Args:
            state: dict from SpectralMAE.encoder_state_dict() — keys like
                   'band_embed.weight', 'pos_embed.weight', 'encoder.layers.N.*'

        Returns:
            (missing_keys, unexpected_keys)
        """
        return self.encoder.load_encoder_state_dict(state)
