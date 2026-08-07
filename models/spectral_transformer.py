"""
Spectral Transformer for per-pixel mineral classification.
Treats each spectral band as a token (with learned positional embedding).
Used both for classification and as the MAE encoder backbone.
"""
import torch
import torch.nn as nn


class SpectralTransformer(nn.Module):
    """
    Transformer operating on a pixel's reflectance spectrum.

    Each of the 59 bands is projected to embed_dim, positional embeddings added,
    then processed through n_layers Transformer encoder blocks. A CLS token
    aggregates the sequence for classification.

    Input:  (batch, n_bands)
    Output: (batch, n_classes) — raw logits
    """

    def __init__(
        self,
        n_bands: int = 59,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.embed_dim = embed_dim

        # Project each scalar band value to embed_dim
        self.band_embed = nn.Linear(1, embed_dim)
        # Learned positional embedding for each band position
        self.pos_embed = nn.Embedding(n_bands + 1, embed_dim)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands)
        B = x.shape[0]
        # Embed each band: (batch, n_bands, embed_dim)
        tokens = self.band_embed(x.unsqueeze(-1))
        # Add positional embeddings for band positions 1..n_bands
        pos_ids = torch.arange(1, self.n_bands + 1, device=x.device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)
        # Prepend CLS token (position 0)
        cls = self.cls_token.expand(B, -1, -1)
        cls = cls + self.pos_embed(torch.zeros(1, device=x.device, dtype=torch.long))
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_bands+1, embed_dim)
        # Transformer
        out = self.encoder(tokens)
        cls_out = self.norm(out[:, 0])             # CLS token
        return self.head(cls_out)

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """
        Return two optimizer param groups for differential LR fine-tuning.

        When loaded from a MAE checkpoint, the encoder (band_embed, pos_embed,
        cls_token, encoder layers, norm) has pretrained weights that should be
        fine-tuned gently. The classification head is randomly initialized and
        can use a higher LR.

        Returns:
            [{'params': encoder_params, 'lr': encoder_lr},
             {'params': head_params,    'lr': head_lr}]
        """
        head_params = list(self.head.parameters())
        head_param_ids = {id(p) for p in head_params}
        encoder_params = [p for p in self.parameters() if id(p) not in head_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        """
        Load encoder weights from a pre-trained SpectralMAE.
        Skips the classification head (which is randomly initialized).
        Returns (missing_keys, unexpected_keys).
        """
        own = self.state_dict()
        unexpected = [k for k in state if k not in own]
        missing = [k for k in own if k not in state and not k.startswith('head.')]
        for k, v in state.items():
            if k in own:
                own[k] = v
        self.load_state_dict(own)
        return missing, unexpected
