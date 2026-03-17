"""
Spatial-Spectral Transformer for CRISM hyperspectral patch data.

Each pixel in a spatial patch is a token; its 59-band spectrum is projected
to embed_dim. Used as the MAE pre-training encoder and downstream classifier.
"""
import torch
import torch.nn as nn


class SpatialSpectralTransformer(nn.Module):
    """
    Transformer over a spatial patch of spectral pixels.

    Input:  (batch, patch_size, patch_size, n_bands)
    Output: (batch, n_tokens+1, embed_dim)  — CLS token first, then spatial tokens
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.n_tokens = patch_size * patch_size     # 49 for 7×7
        self.embed_dim = embed_dim

        # Project each pixel's 59-band spectrum to embed_dim
        self.band_embed = nn.Linear(n_bands, embed_dim)
        # Learned positional embedding: 0=CLS, 1..n_tokens=spatial positions
        self.pos_embed = nn.Embedding(self.n_tokens + 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _prepend_cls(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prepend CLS token (with pos 0 embedding) to token sequence."""
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        cls = cls + self.pos_embed(torch.zeros(1, device=tokens.device, dtype=torch.long))
        return torch.cat([cls, tokens], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass — all 49 spatial tokens visible.
        x: (B, patch_size, patch_size, n_bands)  [normal path]
           or (B, n_bands)  [1-D spectral path: spectrum is broadcast to all tokens]
        Returns: (B, n_tokens+1, embed_dim)  — all embeddings after encoder+norm
        """
        B = x.shape[0]
        if x.ndim == 2:
            # 1-D spectral input: broadcast single spectrum across all spatial tokens
            tokens_in = x.unsqueeze(1).expand(B, self.n_tokens, self.n_bands)
        else:
            tokens_in = x.reshape(B, self.n_tokens, self.n_bands)  # (B, 49, 59)
        tokens = self.band_embed(tokens_in)                      # (B, 49, embed_dim)
        pos_ids = torch.arange(1, self.n_tokens + 1, device=x.device)
        tokens = tokens + self.pos_embed(pos_ids)
        seq = self._prepend_cls(tokens)                          # (B, 50, embed_dim)
        return self.norm(self.encoder(seq))

    def encode_visible(self, x: torch.Tensor, visible_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode only a subset of spatial tokens (for MAE pre-training).

        x:           (B, patch_size, patch_size, n_bands)
        visible_ids: (B, n_visible)  — 0-indexed spatial positions to keep

        Returns: (B, n_visible+1, embed_dim)  — [CLS, visible_0, visible_1, ...]
                 The i-th visible output corresponds to visible_ids[:, i].
        """
        B = x.shape[0]
        tokens_in = x.reshape(B, self.n_tokens, self.n_bands)  # (B, 49, 59)

        # Gather spectra at visible positions: (B, n_visible, 59)
        gather_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.n_bands)
        visible_spectra = tokens_in.gather(1, gather_idx)

        tokens = self.band_embed(visible_spectra)                    # (B, n_visible, embed_dim)
        tokens = tokens + self.pos_embed(visible_ids + 1)            # true positional embeddings
        seq = self._prepend_cls(tokens)                              # (B, n_visible+1, embed_dim)
        return self.norm(self.encoder(seq))

    def load_encoder_state_dict(self, state: dict):
        """Load encoder weights from a SpatialSpectralMAE checkpoint."""
        own = self.state_dict()
        unexpected = [k for k in state if k not in own]
        missing = [k for k in own if k not in state]
        for k, v in state.items():
            if k in own:
                own[k].copy_(v)
        return missing, unexpected


class SpatialSpectralClassifier(nn.Module):
    """
    Downstream mineral classifier using SpatialSpectralTransformer encoder.

    Uses the center-pixel token (position patch_size²//2 + 1 with CLS offset)
    for per-pixel mineral prediction.

    Input:  (batch, patch_size, patch_size, n_bands)
    Output: (batch, n_classes) logits
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
    ):
        super().__init__()
        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, n_classes)
        # Center token index: CLS is slot 0; spatial token i is slot i+1.
        # Center of patch_size×patch_size grid = flat index (n_tokens//2).
        self._center_idx = self.encoder.n_tokens // 2 + 1  # +1 for CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, patch_size, patch_size, n_bands)
        out = self.encoder(x)               # (B, n_tokens+1, embed_dim)
        center = out[:, self._center_idx]   # (B, embed_dim)
        return self.head(center)

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """Return param groups for differential LR fine-tuning."""
        head_params = list(self.head.parameters())
        head_param_ids = {id(p) for p in head_params}
        encoder_params = [p for p in self.parameters() if id(p) not in head_param_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        """Load encoder weights from a SpatialSpectralMAE checkpoint."""
        return self.encoder.load_encoder_state_dict(state)
