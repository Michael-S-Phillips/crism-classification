"""
Spatial-Spectral Masked Autoencoder for CRISM mrral hyperspectral patches.

Pre-trains a SpatialSpectralTransformer encoder to reconstruct randomly
masked spatial pixels in a 7×7 patch. After pre-training, call
encoder_state_dict() to extract encoder weights for fine-tuning.

Reference: He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
           adapted for spatial hyperspectral patch data.
"""
import torch
import torch.nn as nn
from models.spatial_spectral_transformer import SpatialSpectralTransformer


class SpatialSpectralMAE(nn.Module):
    """
    Masked Autoencoder for spatial patches of spectral data.

    Forward pass:
      1. Mask mask_ratio fraction of spatial tokens (pixels)
      2. Encode visible tokens with SpatialSpectralTransformer (CLS + visible)
      3. Project encoder output to decoder_dim
      4. Build full decoder input: projected visible tokens + mask tokens, all
         with decoder positional embeddings, in original spatial order
      5. Decode with 2-layer transformer, reconstruct all 49 pixel spectra
      6. Loss: MSE on masked pixels only

    After pre-training:
      - Call encoder_state_dict() to extract encoder weights
      - Load into SpatialSpectralClassifier.load_encoder_state_dict()
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
        dropout: float = 0.0,   # no dropout during MAE pre-training
    ):
        super().__init__()
        self.n_bands = n_bands
        self.mask_ratio = mask_ratio
        self.n_tokens = patch_size * patch_size  # 49
        self.decoder_dim = decoder_dim

        # Encoder
        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )

        # Project encoder output to decoder_dim
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)

        # Learnable mask token (one vector, broadcast to all masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Decoder positional embeddings (separate from encoder's)
        # Index 0 unused; 1..n_tokens for spatial positions
        self.decoder_pos_embed = nn.Embedding(self.n_tokens + 1, decoder_dim)

        # Lightweight decoder transformer
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=max(1, decoder_dim // 16),
            dim_feedforward=decoder_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)

        # Reconstruction head: decoder_dim → n_bands per masked pixel
        self.reconstruction_head = nn.Linear(decoder_dim, n_bands)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _mask_tokens(self, B: int, device: torch.device):
        """
        Generate random mask for B samples.
        Returns:
          visible_ids: (B, n_visible)  — sorted spatial positions kept
          masked_ids:  (B, n_masked)   — sorted spatial positions masked
          mask:        (B, n_tokens) bool — True = was masked
        """
        N = self.n_tokens
        n_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=device)
        ids = torch.argsort(noise, dim=1)
        masked_ids  = ids[:, :n_mask].sort(dim=1).values
        visible_ids = ids[:, n_mask:].sort(dim=1).values
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_ids, True)
        return visible_ids, masked_ids, mask

    def forward(self, x: torch.Tensor):
        """
        Returns: (loss, recon, mask)
          loss:  scalar MSE on masked pixels
          recon: (B, n_tokens, n_bands) — reconstructed spectra for all positions
          mask:  (B, n_tokens) bool — True = was masked
        """
        B = x.shape[0]
        device = x.device
        N = self.n_tokens

        visible_ids, masked_ids, mask = self._mask_tokens(B, device)

        # Encode visible tokens: (B, n_visible+1, embed_dim)
        enc_out = self.encoder.encode_visible(x, visible_ids)
        # Skip CLS (slot 0), project visible tokens: (B, n_visible, decoder_dim)
        enc_proj = self.enc_to_dec(enc_out[:, 1:])

        # Build full decoder sequence in original spatial order (B, N, decoder_dim)
        # Start with mask tokens everywhere, fill in visible positions
        decoder_tokens = self.mask_token.expand(B, N, -1).clone()
        scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        decoder_tokens.scatter_(1, scatter_idx, enc_proj)

        # Add decoder positional embeddings to all positions
        pos_ids = torch.arange(1, N + 1, device=device)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed(pos_ids)

        # Decode: (B, N, decoder_dim)
        decoded = self.decoder(decoder_tokens)

        # Reconstruct: (B, N, n_bands)
        recon = self.reconstruction_head(decoded)

        # MSE loss on masked pixels only
        x_flat = x.reshape(B, N, self.n_bands)  # (B, 49, 59)
        per_pixel_loss = ((recon - x_flat) ** 2).mean(dim=-1)  # (B, N)
        loss = per_pixel_loss[mask].mean()

        return loss, recon, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract center-pixel embedding without masking. Shape: (B, embed_dim)."""
        out = self.encoder(x)                              # (B, 50, embed_dim)
        center_idx = self.n_tokens // 2 + 1               # +1 for CLS
        return out[:, center_idx]

    def encoder_state_dict(self) -> dict:
        """Return encoder weights for loading into SpatialSpectralClassifier."""
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}
