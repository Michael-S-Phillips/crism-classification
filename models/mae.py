"""
Spectral Masked Autoencoder (MAE) for CRISM mrral data.

Pre-trains a SpectralTransformer encoder to reconstruct randomly masked bands.
The encoder can then be loaded into SpectralTransformer for fine-tuning.

Reference: He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
           adapted for 1D spectral data.
"""
import torch
import torch.nn as nn
from models.spectral_transformer import SpectralTransformer


class SpectralMAE(nn.Module):
    """
    Masked Autoencoder for spectral data.

    Forward pass:
      1. Randomly mask mask_ratio fraction of bands (set to 0)
      2. Encode masked spectrum with SpectralTransformer encoder
      3. Decode CLS embedding to predict ALL 59 band values
      4. Compute MSE loss on masked bands only

    After pre-training:
      - Call encoder_state_dict() to extract encoder weights
      - Load into SpectralTransformer.load_encoder_state_dict()
    """

    def __init__(
        self,
        n_bands: int = 59,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        decoder_dim: int = 64,
        mask_ratio: float = 0.40,
        dropout: float = 0.0,   # no dropout during MAE pre-training
    ):
        super().__init__()
        self.n_bands = n_bands
        self.mask_ratio = mask_ratio

        # Encoder: shared with downstream SpectralTransformer
        self.encoder = SpectralTransformer(
            n_bands=n_bands, n_classes=embed_dim,  # head output = embed_dim (replaced below)
            embed_dim=embed_dim, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )
        # Replace classification head with identity (we use CLS embed directly)
        self.encoder.head = nn.Identity()

        # Decoder: lightweight MLP that predicts all band values from CLS token
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, n_bands),
        )

    def _random_mask(self, x: torch.Tensor) -> tuple:
        """Returns (masked_x, mask) where mask=True means band was masked."""
        B, N = x.shape
        n_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def forward(self, x: torch.Tensor):
        """
        Returns: (loss, pred, mask)
          loss: scalar MSE on masked bands
          pred: (B, n_bands) reconstructed spectrum
          mask: (B, n_bands) bool, True = was masked
        """
        x_masked, mask = self._random_mask(x)
        cls_embed = self.encoder(x_masked)  # (B, embed_dim) — from encoder.head=Identity
        pred = self.decoder(cls_embed)       # (B, n_bands)
        # MSE only on masked bands
        loss = ((pred - x) ** 2)[mask].mean()
        return loss, pred, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS embedding without masking. Shape: (B, embed_dim)."""
        return self.encoder(x)

    def encoder_state_dict(self) -> dict:
        """Return encoder weights (excluding the replaced head)."""
        return {k: v for k, v in self.encoder.state_dict().items()
                if not k.startswith('head.')}
