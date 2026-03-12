# Spatial MAE Pre-training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the per-pixel `SpectralMAE` with a spatial patch MAE (`SpatialSpectralMAE`) that pre-trains on the full global MRDR dataset (~3.87B pixels across 1,764 mrral tiles) by streaming random 7×7 patches, producing a richer encoder that understands spatial geological context.

**Architecture:** `SpatialSpectralTransformer` treats each pixel in a 7×7 patch as one token (49 tokens + CLS), projecting its 59-band spectrum to `embed_dim` via a linear layer, then processing through n_layers pre-norm transformer blocks. `SpatialSpectralMAE` masks 75% of spatial tokens and reconstructs their spectra with a lightweight 2-layer transformer decoder. `CRISMGlobalPatchDataset` is an IterableDataset that streams patches from raw mrral `.img` files across all mc## tile directories with no pre-extraction.

**Tech Stack:** PyTorch (TransformerEncoderLayer, IterableDataset), rasterio (ENVI tile reading), wandb (logging), conda env `crism`, Python 3.11

**All commands run from:** `/mnt/crism/MRDR/crism_classification/`

---

## Task 1: `SpatialSpectralTransformer` encoder + downstream classifier

**Files:**
- Create: `models/spatial_spectral_transformer.py`
- Test: `tests/test_spatial_transformer.py`

**Step 1: Write the failing tests**

Create `tests/test_spatial_transformer.py`:

```python
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_encoder_forward_shape():
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    enc = SpatialSpectralTransformer(n_bands=59, patch_size=7, embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    out = enc(x)
    # CLS + 49 spatial tokens
    assert out.shape == (4, 50, 64), f"Got {out.shape}"


def test_encoder_encode_visible():
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    enc = SpatialSpectralTransformer(n_bands=59, patch_size=7, embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    # Keep 12 visible tokens out of 49
    visible_ids = torch.stack([torch.randperm(49)[:12] for _ in range(4)])
    out = enc.encode_visible(x, visible_ids)
    assert out.shape == (4, 13, 64), f"Got {out.shape}"  # CLS + 12 visible


def test_classifier_output_shape():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    out = clf(x)
    assert out.shape == (4, 5), f"Got {out.shape}"


def test_classifier_param_groups():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2)
    groups = clf.get_param_groups(head_lr=1e-3, encoder_lr=1e-4)
    assert len(groups) == 2
    assert groups[0]['lr'] == 1e-4   # encoder
    assert groups[1]['lr'] == 1e-3   # head
    # All params accounted for
    all_ids = {id(p) for g in groups for p in g['params']}
    assert all_ids == {id(p) for p in clf.parameters()}


def test_classifier_deterministic_in_eval():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2, dropout=0.1)
    clf.eval()
    x = torch.randn(4, 7, 7, 59)
    with torch.no_grad():
        assert torch.allclose(clf(x), clf(x)), "eval() should be deterministic"
```

**Step 2: Run to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_spatial_transformer.py -v
```
Expected: 5 errors — `ModuleNotFoundError: No module named 'models.spatial_spectral_transformer'`

**Step 3: Implement `models/spatial_spectral_transformer.py`**

```python
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
        x: (B, patch_size, patch_size, n_bands)
        Returns: (B, n_tokens+1, embed_dim)  — all embeddings after encoder+norm
        """
        B = x.shape[0]
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
```

**Step 4: Run tests**

```bash
conda run -n crism python -m pytest tests/test_spatial_transformer.py -v
```
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add models/spatial_spectral_transformer.py tests/test_spatial_transformer.py
git commit -m "feat: add SpatialSpectralTransformer encoder and classifier"
```

---

## Task 2: `SpatialSpectralMAE` wrapper

**Files:**
- Create: `models/spatial_mae.py`
- Test: `tests/test_spatial_mae.py`

**Step 1: Write the failing tests**

Create `tests/test_spatial_mae.py`:

```python
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mae_forward_returns_scalar_loss():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2, mask_ratio=0.75)
    x = torch.randn(4, 7, 7, 59)
    loss, recon, mask = model(x)
    assert loss.shape == (), f"Loss must be scalar, got {loss.shape}"
    assert loss.item() > 0


def test_mae_reconstruction_shape():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2, mask_ratio=0.75)
    x = torch.randn(4, 7, 7, 59)
    loss, recon, mask = model(x)
    assert recon.shape == (4, 49, 59), f"Got {recon.shape}"
    assert mask.shape == (4, 49), f"Got {mask.shape}"


def test_mae_mask_ratio():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2, mask_ratio=0.75)
    x = torch.randn(8, 7, 7, 59)
    _, _, mask = model(x)
    # Each sample should have ~75% of 49 tokens masked = ~37
    per_sample = mask.float().sum(dim=1)
    assert (per_sample == 37).all(), f"Expected 37 masked per sample, got {per_sample}"


def test_mae_encoder_state_dict_loads_into_classifier():
    from models.spatial_mae import SpatialSpectralMAE
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    mae = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                              n_heads=4, n_layers=2)
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2)
    state = mae.encoder_state_dict()
    missing, unexpected = clf.load_encoder_state_dict(state)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    assert len(missing) == 0, f"Missing keys: {missing}"


def test_mae_encode_no_masking():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    emb = model.encode(x)
    # encode() returns center-pixel embedding
    assert emb.shape == (4, 64), f"Got {emb.shape}"
```

**Step 2: Run to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_spatial_mae.py -v
```
Expected: 5 errors — `ModuleNotFoundError: No module named 'models.spatial_mae'`

**Step 3: Implement `models/spatial_mae.py`**

```python
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
        scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, enc_proj.shape[-1])
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
```

**Step 4: Run tests**

```bash
conda run -n crism python -m pytest tests/test_spatial_mae.py -v
```
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add models/spatial_mae.py tests/test_spatial_mae.py
git commit -m "feat: add SpatialSpectralMAE with 75% spatial masking"
```

---

## Task 3: `CRISMGlobalPatchDataset` — streaming from all tiles

**Files:**
- Create: `data/global_patch_dataset.py`
- Test: `tests/test_global_patch_dataset.py`

**Step 1: Write the failing tests**

Create `tests/test_global_patch_dataset.py`:

```python
import torch
import numpy as np
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MRRAL_DIR = '/mnt/crism/MRDR'
HDR_FILES = sorted(glob.glob(os.path.join(MRRAL_DIR, 'mc*/t*mrral*.hdr')))


def test_dataset_yields_correct_shape():
    """Each yielded patch should be (7, 7, 59) float32."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(HDR_FILES[:5], patch_size=7)
    it = iter(ds)
    patch = next(it)
    assert patch.shape == (7, 7, 59), f"Got {patch.shape}"
    assert patch.dtype == torch.float32


def test_dataset_values_clipped():
    """Values should be in [0.0, 0.5] after normalization."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(HDR_FILES[:5], patch_size=7)
    it = iter(ds)
    for _ in range(20):
        patch = next(it)
        assert patch.min().item() >= 0.0, f"Min below 0: {patch.min()}"
        assert patch.max().item() <= 0.5, f"Max above 0.5: {patch.max()}"


def test_dataset_no_nodata_in_output():
    """No 65535 values should appear in yielded patches."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(HDR_FILES[:5], patch_size=7)
    it = iter(ds)
    for _ in range(10):
        patch = next(it)
        assert not (patch == 65535.0).any(), "Nodata value 65535 found in output"


def test_dataloader_multiworker():
    """DataLoader with num_workers=2 should yield patches without error."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    from torch.utils.data import DataLoader
    ds = CRISMGlobalPatchDataset(HDR_FILES[:10], patch_size=7)
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    assert batch.shape == (4, 7, 7, 59)
```

**Step 2: Run to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_global_patch_dataset.py -v
```
Expected: 4 errors — `ModuleNotFoundError: No module named 'data.global_patch_dataset'`

**Step 3: Implement `data/global_patch_dataset.py`**

```python
"""
CRISMGlobalPatchDataset — streams random 7×7 spatial patches from all mrral tiles.

Uses IterableDataset with per-worker tile sharding. No pre-extraction needed;
training can begin immediately. Each worker randomly samples tiles (weighted by
area) and random valid patch centers within each tile.

Usage:
    import glob
    from data.global_patch_dataset import CRISMGlobalPatchDataset

    hdr_files = sorted(glob.glob('/mnt/crism/MRDR/mc*/t*mrral*.hdr'))
    ds = CRISMGlobalPatchDataset(hdr_files)
    loader = DataLoader(ds, batch_size=512, num_workers=8, pin_memory=True)
"""
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset

try:
    import rasterio
    import rasterio.windows
except ImportError:
    raise ImportError("rasterio is required: conda install -c conda-forge rasterio")

NODATA = 65535.0
N_BANDS = 59          # mrral bands 1–59 (410–2457 nm); ignore bands 60–72
CLIP_MAX = 0.5        # reflectance clip — covers P99 with headroom
MIN_VALID_FRAC = 0.8  # fraction of patch pixels that must be non-NaN/nodata


def _tile_shape(hdr_path: str):
    """Return (height, width) of a tile without loading data."""
    img_path = hdr_path.replace('.hdr', '.img')
    try:
        with rasterio.open(img_path) as src:
            return src.height, src.width
    except Exception:
        return 0, 0


class CRISMGlobalPatchDataset(IterableDataset):
    """
    Infinite stream of (patch_size, patch_size, N_BANDS) float32 tensors.

    Two-level random sampling:
      1. Sample a tile with probability proportional to its pixel area.
      2. Sample a random valid patch center (no NaN-heavy patches) within that tile.

    Workers each receive a shard of the tile list; file handles are cached
    per-worker (no shared state needed between workers).
    """

    def __init__(
        self,
        hdr_paths: list,
        patch_size: int = 7,
        min_valid_frac: float = MIN_VALID_FRAC,
        clip_max: float = CLIP_MAX,
        max_retries: int = 20,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.hdr_paths = list(hdr_paths)
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.min_valid_frac = min_valid_frac
        self.clip_max = clip_max
        self.max_retries = max_retries

        # Precompute tile areas for weighted sampling (skip inaccessible tiles)
        sizes = [_tile_shape(p) for p in self.hdr_paths]
        areas = [float(h * w) for h, w in sizes]
        total = sum(areas) or 1.0
        self._weights = np.array([a / total for a in areas], dtype=np.float64)
        # Zero out tiles with zero area (inaccessible)
        self._weights[self._weights == 0] = 0.0
        if self._weights.sum() > 0:
            self._weights /= self._weights.sum()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Shard: this worker handles every num_workers-th tile
            indices = list(range(worker_info.id, len(self.hdr_paths), worker_info.num_workers))
        else:
            indices = list(range(len(self.hdr_paths)))

        if not indices:
            return

        hdr_paths = [self.hdr_paths[i] for i in indices]
        weights = self._weights[indices]
        total = weights.sum()
        if total == 0:
            return
        weights = weights / total

        rng = np.random.default_rng()
        handles: dict = {}  # hdr_path -> rasterio.DatasetReader

        while True:
            # Sample a tile
            tile_idx = int(rng.choice(len(hdr_paths), p=weights))
            hdr = hdr_paths[tile_idx]
            img_path = hdr.replace('.hdr', '.img')

            # Open/cache file handle
            if hdr not in handles:
                try:
                    handles[hdr] = rasterio.open(img_path)
                except Exception:
                    continue
            src = handles[hdr]
            H, W = src.height, src.width
            if H < self.patch_size or W < self.patch_size:
                continue

            # Sample a valid patch center (with retries)
            for _ in range(self.max_retries):
                r = int(rng.integers(self.half, H - self.half))
                c = int(rng.integers(self.half, W - self.half))

                window = rasterio.windows.Window(
                    c - self.half, r - self.half,
                    self.patch_size, self.patch_size,
                )
                try:
                    # Read bands 1–59 (rasterio is 1-indexed)
                    patch = src.read(list(range(1, N_BANDS + 1)), window=window)
                    patch = patch.astype(np.float32)  # (59, 7, 7)
                except Exception:
                    continue

                # Validity check: fraction of pixels where all bands are valid
                nodata_mask = (patch == NODATA) | ~np.isfinite(patch)
                any_nodata = nodata_mask.any(axis=0)  # (7, 7) — True if any band is bad
                valid_frac = float(1.0 - any_nodata.mean())
                if valid_frac < self.min_valid_frac:
                    continue

                # Replace nodata/NaN with 0.0 (these positions are masked anyway)
                patch[nodata_mask] = 0.0

                # Normalize: clip to [0, clip_max]
                patch = np.clip(patch, 0.0, self.clip_max)

                # (59, 7, 7) → (7, 7, 59) — spatial-first for transformer
                patch = patch.transpose(1, 2, 0)

                yield torch.from_numpy(patch.copy())
                break
```

**Step 4: Run tests**

```bash
conda run -n crism python -m pytest tests/test_global_patch_dataset.py -v
```
Expected: 4 PASSED (may be slow on first run; test uses only first 5–10 tiles)

**Step 5: Commit**

```bash
git add data/global_patch_dataset.py tests/test_global_patch_dataset.py
git commit -m "feat: add CRISMGlobalPatchDataset streaming from all mrral tiles"
```

---

## Task 4: Pre-training script

**Files:**
- Create: `scripts/pretrain_spatial_mae.py`

No unit test for the training script itself — the integration is validated by running it briefly and checking loss decreases. See Step 3.

**Step 1: Implement `scripts/pretrain_spatial_mae.py`**

```python
"""
Spatial MAE pre-training on all global CRISM mrral tiles.

Streams random 7×7 patches from all 1,764 mrral tiles. One "epoch" = 1M patches.
Saves best checkpoint (lowest reconstruction loss) and periodic checkpoints.

Usage:
    conda run -n crism python scripts/pretrain_spatial_mae.py

    # Custom config:
    conda run -n crism python scripts/pretrain_spatial_mae.py \\
        --epochs 400 --embed_dim 128 --n_layers 6 --mask_ratio 0.75 \\
        --batch_size 512 --no_wandb

Checkpoint: checkpoints/spatial_mae_{embed_dim}d_{n_layers}l_best.pt
Format:     {'encoder_state': ..., 'mae_state': ..., 'mae_loss': ...,
             'epoch': ..., 'config': {...}}
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRRAL_GLOB = '/mnt/crism/MRDR/mc*/t*mrral*.hdr'
PATCHES_PER_EPOCH = 1_000_000
SAVE_EVERY = 50  # save periodic checkpoint every N epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=400)
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--embed_dim',   type=int,   default=128)
    parser.add_argument('--n_heads',     type=int,   default=4)
    parser.add_argument('--n_layers',    type=int,   default=6)
    parser.add_argument('--decoder_dim', type=int,   default=64)
    parser.add_argument('--mask_ratio',  type=float, default=0.75)
    parser.add_argument('--warmup',      type=int,   default=40)
    parser.add_argument('--num_workers', type=int,   default=8)
    parser.add_argument('--no_wandb',    action='store_true')
    parser.add_argument('--resume',      type=str,   default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    import yaml
    cfg_path = os.path.join(PROJ, 'config.yaml')
    cfg = yaml.safe_load(open(cfg_path))
    ckpt_dir = cfg['checkpoints_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    run_name = f'spatial_mae_{args.embed_dim}d_{args.n_layers}l'

    # ── Data ──────────────────────────────────────────────────────────────
    hdr_files = sorted(glob.glob(MRRAL_GLOB))
    if not hdr_files:
        raise FileNotFoundError(f"No mrral HDR files found at {MRRAL_GLOB}")
    log.info(f"Found {len(hdr_files)} mrral tiles")

    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(hdr_files, patch_size=7, min_valid_frac=0.8)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=args.decoder_dim, mask_ratio=args.mask_ratio,
    ).to(device)

    # ── Optimizer & schedule ──────────────────────────────────────────────
    base_lr = 1.5e-4 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        betas=(0.9, 0.95), weight_decay=0.05,
    )
    # Linear warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['mae_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('mae_loss', float('inf'))
        log.info(f"Resumed from {args.resume} at epoch {start_epoch}, loss={best_loss:.6f}")

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project='crism-mineral-classification', name=run_name,
                       config=vars(args), resume='allow')
        except Exception as e:
            log.warning(f"wandb init failed ({e}), continuing without")
            use_wandb = False

    # ── Sanity patches for per-epoch visual log ───────────────────────────
    sanity_patches = None  # filled on first batch

    # ── Training loop ─────────────────────────────────────────────────────
    batches_per_epoch = PATCHES_PER_EPOCH // args.batch_size
    data_iter = iter(loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(batches_per_epoch):
            try:
                patches = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                patches = next(data_iter)

            if sanity_patches is None:
                sanity_patches = patches[:4].clone()

            patches = patches.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(patches)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch}/{args.epochs} | mae_loss={mean_loss:.6f} | lr={lr_now:.2e}")

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'mae_loss': mean_loss, 'lr': lr_now})

        # Save best checkpoint
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({
                'encoder_state': model.encoder_state_dict(),
                'mae_state': model.state_dict(),
                'mae_loss': best_loss,
                'epoch': epoch,
                'config': vars(args),
            }, best_path)
            log.info(f"  → New best: {best_loss:.6f}  saved to {best_path}")

        # Save periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            periodic_path = os.path.join(ckpt_dir, f'{run_name}_epoch{epoch}.pt')
            torch.save({
                'encoder_state': model.encoder_state_dict(),
                'mae_state': model.state_dict(),
                'mae_loss': mean_loss,
                'epoch': epoch,
                'config': vars(args),
            }, periodic_path)
            log.info(f"  Periodic checkpoint → {periodic_path}")

    log.info(f"Pre-training complete. Best MAE loss: {best_loss:.6f}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
```

**Step 2: Smoke test — run 2 epochs to verify loss decreases**

```bash
conda run -n crism python scripts/pretrain_spatial_mae.py \
    --epochs 2 --batch_size 64 --n_layers 2 --embed_dim 64 \
    --num_workers 2 --no_wandb \
    2>&1 | tee logs/pretrain_smoke_test.log
```

Expected output (approximately):
```
Found 1764 mrral tiles
Using device: cuda
Epoch 1/2 | mae_loss=0.0XXXXX | lr=...
Epoch 2/2 | mae_loss=0.0XXXXX | lr=...
```
Verify: both epochs complete without error, loss is a finite positive number.

**Step 3: Commit**

```bash
git add scripts/pretrain_spatial_mae.py
git commit -m "feat: add pretrain_spatial_mae.py script with streaming global dataset"
```

---

## Task 5: `CRISMSpectralPatchDataset` for fine-tuning

Add a new dataset class that reads mrral patches (7×7×59) from tiles for
fine-tuning `SpatialSpectralClassifier` on labeled data.

**Files:**
- Modify: `data/dataset.py` (append new class)
- Modify: `tests/test_dataset.py` (add tests)

**Step 1: Check existing test structure**

```bash
conda run -n crism python -m pytest tests/test_dataset.py -v
```
Note: confirm all existing tests pass before modifying.

**Step 2: Add failing tests to `tests/test_dataset.py`**

Append to the end of `tests/test_dataset.py`:

```python
def test_spectral_patch_dataset_shape():
    """CRISMSpectralPatchDataset should yield (7, 7, 59) patches."""
    import glob
    from data.dataset import CRISMSpectralPatchDataset
    import pandas as pd

    mrral_files = sorted(glob.glob('/mnt/crism/MRDR/mc*/t*mrral*.hdr'))[:5]
    mrral_map = {}
    for hdr in mrral_files:
        basename = os.path.basename(hdr)
        # tile_id is the obs-id part: t{id}_mrral_...
        tile_id = basename.split('_mrral_')[0]
        mrral_map[tile_id] = hdr.replace('.hdr', '.img')

    if not mrral_map:
        pytest.skip("No mrral tiles found")

    # Use a single known valid tile
    tile_id = list(mrral_map.keys())[0]
    import rasterio
    with rasterio.open(mrral_map[tile_id]) as src:
        H, W = src.height, src.width

    df = pd.DataFrame({
        'tile_id': [tile_id] * 4,
        'pixel_row': [H // 2] * 4,
        'pixel_col': [W // 2] * 4,
        'olivine_t1': [0.0] * 4, 'olivine_t2': [0.0] * 4,
        'lcp': [1.0] * 4, 'hcp': [0.0] * 4,
        'plagioclase': [0.0] * 4, 'other': [0.0] * 4,
        'confidence_weight': [1.0] * 4,
        'confidence_tier': ['High'] * 4,
        'split': ['train'] * 4,
    })

    ds = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7)
    patch, labels, weights = ds[0]
    assert patch.shape == (7, 7, 59), f"Got {patch.shape}"
    assert labels.shape == (5,)
    assert weights.shape == ()
```

**Step 3: Run to verify new test fails**

```bash
conda run -n crism python -m pytest tests/test_dataset.py::test_spectral_patch_dataset_shape -v
```
Expected: FAIL — `ImportError: cannot import name 'CRISMSpectralPatchDataset'`

**Step 4: Add `CRISMSpectralPatchDataset` to `data/dataset.py`**

Append to the end of `data/dataset.py`:

```python
class CRISMSpectralPatchDataset(Dataset):
    """
    Spatial patch dataset for SpatialSpectralClassifier fine-tuning.

    Reads 7×7×59 mrral reflectance patches around labeled pixel centers.
    Applies the same normalization as CRISMGlobalPatchDataset (clip to [0, 0.5]).
    Border pixels are zero-padded. File handles are cached per tile, pid-safe.
    """

    CLIP_MAX = 0.5
    NODATA = 65535.0

    def __init__(
        self,
        df: pd.DataFrame,
        mrral_map: Dict[str, str],
        patch_size: int = 7,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        df = _collapse_labels(df).reset_index(drop=True)
        self.mrral_map = mrral_map
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        self._tile_ids = df['tile_id'].values
        self._pixel_rows = df['pixel_row'].values.astype(np.int64)
        self._pixel_cols = df['pixel_col'].values.astype(np.int64)
        self._n = len(df)
        self._handles: Dict[str, rasterio.DatasetReader] = {}
        self._pid = os.getpid()

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        current_pid = os.getpid()
        if current_pid != self._pid:
            self._handles.clear()
            self._pid = current_pid

        tile_id = self._tile_ids[idx]
        pr = int(self._pixel_rows[idx])
        pc = int(self._pixel_cols[idx])

        if tile_id not in self._handles:
            if tile_id not in self.mrral_map:
                raise KeyError(f"tile_id {tile_id!r} not found in mrral_map")
            self._handles[tile_id] = rasterio.open(self.mrral_map[tile_id])
        src = self._handles[tile_id]

        h = self.half
        r0 = max(0, pr - h);  r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h);  c1 = min(src.width,  pc + h + 1)
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)

        # Read first 59 bands (1-indexed for rasterio)
        chunk = src.read(list(range(1, 60)), window=window).astype(np.float32)  # (59, h, w)

        # Zero-pad to (59, patch_size, patch_size)
        patch = np.zeros((59, self.patch_size, self.patch_size), dtype=np.float32)
        dr0 = pr - h - (pr - h - max(0, pr - h))
        dc0 = pc - h - (pc - h - max(0, pc - h))
        ph, pw = chunk.shape[1], chunk.shape[2]
        patch[:, dr0:dr0 + ph, dc0:dc0 + pw] = chunk

        # Handle nodata
        patch[(patch == self.NODATA) | ~np.isfinite(patch)] = 0.0

        # Normalize: clip to [0, CLIP_MAX]
        patch = np.clip(patch, 0.0, self.CLIP_MAX)

        # (59, 7, 7) → (7, 7, 59)
        patch = patch.transpose(1, 2, 0)

        return torch.from_numpy(patch), self.labels[idx], self.weights[idx]

    def close(self):
        for src in self._handles.values():
            src.close()
        self._handles.clear()
```

**Step 5: Run all dataset tests**

```bash
conda run -n crism python -m pytest tests/test_dataset.py -v
```
Expected: all PASSED including new test.

**Step 6: Commit**

```bash
git add data/dataset.py tests/test_dataset.py
git commit -m "feat: add CRISMSpectralPatchDataset for spatial_vit fine-tuning"
```

---

## Task 6: Wire `spatial_vit` into `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

**Step 1: Add `spatial_vit` to TORCH_MODELS and argument parser**

Find the line:
```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit', 'spectral_hybrid'}
```
Change to:
```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit', 'spectral_hybrid', 'spatial_vit'}
```

**Step 2: Add `spatial_vit` model construction block**

Find the `elif args.model == 'spectral_hybrid':` block. After it (before the final `else`
or `if args.model in TORCH_MODELS:` guard at the bottom), add:

```python
        elif args.model == 'spatial_vit':
            # Build mrral_map: tile_id -> mrral .img path
            import glob as _glob
            mrral_hdrs = sorted(_glob.glob('/mnt/crism/MRDR/mc*/t*mrral*.hdr'))
            mrral_map = {}
            for hdr in mrral_hdrs:
                tid = os.path.basename(hdr).split('_mrral_')[0]
                mrral_map[tid] = hdr.replace('.hdr', '.img')
            logging.info(f"mrral_map: {len(mrral_map)} tiles found")

            mrral_parquet = os.path.join(os.path.dirname(parquet_path), 'mrral_pixels.parquet')
            df_mrral = pd.read_parquet(mrral_parquet)
            dropout = args.dropout if args.dropout is not None else 0.1

            from models.spatial_spectral_transformer import SpatialSpectralClassifier
            from data.dataset import CRISMSpectralPatchDataset
            model = SpatialSpectralClassifier(
                n_bands=59, patch_size=args.patch_size, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f"Loaded spatial MAE encoder from {args.pretrain_ckpt}. "
                    f"Missing: {missing}, Unexpected: {unexpected}"
                )

            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrrsu_map=mrral_map,      # reuses mrrsu_map kwarg; dataset.py checks type
                patch_size=args.patch_size,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                encoder_lr_scale=args.encoder_lr_scale,
            )
```

Note: `train_torch_model` uses `mrrsu_map` to select `CRISMPatchDataset`. For `spatial_vit`,
we need it to use `CRISMSpectralPatchDataset` instead. Open `training/train_torch.py` and find
the dataset selection logic — it checks `if mrrsu_map is not None`. Add a check for model
type, or pass `mrral_map` via a new `mrral_map` kwarg. The simplest fix: in
`training/train_torch.py`, detect `SpatialSpectralClassifier` by checking
`isinstance(model, SpatialSpectralClassifier)` and use `CRISMSpectralPatchDataset` instead
of `CRISMPatchDataset`.

In `training/train_torch.py`, find the dataset construction block (where `CRISMPatchDataset`
is instantiated) and add:

```python
from models.spatial_spectral_transformer import SpatialSpectralClassifier
if isinstance(model, SpatialSpectralClassifier) and mrrsu_map is not None:
    from data.dataset import CRISMSpectralPatchDataset
    train_ds = CRISMSpectralPatchDataset(
        df[df['split'] == 'train'], mrrsu_map, patch_size=patch_size
    )
    val_ds = CRISMSpectralPatchDataset(
        df[df['split'] == 'val'], mrrsu_map, patch_size=patch_size
    )
else:
    # existing CRISMPatchDataset / CRISMPixelDataset logic
    ...
```

**Step 3: Smoke test `spatial_vit` without pretrain checkpoint**

```bash
conda run -n crism python scripts/train.py \
    --model spatial_vit \
    --n_layers 2 --embed_dim 64 \
    --epochs 2 --batch_size 32 --no_wandb \
    2>&1 | tail -10
```
Expected: completes 2 epochs, prints val_mAP.

**Step 4: Commit**

```bash
git add scripts/train.py training/train_torch.py
git commit -m "feat: wire spatial_vit into train.py with CRISMSpectralPatchDataset"
```

---

## Task 7: Launch full pre-training run

**Step 1: Run final test suite to confirm everything passes**

```bash
conda run -n crism python -m pytest tests/test_spatial_transformer.py \
    tests/test_spatial_mae.py tests/test_global_patch_dataset.py -v
```
Expected: all PASSED.

**Step 2: Launch pre-training in background**

```bash
mkdir -p logs
conda run -n crism python scripts/pretrain_spatial_mae.py \
    --epochs 400 \
    --batch_size 512 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --mask_ratio 0.75 \
    --warmup 40 \
    --num_workers 8 \
    > logs/pretrain_spatial_mae_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

**Step 3: Confirm healthy start**

```bash
sleep 60 && tail -20 logs/pretrain_spatial_mae_*.log
```

Expected:
```
Found 1764 mrral tiles
Using device: cuda
Epoch 1/400 | mae_loss=0.0XXXXX | lr=...
```
Loss should be a finite number > 0. If NaN or error, check the log for the failure.

**Step 4: Monitor**

```bash
# Check progress any time:
tail -5 logs/pretrain_spatial_mae_*.log
grep "New best" logs/pretrain_spatial_mae_*.log
```

**Step 5: After pre-training, verify checkpoint loads cleanly**

```bash
conda run -n crism python -c "
import torch
from models.spatial_spectral_transformer import SpatialSpectralClassifier
ck = torch.load('checkpoints/spatial_mae_128d_6l_best.pt', map_location='cpu', weights_only=False)
print('Best loss:', ck['mae_loss'])
print('Epoch:', ck['epoch'])
clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                 embed_dim=128, n_heads=4, n_layers=6)
missing, unexpected = clf.load_encoder_state_dict(ck['encoder_state'])
print('Missing:', missing)
print('Unexpected:', unexpected)
print('OK')
"
```
Expected: `Missing: []  Unexpected: []  OK`

---

## Critical notes

- **n_layers must match downstream**: `--n_layers 6` in `train.py` must match the pre-training config. A mismatch leaves layers randomly initialized while receiving the slow pretrained encoder LR.
- **Same normalization train/fine-tune**: both `CRISMGlobalPatchDataset` and `CRISMSpectralPatchDataset` clip to `[0.0, 0.5]`. Do not change one without the other.
- **Resume**: if pre-training is interrupted, use `--resume checkpoints/spatial_mae_128d_6l_epoch350.pt` to continue.
- **Decoder not transferred**: only `encoder_state` is loaded into the downstream classifier. The decoder is discarded after pre-training.
