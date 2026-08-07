"""Embedding-space SAM-analog.

Run the champion `SpatialSpectralClassifier` encoder per pixel to produce a
128-d embedding raster, compute per-class centroid embeddings from labeled
high-confidence training pixels, then compute the cosine-angle (arccos of
cosine similarity) from every Argyre pixel to each centroid.

The angle math is the same as the spectral SAM core (reused from
`sam_analysis.sam.spectral_angle`) — the only difference is the input
dimensionality.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import numpy as np

from .sam import spectral_angle

N_BANDS = 59
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2
N_CLASSES = 5
NODATA = 65535.0
CLIP_MAX = 0.5
CLASS_NAMES = ("olivine", "lcp", "hcp", "plagioclase", "other")


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(
    ckpt_path: str = "checkpoints/ft_plag_aware_relabeled_best.pt",
    device: "object | None" = None,
) -> Tuple[object, int, "object"]:
    """Load the champion classifier; return (model, center_token_idx, device).

    Returned model is in eval mode. `center_token_idx` is the index of the
    center-pixel token in `encoder(patches)` output (includes CLS offset).
    """
    import torch  # local import to keep CLI helpers lightweight

    from models.spatial_spectral_transformer import SpatialSpectralClassifier

    if device is None:
        device = get_device()

    model = SpatialSpectralClassifier(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
        embed_dim=128, n_heads=4, n_layers=6,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    # Center token: CLS at 0, spatial tokens 1..49 → center at (49 // 2) + 1 = 25
    center_idx = model._center_idx
    return model, center_idx, device


# ---------------------------------------------------------------------------
# Patch extraction (mirrors classify_tile_supervised.py conventions)
# ---------------------------------------------------------------------------

def _normalize_patches(patches: np.ndarray) -> np.ndarray:
    """Match `classify_tile_supervised.normalize_patches` (per-patch z-score)."""
    b = patches.shape[0]
    flat = patches.reshape(b, -1)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((flat - mu) / sigma).reshape(patches.shape)


def _iter_patches(tile: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (patches, flat_pixel_indices) batches over a (H, W, B) tile."""
    h, w, _ = tile.shape
    padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant")
    n_pixels = h * w
    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        rows = np.arange(start, end) // w
        cols = np.arange(start, end) % w
        batch = np.stack([
            padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            for r, c in zip(rows, cols)
        ])
        yield batch.astype(np.float32), np.arange(start, end)


def extract_embeddings(
    cube: np.ndarray,
    model,
    center_idx: int,
    device,
    batch_size: int = 2048,
    cache_path: str | None = None,
) -> np.ndarray:
    """Run the encoder on every 7x7 patch centered on every valid pixel.

    Returns a (H, W, 128) float32 embedding raster. For large tiles this can
    use ~700 MB; prefer `stream_angles_to_centroids` if you only need angle
    rasters (it never materializes the full embedding cube in RAM).

    cache_path: optional .npz to load/save the full embedding raster from.
    """
    import torch
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from device import get_device
    from tqdm import tqdm

    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["embeddings"]

    h, w, _ = cube.shape
    safe = np.where(np.isfinite(cube), cube, 0.0).astype(np.float32)
    embed_dim = 128
    embeddings = np.zeros((h * w, embed_dim), dtype=np.float32)
    n_pixels = h * w
    n_batches = (n_pixels + batch_size - 1) // batch_size
    with torch.no_grad():
        for patches, idx in tqdm(
            _iter_patches(safe, batch_size),
            total=n_batches, desc="encoder", leave=False,
        ):
            patches = _normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            out = model.encoder(x)
            center = out[:, center_idx]
            embeddings[idx] = center.cpu().numpy()
    raster = embeddings.reshape(h, w, embed_dim)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, embeddings=raster)
    return raster


def stream_angles_to_centroids(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    model,
    center_idx: int,
    device,
    centroids: Dict[str, np.ndarray],
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Memory-efficient: compute per-class cosine-angle rasters in one pass.

    Instead of materializing a (H, W, 128) embedding raster, this streams
    encoder batches and accumulates the per-class angle to each centroid
    directly. Peak RAM is ~3 * (H*W) floats for the angle rasters
    plus one batch worth of embeddings.

    Args:
        cube: (H, W, 59) — NaN-zeroed mrral cube ready for the encoder.
        valid_mask: (H, W) bool; pixels with all invalid bands.
        centroids: dict class_name -> (128,) float centroid embedding.
    Returns:
        dict class_name -> (H, W) float32 angle raster (rad). NaN at invalid pixels.
    """
    import torch
    from tqdm import tqdm

    h, w, _ = cube.shape
    safe = np.where(np.isfinite(cube), cube, 0.0).astype(np.float32)

    # Pre-normalize centroids for fast cosine: store as float64 unit vectors.
    cent_unit = {}
    for k, c in centroids.items():
        c64 = c.astype(np.float64)
        n = np.linalg.norm(c64)
        if n < 1e-12:
            continue
        cent_unit[k] = c64 / n

    # Output angle rasters, NaN where invalid.
    angles = {k: np.full(h * w, np.nan, dtype=np.float32) for k in cent_unit}

    n_pixels = h * w
    n_batches = (n_pixels + batch_size - 1) // batch_size
    with torch.no_grad():
        for patches, idx in tqdm(
            _iter_patches(safe, batch_size),
            total=n_batches, desc="enc-stream", leave=False,
        ):
            patches = _normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            out = model.encoder(x)
            center = out[:, center_idx]            # (B, 128)
            emb = center.cpu().numpy().astype(np.float64)
            # Per-pixel norm of embedding (B,)
            emb_norm = np.linalg.norm(emb, axis=1)
            for k, c in cent_unit.items():
                # cosine sim = (emb @ c) / emb_norm; centroid unit norm already.
                dot = emb @ c                       # (B,)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cos = np.where(emb_norm < 1e-12, np.nan, dot / emb_norm)
                cos = np.clip(cos, -1.0, 1.0)
                ang = np.arccos(cos).astype(np.float32)
                angles[k][idx] = ang
    out_rasters: Dict[str, np.ndarray] = {}
    for k, a in angles.items():
        r = a.reshape(h, w)
        r[~valid_mask] = np.nan
        out_rasters[k] = r
    return out_rasters


# ---------------------------------------------------------------------------
# Class centroids from labeled high-confidence pixels
# ---------------------------------------------------------------------------

def _build_patch_from_row(row, band_cols: list[str]) -> np.ndarray:
    """Build a 7x7x59 patch by tiling the single-pixel mean spectrum.

    The labeled-pixel parquet only stores the center spectrum, not the
    surrounding pixels of the original tile. For centroid estimation we
    tile the spectrum across the 7x7 spatial extent — this matches the
    receptive field but gives an upper bound on the encoder's response
    since neighbours are identical. It is consistent across classes so
    the resulting centroids remain comparable.
    """
    spec = np.asarray([row[c] for c in band_cols], dtype=np.float32)
    return np.broadcast_to(spec, (PATCH_SIZE, PATCH_SIZE, N_BANDS)).copy()


def class_centroids(
    parquet_path: str,
    model,
    center_idx: int,
    device,
    splits: tuple[str, ...] = ("train",),
    conf_tier: str = "High",
    max_per_class: int = 5000,
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """Compute mean 128-d embedding per class on high-confidence pixels.

    Returns dict keyed by class name in {olivine, lcp, hcp, plagioclase}.
    Skips 'other' to keep things focused on the diagnostic mineral classes.
    """
    import pandas as pd
    import torch

    band_cols = [f"m{i}" for i in range(N_BANDS)]
    cols = (
        ["split", "confidence_tier", "olivine_t1", "olivine_t2", "lcp", "hcp",
         "plagioclase", "other"]
        + band_cols
    )
    df = pd.read_parquet(parquet_path, columns=cols)
    df = df[df["split"].isin(splits) & (df["confidence_tier"] == conf_tier)]

    class_to_filter = {
        "olivine": (df["olivine_t1"] + df["olivine_t2"]) >= 0.7,
        "lcp": df["lcp"] >= 0.7,
        "hcp": df["hcp"] >= 0.7,
        "plagioclase": df["plagioclase"] >= 0.7,
    }

    centroids: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for cls, mask in class_to_filter.items():
            sel = df[mask]
            if len(sel) == 0:
                continue
            if len(sel) > max_per_class:
                idx = rng.choice(len(sel), size=max_per_class, replace=False)
                sel = sel.iloc[idx]
            patches = np.stack(
                [_build_patch_from_row(r, band_cols) for _, r in sel.iterrows()],
                axis=0,
            ).astype(np.float32)
            patches = _normalize_patches(patches)
            embeds = []
            for start in range(0, len(patches), batch_size):
                x = torch.from_numpy(patches[start:start + batch_size]).to(device)
                out = model.encoder(x)
                center = out[:, center_idx]
                embeds.append(center.cpu().numpy())
            emb = np.concatenate(embeds, axis=0)
            centroids[cls] = emb.mean(axis=0).astype(np.float32)
    return centroids


def angle_to_centroids(
    embeddings: np.ndarray, centroids: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Per-class (H, W) cosine-angle rasters from per-pixel embeddings.

    Returns dict mapping class -> (H, W) float32 angle in radians.
    Invalid pixels (zero-norm embedding) return NaN.
    """
    h, w, d = embeddings.shape
    flat = embeddings.reshape(h * w, d).astype(np.float64)
    # Zero-embedding pixels: cause NaN (no information). Use spectral_angle's
    # zero-norm safeguard.
    out = {}
    for name, c in centroids.items():
        angles = spectral_angle(flat, c.astype(np.float64))
        out[name] = angles.reshape(h, w).astype(np.float32)
    return out
