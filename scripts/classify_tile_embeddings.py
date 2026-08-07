"""
Unsupervised classification of a CRISM mrral tile using spatial MAE embeddings.

For every pixel in the tile, extracts a 7×7 spectral patch, encodes it with
the pretrained SpatialSpectralMAE encoder, then clusters the embeddings with
k-means and visualizes the result.

Usage:
    python scripts/classify_tile_embeddings.py --tile /path/to/t0435_mrral_*.img
    python scripts/classify_tile_embeddings.py --tile /path/to/t0435_mrral_*.img --n_clusters 10
"""
import argparse
import os
import sys
import numpy as np
import torch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import get_device
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_mae import SpatialSpectralMAE

NODATA = 65535.0
CLIP_MAX = 0.5
N_BANDS = 59
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2  # 3


def load_tile(img_path: str) -> np.ndarray:
    """Load first 59 bands, clip, replace nodata with 0. Returns (H, W, 59) float32."""
    import rasterio
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)  # (59, H, W)
    nodata_mask = (data == NODATA) | ~np.isfinite(data)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata_mask] = 0.0
    valid_mask_2d = ~nodata_mask.any(axis=0)  # (H, W) True = all bands valid
    return data.transpose(1, 2, 0), valid_mask_2d  # (H, W, 59), (H, W)


def extract_patches_batched(tile: np.ndarray, batch_size: int = 4096):
    """
    Yield batches of (7, 7, 59) patches for every pixel, row-major order.
    Tile is padded with zeros. Yields (patches, flat_indices).
    """
    H, W, C = tile.shape
    padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')

    n_pixels = H * W
    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        rows = np.arange(start, end) // W
        cols = np.arange(start, end) % W

        # Extract patches: (B, 7, 7, 59)
        batch = np.stack([
            padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            for r, c in zip(rows, cols)
        ])
        yield batch.astype(np.float32), np.arange(start, end)


def normalize_patches(patches: np.ndarray) -> np.ndarray:
    """Per-patch zero-mean unit-variance normalization. (B, 7, 7, 59) → same shape."""
    B = patches.shape[0]
    flat = patches.reshape(B, -1)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((patches.reshape(B, -1) - mu) / sigma).reshape(patches.shape)


def load_model(ckpt_path: str, device: torch.device) -> SpatialSpectralMAE:
    model = SpatialSpectralMAE(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, embed_dim=128,
        n_heads=4, n_layers=6, decoder_dim=64, mask_ratio=0.85,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'mae_state' in state:
        state = state['mae_state']
    model.load_state_dict(state)
    model.eval()
    return model


def embed_tile(tile, model, device, batch_size=4096):
    """Run encoder over all pixels. Returns (H*W, embed_dim) float32."""
    H, W, _ = tile.shape
    n_pixels = H * W
    embed_dim = 128
    embeddings = np.zeros((n_pixels, embed_dim), dtype=np.float32)

    n_batches = (n_pixels + batch_size - 1) // batch_size
    with torch.no_grad():
        for patches, idx in tqdm(extract_patches_batched(tile, batch_size),
                                  total=n_batches, desc='Embedding'):
            patches = normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)  # (B, 7, 7, 59)
            emb = model.encode(x)                      # (B, embed_dim)
            embeddings[idx] = emb.cpu().numpy()

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', required=True, help='Path to mrral .img file')
    parser.add_argument('--ckpt', default='/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/checkpoints/spatial_mae_128d_6l_best.pt')
    parser.add_argument('--n_clusters', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--out', default=None, help='Output path (default: reports/)')
    parser.add_argument('--save_embeddings', default=None, metavar='PATH',
                        help='Save embeddings + valid_mask to .npz for reuse')
    parser.add_argument('--load_embeddings', default=None, metavar='PATH',
                        help='Load pre-saved embeddings instead of recomputing')
    args = parser.parse_args()

    tile_name = os.path.splitext(os.path.basename(args.tile))[0]
    out_path = args.out or f'/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/{tile_name}_clusters_k{args.n_clusters}.png'

    device = get_device()
    print(f'Device: {device}')

    print(f'Loading tile: {args.tile}')
    tile, valid_mask = load_tile(args.tile)
    H, W, _ = tile.shape
    print(f'Tile shape: {H} × {W} × {N_BANDS}  ({H*W:,} pixels)')

    if args.load_embeddings:
        print(f'Loading embeddings from {args.load_embeddings}')
        npz = np.load(args.load_embeddings)
        embeddings = npz['embeddings']
        valid_mask = npz['valid_mask']
    else:
        print('Loading model...')
        model = load_model(args.ckpt, device)
        print('Computing embeddings...')
        embeddings = embed_tile(tile, model, device, args.batch_size)  # (H*W, 128)
        if args.save_embeddings:
            np.savez_compressed(args.save_embeddings, embeddings=embeddings, valid_mask=valid_mask)
            print(f'Saved embeddings → {args.save_embeddings}')

    print(f'Clustering into {args.n_clusters} clusters (MiniBatchKMeans)...')
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=42,
                             batch_size=8192, n_init=5, max_iter=300)
    labels = kmeans.fit_predict(embeddings)  # (H*W,)
    label_map = labels.reshape(H, W)

    # Mask invalid pixels
    label_map_masked = label_map.astype(float)
    label_map_masked[~valid_mask] = np.nan

    print('Plotting...')
    cmap = plt.cm.get_cmap('tab10' if args.n_clusters <= 10 else 'tab20', args.n_clusters)
    cmap.set_bad(color='black')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # Left: false-color RGB from bands ~R:600nm, G:1000nm, B:2300nm
    r_idx = np.argmin(np.abs(np.array([410.12, 442.63, 533.74, 598.86, 650.99,
        683.59, 709.68, 742.30, 774.92, 801.04, 833.68, 859.81, 892.48, 925.16,
        951.31, 984.01, 1021.00, 1023.27, 1047.20, 1055.99, 1079.96, 1152.06,
        1211.09, 1250.45, 1257.01, 1263.57, 1276.70, 1329.21, 1368.61, 1394.89,
        1427.73, 1467.16, 1500.03, 1506.61, 1559.21, 1625.00, 1657.91, 1690.82,
        1750.09, 1809.39, 1875.30, 1928.06, 1974.24, 1980.84, 2007.23, 2066.64,
        2119.48, 2139.30, 2165.72, 2205.38, 2231.82, 2251.65, 2291.33, 2317.79,
        2331.02, 2350.87, 2390.58, 2430.30, 2456.79]) - 600))
    g_idx = 16  # ~1021 nm
    b_idx = 53  # ~2317 nm

    rgb = np.stack([tile[:, :, r_idx], tile[:, :, g_idx], tile[:, :, b_idx]], axis=-1)
    rgb = rgb / (np.percentile(rgb[valid_mask], 98, axis=0) + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb[~valid_mask] = 0.0

    axes[0].imshow(rgb, origin='upper')
    axes[0].set_title(f'{tile_name}\nFalse color (R:600nm G:1021nm B:2317nm)', fontsize=10)
    axes[0].axis('off')

    # Right: cluster map
    im = axes[1].imshow(label_map_masked, cmap=cmap,
                        vmin=-0.5, vmax=args.n_clusters - 0.5, origin='upper')
    axes[1].set_title(f'Spatial MAE embeddings — {args.n_clusters} clusters (k-means)', fontsize=10)
    axes[1].axis('off')
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(args.n_clusters), shrink=0.8)
    cbar.set_label('Cluster')

    fig.suptitle(f'Unsupervised classification of {tile_name}', fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
