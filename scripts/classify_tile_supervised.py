"""
Supervised per-pixel mineral classification of a CRISM mrral tile using the
fine-tuned SpatialSpectralClassifier.

Produces a comparison figure showing:
  - False-color RGB
  - Dominant predicted mineral class map
  - Per-class sigmoid probability heatmaps (5 classes)
  - Unsupervised PCA-filtered cluster map (drop PCs 0-3) for comparison

GeoPackage mineral polygon outlines are overlaid on the class maps.

Usage:
    python scripts/classify_tile_supervised.py \\
        --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \\
        --ckpt checkpoints/spvit_lrscale001_best.pt \\
        --embeddings /tmp/t0435_embeddings.npz \\
        --gpkg /mnt/mrdr/categorized_mineral_units/T0435.gpkg
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_spectral_transformer import SpatialSpectralClassifier

NODATA = 65535.0
CLIP_MAX = 0.5
N_BANDS = 59
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2  # 3
N_CLASSES = 5
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
CLASS_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#aaaaaa']

# GeoPackage category → display color mapping
GPKG_CATEGORY_COLORS = {
    'olivine':       '#e6194b',
    'lcp':           '#4363d8',
    'hcp':           '#3cb44b',
    'plagioclase':   '#f58231',
    'hcp+olivine':   '#911eb4',
    'olivine+plagio':'#42d4f4',
    'alteration':    '#ffe119',
    'other':         '#aaaaaa',
}


def load_tile(img_path):
    import rasterio
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
        transform = src.transform
        crs = src.crs
    nodata_mask = (data == NODATA) | ~np.isfinite(data)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata_mask] = 0.0
    valid_mask = ~nodata_mask.any(axis=0)
    return data.transpose(1, 2, 0), valid_mask, transform, crs


def extract_patches_batched(tile, batch_size=4096):
    H, W, C = tile.shape
    padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')
    n_pixels = H * W
    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        rows = np.arange(start, end) // W
        cols = np.arange(start, end) % W
        batch = np.stack([
            padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            for r, c in zip(rows, cols)
        ])
        yield batch.astype(np.float32), np.arange(start, end)


def normalize_patches(patches):
    B = patches.shape[0]
    flat = patches.reshape(B, -1)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((patches.reshape(B, -1) - mu) / sigma).reshape(patches.shape)


def load_classifier(ckpt_path, device):
    model = SpatialSpectralClassifier(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
        embed_dim=128, n_heads=4, n_layers=6,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        val_map = state.get('val_mAP', None)
        state = state['model_state']
        print(f'  val_mAP from checkpoint: {val_map:.4f}' if val_map else '')
    model.load_state_dict(state)
    model.eval()
    return model


def run_supervised(tile, model, device, batch_size=4096):
    """Returns prob_maps: (H*W, N_CLASSES) float32 in [0,1]."""
    H, W, _ = tile.shape
    n_pixels = H * W
    n_batches = (n_pixels + batch_size - 1) // batch_size
    probs = np.zeros((n_pixels, N_CLASSES), dtype=np.float32)

    from tqdm import tqdm
    with torch.no_grad():
        for patches, idx in tqdm(extract_patches_batched(tile, batch_size),
                                  total=n_batches, desc='Classifying'):
            patches = normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            logits = model(x)                         # (B, 5)
            p = torch.sigmoid(logits).cpu().numpy()   # (B, 5)
            probs[idx] = p

    return probs


def save_probs(path: str, probs_hw4: np.ndarray, valid_mask: np.ndarray,
               transform_arr: np.ndarray, crs_wkt: str) -> None:
    """Save (H,W,4) mineral probability raster to .npz for downstream vectorization.

    Args:
        path: output .npz path
        probs_hw4: (H, W, 4) float32 probabilities for olivine/lcp/hcp/plagioclase
        valid_mask: (H, W) bool, True = valid pixel
        transform_arr: (6,) float64 rasterio Affine coefficients (a,b,c,d,e,f)
        crs_wkt: CRS as WKT string
    """
    np.savez_compressed(
        path,
        probs=probs_hw4,
        valid_mask=valid_mask,
        transform=transform_arr,
        crs_wkt=crs_wkt,
    )


def build_pca_clusters(embeddings, valid_mask, n_components=20, drop_pcs=(0, 1, 2, 3),
                        n_clusters=8):
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(embeddings)
    keep = [i for i in range(n_components) if i not in drop_pcs]
    scores_clean = scores[:, keep]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                         batch_size=8192, n_init=5, max_iter=300)
    labels = km.fit_predict(scores_clean)
    return labels


def load_gpkg_polygons(gpkg_path, transform, shape):
    """Load GeoPackage polygons, return list of (category, pixel_coords) tuples."""
    try:
        import geopandas as gpd
        import rasterio.transform as rt
    except ImportError:
        print('geopandas not available — skipping polygon overlay')
        return []

    gdf = gpd.read_file(gpkg_path)
    H, W = shape
    result = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        cat = str(row.get('Category', row.get('category', 'other'))).lower().strip()
        # Normalize category names
        if 'olivine' in cat and 'plagio' in cat:
            cat = 'olivine+plagio'
        elif 'hcp' in cat and 'olivine' in cat:
            cat = 'hcp+olivine'
        elif 'olivine' in cat:
            cat = 'olivine'
        elif 'lcp' in cat:
            cat = 'lcp'
        elif 'hcp' in cat:
            cat = 'hcp'
        elif 'plagio' in cat or 'plagio' in cat:
            cat = 'plagioclase'
        elif 'alter' in cat:
            cat = 'alteration'
        else:
            cat = 'other'

        # Get polygon exterior coordinates in pixel space
        if geom.geom_type == 'Polygon':
            polygons = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            xs, ys = poly.exterior.xy
            rows_px, cols_px = rt.rowcol(transform, xs, ys)
            # Filter to within-bounds
            pts = [(c, r) for r, c in zip(rows_px, cols_px)
                   if 0 <= r < H and 0 <= c < W]
            if len(pts) >= 3:
                result.append((cat, pts))

    return result


def overlay_polygons(ax, polygons, alpha=0.0, edge_alpha=0.8, lw=0.8):
    """Draw polygon outlines on an axes."""
    for cat, pts in polygons:
        color = GPKG_CATEGORY_COLORS.get(cat, '#ffffff')
        poly = mpatches.Polygon(pts, closed=True,
                                facecolor='none',
                                edgecolor=color,
                                linewidth=lw,
                                alpha=edge_alpha)
        ax.add_patch(poly)


def make_false_color(tile, valid_mask):
    WAVELENGTHS = np.array([
        410.12, 442.63, 533.74, 598.86, 650.99, 683.59, 709.68, 742.30, 774.92,
        801.04, 833.68, 859.81, 892.48, 925.16, 951.31, 984.01, 1021.00, 1023.27,
        1047.20, 1055.99, 1079.96, 1152.06, 1211.09, 1250.45, 1257.01, 1263.57,
        1276.70, 1329.21, 1368.61, 1394.89, 1427.73, 1467.16, 1500.03, 1506.61,
        1559.21, 1625.00, 1657.91, 1690.82, 1750.09, 1809.39, 1875.30, 1928.06,
        1974.24, 1980.84, 2007.23, 2066.64, 2119.48, 2139.30, 2165.72, 2205.38,
        2231.82, 2251.65, 2291.33, 2317.79, 2331.02, 2350.87, 2390.58, 2430.30, 2456.79,
    ])
    r_idx = int(np.argmin(np.abs(WAVELENGTHS - 600)))
    g_idx = 16   # ~1021 nm
    b_idx = 53   # ~2317 nm
    rgb = np.stack([tile[:, :, r_idx], tile[:, :, g_idx], tile[:, :, b_idx]], axis=-1)
    rgb = rgb / (np.percentile(rgb[valid_mask], 98, axis=0) + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb[~valid_mask] = 0.0
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', required=True)
    parser.add_argument('--ckpt', default='checkpoints/spvit_lrscale001_best.pt')
    parser.add_argument('--embeddings', default=None, metavar='PATH',
                        help='Pre-saved embeddings .npz for comparison cluster map')
    parser.add_argument('--gpkg', default=None, metavar='PATH',
                        help='GeoPackage with mineral polygon labels')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_clusters', type=int, default=8)
    parser.add_argument('--drop_pcs', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--out_dir', default='/mnt/mrdr/crism_classification/reports')
    parser.add_argument('--save_probs', default=None, metavar='PATH',
                        help='Save (H,W,4) mineral prob raster to .npz for vectorization')
    parser.add_argument('--out', default=None, metavar='PATH',
                        help='Output figure path (overrides --out_dir naming)')
    args = parser.parse_args()

    tile_name = os.path.splitext(os.path.basename(args.tile))[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading tile: {args.tile}')
    tile, valid_mask, transform, crs = load_tile(args.tile)
    H, W = valid_mask.shape
    print(f'Tile: {H}×{W}, {valid_mask.sum():,} valid pixels')

    print(f'Loading classifier: {args.ckpt}')
    model = load_classifier(args.ckpt, device)

    print('Running supervised inference...')
    probs_flat = run_supervised(tile, model, device, args.batch_size)  # (H*W, 5)
    probs = probs_flat.reshape(H, W, N_CLASSES)  # (H, W, 5)

    if args.save_probs:
        # transform and crs already extracted by load_tile() above
        transform_arr = np.array([transform.a, transform.b, transform.c,
                                   transform.d, transform.e, transform.f],
                                  dtype=np.float64)
        crs_wkt = crs.to_wkt()
        probs_hw4 = probs[:, :, :4]  # drop "other" class (index 4)
        save_probs(args.save_probs, probs_hw4, valid_mask, transform_arr, crs_wkt)
        print(f'Saved probs → {args.save_probs}')

    # Dominant class per pixel (argmax over classes)
    dom_class = np.argmax(probs, axis=2).astype(float)   # (H, W)
    dom_class[~valid_mask] = np.nan

    # Mask invalid pixels in prob maps
    probs_masked = probs.copy()
    for c in range(N_CLASSES):
        probs_masked[:, :, c][~valid_mask] = np.nan

    # Load GeoPackage
    polygons = []
    if args.gpkg and os.path.exists(args.gpkg):
        print(f'Loading GeoPackage: {args.gpkg}')
        polygons = load_gpkg_polygons(args.gpkg, transform, (H, W))
        print(f'  {len(polygons)} polygons loaded')

    # Build unsupervised comparison (PCA-filtered k-means)
    labels_clean = None
    if args.embeddings and os.path.exists(args.embeddings):
        print(f'Loading embeddings for unsupervised comparison: {args.embeddings}')
        npz = np.load(args.embeddings)
        embeddings = npz['embeddings']
        valid_mask_emb = npz['valid_mask']
        emb_valid = embeddings[valid_mask_emb.ravel()]
        print(f'Running PCA (drop PCs {args.drop_pcs}) + k-means ({args.n_clusters} clusters)...')
        labels_all = np.full(H * W, np.nan)
        labels_valid = build_pca_clusters(emb_valid, valid_mask_emb,
                                           drop_pcs=args.drop_pcs,
                                           n_clusters=args.n_clusters)
        labels_all_flat = np.full(H * W, np.nan)
        labels_all_flat[valid_mask_emb.ravel()] = labels_valid
        labels_clean = labels_all_flat.reshape(H, W)

    # -----------------------------------------------------------------------
    # Figure layout
    # Row 0: false color | dominant class map | (unsupervised if available)
    # Row 1: per-class probability maps (5 panels)
    # -----------------------------------------------------------------------
    has_unsup = labels_clean is not None
    ncols_top = 3 if has_unsup else 2
    nrows = 2
    fig_w = ncols_top * 5
    fig_h = 10

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows, max(ncols_top, N_CLASSES))

    # --- Row 0 ---
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_dom = fig.add_subplot(gs[0, 1])

    false_color = make_false_color(tile, valid_mask)
    ax_rgb.imshow(false_color, origin='upper')
    ax_rgb.set_title(f'{tile_name}\nFalse color (R:600 G:1021 B:2317 nm)', fontsize=9)
    ax_rgb.axis('off')
    if polygons:
        overlay_polygons(ax_rgb, polygons)

    # Dominant class colormap: one color per class
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_dom = ListedColormap(CLASS_COLORS)
    bounds = np.arange(-0.5, N_CLASSES)
    norm_dom = BoundaryNorm(bounds, cmap_dom.N)
    cmap_dom.set_bad('black')

    im_dom = ax_dom.imshow(dom_class, cmap=cmap_dom, norm=norm_dom, origin='upper')
    ax_dom.set_title('Supervised: dominant class\n(spvit_lrscale001)', fontsize=9)
    ax_dom.axis('off')
    if polygons:
        overlay_polygons(ax_dom, polygons)
    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(N_CLASSES)]
    ax_dom.legend(handles=handles, loc='lower right', fontsize=7,
                  framealpha=0.7, ncol=2)

    if has_unsup:
        ax_unsup = fig.add_subplot(gs[0, 2])
        cmap_unsup = plt.colormaps.get_cmap('tab10').resampled(args.n_clusters)
        cmap_unsup.set_bad('black')
        ax_unsup.imshow(labels_clean, cmap=cmap_unsup,
                        vmin=-0.5, vmax=args.n_clusters - 0.5, origin='upper')
        ax_unsup.set_title(f'Unsupervised: MAE embeddings\nPCA drop {args.drop_pcs}, k={args.n_clusters}',
                           fontsize=9)
        ax_unsup.axis('off')
        if polygons:
            overlay_polygons(ax_unsup, polygons)

    # --- Row 1: per-class probability maps ---
    for ci in range(N_CLASSES):
        ax = fig.add_subplot(gs[1, ci])
        pmap = probs_masked[:, :, ci]
        im = ax.imshow(pmap, cmap='hot', vmin=0, vmax=1, origin='upper')
        ax.set_title(f'{CLASS_NAMES[ci]}\n(prob)', fontsize=9)
        ax.axis('off')
        if polygons:
            overlay_polygons(ax, polygons)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    # GeoPackage legend
    if polygons:
        gpkg_handles = [mpatches.Patch(color=v, label=k)
                        for k, v in GPKG_CATEGORY_COLORS.items()]
        fig.legend(handles=gpkg_handles, title='GeoPackage labels',
                   loc='lower center', ncol=len(GPKG_CATEGORY_COLORS),
                   fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Supervised vs. Unsupervised Classification — {tile_name}', fontsize=12)

    drop_str = ''.join(str(p) for p in args.drop_pcs)
    out_name = f'{tile_name}_supervised_vs_unsup_drop{drop_str}_k{args.n_clusters}.png'
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(args.out_dir, out_name)
        os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
