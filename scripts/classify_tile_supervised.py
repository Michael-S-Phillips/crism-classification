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
# Defaults for 5-class checkpoints; _set_n_classes() rebinds all three from
# the checkpoint's head shape so 6-class (--with_alteration) ckpts work.
N_CLASSES = 5
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
CLASS_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#aaaaaa']
_CLASS_NAMES_6 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']
_CLASS_COLORS_6 = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#aaaaaa',
                   '#ffe119']  # alteration yellow, matches GPKG_CATEGORY_COLORS
_CLASS_NAMES_7 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
_CLASS_COLORS_7 = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#aaaaaa',
                   '#ffe119', '#808080']


def _set_n_classes(state):
    """Rebind N_CLASSES / CLASS_NAMES / CLASS_COLORS from a checkpoint
    state_dict's head.weight shape (5-, 6-, and 7-class supported)."""
    global N_CLASSES, CLASS_NAMES, CLASS_COLORS
    head_w = state.get('head.weight')
    if head_w is None:
        raise KeyError(f'no head.weight in checkpoint — not a classifier')
    n = int(head_w.shape[0])
    if n == N_CLASSES:
        return
    if n == 6:
        N_CLASSES, CLASS_NAMES, CLASS_COLORS = 6, _CLASS_NAMES_6, _CLASS_COLORS_6
    elif n == 7:
        N_CLASSES, CLASS_NAMES, CLASS_COLORS = 7, _CLASS_NAMES_7, _CLASS_COLORS_7
    elif n == 5:
        pass
    else:
        raise ValueError(f'unsupported head size {n} (expected 5, 6, or 7)')
    print(f'  checkpoint head: {N_CLASSES}-class {CLASS_NAMES}')

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


def derive_mrrsu_path(mrral_path):
    """t..._mrral_..._.img -> t..._mrrsu_..._.img in the same directory."""
    base = os.path.basename(mrral_path)
    return os.path.join(os.path.dirname(mrral_path), base.replace('_mrral_', '_mrrsu_'))


def load_mrrsu_aux_rasters(mrrsu_path, stats_json, patch_size=PATCH_SIZE):
    """Return (H, W, 2) float32 of normalized [mean7x7 RPEAK1, mean7x7 BD1300].

    Branches on ``stats_json["mode"]`` to mirror exactly the transform used at
    training time:

      - ``zscore``        : (x - mean) / std
      - ``minmax``        : clip((x - min) / (max - min), 0, 1)
      - ``pertile_zscore``: per-tile z-score where the tile's mean / std is
        computed on-the-fly from physically-valid raster pixels. If the tile has
        fewer than ``min_valid_per_tile`` physically-valid pixels, the global
        ``fallback_mean`` / ``fallback_std`` from the stats JSON are used --
        this matches the dataset's tile-level fallback behavior.

    Stats JSONs without ``version == 2`` are rejected (the dataset would refuse
    them at train time; we refuse them here for symmetry).

    Non-finite pixels (NaN/inf) after normalization map to 0.0 (the sample mean
    post-transform, i.e. "no information")."""
    import json
    import rasterio
    from data.mrrsu_aux import (
        AUX_BAND_ORDER,
        BD1300_BAND,
        NODATA as ND,
        RPEAK1_BAND,
        apply_invalid_to_nan,
        mean_pool_nodata,
        physically_valid_mask,
    )
    with open(stats_json) as f:
        st = json.load(f)
    version = st.get('version')
    if version != 2:
        raise ValueError(
            f"{stats_json} has version={version!r}; expected 2. Regenerate "
            "the aux cache with `scripts/build_mrrsu_aux.py`."
        )
    mode = st.get('mode', 'zscore')

    with rasterio.open(mrrsu_path) as src:
        rpeak = src.read(RPEAK1_BAND + 1).astype(np.float32)
        bd = src.read(BD1300_BAND + 1).astype(np.float32)
    # NaN-mask physically-implausible / sentinel pixels before pooling so they
    # don't pollute neighbour averages.
    rpeak = apply_invalid_to_nan(rpeak, "RPEAK1")
    bd = apply_invalid_to_nan(bd, "BD1300")
    rpeak_m = mean_pool_nodata(rpeak, patch_size=patch_size, nodata=ND)
    bd_m = mean_pool_nodata(bd, patch_size=patch_size, nodata=ND)
    # Belt-and-braces: pooled means that fall outside the physical range
    # (rare, e.g. at steep gradients) are also NaN.
    rpeak_m = np.where(physically_valid_mask(rpeak_m, "RPEAK1"), rpeak_m, np.nan)
    bd_m = np.where(physically_valid_mask(bd_m, "BD1300"), bd_m, np.nan)
    aux = np.stack([rpeak_m, bd_m], axis=-1).astype(np.float32)

    if mode == 'zscore':
        mean = np.asarray(st['mean'], dtype=np.float32)
        std = np.asarray(st['std'], dtype=np.float32)
        z = (aux - mean) / std
    elif mode == 'minmax':
        mn = np.asarray(st['min'], dtype=np.float32)
        mx = np.asarray(st['max'], dtype=np.float32)
        denom = np.where((mx - mn) < 1e-8, np.float32(1.0), (mx - mn))
        z = np.clip((aux - mn) / denom, 0.0, 1.0)
    elif mode == 'pertile_zscore':
        fallback_mean = np.asarray(st['fallback_mean'], dtype=np.float32)
        fallback_std = np.asarray(st['fallback_std'], dtype=np.float32)
        min_valid = int(st.get('min_valid_per_tile', 1000))
        # Build a per-band validity mask on the pooled raster -- counts only
        # physically-valid pixels in the *current* tile.
        flat = aux.reshape(-1, 2)
        valid_per_band = np.stack([
            physically_valid_mask(flat[:, j], AUX_BAND_ORDER[j])
            for j in range(2)
        ], axis=1)
        n_valid_per_band = valid_per_band.sum(axis=0)
        tile_mean = np.empty(2, dtype=np.float32)
        tile_std = np.empty(2, dtype=np.float32)
        for j in range(2):
            if n_valid_per_band[j] >= min_valid:
                vals = flat[valid_per_band[:, j], j]
                tile_mean[j] = vals.mean()
                tile_std[j] = vals.std() + np.float32(1e-8)
            else:
                tile_mean[j] = fallback_mean[j]
                tile_std[j] = fallback_std[j]
        z = (aux - tile_mean) / tile_std
    else:
        raise ValueError(
            f"unsupported norm mode {mode!r} in {stats_json}; expected one of "
            "{'zscore', 'minmax', 'pertile_zscore'}"
        )
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


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
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        val_map = state.get('val_mAP', None)
        state = state['model_state']
        print(f'  val_mAP from checkpoint: {val_map:.4f}' if val_map else '')
    _set_n_classes(state)
    model = SpatialSpectralClassifier(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
        embed_dim=128, n_heads=4, n_layers=6,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_supervised(tile, model, device, batch_size=4096, aux_rasters=None):
    """Returns prob_maps: (H*W, N_CLASSES) float32 in [0,1].

    If aux_rasters is provided (H, W, 2) float32, feeds per-pixel aux features
    to SpatialSpectralClassifierAux using the same row-major pixel ordering as
    extract_patches_batched (which yields idx = np.arange(start, end)).
    """
    H, W, _ = tile.shape
    n_pixels = H * W
    n_batches = (n_pixels + batch_size - 1) // batch_size
    probs = np.zeros((n_pixels, N_CLASSES), dtype=np.float32)

    # Flatten aux_rasters row-major to (H*W, 2) once; sliced per batch via idx.
    aux_flat = aux_rasters.reshape(-1, 2) if aux_rasters is not None else None

    from tqdm import tqdm
    with torch.no_grad():
        for patches, idx in tqdm(extract_patches_batched(tile, batch_size),
                                  total=n_batches, desc='Classifying'):
            patches = normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            if aux_flat is not None:
                aux_batch = torch.from_numpy(aux_flat[idx]).to(device)
                logits = model(x, aux_batch)              # (B, 5) aux path
            else:
                logits = model(x)                         # (B, 5)
            p = torch.sigmoid(logits).cpu().numpy()       # (B, 5)
            probs[idx] = p

    return probs


def save_probs(path: str, probs_hw: np.ndarray, valid_mask: np.ndarray,
               transform_arr: np.ndarray, crs_wkt: str) -> None:
    """Save (H,W,N) mineral probability raster to .npz for downstream vectorization.

    Args:
        path: output .npz path
        probs_hw: (H, W, N) float32 probabilities, N = len(CLASS_NAMES)
            (5-class legacy, or 6 with alteration)
        valid_mask: (H, W) bool, True = valid pixel
        transform_arr: (6,) float64 rasterio Affine coefficients (a,b,c,d,e,f)
        crs_wkt: CRS as WKT string
    """
    np.savez_compressed(
        path,
        probs=probs_hw,
        valid_mask=valid_mask,
        transform=transform_arr,
        crs_wkt=crs_wkt,
        # channel names — lets downstream vectorize scripts detect 5- vs
        # 6-class (alteration) outputs instead of assuming 5
        class_names=np.array(CLASS_NAMES),
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
                        help='Save (H,W,5) mineral prob raster to .npz for vectorization')
    parser.add_argument('--out', default=None, metavar='PATH',
                        help='Output figure path (overrides --out_dir naming)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip figure generation (implied when --save_probs is set without --out)')
    parser.add_argument('--mrrsu_aux', action='store_true',
                        help='Use SpatialSpectralClassifierAux with smoothed mrrsu '
                             'RPEAK1/BD1300 features.')
    parser.add_argument('--mrrsu_tile', type=str, default=None,
                        help='Paired mrrsu .img (default: derive from --tile path).')
    parser.add_argument('--mrrsu_aux_stats', type=str,
                        default='data/patch_cache/mrrsu_aux_stats.json',
                        help='z-score stats json from build_mrrsu_aux.py.')
    args = parser.parse_args()

    tile_name = os.path.splitext(os.path.basename(args.tile))[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading tile: {args.tile}')
    tile, valid_mask, transform, crs = load_tile(args.tile)
    H, W = valid_mask.shape
    print(f'Tile: {H}×{W}, {valid_mask.sum():,} valid pixels')

    print(f'Loading classifier: {args.ckpt}')
    if args.mrrsu_aux:
        from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            val_map = state.get('val_mAP', None)
            state = state['model_state']
            print(f'  val_mAP from checkpoint: {val_map:.4f}' if val_map else '')
        _set_n_classes(state)
        model = SpatialSpectralClassifierAux(
            n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
            embed_dim=128, n_heads=4, n_layers=6,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        mrrsu_path = args.mrrsu_tile or derive_mrrsu_path(args.tile)
        print(f'Loading mrrsu aux tile: {mrrsu_path}')
        aux_rasters = load_mrrsu_aux_rasters(mrrsu_path, args.mrrsu_aux_stats)
    else:
        model = load_classifier(args.ckpt, device)
        aux_rasters = None

    print('Running supervised inference...')
    probs_flat = run_supervised(tile, model, device, args.batch_size,
                                aux_rasters=aux_rasters)  # (H*W, 5)
    probs = probs_flat.reshape(H, W, N_CLASSES)  # (H, W, 5)

    if args.save_probs:
        # transform and crs already extracted by load_tile() above
        transform_arr = np.array([transform.a, transform.b, transform.c,
                                   transform.d, transform.e, transform.f],
                                  dtype=np.float64)
        crs_wkt = crs.to_wkt()
        save_probs(args.save_probs, probs, valid_mask, transform_arr, crs_wkt)
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
    skip_plot = args.no_plot or (args.save_probs and not args.out)
    if skip_plot:
        return
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
