"""
PCA analysis of spatial MAE tile embeddings.

Step 1 (explore): Visualize spatial maps of the top N principal components
  to identify which PCs capture edge/artifact variance.

Step 2 (filter): Drop artifact PCs, re-cluster on clean subspace, compare.

Usage:
  # First: embed the tile and save
  python scripts/classify_tile_embeddings.py \
      --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
      --save_embeddings /tmp/t0435_embeddings.npz

  # Step 1: explore PCs
  python scripts/analyze_embedding_pca.py \
      --embeddings /tmp/t0435_embeddings.npz \
      --n_components 20 --mode explore

  # Step 2: re-cluster dropping artifact PCs
  python scripts/analyze_embedding_pca.py \
      --embeddings /tmp/t0435_embeddings.npz \
      --n_components 20 --mode cluster --drop_pcs 0 3 --n_clusters 8
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import binary_dilation

OUT_DIR = '/mnt/mrdr/crism_classification/reports'


def load_embeddings(path):
    npz = np.load(path)
    return npz['embeddings'], npz['valid_mask']


def build_edge_mask(valid_mask, dilation_px=5):
    """True where pixel is valid but within dilation_px of a nodata pixel."""
    nodata = ~valid_mask
    near_nodata = binary_dilation(nodata, iterations=dilation_px)
    return valid_mask & near_nodata


def artifact_scores(pc_maps, valid_mask, edge_mask):
    """
    For each PC spatial map, compute:
      edge_corr:   |correlation| of PC scores with edge mask (valid pixels only)
      stripe_var:  fraction of variance explained by row-mean vs pixel variance
                   (high = horizontal stripe pattern typical of CRISM artifacts)
    Returns arrays of shape (n_pcs,).
    """
    n_pcs = pc_maps.shape[2]
    edge_corr = np.zeros(n_pcs)
    stripe_frac = np.zeros(n_pcs)

    edge_flat = edge_mask[valid_mask].astype(float)
    edge_flat -= edge_flat.mean()

    for i in range(n_pcs):
        pc_map = pc_maps[:, :, i]
        vals = pc_map[valid_mask]
        vals_z = vals - vals.mean()

        # Edge correlation
        denom = np.std(vals_z) * np.std(edge_flat)
        if denom > 1e-9:
            edge_corr[i] = abs(np.mean(vals_z * edge_flat) / denom)

        # Stripe fraction: how much of valid-region variance is in row means
        H = pc_map.shape[0]
        row_means = np.array([
            pc_map[r, valid_mask[r]].mean() if valid_mask[r].any() else 0.0
            for r in range(H)
        ])
        # Variance of row means (repeated per valid pixel) vs total variance
        row_mean_expanded = np.array([
            row_means[r] for r in range(H) for c in range(pc_map.shape[1])
            if valid_mask[r, c]
        ])
        total_var = vals.var()
        stripe_frac[i] = row_mean_expanded.var() / (total_var + 1e-9)

    return edge_corr, stripe_frac


def explore(embeddings, valid_mask, n_components, tile_name, out_dir):
    H, W = valid_mask.shape
    print(f'Running PCA ({n_components} components) on {embeddings.shape[0]:,} pixels...')
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(embeddings)  # (H*W, n_components)

    # Reshape to spatial maps
    pc_maps = scores.reshape(H, W, n_components)

    edge_mask = build_edge_mask(valid_mask)
    edge_corr, stripe_frac = artifact_scores(pc_maps, valid_mask, edge_mask)

    print(f'\n{"PC":>3}  {"Expl.Var%":>9}  {"EdgeCorr":>8}  {"StripeFrac":>10}  Artifact?')
    print('-' * 55)
    for i in range(n_components):
        flag = ''
        if edge_corr[i] > 0.15:
            flag += ' EDGE'
        if stripe_frac[i] > 0.3:
            flag += ' STRIPE'
        print(f'{i:>3}  {100*pca.explained_variance_ratio_[i]:>9.2f}  '
              f'{edge_corr[i]:>8.3f}  {stripe_frac[i]:>10.3f}  {flag}')

    # Plot grid of PC spatial maps
    ncols = 5
    nrows = (n_components + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5),
                             constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_components):
        ax = axes[i]
        pc_map = pc_maps[:, :, i].copy().astype(float)
        pc_map[~valid_mask] = np.nan
        # Robust percentile scaling
        vals = pc_map[valid_mask]
        vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
        im = ax.imshow(pc_map, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper')

        flag = ''
        if edge_corr[i] > 0.15: flag += ' E'
        if stripe_frac[i] > 0.3: flag += ' S'
        ax.set_title(f'PC{i}  {100*pca.explained_variance_ratio_[i]:.1f}%{flag}',
                     fontsize=8, color='red' if flag else 'black')
        ax.axis('off')

    for j in range(n_components, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'{tile_name} — PCA of MAE embeddings\n'
        f'E=edge artifact  S=stripe artifact  (drop suspicious PCs then re-run with --mode cluster --drop_pcs ...)',
        fontsize=10
    )
    out_path = os.path.join(out_dir, f'{tile_name}_pca_explore_n{n_components}.png')
    plt.savefig(out_path, dpi=130)
    print(f'\nSaved → {out_path}')
    return pca, pc_maps


def cluster(embeddings, valid_mask, n_components, drop_pcs, n_clusters, tile_name, out_dir):
    H, W = valid_mask.shape
    print(f'Running PCA ({n_components} components)...')
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(embeddings)  # (H*W, n_components)

    keep = [i for i in range(n_components) if i not in drop_pcs]
    print(f'Dropping PCs {drop_pcs}, keeping {keep} ({len(keep)} components)')
    scores_clean = scores[:, keep]

    print(f'Clustering into {n_clusters} clusters...')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             batch_size=8192, n_init=5, max_iter=300)

    # Cluster on raw embeddings too for comparison
    labels_raw = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                  batch_size=8192, n_init=5, max_iter=300).fit_predict(embeddings)
    labels_clean = kmeans.fit_predict(scores_clean)

    def make_map(labels):
        m = labels.reshape(H, W).astype(float)
        m[~valid_mask] = np.nan
        return m

    cmap = plt.colormaps.get_cmap('tab10' if n_clusters <= 10 else 'tab20').resampled(n_clusters)
    cmap.set_bad('black')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, lbl, title in [
        (axes[0], labels_raw, f'Raw embeddings (128-dim)\n{n_clusters} clusters'),
        (axes[1], labels_clean, f'PCA filtered (drop PCs {drop_pcs})\n{n_clusters} clusters'),
    ]:
        ax.imshow(make_map(lbl), cmap=cmap, vmin=-0.5, vmax=n_clusters - 0.5, origin='upper')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle(f'Artifact-filtered clustering — {tile_name}', fontsize=12)
    drop_str = '_'.join(str(p) for p in drop_pcs)
    out_path = os.path.join(out_dir, f'{tile_name}_pca_filtered_drop{drop_str}_k{n_clusters}.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True, help='.npz file from classify_tile_embeddings.py')
    parser.add_argument('--n_components', type=int, default=20)
    parser.add_argument('--mode', choices=['explore', 'cluster'], default='explore')
    parser.add_argument('--drop_pcs', type=int, nargs='+', default=[],
                        help='PC indices to drop before clustering (--mode cluster only)')
    parser.add_argument('--n_clusters', type=int, default=8)
    parser.add_argument('--out_dir', default=OUT_DIR)
    args = parser.parse_args()

    tile_name = os.path.splitext(os.path.basename(args.embeddings))[0].replace('_embeddings', '')
    embeddings, valid_mask = load_embeddings(args.embeddings)

    if args.mode == 'explore':
        explore(embeddings, valid_mask, args.n_components, tile_name, args.out_dir)
    else:
        cluster(embeddings, valid_mask, args.n_components, args.drop_pcs,
                args.n_clusters, tile_name, args.out_dir)


if __name__ == '__main__':
    main()
