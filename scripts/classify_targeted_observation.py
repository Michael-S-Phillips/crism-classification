"""
Run the v3 (denoising) classifier on a CRISM TARGETED observation
(e.g. FRT / FRS / HRL / HRS) by subsampling its native ~545-band spectral
axis down to the 59 MRDR mrral bands the model expects.

The targeted (TRR3) products from Mars ODE are higher spectral resolution
(~6.55 nm/band) and higher spatial resolution (~18 m/px) than the MRDR
mrral tiles (~33 nm/band, ~180 m/px) the classifier was trained on. This
script handles the spectral resampling. Spatial resampling is optional
(--downsample N).

Spectral subsampling: for each MRDR wavelength λ_i, pick the targeted band
whose center wavelength is closest to λ_i (nearest-neighbor). MRDR
wavelengths are read from any /mnt/mrdr/mc*/t*mrral*.hdr file.

Output is the same .npz + figure pair as classify_tile_supervised.py, so
downstream pipelines (vectorize, threshold compare, etc.) consume it
unchanged.

ODE-downloaded file structure (expected):
    frt00009d44_07_if165j_mtr3.img    (binary ENVI cube)
    frt00009d44_07_if165j_mtr3.hdr    (ENVI header with `wavelength` field)
    frt00009d44_07_if165j_mtr3.lbl    (PDS3 label, not required if .hdr present)

If only a .lbl is present (no .hdr), regenerate the .hdr first via
ISIS' `pds2isis` or `crism2isis` workflow, or use rasterio's GDAL driver
which can read ENVI-style binary directly if accompanied by a sidecar.

Usage:
    conda run -n crism python scripts/classify_targeted_observation.py \\
        --tile /path/to/frt00009d44_07_if165j_mtr3.img \\
        --ckpt checkpoints/ft_v3_denoising_lrscale001_best.pt \\
        --save_probs /tmp/frt00009d44_probs.npz \\
        --out /tmp/frt00009d44_classify_panel.png

With spatial 10× downsampling to approximate MRDR resolution:
        --downsample 10

NOTE: even with spectral resampling the model sees a different spatial
context than during training. The 7×7 patch at native targeted resolution
covers ~126 m × 126 m of surface, vs ~1.26 km × 1.26 km at MRDR
resolution. Predictions on targeted data should be regarded as a
research-grade exploration of generalization, not a calibrated product.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import spectral.io.envi as envi
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_spectral_transformer import SpatialSpectralClassifier

# Match classify_tile_supervised.py conventions
NODATA       = 65535.0
CLIP_MAX     = 0.5
N_BANDS      = 59
PATCH_SIZE   = 7
PAD          = PATCH_SIZE // 2
N_CLASSES    = 5
CLASS_NAMES  = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
CLASS_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#aaaaaa']


# ── wavelength + band-mapping helpers ─────────────────────────────────────────

def get_mrdr_wavelengths() -> np.ndarray:
    """Read the 59 MRDR mrral wavelengths from the first available mrral .hdr."""
    import glob
    hdrs = sorted(glob.glob('/mnt/mrdr/mc*/t*mrral*.hdr'))
    if not hdrs:
        raise RuntimeError(
            'No mrral .hdr files found under /mnt/mrdr/mc*/. Pass --mrdr_hdr '
            'to override.'
        )
    hdr = envi.open(hdrs[0])
    return np.array([float(w) for w in hdr.metadata['wavelength']][:N_BANDS],
                    dtype=np.float32)


def read_targeted_wavelengths(hdr_path: str) -> np.ndarray:
    """Read all wavelengths (nm) from a CRISM TRR3 ENVI .hdr."""
    hdr = envi.open(hdr_path)
    wls = np.array([float(w) for w in hdr.metadata['wavelength']], dtype=np.float32)
    units = hdr.metadata.get('wavelength units', 'nm').lower()
    if units in ('micrometers', 'um', 'μm', 'micron'):
        wls = wls * 1000.0
    return wls


def find_band_mapping(target_wls: np.ndarray, mrdr_wls: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """For each MRDR wavelength, return (band_index, residual_nm) in the
    target observation. band_index is 0-indexed."""
    band_idx  = np.empty(len(mrdr_wls), dtype=np.int32)
    residuals = np.empty(len(mrdr_wls), dtype=np.float32)
    for i, lam in enumerate(mrdr_wls):
        diffs = np.abs(target_wls - lam)
        bi    = int(np.argmin(diffs))
        band_idx[i]  = bi
        residuals[i] = diffs[bi]
    return band_idx, residuals


# ── data loading ──────────────────────────────────────────────────────────────

def load_targeted(img_path: str, band_idx_1based: list, downsample: int = 1
                  ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read the 59 selected bands from a targeted observation, apply MRDR-style
    preprocessing, and (optionally) block-average spatially.

    Returns:
        cube       : (H, W, 59) float32, NaN-zeroed and clipped to [0, CLIP_MAX]
        valid_mask : (H, W) bool, True where ALL 59 bands were valid
        profile    : rasterio src profile (transform, crs, etc.)
    """
    with rasterio.open(img_path) as src:
        if max(band_idx_1based) > src.count:
            raise ValueError(
                f'Band index {max(band_idx_1based)} exceeds file band count '
                f'{src.count}. Re-check wavelength axis or .hdr metadata.'
            )
        data = src.read(band_idx_1based).astype(np.float32)   # (59, H, W)
        profile = {
            'height':    src.height,
            'width':     src.width,
            'transform': src.transform,
            'crs':       src.crs,
        }

    nodata_mask = (data == NODATA) | ~np.isfinite(data) | (data < 0)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata_mask] = 0.0
    valid_mask = ~nodata_mask.any(axis=0)   # (H, W) — bool

    # (59, H, W) → (H, W, 59) for the classifier
    cube = data.transpose(1, 2, 0)

    if downsample > 1:
        cube, valid_mask, profile = _block_average(cube, valid_mask, profile, downsample)
    return cube, valid_mask, profile


def _block_average(cube, valid_mask, profile, factor):
    """Block-average a (H, W, B) cube by `factor` along H and W. Crops to
    a multiple of `factor`. Updates transform to reflect coarser pixels."""
    H, W, B = cube.shape
    H2 = (H // factor) * factor
    W2 = (W // factor) * factor
    cube = cube[:H2, :W2]
    valid_mask = valid_mask[:H2, :W2]

    cube_r = cube.reshape(H2 // factor, factor, W2 // factor, factor, B)
    cube_ds = cube_r.mean(axis=(1, 3))
    mask_r = valid_mask.reshape(H2 // factor, factor, W2 // factor, factor)
    mask_ds = mask_r.all(axis=(1, 3))

    new_transform = profile['transform'] * profile['transform'].scale(factor, factor)
    profile['transform'] = new_transform
    profile['height']    = cube_ds.shape[0]
    profile['width']     = cube_ds.shape[1]
    return cube_ds.astype(np.float32), mask_ds, profile


# ── classifier (mirrors classify_tile_supervised.py) ──────────────────────────

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
        yield start, end, batch


def normalize_patches(patches):
    out = patches.astype(np.float32, copy=True)
    flat = out.reshape(out.shape[0], -1)
    valid_count = np.sum(flat != 0, axis=1).clip(min=1)
    s  = flat.sum(axis=1) / valid_count
    s2 = (flat * flat).sum(axis=1) / valid_count
    mu = s.reshape(-1, 1, 1, 1)
    sd = np.sqrt(np.clip(s2 - s * s, 1e-12, None)).reshape(-1, 1, 1, 1)
    nonzero = (out != 0)
    out = np.where(nonzero, (out - mu) / sd, 0.0)
    return out


def load_classifier(ckpt_path, device):
    model = SpatialSpectralClassifier(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
        embed_dim=128, n_heads=4, n_layers=6,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt['state_dict']
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def run_inference(tile, model, device, batch_size=4096):
    from tqdm import tqdm
    H, W, _ = tile.shape
    probs = np.zeros((H * W, N_CLASSES), dtype=np.float32)
    n_batches = (H * W + batch_size - 1) // batch_size
    with torch.no_grad():
        for start, end, patches in tqdm(
            extract_patches_batched(tile, batch_size),
            total=n_batches, desc='Classifying',
        ):
            patches = normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()
            probs[start:end] = p
    return probs.reshape(H, W, N_CLASSES)


# ── output helpers ────────────────────────────────────────────────────────────

def save_probs(path, probs_hw, valid_mask, transform_arr, crs_wkt):
    np.savez_compressed(
        path, probs=probs_hw, valid_mask=valid_mask,
        transform=transform_arr, crs_wkt=crs_wkt,
    )


def render_panel(probs, valid_mask, cube, out_path, run_label, band_residuals):
    """Brief multi-panel: false-color RGB + dominant class + per-class heatmaps."""
    H, W, _ = probs.shape

    # False-color RGB from selected bands matching what classify_tile_supervised does
    r = cube[:, :, 46]; g = cube[:, :, 33]; b = cube[:, :, 20]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb / np.percentile(rgb[valid_mask], 98) if valid_mask.any() else rgb
    rgb = np.clip(rgb, 0, 1)
    rgb[~valid_mask] = 0

    # Dominant class (argmax above 0.5; else "other")
    dom = np.argmax(probs, axis=-1)
    dom[probs.max(axis=-1) < 0.5] = N_CLASSES - 1   # 'other'
    dom_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ci, hex_c in enumerate(CLASS_COLORS):
        rgb_c = np.array([int(hex_c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        dom_rgb[dom == ci] = rgb_c
    dom_rgb[~valid_mask] = 0

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    axes[0, 0].imshow(rgb); axes[0, 0].set_title('false color')
    axes[0, 1].imshow(dom_rgb); axes[0, 1].set_title('dominant class (>0.5)')

    for ci, cname in enumerate(CLASS_NAMES):
        ax = axes[0 + ci // 4, 2 + (ci % 4) if ci < 2 else (ci - 2) % 4]
        if ci < 2:
            ax = axes[0, 2 + ci]
        else:
            ax = axes[1, ci - 2]
        im = ax.imshow(probs[:, :, ci], cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'p({cname})', fontsize=9)

    legend = [mpatches.Patch(color=c, label=n)
              for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    axes[0, 1].legend(handles=legend, loc='lower right', fontsize=6, framealpha=0.85)

    band_mae = float(np.mean(band_residuals))
    fig.suptitle(f'{run_label}  |  mean band-mapping residual: {band_mae:.2f} nm',
                 fontsize=10)
    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tile', required=True,
                   help='Path to the targeted observation .img file '
                        '(needs matching .hdr alongside)')
    p.add_argument('--ckpt', required=True,
                   help='Path to the fine-tuned SpatialSpectralClassifier checkpoint')
    p.add_argument('--save_probs', required=True,
                   help='Output .npz path for per-pixel probabilities')
    p.add_argument('--out', default=None,
                   help='Output .png path for the diagnostic panel')
    p.add_argument('--mrdr_hdr', default=None,
                   help='Override path to an mrral .hdr to read 59 reference '
                        'wavelengths from (default: auto-discover under /mnt/mrdr/mc*).')
    p.add_argument('--downsample', type=int, default=1,
                   help='Block-average factor over the spatial axes before '
                        'inference (1=no downsample, 10≈MRDR resolution).')
    p.add_argument('--batch_size', type=int, default=4096)
    args = p.parse_args()

    hdr_path = args.tile.rsplit('.', 1)[0] + '.hdr'
    if not os.path.exists(hdr_path):
        # Try uppercased extension
        for cand in (args.tile.rsplit('.', 1)[0] + '.HDR', args.tile + '.hdr'):
            if os.path.exists(cand):
                hdr_path = cand
                break
        else:
            sys.exit(
                f'No ENVI header found alongside {args.tile}. Expected '
                f'{hdr_path}. Regenerate via ISIS or rasterio if needed.'
            )

    # 1. Wavelength axes
    print(f'Reading targeted wavelengths from {hdr_path} ...')
    target_wls = read_targeted_wavelengths(hdr_path)
    print(f'  {len(target_wls)} bands span {target_wls.min():.1f} – {target_wls.max():.1f} nm')

    if args.mrdr_hdr:
        h = envi.open(args.mrdr_hdr)
        mrdr_wls = np.array([float(w) for w in h.metadata['wavelength']][:N_BANDS],
                             dtype=np.float32)
    else:
        mrdr_wls = get_mrdr_wavelengths()
    print(f'Reference MRDR axis: {len(mrdr_wls)} bands, '
          f'{mrdr_wls.min():.1f} – {mrdr_wls.max():.1f} nm')

    # 2. Map each MRDR band to nearest targeted band
    band_idx0, residuals = find_band_mapping(target_wls, mrdr_wls)
    band_idx1 = (band_idx0 + 1).tolist()   # rasterio is 1-indexed
    print(f'Band-mapping residuals: mean {residuals.mean():.2f} nm, '
          f'max {residuals.max():.2f} nm')
    if residuals.max() > 50.0:
        print(f'  WARNING: largest residual {residuals.max():.1f} nm — check that the '
              'targeted .hdr wavelength axis covers the full MRDR range.')

    # 3. Load + preprocess the targeted cube
    print(f'Loading targeted cube ...')
    cube, valid_mask, profile = load_targeted(args.tile, band_idx1, args.downsample)
    print(f'  shape after preprocessing: {cube.shape}  '
          f'valid pixels: {int(valid_mask.sum()):,}')

    # 4. Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading classifier from {args.ckpt} ...')
    model = load_classifier(args.ckpt, device)
    print('Running inference ...')
    probs = run_inference(cube, model, device, batch_size=args.batch_size)

    # 5. Save probs
    transform_arr = np.array(profile['transform'].to_gdal(), dtype=np.float64)
    crs_wkt = profile['crs'].to_wkt() if profile['crs'] else ''
    save_probs(args.save_probs, probs, valid_mask, transform_arr, crs_wkt)
    print(f'Wrote {args.save_probs}')

    # 6. Optional diagnostic figure
    if args.out:
        run_label = os.path.basename(args.tile).rsplit('.', 1)[0]
        if args.downsample > 1:
            run_label += f' (×{args.downsample} spatial downsample)'
        render_panel(probs, valid_mask, cube, args.out, run_label, residuals)
        print(f'Wrote {args.out}')

    # 7. Class summary
    print('\nPer-class fraction of valid pixels with p > 0.5:')
    for ci, cname in enumerate(CLASS_NAMES):
        frac = float(((probs[:, :, ci] > 0.5) & valid_mask).sum() / max(1, valid_mask.sum()))
        print(f'  {cname:<12}  {frac * 100:>6.2f}%')


if __name__ == '__main__':
    main()
