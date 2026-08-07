"""
Visualize spatial MAE reconstructions vs. original CRISM spectra.

Loads a SpatialSpectralMAE checkpoint, samples a few patches from the
val patch cache, runs a forward pass, and plots original vs. reconstructed
spectra for visible and masked pixels.

Usage:
    python scripts/visualize_mae_reconstructions.py
    python scripts/visualize_mae_reconstructions.py --ckpt checkpoints/spatial_mae_128d_6l_best.pt
    python scripts/visualize_mae_reconstructions.py --n_examples 6 --seed 1
"""
import argparse
import sys
import os
import numpy as np
import torch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import get_device
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_mae import SpatialSpectralMAE

WAVELENGTHS = [
    410.12, 442.63, 533.74, 598.86, 650.99, 683.59, 709.68, 742.30, 774.92,
    801.04, 833.68, 859.81, 892.48, 925.16, 951.31, 984.01, 1021.00, 1023.27,
    1047.20, 1055.99, 1079.96, 1152.06, 1211.09, 1250.45, 1257.01, 1263.57,
    1276.70, 1329.21, 1368.61, 1394.89, 1427.73, 1467.16, 1500.03, 1506.61,
    1559.21, 1625.00, 1657.91, 1690.82, 1750.09, 1809.39, 1875.30, 1928.06,
    1974.24, 1980.84, 2007.23, 2066.64, 2119.48, 2139.30, 2165.72, 2205.38,
    2231.82, 2251.65, 2291.33, 2317.79, 2331.02, 2350.87, 2390.58, 2430.30,
    2456.79,
]  # 59 bands, 410–2457 nm


def load_model(ckpt_path: str, device: torch.device) -> SpatialSpectralMAE:
    model = SpatialSpectralMAE(
        n_bands=59, patch_size=7, embed_dim=128,
        n_heads=4, n_layers=6, decoder_dim=64, mask_ratio=0.75,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Handle checkpoints saved with 'model_state_dict' key
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return model


def load_patches(cache_dir: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Load n random patches from val cache. Returns (n, 7, 7, 59)."""
    val_path = os.path.join(cache_dir, 'mrral_val_patches_p7.npy')
    patches = np.load(val_path, mmap_mode='r')
    idx = rng.choice(len(patches), size=n, replace=False)
    return patches[idx].astype(np.float32)


def plot_reconstructions(patches, recon, mask, out_path):
    """
    patches: (n, 49, 59) original
    recon:   (n, 49, 59) reconstructed
    mask:    (n, 49)     bool True=masked
    """
    n = len(patches)
    wl = np.array(WAVELENGTHS)
    center = 49 // 2  # pixel 24 = center of 7×7 patch

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        orig = patches[i, center]       # (59,)
        rec  = recon[i, center]         # (59,)
        was_masked = mask[i, center]

        ax.plot(wl, orig, color='steelblue', lw=1.5, label='Original')
        ax.plot(wl, rec,  color='tomato',    lw=1.5, linestyle='--',
                label=f'Reconstructed (center {"masked" if was_masked else "visible"})')

        # Shade masked bands conceptually (mask is per-pixel, not per-band)
        status = 'MASKED' if was_masked else 'VISIBLE'
        ax.set_title(f'Example {i+1} — center pixel {status}', fontsize=11)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.legend(fontsize=9)
        ax.set_xlim(wl[0], wl[-1])

        # Annotate key absorption regions
        for xpos, label in [(1000, '1 μm\n(olivine/px)'), (2300, '2.3 μm\n(pyroxene)')]:
            ax.axvline(xpos, color='gray', lw=0.8, linestyle=':')
            ax.text(xpos + 15, ax.get_ylim()[1] * 0.92, label, fontsize=7, color='gray')

    fig.suptitle('Spatial MAE — Original vs. Reconstructed Center Pixel Spectra', fontsize=13)
    plt.savefig(out_path, dpi=150)
    print(f'Saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/spatial_mae_128d_6l_best.pt')
    parser.add_argument('--cache_dir', default='data/patch_cache')
    parser.add_argument('--n_examples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='reports/mae_reconstructions.png')
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    model = load_model(args.ckpt, device)
    print(f'Loaded checkpoint: {args.ckpt}')

    rng = np.random.default_rng(args.seed)
    raw = load_patches(args.cache_dir, args.n_examples, rng)  # (n, 7, 7, 59)
    print(f'Loaded {args.n_examples} patches from val cache')

    x = torch.from_numpy(raw).to(device)  # (n, 7, 7, 59)
    with torch.no_grad():
        loss, recon, mask = model(x)

    print(f'Val reconstruction loss (masked pixels only): {loss.item():.4f}')

    # Convert to numpy: (n, 49, 59)
    x_flat  = x.reshape(args.n_examples, 49, 59).cpu().numpy()
    recon_np = recon.cpu().numpy()
    mask_np  = mask.cpu().numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_reconstructions(x_flat, recon_np, mask_np, args.out)


if __name__ == '__main__':
    main()
