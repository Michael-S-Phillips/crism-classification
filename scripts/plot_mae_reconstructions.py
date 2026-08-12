"""Plot MAE reconstructions of held-out spectra, per mineral class.

What this answers: the dual-CR MAE sees 25% of a patch's 49 spatial positions and
must reconstruct the rest. Does its reconstruction preserve the DIAGNOSTIC SHAPE
of each mineral class -- and specifically, does the linear-CR block carry
alteration's broad 1-2 um arch, which is the entire premise of the 118-channel
representation? Upper-hull CR retains only 41% of that arch (a broad convex arch
approximately IS the hull, so hull-CR divides it out), while linear CR cannot
remove curvature at all because a line has none.

Reconstructions are plotted ONLY at MASKED positions. A reconstruction at a
visible position is not a test of anything -- the model was shown the answer.

Both channel blocks are un-standardised back to physical ratio units before
plotting (each block is divided by its own global std at cache-build time, which
puts hull near 1/0.0705 = 14.2 and linear near 1/0.172 = 5.8 -- readable as
neither reflectance nor a band-depth ratio). Blocks are shown in SEPARATE panels,
never twinned on one pair of axes: they are different transforms on different
scales, and overlaying them on a shared y-axis would invite exactly the
apples-to-oranges reading this figure exists to avoid.

Usage
    python scripts/plot_mae_reconstructions.py \
        --ckpt checkpoints/spatial_mae_dualcr_denoising_256d_6l_best.pt \
        --parquet data/mrral_pixels.parquet \
        --out reports/mae_dualcr_reconstructions.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import CR_SCALES, WAVELENGTHS_59, N_BANDS  # noqa: E402

# Validated with dataviz/scripts/validate_palette.js:
#   "#2a78d6,#eb6834" --mode light --pairs all -> ALL CHECKS PASS
#   CVD separation worst all-pairs dE 24.7 (protan), normal-vision 33.6,
#   both well clear of the 8 / 15 floors; contrast >= 3:1 on #fcfcfb.
# Line style (solid vs dashed) is a deliberate second encoding on top of hue, so
# the pair survives greyscale printing and any CVD.
C_TRUE, C_RECON = '#2a78d6', '#eb6834'
SURFACE, INK, INK_2 = '#fcfcfb', '#0b0b0b', '#52514e'
GRID = '#e4e3df'
# The trivial baseline is a REFERENCE, not a peer series, so it wears a muted ink
# token rather than a third categorical hue -- a third hue would imply three
# comparable things being measured.
C_BASE = '#9a9890'

# The 1-2 um window the design argument turns on (arch measured at 1625 nm
# against the 984-2205 nm chord).
ARCH_LO, ARCH_HI = 984.0, 2205.0


def build_mrral_map(data_root: str) -> dict[str, str]:
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
    return {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def load_mae(ckpt_path: str, device):
    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    n_bands = cfg['n_bands']
    if n_bands != 2 * N_BANDS:
        raise SystemExit(
            f'{ckpt_path} is a {n_bands}-band MAE; this figure is for the '
            f'118-channel dual-CR representation. Nothing here would fail on a '
            f'59-band checkpoint -- it would just silently plot one block.')
    mae = DenoisingSpatialSpectralMAE(
        n_bands=n_bands, patch_size=7,
        embed_dim=cfg['embed_dim'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], decoder_dim=cfg['decoder_dim'],
        decoder_layers=cfg['decoder_layers'], mask_ratio=cfg['mask_ratio'],
        n_channel_blocks=cfg.get('n_channel_blocks', 1),
        sigma_gauss=cfg['sigma_gauss'], sigma_spike=cfg['sigma_spike'],
        sigma_column=cfg['sigma_column'],
    )
    missing, unexpected = mae.load_state_dict(ck['mae_state'], strict=False)
    if missing or unexpected:
        raise SystemExit(f'state_dict mismatch: missing={missing[:4]} '
                         f'unexpected={unexpected[:4]}')
    # eval() makes CrismNoiseAugmentation a no-op by design, so this measures
    # reconstruction from a CLEAN input. The model was trained denoising; adding
    # corruption here would conflate two questions.
    return mae.to(device).eval(), ck, cfg


def pick_rows(parquet: str, classes: list[str], per_class: int, seed: int):
    """One confidently single-class pixel set per class, from distinct tiles."""
    from data.dataset import _collapse_labels
    df = _collapse_labels(pd.read_parquet(parquet))
    present = [c for c in classes if c in df.columns]
    rng = np.random.default_rng(seed)
    out = {}
    for cls in present:
        others = [c for c in present if c != cls]
        # Single-label only: a dual-labelled pixel would make "does the
        # reconstruction look like class X" unanswerable.
        m = (df[cls] > 0)
        if others:
            m &= (df[others].to_numpy() == 0).all(axis=1)
        sub = df[m]
        if not len(sub):
            continue
        # Spread across tiles so one anomalous scene cannot stand for the class.
        sub = sub.groupby('tile_id', group_keys=False).head(max(1, per_class // 3))
        take = min(per_class, len(sub))
        out[cls] = sub.iloc[rng.choice(len(sub), take, replace=False)].copy()
    return out


def reconstruct(mae, patches: np.ndarray, device, seed: int):
    """Return (true, recon, mask) as (B, 49, 118) / (B, 49, 118) / (B, 49) bool."""
    x = torch.from_numpy(patches.astype(np.float32)).to(device)
    torch.manual_seed(seed)          # the mask is random; make the figure reproducible
    with torch.no_grad():
        _loss, recon, mask = mae(x)
    B = x.shape[0]
    return (x.reshape(B, 49, -1).cpu().numpy(),
            recon.cpu().numpy(),
            mask.cpu().numpy().astype(bool))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--parquet', default='data/mrral_pixels.parquet')
    ap.add_argument('--out', default='reports/mae_dualcr_reconstructions.png')
    ap.add_argument('--classes', nargs='+',
                    default=['alteration', 'plagioclase', 'lcp', 'hcp', 'other'])
    ap.add_argument('--per_class', type=int, default=24,
                    help='patches per class; the median over their masked '
                         'positions is plotted, so this smooths one-pixel luck')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_root', default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from config_loader import load_config
    from data.dataset import CRISMSpectralPatchDataset
    from device import get_device

    cfg = load_config()
    root = args.data_root or cfg['data_root']
    device = get_device()
    mae, ck, mcfg = load_mae(args.ckpt, device)
    print(f'MAE  epoch {ck["epoch"]}  mae_loss {ck["mae_loss"]:.5f}  '
          f'n_bands {mcfg["n_bands"]}  blocks {mcfg.get("n_channel_blocks")}  '
          f'mask_ratio {mcfg["mask_ratio"]}  device {device}')

    mrral_map = build_mrral_map(root)
    picks = pick_rows(args.parquet, args.classes, args.per_class, args.seed)
    if not picks:
        raise SystemExit(f'no single-label rows for any of {args.classes}')

    wl = np.asarray(WAVELENGTHS_59, dtype=float)
    scale = {'hull': CR_SCALES['hull_std'], 'linear': CR_SCALES['linear_std']}
    rows = list(picks)
    # Three panels, not two. The full-range linear-CR panel is dominated by the
    # 410 nm blue edge (linear-CR there is ~0.2 against ~1.0 across the rest),
    # and that single artifact band sets the y-scale, compressing the 1-2 um arch
    # -- the whole reason this representation exists -- into a few pixels. The
    # third panel is the same linear-CR series restricted to the arch window with
    # its own scale. Separate panels rather than a second y-axis on the same
    # plot: a twinned axis is the one chart form that reliably misleads.
    COLS = (('hull', 'hull-CR', None),
            ('linear', 'linear-CR', None),
            ('linear', 'linear-CR · 1–2 µm zoom', (ARCH_LO, ARCH_HI)))
    fig, axes = plt.subplots(len(rows), 3, figsize=(16.5, 2.5 * len(rows)),
                             squeeze=False, facecolor=SURFACE)
    stats = []

    for r, cls in enumerate(rows):
        sub = picks[cls]
        # Same reader the training run uses -- identical clipping, nodata
        # handling, padding and dual_continuum call. Re-implementing patch
        # extraction here would risk showing a representation the model was
        # never trained on.
        ds = CRISMSpectralPatchDataset(
            sub, mrral_map, patch_size=7,
            continuum_removed=True, dual_cr=True)
        patches = np.stack([ds[i][0].numpy() for i in range(len(ds))])
        true, recon, mask = reconstruct(mae, patches, device, args.seed)

        for c, (block, title, xlim) in enumerate(COLS):
            sl = slice(0, N_BANDS) if block == 'hull' else slice(N_BANDS, 2 * N_BANDS)
            # Masked positions only, pooled across the class's patches.
            t_all = true[:, :, sl]
            # The honest yardstick: predict every masked position as the mean of
            # the VISIBLE positions in the SAME patch. A 7x7 CRISM window spans
            # ~1 km and adjacent pixels are near-duplicates, so this trivial rule
            # is already strong -- without it, "reconstruction tracks actual" is
            # a statement about CRISM's spatial autocorrelation, not the model.
            nb = np.zeros_like(t_all)
            for b in range(t_all.shape[0]):
                vis = t_all[b][~mask[b]]
                nb[b, :, :] = vis.mean(axis=0) if len(vis) else 0.0

            t = t_all[mask] * scale[block]
            p = recon[:, :, sl][mask] * scale[block]
            n = nb[mask] * scale[block]
            rmse = float(np.sqrt(np.mean((t - p) ** 2)))
            rmse_nb = float(np.sqrt(np.mean((t - n) ** 2)))
            if xlim is None:                      # don't double-count the zoom
                stats.append((cls, block, len(t), rmse, rmse_nb))

            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            if xlim is None:
                ax.axvspan(ARCH_LO, ARCH_HI, color='#000000', alpha=0.035, lw=0)
            else:
                ax.set_xlim(*xlim)
                inw = (wl >= xlim[0]) & (wl <= xlim[1])
                lo = min(np.percentile(t[:, inw], 1), np.percentile(p[:, inw], 1))
                hi = max(np.percentile(t[:, inw], 99), np.percentile(p[:, inw], 99))
                pad = 0.12 * (hi - lo) or 0.01
                ax.set_ylim(lo - pad, hi + pad)
            # Median over masked positions, with the interquartile band: one
            # spectrum would be anecdote, and the band shows whether the
            # reconstruction tracks the spread or just the centre.
            ax.plot(wl, np.median(n, axis=0), color=C_BASE, lw=1.4, ls=':',
                    label='baseline: mean of visible', zorder=2)
            for arr, colr, lbl, ls in ((t, C_TRUE, 'actual', '-'),
                                       (p, C_RECON, 'MAE reconstruction', '--')):
                med = np.median(arr, axis=0)
                ax.plot(wl, med, color=colr, lw=2.0, ls=ls, label=lbl, zorder=3)
                ax.fill_between(wl, np.percentile(arr, 25, axis=0),
                                np.percentile(arr, 75, axis=0),
                                color=colr, alpha=0.13, lw=0, zorder=1)
            ax.grid(True, color=GRID, lw=0.6, zorder=0)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=8)
            if c == 0:
                ax.set_ylabel(f'{cls}\n({len(sub)} patches)', color=INK,
                              fontsize=9.5, fontweight='medium')
            if r == 0:
                ax.set_title(f'{title}   ch {sl.start}–{sl.stop - 1}',
                             color=INK, fontsize=10.5, fontweight='medium', pad=8)
            if r == len(rows) - 1:
                ax.set_xlabel('wavelength (nm)', color=INK_2, fontsize=9)
            # Bottom-LEFT: the legend sits bottom-right of the top row, and at
            # bottom-right these two overlapped into unreadable mush.
            ax.text(0.035, 0.035,
                    f'RMSE {rmse:.4f}   vs baseline {rmse_nb:.4f}'
                    f'  ({rmse_nb / rmse:.2f}×)',
                    transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=8, color=INK_2,
                    bbox=dict(facecolor=SURFACE, edgecolor='none',
                              boxstyle='square,pad=0.25'))

    # One legend for the figure -- 2 series, so identity is never colour-alone
    # (line style also differs) and a per-panel legend would be 15x redundant.
    h, l = axes[0][0].get_legend_handles_labels()
    # Figure-level, in the margin: inside any panel it sat on top of the data.
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.947), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(
        'Dual-CR MAE: reconstruction at MASKED positions, by mineral class',
        color=INK, fontsize=13, fontweight='semibold', y=0.998)
    fig.text(0.5, 0.974,
             f'{os.path.basename(args.ckpt)}  ·  epoch {ck["epoch"]}  ·  '
             f'{int(mcfg["mask_ratio"] * 100)}% of 49 positions hidden  ·  '
             f'shaded band = IQR  ·  grey column = 1–2 µm arch window',
             ha='center', color=INK_2, fontsize=8.5)
    fig.text(0.5, 0.958,
             'Adjacent CRISM pixels are near-duplicates, so most of the apparent '
             'fidelity is spatial autocorrelation: read the ×  ratio, not the curves. '
             'The MAE beats the trivial baseline consistently but modestly.',
             ha='center', color=INK_2, fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0, 1, 0.930])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f'wrote {args.out}')

    print(f'\n{"class":<14}{"block":<9}{"masked":>9}{"MAE":>9}{"baseline":>10}{"gain":>8}')
    print('-' * 59)
    for cls, block, n, rmse, rmse_nb in stats:
        print(f'{cls:<14}{block:<9}{n:>9,}{rmse:>9.4f}{rmse_nb:>10.4f}'
              f'{rmse_nb / rmse:>7.2f}x')
    for block in ('hull', 'linear'):
        v = [(r, b) for _c, bl, _n, r, b in stats if bl == block]
        m, mb = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print(f'{"MEAN":<14}{block:<9}{"":>9}{m:>9.4f}{mb:>10.4f}{mb / m:>7.2f}x')


if __name__ == '__main__':
    main()
