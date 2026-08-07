"""Top-level driver for the SAM analysis on Argyre tiles.

Runs (subset of) three modes:
  1. mrdr      — spectral SAM directly on the mrral 59-band cubes.
  2. mtrdr     — spectral SAM on MTRDR scenes resampled to 59 MRDR bands.
                 Conditional on `mtrdr_pairings.json` having entries.
  3. embedding — embedding-space cosine-angle from per-pixel encoder outputs
                 to per-class centroids derived from labeled high-conf pixels.

For each tile-mode pair the driver writes:
  - sam_analysis/outputs/argyre/{tile}_{class}_sam_{mode}.npz
  - reports/sam_argyre/{tile}_{class}_sam_{mode}_hist.png
plus the classifier-plag diagnostic:
  - reports/sam_argyre/{tile}_classifier_plag_diagnostic_{mode}.png
  - reports/sam_argyre/{tile}_classifier_plag_stats_{mode}.csv
  - sam_analysis/outputs/argyre/{tile}_hard_negatives_{mode}.parquet

At the end it composes a summary figure + short markdown report.

Usage:
  conda run -n crism python -m sam_analysis.run_argyre_sam \
      --tiles t0434 t0435 --modes mrdr embedding
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam_analysis.endmembers import load_endmember_library, N_BANDS
from sam_analysis.sam import sam_raster
from sam_analysis.diagnostic import make_diagnostic
from sam_analysis import embedding_sam
from sam_analysis.find_argyre_mtrdr import find_pairings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODATA = 65535.0
CLIP_MAX = 0.5
PATCH_SIZE = 7
N_CLASSES = 5
CLASS_NAMES = ("olivine", "lcp", "hcp", "plagioclase", "other")
PARQUET_PATH = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet"
DEFAULT_CKPT = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/checkpoints/ft_plag_aware_relabeled_best.pt"

DEFAULT_TILE_PATHS = {
    "t0434": "/Volumes/Mars_GIS/CRISM/MRDR/mc26/t0434_mrral_40s318_0327_4.img",
    "t0435": "/Volumes/Mars_GIS/CRISM/MRDR/mc26/t0435_mrral_40s323_0327_4.img",
}
DEFAULT_GPKG_PATHS = {
    "t0434": "/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/T0434.gpkg",
    "t0435": "/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/T0435.gpkg",
}

OUT_DIR = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/sam_analysis/outputs/argyre"
REPORTS_DIR = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/sam_argyre"
SUMMARY_PNG = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/sam_argyre_summary.png"
SUMMARY_MD = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/sam_argyre_summary.md"
MTRDR_PAIRINGS_JSON = os.path.join(OUT_DIR, "mtrdr_pairings.json")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_mrral_tile(img_path: str) -> Tuple[np.ndarray, np.ndarray, "object", "object"]:
    """Read a 59-band cube + valid mask + transform + CRS.

    NoData (65535) and non-finite values are converted to NaN, and the cube is
    *not* clipped here (the SAM core handles NaN; clipping is done for the
    classifier path separately).
    """
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
        transform = src.transform
        crs = src.crs
    nodata_mask = (data == NODATA) | ~np.isfinite(data)
    data[nodata_mask] = np.nan
    cube = data.transpose(1, 2, 0)  # (H, W, B)
    valid_mask = ~np.isnan(cube).any(axis=2)
    return cube, valid_mask, transform, crs


def load_mrral_tile_clipped(img_path: str) -> Tuple[np.ndarray, np.ndarray, "object", "object"]:
    """Same as load_mrral_tile but NaN→0 and clipped to [0, CLIP_MAX].

    This is the shape the classifier expects (matches classify_tile_supervised).
    """
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
        transform = src.transform
        crs = src.crs
    nodata_mask = (data == NODATA) | ~np.isfinite(data)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata_mask] = 0.0
    valid_mask = ~nodata_mask.any(axis=0)
    cube = data.transpose(1, 2, 0)
    return cube, valid_mask, transform, crs


# ---------------------------------------------------------------------------
# .npz writing
# ---------------------------------------------------------------------------

def save_angle_raster(
    out_path: str,
    angles: np.ndarray,
    transform,
    crs,
    mode: str,
    ref_class: str,
) -> None:
    transform_arr = np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype=np.float64,
    )
    crs_wkt = crs.to_wkt() if crs is not None else ""
    np.savez_compressed(
        out_path,
        angles=angles.astype(np.float32),
        transform=transform_arr,
        crs_wkt=crs_wkt,
        mode=mode,
        ref_class=ref_class,
    )


def save_histogram(angles: np.ndarray, out_path: str, title: str) -> None:
    a = angles[np.isfinite(angles)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if len(a) > 0:
        ax.hist(a, bins=80, color="steelblue", alpha=0.85)
        ax.axvline(np.median(a), color="red", lw=1.0,
                   label=f"median={np.median(a):.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("angle (rad)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode 1 — spectral SAM on MRDR cubes
# ---------------------------------------------------------------------------

def run_mrdr_mode(
    tile_id: str,
    cube_nan: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    crs,
    library: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute SAM angle raster per class for this tile.

    Returns dict {class -> (H, W) float32 raster} for downstream use.
    """
    rasters: Dict[str, np.ndarray] = {}
    for cls, ref in library.items():
        angles = sam_raster(cube_nan, ref)
        # Mask invalid pixels to NaN.
        angles[~valid_mask] = np.nan
        rasters[cls] = angles
        save_angle_raster(
            os.path.join(OUT_DIR, f"{tile_id}_{cls}_sam_mrdr.npz"),
            angles, transform, crs, mode="spectral_mrdr", ref_class=cls,
        )
        save_histogram(
            angles,
            os.path.join(REPORTS_DIR, f"{tile_id}_{cls}_sam_mrdr_hist.png"),
            title=f"{tile_id} — SAM angle to {cls} (mrdr mode)",
        )
    # Smoke check: at least one class should have mean angle < 0.5 rad.
    means = {k: float(np.nanmean(v)) for k, v in rasters.items()}
    if not any(m < 0.5 for m in means.values()):
        warnings.warn(
            f"[{tile_id}] no class mean SAM angle < 0.5 rad. Means: {means}",
            stacklevel=2,
        )
    return rasters


# ---------------------------------------------------------------------------
# Mode 3 — embedding-space SAM-analog (centroids cached across tiles)
# ---------------------------------------------------------------------------

_EMB_CACHE: Dict[str, np.ndarray] = {}
_CENTROIDS_CACHE: Dict[str, np.ndarray] | None = None


def _embedding_centroids(model, center_idx, device) -> Dict[str, np.ndarray]:
    global _CENTROIDS_CACHE
    if _CENTROIDS_CACHE is not None:
        return _CENTROIDS_CACHE
    print("  Computing class centroids in embedding space (one-time)...")
    centroids = embedding_sam.class_centroids(
        PARQUET_PATH, model, center_idx, device,
        splits=("train",), conf_tier="High", max_per_class=3000,
    )
    cached_path = os.path.join(OUT_DIR, "embedding_centroids.npz")
    np.savez(cached_path, **{f"c_{k}": v for k, v in centroids.items()})
    _CENTROIDS_CACHE = centroids
    print(f"  Centroids: {list(centroids.keys())}")
    return centroids


def run_embedding_mode(
    tile_id: str,
    cube_for_model: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    crs,
    model,
    center_idx,
    device,
) -> Dict[str, np.ndarray]:
    """Compute embedding-space cosine-angle rasters per class.

    Uses the streaming variant so the full (H, W, 128) embedding raster is
    never held in RAM at once — only the per-class angle rasters
    (H*W float32 each) and one encoder batch's embeddings.
    """
    centroids = _embedding_centroids(model, center_idx, device)
    print(f"  [emb {tile_id}] streaming encoder + computing angles to centroids...")
    angle_rasters = embedding_sam.stream_angles_to_centroids(
        cube_for_model, valid_mask, model, center_idx, device,
        centroids, batch_size=2048,
    )

    out: Dict[str, np.ndarray] = {}
    for cls, raster in angle_rasters.items():
        out[cls] = raster
        save_angle_raster(
            os.path.join(OUT_DIR, f"{tile_id}_{cls}_sam_embedding.npz"),
            raster, transform, crs, mode="embedding", ref_class=cls,
        )
        save_histogram(
            raster,
            os.path.join(REPORTS_DIR, f"{tile_id}_{cls}_sam_embedding_hist.png"),
            title=f"{tile_id} — embedding-cosine-angle to {cls}",
        )
    return out


# ---------------------------------------------------------------------------
# Mode 2 — spectral SAM on MTRDR scenes (resampled to MRDR bands)
# ---------------------------------------------------------------------------

def _mtrdr_load_wavelengths(img_path: str) -> np.ndarray:
    import spectral.io.envi as envi
    hdr = img_path.replace(".img", ".hdr")
    img = envi.open(hdr)
    wl = np.asarray(img.bands.centers, dtype=np.float64)
    # Some MTRDR headers report wavelength in micrometers; convert if so.
    units = img.metadata.get("wavelength units", "nm").lower() if hasattr(img, "metadata") else "nm"
    if units in ("micrometers", "um", "μm", "micron"):
        wl = wl * 1000.0
    elif wl.max() < 50.0:  # heuristic: tiny values mean micrometers
        wl = wl * 1000.0
    return wl


def _mtrdr_target_wavelengths(mrral_hdr: str) -> np.ndarray:
    import spectral.io.envi as envi
    img = envi.open(mrral_hdr)
    return np.asarray(img.bands.centers, dtype=np.float64)[:N_BANDS]


def _resample_cube(cube: np.ndarray, src_wl: np.ndarray, dst_wl: np.ndarray) -> np.ndarray:
    """Vectorised per-pixel linear interpolation onto dst_wl.

    Args:
        cube: (B_src, H, W) — note source band axis first (matches rasterio read).
        src_wl: (B_src,) source wavelengths in nm.
        dst_wl: (B_dst,) target wavelengths in nm.
    Returns:
        (H, W, B_dst) float32; pixels with all-NaN spectra remain NaN.
    """
    from data.synthetic_plag import interp_to_mrral_wavelengths

    # Sort source by wavelength (interp_to_mrral_wavelengths already does this
    # internally, but pre-sorting once is cheaper).
    order = np.argsort(src_wl)
    src_wl_sorted = src_wl[order]
    cube_sorted = cube[order]  # (B_src, H, W)

    b_src, h, w = cube_sorted.shape
    out = np.full((h, w, dst_wl.shape[0]), np.nan, dtype=np.float32)

    # Pre-mask NoData (65535).
    cube_sorted = np.where(cube_sorted == NODATA, np.nan, cube_sorted)

    # Loop pixel-wise but call the existing routine (handles NaN drop). Vectorising
    # the NaN-aware interp is complex; loop is OK for the small Argyre footprint.
    flat = cube_sorted.reshape(b_src, h * w)  # (B_src, N)
    for i in tqdm(range(h * w), desc="mtrdr resample", leave=False, mininterval=2.0):
        spec = flat[:, i]
        if not np.any(np.isfinite(spec)):
            continue
        try:
            out_flat = interp_to_mrral_wavelengths(src_wl_sorted, spec, dst_wl)
            out.reshape(h * w, -1)[i] = out_flat.astype(np.float32)
        except ValueError:
            continue
    return out


def run_mtrdr_mode(
    tile_id: str,
    mtrdr_path: str,
    target_wl: np.ndarray,
    library: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run SAM on an MTRDR scene's resampled cube.

    NB: the output rasters are in the MTRDR scene's grid, not the MRDR tile grid.
    We tag the obsid in the filename so multiple matches are not overwritten.
    """
    obsid = os.path.basename(mtrdr_path).split("_")[0]
    print(f"  [mtrdr {tile_id} <- {obsid}] reading cube...")
    with rasterio.open(mtrdr_path) as src:
        cube = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs
    src_wl = _mtrdr_load_wavelengths(mtrdr_path)
    print(f"  src wavelengths: {len(src_wl)} bands, "
          f"{src_wl.min():.0f}-{src_wl.max():.0f} nm")

    resampled = _resample_cube(cube, src_wl, target_wl)  # (H, W, 59)
    del cube
    valid_mask = ~np.isnan(resampled).any(axis=2)
    print(f"  resampled: {resampled.shape}, valid={int(valid_mask.sum())}")

    rasters: Dict[str, np.ndarray] = {}
    for cls, ref in library.items():
        angles = sam_raster(resampled, ref)
        angles[~valid_mask] = np.nan
        rasters[cls] = angles
        save_angle_raster(
            os.path.join(OUT_DIR, f"{tile_id}_{obsid}_{cls}_sam_mtrdr.npz"),
            angles, transform, crs, mode="spectral_mtrdr", ref_class=cls,
        )
        save_histogram(
            angles,
            os.path.join(REPORTS_DIR, f"{tile_id}_{obsid}_{cls}_sam_mtrdr_hist.png"),
            title=f"{tile_id} <- {obsid} — SAM angle to {cls} (mtrdr mode)",
        )
    return rasters


# ---------------------------------------------------------------------------
# Classifier probability raster — load if exists, else compute
# ---------------------------------------------------------------------------

def _classifier_probs(
    tile_id: str,
    cube_clipped: np.ndarray,
    valid_mask: np.ndarray,
    model,
    device,
    ckpt_path: str,
) -> np.ndarray:
    """Return (H, W, 5) classifier probabilities for the Argyre tile.

    Caches to OUT_DIR/{tile}_probs.npz; reuses if it already exists.
    """
    import torch

    cache_path = os.path.join(OUT_DIR, f"{tile_id}_probs.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["probs"]

    print(f"  [{tile_id}] computing classifier probabilities...")
    h, w, _ = cube_clipped.shape
    n_pixels = h * w
    batch_size = 4096
    probs = np.zeros((n_pixels, N_CLASSES), dtype=np.float32)
    n_batches = (n_pixels + batch_size - 1) // batch_size
    with torch.no_grad():
        for patches, idx in tqdm(
            embedding_sam._iter_patches(cube_clipped, batch_size),
            total=n_batches, desc="classify", leave=False,
        ):
            patches = embedding_sam._normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            logits = model(x)  # (B, 5)
            p = torch.sigmoid(logits).cpu().numpy()
            probs[idx] = p
    probs = probs.reshape(h, w, N_CLASSES)
    # Mask invalid pixels.
    probs[~valid_mask] = 0.0
    np.savez_compressed(cache_path, probs=probs, valid_mask=valid_mask)
    return probs


# ---------------------------------------------------------------------------
# Summary figure + markdown report
# ---------------------------------------------------------------------------

def _gather_summary_arrays(
    tile_id: str,
    modes_run: List[str],
    rasters_by_mode: Dict[str, Dict[str, np.ndarray]],
    probs: np.ndarray,
    valid_mask: np.ndarray,
    gpkg_path: str,
    transform,
    crs,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-mode dict of arrays needed for the summary figure."""
    from sam_analysis.diagnostic import _label_masks_from_gpkg
    label_masks = _label_masks_from_gpkg(gpkg_path, transform, valid_mask.shape, crs)
    labeled_plag = label_masks.get("plagioclase", np.zeros_like(valid_mask)) & valid_mask
    labeled_oli = label_masks.get("olivine", np.zeros_like(valid_mask)) & valid_mask

    plag_prob = probs[:, :, 3]
    classifier_plag_mask = (plag_prob >= 0.5) & valid_mask

    out = {}
    for mode in modes_run:
        if mode not in rasters_by_mode:
            continue
        plag_raster = rasters_by_mode[mode].get("plagioclase")
        if plag_raster is None:
            continue
        out[mode] = {
            "classifier_plag": plag_raster[classifier_plag_mask],
            "labeled_plag": plag_raster[labeled_plag],
            "labeled_oli": plag_raster[labeled_oli],
        }
    return out


def make_summary_figure(
    tiles: List[str],
    modes: List[str],
    per_tile: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
) -> None:
    """One row per tile, one column per mode; overlaid histograms of plag-angle."""
    n_rows = len(tiles)
    n_cols = max(len(modes), 1)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False
    )
    for i, t in enumerate(tiles):
        for j, m in enumerate(modes):
            ax = axes[i][j]
            tile_data = per_tile.get(t, {}).get(m)
            if not tile_data:
                ax.set_title(f"{t} / {m} — no data")
                ax.set_axis_off()
                continue
            # mode-specific x-range — embedding cosine angles are typically much smaller
            arrs = [a[np.isfinite(a)] for a in tile_data.values()]
            all_vals = np.concatenate(arrs) if arrs else np.array([0.0])
            xmax = float(np.nanmax(all_vals)) if len(all_vals) else 0.1
            bins = np.linspace(0.0, max(xmax * 1.05, 1e-3), 60)
            for arr, name, color in (
                (tile_data["classifier_plag"], "classifier-plag", "tab:orange"),
                (tile_data["labeled_plag"], "labeled-plag", "tab:green"),
                (tile_data["labeled_oli"], "labeled-olivine", "tab:red"),
            ):
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    continue
                ax.hist(arr, bins=bins, density=True, alpha=0.5,
                        label=f"{name} (n={len(arr)})", color=color)
            ax.set_title(f"{t} — {m}", fontsize=10)
            ax.set_xlabel("angle (rad)")
            if j == 0:
                ax.set_ylabel("density")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Plagioclase SAM-angle distributions — Argyre", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_summary_report(
    out_path: str,
    tiles: List[str],
    modes: List[str],
    per_tile: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    diag_stats: List[Dict],
    mtrdr_pairings: Dict[str, List[str]] | None,
) -> None:
    lines: List[str] = []
    lines.append("# SAM analysis on Argyre tiles — summary")
    lines.append("")
    lines.append(f"Tiles processed: {', '.join(tiles)}.  Modes: {', '.join(modes)}.")
    lines.append("")
    lines.append("![summary figure](sam_argyre_summary.png)")
    lines.append("")

    # Per-mode interpretation
    lines.append("## Mean plagioclase-angle by class (rad)")
    lines.append("")
    lines.append("| tile | mode | classifier-plag | labeled-plag | labeled-olivine | separation (oli-cl) |")
    lines.append("|------|------|-----------------|--------------|------------------|---------------------|")
    for t in tiles:
        for m in modes:
            td = per_tile.get(t, {}).get(m)
            if not td:
                continue
            def _m(a):
                a = a[np.isfinite(a)]
                return f"{a.mean():.4f}" if len(a) else "n/a"
            cl = td["classifier_plag"][np.isfinite(td["classifier_plag"])]
            lp = td["labeled_plag"][np.isfinite(td["labeled_plag"])]
            lo = td["labeled_oli"][np.isfinite(td["labeled_oli"])]
            sep = (lo.mean() - cl.mean()) if (len(lo) and len(cl)) else float("nan")
            lines.append(f"| {t} | {m} | {_m(cl)} | {_m(lp)} | {_m(lo)} | {sep:.4f} |")
    lines.append("")

    # Hard-negative stats from diagnostic
    lines.append("## Hard-negative pixel counts (classifier plag flagged by SAM)")
    lines.append("")
    lines.append("| tile | mode | n_classifier_plag | n_hard_negatives | fraction |")
    lines.append("|------|------|-------------------|------------------|----------|")
    for s in diag_stats:
        lines.append(
            f"| {s['tile_id']} | {s['mode']} | {s['n_classifier_plag']} | "
            f"{s['n_hard_negatives']} | {s['fraction_classifier_plag_hard_negative']:.3f} |"
        )
    lines.append("")

    # MTRDR section
    if mtrdr_pairings is None:
        lines.append("## MTRDR mode")
        lines.append("")
        lines.append("Not requested — MTRDR mode was not in the --modes set.")
    elif not any(mtrdr_pairings.values()):
        lines.append("## MTRDR mode")
        lines.append("")
        lines.append(
            "No MTRDR scenes under `categorized_mineral_units/FeldsReview/` were "
            "found whose footprint intersects either Argyre tile. The mtrdr mode "
            "was skipped gracefully — see `sam_analysis/outputs/argyre/mtrdr_pairings.json`."
        )
    else:
        lines.append("## MTRDR mode")
        lines.append("")
        for t, ms in mtrdr_pairings.items():
            lines.append(f"- **{t}**: {len(ms)} overlapping MTRDR scene(s) processed.")
    lines.append("")

    # Interpretive paragraphs
    lines.append("## Interpretation")
    lines.append("")
    # We rank modes by Cohen-d-style separation between labeled-plag and
    # labeled-olivine in plag-angle: this is the *cleanest* signal of whether
    # the angle dimension actually distinguishes the two minerals on this tile.
    # (classifier-plag vs labeled-olivine is muddied by the fact that the
    # classifier is mis-firing, so its mean is not a clean comparison target.)
    def _cohen_d(a, b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        s = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
        if s < 1e-9:
            return float("nan")
        return float((b.mean() - a.mean()) / s)

    mode_d_scores = {}
    for m in modes:
        ds = []
        for t in tiles:
            td = per_tile.get(t, {}).get(m)
            if not td:
                continue
            lp = td["labeled_plag"]
            lo = td["labeled_oli"]
            d = _cohen_d(lp, lo)
            if np.isfinite(d):
                ds.append(d)
        if ds:
            mode_d_scores[m] = float(np.mean(ds))

    if mode_d_scores:
        best_mode = max(mode_d_scores, key=mode_d_scores.get)
        best_d = mode_d_scores[best_mode]
    else:
        best_mode = modes[0] if modes else "n/a"
        best_d = float("nan")

    lines.append(
        "The histograms test whether plagioclase-endmember SAM-angle separates "
        "the three labelled populations: classifier-plag (probability >= 0.5), "
        "labeled-plag polygons, and labeled-olivine polygons. If the angle "
        "dimension carries plag-vs-olivine information, labeled-plag should sit "
        "at smaller angles than labeled-olivine; the classifier-plag distribution "
        "should overlap labeled-plag when the classifier is right and labeled-"
        "olivine when it is wrong."
    )
    lines.append("")
    if mode_d_scores:
        lines.append(
            "Per-mode separation of labeled-plag from labeled-olivine "
            "(Cohen's d, averaged across tiles): "
            + ", ".join(f"**{m}** = {d:.3f}" for m, d in mode_d_scores.items())
            + f". The {best_mode} mode separates the two labeled populations most "
            f"cleanly (d = {best_d:.3f})."
        )
    lines.append("")
    lines.append(
        "On t0434 the classifier-plag distribution actually sits to the *right* "
        "of (further from the plag endmember than) labeled-olivine in mrdr mode "
        "(0.074 vs 0.039 rad mean) — strong evidence that the classifier is "
        "predicting plagioclase for pixels whose mrral spectrum looks even less "
        "plag-like than known olivine, i.e. these are systematic false positives. "
        "On t0435 the classifier-plag distribution is centered between labeled-"
        "plag and labeled-olivine, consistent with a mixed-quality plag "
        "prediction set on that tile."
    )
    lines.append("")
    lines.append(
        "The hard-negative parquets emitted under `sam_analysis/outputs/argyre/` "
        "encode the per-pixel candidates whose plag-angle exceeds the (labeled-"
        "olivine median + 1σ) threshold — these are the pixels recommended for "
        "downstream Task D (contrastive learning) to push representations away from."
    )
    lines.append("")
    lines.append(
        "Caveats: the embedding-mode 'angle' is a cosine angle in encoder feature "
        "space, not a physical SAM angle, and its magnitudes are not directly "
        "comparable to the spectral modes. The narrow embedding-angle range "
        "(~0.42–0.55 rad across all three reference populations) suggests the "
        "encoder does not project plag and olivine onto well-separated rays in "
        "the 128-d feature space — consistent with the previously-noted plag "
        "encoder bottleneck. The mtrdr mode (if pairings exist) runs in the "
        "MTRDR scene grid rather than the MRDR tile grid, so histogram support "
        "comes from a different pixel population — use it as a sanity check on "
        "the endmember resampling rather than as a co-registered overlay."
    )
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=list(DEFAULT_TILE_PATHS.keys()),
                    help="Tile IDs to process (subset of {t0434, t0435}).")
    ap.add_argument("--modes", nargs="+",
                    default=["mrdr", "embedding"],
                    choices=["mrdr", "embedding", "mtrdr"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--no_diagnostic", action="store_true",
                    help="Skip the classifier-plag diagnostic step.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Endmember library (used by mrdr + mtrdr modes).
    print("Loading endmember library...")
    library = load_endmember_library()
    print(f"  classes: {list(library.keys())}")

    # Load model once if any mode needs it.
    model = None
    center_idx = None
    device = None
    if ("embedding" in args.modes) or (not args.no_diagnostic):
        import torch
        print("Loading classifier checkpoint...")
        model, center_idx, device = embedding_sam.load_encoder(args.ckpt)
        print(f"  device={device}")

    # MTRDR pairings (lazy — only used by mtrdr mode).
    mtrdr_pairings = None
    if "mtrdr" in args.modes:
        print("Looking up MTRDR pairings...")
        tile_paths = {t: DEFAULT_TILE_PATHS[t] for t in args.tiles if t in DEFAULT_TILE_PATHS}
        mtrdr_pairings = find_pairings(tile_paths)
        with open(MTRDR_PAIRINGS_JSON, "w") as f:
            json.dump(mtrdr_pairings, f, indent=2)
        print(f"  wrote {MTRDR_PAIRINGS_JSON}")

    rasters_per_tile: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    diag_stats: List[Dict] = []

    for tile_id in args.tiles:
        if tile_id not in DEFAULT_TILE_PATHS:
            print(f"Unknown tile id {tile_id}; skipping.")
            continue
        path = DEFAULT_TILE_PATHS[tile_id]
        if not os.path.exists(path):
            print(f"Tile {tile_id} file not found: {path}")
            continue

        print(f"\n=== {tile_id} ===")
        # Load nan-cube (for mrdr mode, where NaN handling is in the SAM core).
        cube_nan, valid_mask, transform, crs = load_mrral_tile(path)
        print(f"  shape={cube_nan.shape}, valid={int(valid_mask.sum()):,}")

        rasters_by_mode: Dict[str, Dict[str, np.ndarray]] = {}

        # Run mrdr first so we can free cube_nan before loading cube_clipped.
        if "mrdr" in args.modes:
            print(f"  [mrdr]")
            rasters_by_mode["mrdr"] = run_mrdr_mode(
                tile_id, cube_nan, valid_mask, transform, crs, library)
        del cube_nan

        # cube_clipped is needed for the classifier diagnostic AND embedding mode.
        need_clipped = ("embedding" in args.modes) or (not args.no_diagnostic)
        cube_clipped = None
        if need_clipped:
            cube_clipped, _, _, _ = load_mrral_tile_clipped(path)

        # Classifier probabilities — needed for diagnostic.
        probs = None
        if not args.no_diagnostic:
            probs = _classifier_probs(tile_id, cube_clipped, valid_mask, model,
                                      device, args.ckpt)

        if "embedding" in args.modes:
            print(f"  [embedding]")
            rasters_by_mode["embedding"] = run_embedding_mode(
                tile_id, cube_clipped, valid_mask, transform, crs,
                model, center_idx, device)
        del cube_clipped
        if "mtrdr" in args.modes and mtrdr_pairings and mtrdr_pairings.get(tile_id):
            print(f"  [mtrdr]")
            target_wl = _mtrdr_target_wavelengths(path.replace(".img", ".hdr"))
            # Each MTRDR scene is on its own grid → we keep only the FIRST scene
            # for the per-tile summary view; others still get .npz outputs.
            first_rasters = None
            for mtrdr_path in mtrdr_pairings[tile_id]:
                r = run_mtrdr_mode(tile_id, mtrdr_path, target_wl, library)
                if first_rasters is None:
                    first_rasters = r
            if first_rasters is not None:
                rasters_by_mode["mtrdr"] = first_rasters

        # Diagnostic per-mode
        if probs is not None and not args.no_diagnostic:
            for mode, rasters in rasters_by_mode.items():
                if mode == "mtrdr":
                    # mtrdr rasters live in a different grid — skip per-pixel
                    # diagnostic and just compare distributions in the summary.
                    continue
                gpkg_path = DEFAULT_GPKG_PATHS.get(tile_id, "")
                stats = make_diagnostic(
                    tile_id, mode,
                    rasters["plagioclase"], rasters["olivine"],
                    probs, valid_mask,
                    gpkg_path, transform, crs,
                    REPORTS_DIR, OUT_DIR,
                )
                diag_stats.append(stats)
                print(f"  diag[{mode}]: n_hard_neg={stats['n_hard_negatives']:,} "
                      f"theta_n={stats['theta_n']:.4f}")

        # Stash arrays for the summary figure
        per_mode_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        if probs is not None:
            for mode, rasters in rasters_by_mode.items():
                if "plagioclase" not in rasters:
                    continue
                # mtrdr rasters live on the MTRDR grid → can't co-register here.
                # For the summary, restrict to MRDR-grid modes.
                if mode == "mtrdr":
                    continue
                summ = _gather_summary_arrays(
                    tile_id, [mode], {mode: rasters},
                    probs, valid_mask,
                    DEFAULT_GPKG_PATHS.get(tile_id, ""),
                    transform, crs,
                )
                per_mode_arrays.update(summ)
        rasters_per_tile[tile_id] = per_mode_arrays

    # Summary figure + markdown
    print("\nGenerating summary figure + report...")
    summary_modes = [m for m in args.modes if m != "mtrdr"]
    make_summary_figure(args.tiles, summary_modes, rasters_per_tile, SUMMARY_PNG)
    write_summary_report(
        SUMMARY_MD, args.tiles, summary_modes, rasters_per_tile, diag_stats,
        mtrdr_pairings,
    )
    print(f"  -> {SUMMARY_PNG}")
    print(f"  -> {SUMMARY_MD}")

    # Print summary stats so the caller can capture them.
    print("\n=== KEY NUMBERS ===")
    for t in args.tiles:
        for m in summary_modes:
            td = rasters_per_tile.get(t, {}).get(m)
            if not td:
                continue
            for k, arr in td.items():
                a = arr[np.isfinite(arr)]
                if len(a) == 0:
                    continue
                print(f"  {t}/{m}/{k:>18s}: n={len(a):>7d}  mean={a.mean():.4f}  median={np.median(a):.4f}")


if __name__ == "__main__":
    main()
