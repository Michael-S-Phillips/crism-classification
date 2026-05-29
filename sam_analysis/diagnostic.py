"""Classifier-plag-in-SAM diagnostic.

For each Argyre tile and each SAM mode, compare:
  (a) histogram of plag-endmember angle for classifier-plag pixels (prob >= 0.5)
  (b) labeled-plag pixels in the tile  (from GeoPackage)
  (c) labeled-olivine pixels in the tile

Outputs:
  - reports/sam_argyre/{tile}_classifier_plag_diagnostic_{mode}.png
  - reports/sam_argyre/{tile}_classifier_plag_stats_{mode}.csv
  - sam_analysis/outputs/argyre/{tile}_hard_negatives_{mode}.parquet
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_NAMES = ("olivine", "lcp", "hcp", "plagioclase", "other")
PLAG_IDX = 3
OLIVINE_IDX = 0


def _label_masks_from_gpkg(
    gpkg_path: str, transform, shape: Tuple[int, int], crs
) -> Dict[str, np.ndarray]:
    """Rasterize the labeled GeoPackage polygons into boolean (H, W) masks.

    Returns dict with keys 'plagioclase' and 'olivine'. Both masks include
    high+moderate confidence polygons (Low excluded) and unions any 'X+Y'
    mixed polygons that include the target mineral.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    if not os.path.exists(gpkg_path):
        return {}

    gdf = gpd.read_file(gpkg_path)
    if gdf.crs != crs:
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            # Some Mars CRS conversions are flaky; we'll skip if it fails.
            return {}

    masks: Dict[str, np.ndarray] = {}
    for target in ("plagioclase", "olivine"):
        cat = gdf["Category"].astype(str).str.lower()
        # Exclude 'Low' confidence.
        # Match the target name (so 'olivine + plagioclase' counts for both).
        is_target = cat.str.contains(target) & ~cat.str.contains(r"\(low\)", regex=True)
        sub = gdf[is_target]
        if len(sub) == 0:
            masks[target] = np.zeros(shape, dtype=bool)
            continue
        shapes = [(g, 1) for g in sub.geometry if g is not None and not g.is_empty]
        if not shapes:
            masks[target] = np.zeros(shape, dtype=bool)
            continue
        rast = rasterize(
            shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8"
        )
        masks[target] = rast.astype(bool)
    return masks


def _make_histograms(
    classifier_plag_angles: np.ndarray,
    labeled_plag_angles: np.ndarray,
    labeled_olivine_angles: np.ndarray,
    out_path: str,
    title: str,
    mode: str,
) -> None:
    """Side-by-side: (1) histogram (2) scatter prob vs angle. Saves to PNG."""
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    bins = np.linspace(0.0, np.pi / 2, 60)
    for arr, name, color in (
        (classifier_plag_angles, "classifier-plag (prob>=0.5)", "tab:orange"),
        (labeled_plag_angles, "labeled-plag", "tab:green"),
        (labeled_olivine_angles, "labeled-olivine", "tab:red"),
    ):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        ax_h.hist(arr, bins=bins, alpha=0.5, label=f"{name} (n={len(arr)})",
                  color=color, density=True)
    ax_h.set_xlabel("angle to plag endmember (rad)")
    ax_h.set_ylabel("density")
    ax_h.set_title(f"{title} — mode={mode}")
    ax_h.legend(fontsize=8, loc="upper right")

    # Scatter is filled in by the caller via attribute access — we just leave
    # placeholder axis here; the actual scatter is drawn by `make_diagnostic`.
    ax_s.set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic(
    tile_id: str,
    mode: str,
    sam_plag: np.ndarray,
    sam_olivine: np.ndarray,
    probs: np.ndarray,
    valid_mask: np.ndarray,
    gpkg_path: str,
    transform,
    crs,
    out_dir_reports: str,
    out_dir_parquet: str,
) -> Dict[str, float]:
    """Build diagnostic figure + stats CSV + hard-negative parquet.

    Args:
        sam_plag, sam_olivine: (H, W) float32 angle rasters from this mode.
        probs: (H, W, 5) classifier probability raster.
        valid_mask: (H, W) bool.
        gpkg_path: labeled polygon GeoPackage path (may be missing).
    Returns:
        Dict of summary stats (means, counts) for the run.
    """
    h, w = valid_mask.shape
    # Defensive shape checks
    if sam_plag.shape != (h, w) or sam_olivine.shape != (h, w):
        raise ValueError(
            f"SAM raster shape mismatch — plag={sam_plag.shape}, "
            f"olivine={sam_olivine.shape}, valid_mask={(h, w)}"
        )
    if probs.shape != (h, w, 5):
        raise ValueError(f"probs shape {probs.shape}; expected ({h},{w},5)")

    plag_prob = probs[:, :, PLAG_IDX]
    classifier_plag_mask = (plag_prob >= 0.5) & valid_mask

    label_masks = _label_masks_from_gpkg(gpkg_path, transform, (h, w), crs)
    labeled_plag_mask = label_masks.get("plagioclase", np.zeros((h, w), dtype=bool)) & valid_mask
    labeled_oli_mask = label_masks.get("olivine", np.zeros((h, w), dtype=bool)) & valid_mask

    cl_angles = sam_plag[classifier_plag_mask]
    lp_angles = sam_plag[labeled_plag_mask]
    lo_angles = sam_plag[labeled_oli_mask]

    # ---- Figure: histogram + scatter ----
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0.0, np.pi / 2, 60)
    for arr, name, color in (
        (cl_angles, "classifier-plag (prob>=0.5)", "tab:orange"),
        (lp_angles, "labeled-plag", "tab:green"),
        (lo_angles, "labeled-olivine", "tab:red"),
    ):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        ax_h.hist(arr, bins=bins, alpha=0.5,
                  label=f"{name} (n={len(arr)})", color=color, density=True)
    ax_h.set_xlabel("angle to plag endmember (rad)")
    ax_h.set_ylabel("density")
    ax_h.set_title(f"{tile_id} — plag SAM angle histogram (mode={mode})")
    ax_h.legend(fontsize=8, loc="upper right")

    # Scatter — subsample for plotting
    samp_mask = valid_mask & np.isfinite(sam_plag)
    if samp_mask.sum() > 200_000:
        # Random subsample for speed
        idx = np.flatnonzero(samp_mask)
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=200_000, replace=False)
        scat_x = plag_prob.ravel()[idx]
        scat_y = sam_plag.ravel()[idx]
    else:
        scat_x = plag_prob[samp_mask]
        scat_y = sam_plag[samp_mask]
    ax_s.scatter(scat_x, scat_y, s=1, alpha=0.15, color="gray")
    ax_s.set_xlabel("classifier plag probability")
    ax_s.set_ylabel("SAM angle to plag endmember (rad)")
    ax_s.set_title(f"{tile_id} — prob vs SAM angle (mode={mode})")
    ax_s.axhline(np.nanmedian(lo_angles) if len(lo_angles) else np.nan,
                 color="red", lw=0.8, ls="--",
                 label="median labeled-oli angle")
    ax_s.axvline(0.5, color="black", lw=0.6, ls=":")
    if len(lo_angles) > 0:
        ax_s.legend(fontsize=8)

    out_png = os.path.join(out_dir_reports,
                           f"{tile_id}_classifier_plag_diagnostic_{mode}.png")
    os.makedirs(out_dir_reports, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Hard-negative threshold ----
    if len(lo_angles[np.isfinite(lo_angles)]) > 10:
        lo_clean = lo_angles[np.isfinite(lo_angles)]
        theta_n = float(np.median(lo_clean) + np.std(lo_clean))
    else:
        # Fallback: median of classifier-plag distribution + 1 std.
        if len(cl_angles[np.isfinite(cl_angles)]) > 0:
            cl_clean = cl_angles[np.isfinite(cl_angles)]
            theta_n = float(np.median(cl_clean) + np.std(cl_clean))
        else:
            theta_n = float("nan")

    # Hard negatives: classifier says plag, but SAM angle to plag exceeds theta_n.
    if np.isfinite(theta_n):
        hard_mask = classifier_plag_mask & (sam_plag > theta_n) & np.isfinite(sam_plag)
    else:
        hard_mask = np.zeros_like(classifier_plag_mask)

    rows, cols = np.where(hard_mask)
    if len(rows) > 0:
        hn_df = pd.DataFrame({
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "tile_id": tile_id,
            "plag_prob": plag_prob[rows, cols].astype(np.float32),
            "sam_angle_plag": sam_plag[rows, cols].astype(np.float32),
            "sam_angle_olivine": sam_olivine[rows, cols].astype(np.float32),
            "mode": mode,
        })
    else:
        hn_df = pd.DataFrame(
            columns=["row", "col", "tile_id", "plag_prob", "sam_angle_plag",
                     "sam_angle_olivine", "mode"]
        )
    os.makedirs(out_dir_parquet, exist_ok=True)
    hn_parquet = os.path.join(
        out_dir_parquet, f"{tile_id}_hard_negatives_{mode}.parquet"
    )
    hn_df.to_parquet(hn_parquet, index=False)

    # ---- Stats CSV ----
    stats = {
        "tile_id": tile_id,
        "mode": mode,
        "n_valid": int(valid_mask.sum()),
        "n_classifier_plag": int(classifier_plag_mask.sum()),
        "n_labeled_plag": int(labeled_plag_mask.sum()),
        "n_labeled_olivine": int(labeled_oli_mask.sum()),
        "mean_angle_classifier_plag": float(np.nanmean(cl_angles)) if len(cl_angles) else float("nan"),
        "mean_angle_labeled_plag": float(np.nanmean(lp_angles)) if len(lp_angles) else float("nan"),
        "mean_angle_labeled_oli": float(np.nanmean(lo_angles)) if len(lo_angles) else float("nan"),
        "theta_n": theta_n,
        "n_hard_negatives": int(hard_mask.sum()),
        "fraction_classifier_plag_hard_negative": (
            float(hard_mask.sum() / max(classifier_plag_mask.sum(), 1))
        ),
    }
    stats_csv = os.path.join(
        out_dir_reports, f"{tile_id}_classifier_plag_stats_{mode}.csv"
    )
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)
    return stats
