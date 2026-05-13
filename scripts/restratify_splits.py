"""
Reassign train/val/test splits in pixels.parquet and mrral_pixels.parquet
using stratified-tile assignment that balances HCP/LCP/plagioclase positives
across splits, instead of the random tile-shuffle the original build used.

Motivation: random tile splits land class-rare tiles disproportionately in
val or test, producing pathological per-class AP. See the v4 sweep results
(spvit_lrscale001_v4: val_AP_hcp=0.02, test_AP_hcp=0.35 from the same model).

Usage:
    conda run -n crism python scripts/restratify_splits.py
    conda run -n crism python scripts/restratify_splits.py --dry_run
    conda run -n crism python scripts/restratify_splits.py --seed 7

The script:
1. Reads data/pixels.parquet (mrrsu, source-of-truth for splits).
2. Computes per-tile per-class positive-pixel counts.
3. Bin-packs tiles into train/val/test using greedy stratification on
   ['hcp', 'plagioclase', 'lcp'] (rarest first).
4. Writes the new split assignment back to both parquets in-place
   (atomic write via .tmp).
5. Reports before/after class distribution per split.

Patch caches in data/patch_cache/ must be rebuilt afterward since their row
order encodes the split assignment at the time they were cached.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Order matters for stratification priority — rarest first.
STRATIFY_CLASSES = ("hcp", "plagioclase", "lcp")
# Threshold for "positive" pixel (matches what evaluation/metrics.py uses).
POSITIVE_THRESHOLD = 0.4


def per_tile_positive_counts(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    For each tile_id, return an array of per-class positive-pixel counts in
    STRATIFY_CLASSES order.
    """
    counts: Dict[str, np.ndarray] = {}
    for tid, sub in df.groupby("tile_id"):
        arr = np.array(
            [int((sub[c] > POSITIVE_THRESHOLD).sum()) for c in STRATIFY_CLASSES],
            dtype=np.int64,
        )
        counts[tid] = arr
    return counts


def stratified_tile_splits(
    tile_counts: Dict[str, np.ndarray],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Greedy bin-packing assignment of tiles to train/val/test such that the
    per-class positive-pixel totals in each split end up close to the target
    fractions.

    Algorithm:
      1. Compute global per-class totals and per-split quotas.
      2. Sort tiles by their "rarity score" (per-tile positives weighted by
         inverse-global-frequency) descending — assign hard-to-place tiles first.
      3. For each tile, compute fit score for each split = sum over classes of
         (remaining_quota[split, class] * tile_count[class]). Place in the
         split with the highest fit (random tie-break with `seed`).
      4. Continue until all tiles assigned.

    Returns dict: tile_id -> 'train' | 'val' | 'test'.
    """
    if not tile_counts:
        return {}
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    tile_ids = sorted(tile_counts.keys())
    counts_mat = np.array([tile_counts[t] for t in tile_ids], dtype=np.int64)  # (n_tiles, n_classes)
    totals = counts_mat.sum(axis=0).astype(np.float64)  # global per-class positives

    # Avoid div-by-zero when a class happens to have zero positives.
    inv_freq = np.where(totals > 0, 1.0 / totals, 0.0)
    # Tile rarity = sum over classes of count * inv_freq. Rare-class tiles rank higher.
    rarity = counts_mat.astype(np.float64) @ inv_freq

    quotas = {
        "train": totals * train_frac,
        "val":   totals * val_frac,
        "test":  totals * test_frac,
    }
    accumulated = {s: np.zeros_like(totals) for s in quotas}

    # Sort tiles by rarity descending; ties broken randomly with the seed
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0, 1e-6, size=len(tile_ids))
    order = np.argsort(-(rarity + jitter))

    assignment: Dict[str, str] = {}
    for idx in order:
        tid = tile_ids[idx]
        tc = counts_mat[idx].astype(np.float64)

        # Remaining quota per split per class (clip at 0 to avoid negative
        # contributions if a split has already been overfilled).
        scores = {}
        for s in ("train", "val", "test"):
            remaining = np.clip(quotas[s] - accumulated[s], 0, None)
            scores[s] = float(np.dot(remaining, tc))

        # Tie-break preference: train > val > test (largest first) when scores equal.
        best = max(("train", "val", "test"), key=lambda s: (scores[s], {"train": 2, "val": 1, "test": 0}[s]))
        assignment[tid] = best
        accumulated[best] += tc

    return assignment


def report_distribution(df: pd.DataFrame, label: str) -> None:
    """Print per-split per-class positive-pixel counts."""
    logger.info(f"=== {label} ===")
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        if len(sub) == 0:
            logger.info(f"  {s}: 0 pixels (0 tiles)")
            continue
        per_class = {c: int((sub[c] > POSITIVE_THRESHOLD).sum()) for c in STRATIFY_CLASSES}
        n_tiles = sub["tile_id"].nunique()
        logger.info(
            f"  {s}: {len(sub):>10,} pixels, {n_tiles:>2} tiles, "
            f"hcp={per_class['hcp']:>7,} ({100*per_class['hcp']/len(sub):>5.2f}%), "
            f"plag={per_class['plagioclase']:>7,} ({100*per_class['plagioclase']/len(sub):>5.2f}%), "
            f"lcp={per_class['lcp']:>7,} ({100*per_class['lcp']/len(sub):>5.2f}%)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute and report new splits but do not write to parquets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    from config_loader import load_config
    cfg = load_config()
    mrrsu_path = os.path.join(cfg["output_dir"], "pixels.parquet")
    mrral_path = os.path.join(cfg["output_dir"], "mrral_pixels.parquet")
    if not os.path.exists(mrrsu_path):
        logger.error(f"missing {mrrsu_path}")
        return 1
    if not os.path.exists(mrral_path):
        logger.error(f"missing {mrral_path}")
        return 1

    logger.info(f"Loading {mrrsu_path}")
    mrrsu = pd.read_parquet(mrrsu_path)
    logger.info(f"  {len(mrrsu):,} rows, {mrrsu['tile_id'].nunique()} tiles")

    report_distribution(mrrsu, "BEFORE (mrrsu)")

    # Per-tile stratify counts come from mrrsu (it doesn't include Hellas).
    # Hellas-only tile rows in mrral get default 'train' assignment because
    # build_hellas_dataset.py pins Hellas pixels to train.
    counts = per_tile_positive_counts(mrrsu)
    new_split = stratified_tile_splits(
        counts, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
    )

    # Apply new splits to mrrsu in memory
    mrrsu_new = mrrsu.copy()
    mrrsu_new["split"] = mrrsu_new["tile_id"].map(new_split).fillna("train")
    report_distribution(mrrsu_new, "AFTER (mrrsu)")

    logger.info(f"Loading {mrral_path}")
    mrral = pd.read_parquet(mrral_path)
    logger.info(f"  {len(mrral):,} rows, {mrral['tile_id'].nunique()} tiles")
    report_distribution(mrral, "BEFORE (mrral)")

    mrral_new = mrral.copy()
    # Hellas-tile rows that aren't in `new_split` (because they weren't in
    # mrrsu) keep the 'train' assignment build_hellas_dataset.py gave them.
    # Other tiles use the new stratified assignment.
    mrral_new["split"] = mrral_new.apply(
        lambda r: new_split.get(r["tile_id"], "train"), axis=1,
    )
    report_distribution(mrral_new, "AFTER (mrral)")

    if args.dry_run:
        logger.info("Dry run; not writing.")
        return 0

    # Atomic write via .tmp swap (writing 700MB parquets directly is risky on the 9p mount)
    for path, df in [(mrrsu_path, mrrsu_new), (mrral_path, mrral_new)]:
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        logger.info(f"Wrote {path}")
    logger.info("Done. Remember to delete data/patch_cache/mrral_*_p7.npy and rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
