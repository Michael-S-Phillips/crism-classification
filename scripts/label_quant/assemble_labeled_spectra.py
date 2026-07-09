"""Component 1 — labeled-spectra corpus assembler.

Combines every mineral-label source into a single tidy corpus of one row per
(pixel, class), restricted to the 57-band analysis window m2..m58 (534-2457 nm;
m0/m1 fall below the 450 nm floor). Sources and precedence follow the design
spec (docs/superpowers/specs/2026-07-09-label-quantification-design.md):

  hand        data/mrral_pixels.parquet   (other<=0.5 & any mineral>0.5)
  confirmed   */confirmed_pixels dirs     (all confirmed mineral pixels)
  reassigned  */hard_negatives dirs       (negative_of='' & any mineral>0.5)

A pixel positive for k of the five collapsed classes yields k rows, each
flagged multi=(k>1). Duplicate (tile_id, pixel_row, pixel_col, class) keys are
deduped with precedence reassigned > confirmed > hand.

Two outputs:
  data/labeled_spectra.parquet      full corpus (5 mineral classes)
  data/labeled_spectra_viz.parquet  per-class subsample for the N-D visualizer
                                     (per-polygon cap then class cap) + a bland
                                     reference cloud.

Standalone by design: it does NOT import build_7cls_dataset, but copies the
pyarrow predicate-pushdown pattern from its ``_read_hn_tag`` so the 2.8 GB
hard-negative pool is filtered on read instead of materialised.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

# --- Analysis window: m2..m58 (57 bands, 534-2457 nm). m0/m1 excluded. -------
BAND_COLS = [f"m{i}" for i in range(2, 59)]

# Five collapsed mineral classes analysed by the corpus.
CLASSES = ["olivine", "lcp", "hcp", "plagioclase", "alteration"]

# Raw mineral columns present in the source parquets.
_MINERAL_COLS = ["olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase"]

# Final output schema (pixel_row/pixel_col are internal, dropped before write).
OUTPUT_COLS = ["class", "source", "tile_id", "polygon_id",
               "confidence_weight", "multi"] + BAND_COLS

# Internal columns carried through explode/dedupe then dropped.
_KEY_COLS = ["tile_id", "pixel_row", "pixel_col"]

# Source precedence for dedupe (lower rank wins).
_SOURCE_RANK = {"reassigned": 0, "tag": 1, "confirmed": 2, "hand": 3}

# Columns we ever need to read from a source frame (bands + labels + meta).
_READ_META = ["tile_id", "polygon_id", "pixel_row", "pixel_col",
              "other", "confidence_weight"]


def _default_paths():
    """Repo-relative default paths (module lives in scripts/label_quant/)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    data = os.path.join(root, "data")
    return {
        "hand": os.path.join(data, "mrral_pixels.parquet"),
        "confirmed": [os.path.join(data, "mc13_review", "confirmed_pixels"),
                      os.path.join(data, "mc13_review_7cls_v3",
                                   "confirmed_pixels")],
        "reassigned": [os.path.join(data, "mc13_review", "hard_negatives"),
                       os.path.join(data, "mc13_review_7cls_v3",
                                    "hard_negatives")],
        "out": os.path.join(data, "labeled_spectra.parquet"),
        "viz_out": os.path.join(data, "labeled_spectra_viz.parquet"),
    }


def _as_dirs(dirs):
    if dirs is None:
        return []
    return [dirs] if isinstance(dirs, str) else list(dirs)


def _first_parquet_schema(path):
    """Column names available in a parquet file or dir (from first fragment)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            return []
        return pq.read_schema(files[0]).names
    return pq.read_schema(path).names


def _ensure_labels(df):
    """Guarantee the five mineral columns + alteration exist (fill 0.0).

    Old confirmed files lack ``alteration`` and hand frames are read without
    it; both must degrade to 0 (no bogus alteration rows / no NaN)."""
    for col in _MINERAL_COLS + ["alteration"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)
    return df


def _read_hand(path):
    """Hand labels: other<=0.5 & any mineral>0.5. Alteration intentionally NOT
    read (design table: hand contributes 5 mineral classes only)."""
    if path is None or not os.path.exists(path):
        return pd.DataFrame()
    cols = _READ_META + BAND_COLS + _MINERAL_COLS
    df = pq.read_table(path, columns=cols).to_pandas()
    df = _ensure_labels(df)
    mineral_hit = np.zeros(len(df), dtype=bool)
    for c in _MINERAL_COLS:
        mineral_hit |= df[c].to_numpy() > 0.5
    keep = (df["other"].to_numpy() <= 0.5) & mineral_hit
    df = df.loc[keep].reset_index(drop=True)
    # Hand labels are original annotations at full confidence; discard the
    # tier-derived confidence_weight the training pipeline stamped in.
    df["confidence_weight"] = 1.0
    return df


def _read_confirmed(dirs):
    """Confirmed pixels: read each dir projecting only needed columns; add
    alteration=0 for old-schema dirs that lack it."""
    parts = []
    for d in _as_dirs(dirs):
        if not os.path.exists(d):
            continue
        avail = set(_first_parquet_schema(d))
        want = _READ_META + BAND_COLS + _MINERAL_COLS
        want = [c for c in want if c in avail]
        if "alteration" in avail:
            want.append("alteration")
        df = pq.read_table(d, columns=want).to_pandas()
        parts.append(_ensure_labels(df))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _read_reassigned(dirs):
    """Reassigned pixels from hard_negatives: negative_of is null/'' AND any
    mineral>0.5. Predicate pushdown on negative_of (copied from
    build_7cls_dataset._read_hn_tag) + column projection keeps the 2.8 GB pool
    off the heap."""
    expr = pc.field("negative_of").is_null() | (pc.field("negative_of") == "")
    parts = []
    for d in _as_dirs(dirs):
        if not os.path.exists(d):
            continue
        avail = set(_first_parquet_schema(d))
        want = _READ_META + BAND_COLS + _MINERAL_COLS + ["negative_of"]
        want = [c for c in want if c in avail]
        if "alteration" in avail:
            want.append("alteration")
        df = pq.read_table(d, columns=want, filters=expr).to_pandas()
        df = _ensure_labels(df)
        mineral_hit = np.zeros(len(df), dtype=bool)
        for c in _MINERAL_COLS:
            mineral_hit |= df[c].to_numpy() > 0.5
        parts.append(df.loc[mineral_hit].reset_index(drop=True))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _read_alteration_tags(dirs):
    """Dedicated alteration review tags: hard_negatives rows with
    negative_of='alteration'. These are alteration positives regardless of the
    mineral label columns (the 7cls build stamps alteration=1.0 on them), so we
    force alteration=1.0 rather than relying on the raw label cols. Same
    predicate-pushdown + column-projection pattern as _read_reassigned."""
    expr = pc.field("negative_of") == "alteration"
    parts = []
    for d in _as_dirs(dirs):
        if not os.path.exists(d):
            continue
        avail = set(_first_parquet_schema(d))
        want = _READ_META + BAND_COLS + _MINERAL_COLS + ["negative_of"]
        want = [c for c in want if c in avail]
        if "alteration" in avail:
            want.append("alteration")
        df = pq.read_table(d, columns=want, filters=expr).to_pandas()
        df = _ensure_labels(df)
        # Stamp alteration positive and zero the mineral cols so the class
        # collapse emits exactly one alteration row (multi=False) per pixel.
        for c in _MINERAL_COLS:
            df[c] = 0.0
        df["alteration"] = 1.0
        parts.append(df.reset_index(drop=True))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _class_flags(df):
    """Boolean DataFrame of the five collapsed classes."""
    return pd.DataFrame({
        "olivine": (df["olivine_t1"].to_numpy() > 0.5)
                   | (df["olivine_t2"].to_numpy() > 0.5),
        "lcp": df["lcp"].to_numpy() > 0.5,
        "hcp": df["hcp"].to_numpy() > 0.5,
        "plagioclase": df["plagioclase"].to_numpy() > 0.5,
        "alteration": df["alteration"].to_numpy() > 0.5,
    }, index=df.index)[CLASSES]


def _explode_classes(df, source):
    """One row per (pixel, positive class); multi flags k>1 co-occurrence."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS + _KEY_COLS[1:])
    df = _ensure_labels(df.copy())
    if "confidence_weight" not in df.columns:
        df["confidence_weight"] = 1.0
    df["confidence_weight"] = df["confidence_weight"].fillna(1.0).astype(float)
    if "polygon_id" not in df.columns:
        df["polygon_id"] = ""
    flags = _class_flags(df)
    k = flags.to_numpy().sum(axis=1)
    multi = k > 1
    carry = ["tile_id", "polygon_id", "pixel_row", "pixel_col",
             "confidence_weight"] + BAND_COLS
    parts = []
    for cls in CLASSES:
        mask = flags[cls].to_numpy()
        if not mask.any():
            continue
        sub = df.loc[mask, carry].copy()
        sub.insert(0, "class", cls)
        sub["source"] = source
        sub["multi"] = multi[mask]
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLS + ["pixel_row", "pixel_col"])
    return pd.concat(parts, ignore_index=True)


def _dedupe(df):
    """Precedence dedupe on (tile_id, pixel_row, pixel_col, class); drop the
    internal pixel key columns and return OUTPUT_COLS order."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)
    df = df.copy()
    df["_rank"] = df["source"].map(_SOURCE_RANK).fillna(99).astype(int)
    df = df.sort_values("_rank", kind="stable")
    df = df.drop_duplicates(subset=["tile_id", "pixel_row", "pixel_col",
                                    "class"], keep="first")
    df = df.drop(columns=["_rank", "pixel_row", "pixel_col"])
    df["multi"] = df["multi"].astype(bool)
    return df[OUTPUT_COLS].reset_index(drop=True)


def _per_polygon_cap(df, max_per, seed):
    """Subsample each (tile_id, polygon_id) group to at most max_per rows."""
    if df.empty or max_per is None:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(["tile_id", "polygon_id"], sort=False):
        if len(g) <= max_per:
            parts.append(g)
        else:
            parts.append(g.iloc[rng.choice(len(g), size=max_per,
                                           replace=False)])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _subsample(df, n, seed):
    if len(df) <= n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return df.iloc[rng.choice(len(df), size=n, replace=False)].reset_index(
        drop=True)


def _bland_reference(path, n, seed):
    """Bland reference cloud for the visualizer: mrral_pixels other>0.5,
    class='bland', multi=False. Viz-only (never in the full corpus)."""
    if path is None or not os.path.exists(path):
        return pd.DataFrame(columns=OUTPUT_COLS)
    cols = ["tile_id", "polygon_id", "other", "confidence_weight"] + BAND_COLS
    avail = set(_first_parquet_schema(path))
    cols = [c for c in cols if c in avail]
    df = pq.read_table(path, columns=cols).to_pandas()
    df = df.loc[df["other"].to_numpy() > 0.5].reset_index(drop=True)
    df = _subsample(df, n, seed)
    # Bland is a hand-sourced reference cloud -> full confidence.
    df["confidence_weight"] = 1.0
    if "polygon_id" not in df.columns:
        df["polygon_id"] = ""
    df["class"] = "bland"
    df["source"] = "hand"
    df["multi"] = False
    return df[OUTPUT_COLS].reset_index(drop=True)


def _build_viz(full_df, bland_df, seed, viz_per_class, viz_polygon_cap):
    parts = []
    for cls in CLASSES:
        sub = full_df[full_df["class"] == cls]
        if sub.empty:
            continue
        sub = _per_polygon_cap(sub, viz_polygon_cap, seed)
        sub = _subsample(sub, viz_per_class, seed)
        parts.append(sub)
    if bland_df is not None and not bland_df.empty:
        parts.append(bland_df)
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLS)
    return pd.concat(parts, ignore_index=True)[OUTPUT_COLS]


def assemble(hand_path, confirmed_dirs, reassigned_dirs,
             out_path=None, viz_out_path=None, bland_path="__hand__",
             tag_dirs="__reassigned__",
             seed=42, viz_per_class=5000, viz_polygon_cap=200, write=True):
    """Assemble the labeled-spectra corpus and viz subsample.

    Returns (full_df, viz_df). If ``bland_path`` is the sentinel ``"__hand__"``
    it defaults to ``hand_path``; pass None to skip the bland reference. The
    alteration-tag source (``tag_dirs``) defaults to the same hard_negatives
    dirs as ``reassigned_dirs``.
    """
    if bland_path == "__hand__":
        bland_path = hand_path
    if tag_dirs == "__reassigned__":
        tag_dirs = reassigned_dirs

    exploded = []
    for reader, src in ((_read_hand(hand_path), "hand"),
                        (_read_confirmed(confirmed_dirs), "confirmed"),
                        (_read_reassigned(reassigned_dirs), "reassigned"),
                        (_read_alteration_tags(tag_dirs), "tag")):
        ex = _explode_classes(reader, src)
        if not ex.empty:
            exploded.append(ex)

    if exploded:
        combined = pd.concat(exploded, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OUTPUT_COLS + ["pixel_row",
                                                       "pixel_col"])
    full_df = _dedupe(combined)

    bland_df = _bland_reference(bland_path, viz_per_class, seed)
    viz_df = _build_viz(full_df, bland_df, seed, viz_per_class,
                        viz_polygon_cap)

    if write:
        if out_path:
            full_df.to_parquet(out_path, index=False)
        if viz_out_path:
            viz_df.to_parquet(viz_out_path, index=False)
    return full_df, viz_df


def _report(full_df, viz_df):
    print("\n=== Full corpus: rows per class x source ===")
    if full_df.empty:
        print("(empty)")
    else:
        tab = full_df.groupby(["class", "source"]).size().unstack(
            fill_value=0)
        print(tab.to_string())
        print(f"\nTotal full-corpus rows: {len(full_df):,}")
        print("multi=True rows:",
              int(full_df["multi"].sum()))
    print("\n=== Viz parquet: rows per class ===")
    print(viz_df.groupby("class").size().to_string() if not viz_df.empty
          else "(empty)")
    print(f"Total viz rows: {len(viz_df):,}")


def main(argv=None):
    d = _default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hand_path", default=d["hand"])
    ap.add_argument("--confirmed_dirs", nargs="*", default=d["confirmed"])
    ap.add_argument("--reassigned_dirs", nargs="*", default=d["reassigned"])
    ap.add_argument("--out_path", default=d["out"])
    ap.add_argument("--viz_out_path", default=d["viz_out"])
    ap.add_argument("--bland_path", default=None,
                    help="bland reference source (defaults to hand_path)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--viz_per_class", type=int, default=5000)
    ap.add_argument("--viz_polygon_cap", type=int, default=200)
    args = ap.parse_args(argv)

    bland = args.bland_path if args.bland_path is not None else args.hand_path
    full_df, viz_df = assemble(
        hand_path=args.hand_path,
        confirmed_dirs=args.confirmed_dirs,
        reassigned_dirs=args.reassigned_dirs,
        out_path=args.out_path,
        viz_out_path=args.viz_out_path,
        bland_path=bland,
        seed=args.seed,
        viz_per_class=args.viz_per_class,
        viz_polygon_cap=args.viz_polygon_cap,
        write=True,
    )
    _report(full_df, viz_df)
    print(f"\nWrote {args.out_path}")
    print(f"Wrote {args.viz_out_path}")


if __name__ == "__main__":
    main()
