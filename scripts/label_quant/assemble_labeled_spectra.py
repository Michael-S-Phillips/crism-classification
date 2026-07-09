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
import gc
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

# Final output schema. pixel_row/pixel_col are the true tile coordinates,
# carried through to both outputs (the visualizer's relabel path and the
# patch-cache builder cut 7x7 patches at these coords; m0/m1 back-fill from
# tiles needs them too).
OUTPUT_COLS = ["class", "source", "tile_id", "polygon_id",
               "confidence_weight", "multi", "pixel_row", "pixel_col"] \
    + BAND_COLS

# Source precedence for dedupe (lower rank wins). ndviz (interactive relabel
# session) is authoritative and outranks everything.
_SOURCE_RANK = {"ndviz": 0, "reassigned": 1, "tag": 2, "confirmed": 3,
                "hand": 4}

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
        "ndviz": os.path.join(data, "ndviz_relabels", "hard_negatives"),
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


def _downcast_bands(df):
    """Cast band columns to float32 and pixel coords to int32. Reflectance
    angle math is insensitive to the band fidelity loss and it halves the
    corpus footprint (the bland-tile rows alone add ~877k rows); tile pixel
    coordinates are far below the int32 range, so carrying them as int32 (not
    int64) keeps the two new coord columns from tipping the 15GB budget."""
    for c in BAND_COLS:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    for c in ("pixel_row", "pixel_col"):
        if c in df.columns and len(df):
            df[c] = df[c].astype(np.int32)
    return df


def _read_mrral_pixels(path, include_bland=True):
    """Read mrral_pixels.parquet once and split into two sources:

    - hand minerals: other<=0.5 & any mineral>0.5 (alteration NOT read; per the
      design table hand contributes the 5 mineral classes only).
    - base bland:    other>0.5 (class='bland_dust' later).

    Both carry full confidence (weight 1.0); the tier-derived confidence_weight
    the training pipeline stamped in is discarded. Returns (hand_df, bland_df).
    """
    empty = pd.DataFrame()
    if path is None or not os.path.exists(path):
        return empty, empty
    cols = _READ_META + BAND_COLS + _MINERAL_COLS
    avail = set(_first_parquet_schema(path))
    cols = [c for c in cols if c in avail]
    df = _downcast_bands(_ensure_labels(pq.read_table(
        path, columns=cols).to_pandas()))
    mineral_hit = np.zeros(len(df), dtype=bool)
    for c in _MINERAL_COLS:
        mineral_hit |= df[c].to_numpy() > 0.5
    other = df["other"].to_numpy()
    hand = df.loc[(other <= 0.5) & mineral_hit].reset_index(drop=True)
    hand["confidence_weight"] = 1.0
    bland = empty
    if include_bland:
        bland = df.loc[other > 0.5].reset_index(drop=True)
    del df
    gc.collect()
    return hand, bland


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
        parts.append(_downcast_bands(_ensure_labels(df)))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _mineral_any_expr():
    """pyarrow expression: any of the five mineral cols > 0.5."""
    e = None
    for c in _MINERAL_COLS:
        term = pc.field(c) > 0.5
        e = term if e is None else (e | term)
    return e


def _cat(parts):
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _read_reassigned_minerals(dirs):
    """Reject-pool (negative_of null/'') rows with any mineral>0.5 -> mineral
    reassignments. Both predicates pushed into pyarrow (negative_of pushdown
    copied from build_7cls_dataset._read_hn_tag) so only the ~38k matches
    materialise."""
    expr = (pc.field("negative_of").is_null()
            | (pc.field("negative_of") == "")) & _mineral_any_expr()
    parts = []
    for d in _as_dirs(dirs):
        if not os.path.exists(d):
            continue
        avail = set(_first_parquet_schema(d))
        want = _READ_META + BAND_COLS + _MINERAL_COLS + ["negative_of"]
        want = [c for c in want if c in avail]
        if "alteration" in avail:
            want.append("alteration")
        parts.append(_downcast_bands(_ensure_labels(pq.read_table(
            d, columns=want, filters=expr).to_pandas())))
    return _cat(parts)


def _read_reject_bland(dirs):
    """Reject-pool rows with other>0.5 and NO mineral -> reject->bland
    reassignments. This is the pool's bulk (~8.6M rows), so we push the full
    predicate into pyarrow AND project ONLY the output columns (no mineral/
    other/negative_of), converting Arrow -> pandas with self_destruct to keep
    peak memory bounded."""
    expr = ((pc.field("negative_of").is_null() | (pc.field("negative_of") == ""))
            & (pc.field("other") > 0.5) & ~_mineral_any_expr())
    parts = []
    for d in _as_dirs(dirs):
        if not os.path.exists(d):
            continue
        avail = set(_first_parquet_schema(d))
        want = ["tile_id", "polygon_id", "pixel_row", "pixel_col",
                "confidence_weight"] + BAND_COLS
        want = [c for c in want if c in avail]
        tbl = pq.read_table(d, columns=want, filters=expr)
        parts.append(_downcast_bands(tbl.to_pandas(
            split_blocks=True, self_destruct=True)))
        del tbl
    return _cat(parts)


def _read_tag_rows(dirs, tag):
    """Read hard_negatives rows with negative_of==tag (predicate pushdown +
    projection). Returns raw rows; the caller stamps the fixed class."""
    expr = pc.field("negative_of") == tag
    parts = []
    for d in _as_dirs(dirs):
        if not os.path.exists(d):
            continue
        avail = set(_first_parquet_schema(d))
        want = ["tile_id", "polygon_id", "pixel_row", "pixel_col",
                "confidence_weight", "negative_of"] + BAND_COLS
        want = [c for c in want if c in avail]
        df = pq.read_table(d, columns=want, filters=expr).to_pandas()
        parts.append(_downcast_bands(df))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _read_ndviz(ndviz_dir):
    """Read the interactive-relabel session (all rows, every decision type).

    Absent dir or a dir with no parquet files -> empty frame (no-op). Coords
    downcast to int32 to match the corpus for the suppression anti-join."""
    if not ndviz_dir or not os.path.exists(ndviz_dir):
        return pd.DataFrame()
    avail = set(_first_parquet_schema(ndviz_dir))
    if not avail:
        return pd.DataFrame()
    want = _READ_META + BAND_COLS + _MINERAL_COLS + ["negative_of"]
    want = [c for c in want if c in avail]
    if "alteration" in avail:
        want.append("alteration")
    df = pq.read_table(ndviz_dir, columns=want).to_pandas()
    if df.empty:
        return df
    return _downcast_bands(_ensure_labels(df))


def _build_ndviz_positives(df):
    """Turn ndviz decisions into positive corpus rows (source='ndviz').

    negative_of semantics written by the app:
      '' (empty)   -> reassignment: mineral collapse, else other>0.5 & no
                      mineral -> bland_reject.
      'alteration' -> alteration tag.
      'ambiguous'  -> junk tag.
      anything else (original class name) -> discard: NO positive row (the
                     pixel is still suppressed elsewhere via the key set).
    Returns a list of explode/fixed-shaped frames."""
    if df is None or df.empty:
        return []
    neg = df["negative_of"].fillna("").astype(str) if "negative_of" \
        in df.columns else pd.Series([""] * len(df), index=df.index)
    frames = []
    reassign = df.loc[neg == ""]
    if len(reassign):
        mineral_hit = np.zeros(len(reassign), dtype=bool)
        for c in _MINERAL_COLS:
            mineral_hit |= reassign[c].to_numpy() > 0.5
        frames.append(_explode_classes(reassign.loc[mineral_hit], "ndviz"))
        other = (reassign["other"].to_numpy() if "other" in reassign.columns
                 else np.zeros(len(reassign)))
        bland = reassign.loc[(~mineral_hit) & (other > 0.5)]
        frames.append(_fixed_class_rows(bland, "bland_reject", "ndviz"))
    frames.append(_fixed_class_rows(df.loc[neg == "alteration"],
                                    "alteration", "ndviz"))
    frames.append(_fixed_class_rows(df.loc[neg == "ambiguous"],
                                    "junk", "ndviz"))
    return [f for f in frames if not f.empty]


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
        return pd.DataFrame(columns=OUTPUT_COLS)
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
        return pd.DataFrame(columns=OUTPUT_COLS)
    return _downcast_bands(pd.concat(parts, ignore_index=True))


def _fixed_class_rows(df, class_name, source, force_weight=None):
    """Stamp a raw frame as a single fixed class (bland/junk/alteration tags):
    one row per pixel, multi=False. Produces the same explode-shaped columns so
    it concatenates with _explode_classes output before dedupe."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)
    # Copy only the columns we keep (band data can be millions of rows).
    keep = ["tile_id", "pixel_row", "pixel_col"] + [
        c for c in BAND_COLS if c in df.columns]
    out = df[keep].copy()
    out["polygon_id"] = (df["polygon_id"].to_numpy()
                         if "polygon_id" in df.columns else "")
    if force_weight is not None:
        out["confidence_weight"] = float(force_weight)
    elif "confidence_weight" in df.columns:
        out["confidence_weight"] = df["confidence_weight"].fillna(
            1.0).astype(float).to_numpy()
    else:
        out["confidence_weight"] = 1.0
    out.insert(0, "class", class_name)
    out["source"] = source
    out["multi"] = False
    return _downcast_bands(out)


def _dedupe(df):
    """Precedence dedupe on (tile_id, pixel_row, pixel_col, class), returning
    OUTPUT_COLS order. pixel_row/pixel_col are kept in the output (real tile
    coordinates needed downstream), not dropped.

    Deduping is done on lightweight key/rank arrays (never a copy of the full
    wide frame) so the ~4M-row corpus does not blow the memory budget: stable
    argsort by source rank, then keep the first occurrence of each key."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)
    rank = df["source"].map(_SOURCE_RANK).fillna(99).astype(np.int16).to_numpy()
    order = np.argsort(rank, kind="stable")  # lower rank (higher precedence) 1st
    keys = pd.MultiIndex.from_arrays([
        df["tile_id"].to_numpy()[order],
        df["pixel_row"].to_numpy()[order],
        df["pixel_col"].to_numpy()[order],
        df["class"].to_numpy()[order],
    ])
    keep_positions = order[~keys.duplicated(keep="first")]
    del keys, order, rank
    gc.collect()
    # Materialise deduped rows AND final column order in a single iloc copy
    # (no extra reorder/.copy() passes) to keep peak RAM under the 15GB budget
    # for the 12.7M-row corpus.
    col_idx = [df.columns.get_loc(c) for c in OUTPUT_COLS]
    out = df.iloc[keep_positions, col_idx].reset_index(drop=True)
    out["multi"] = out["multi"].astype(bool)
    return out


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


def _build_viz(full_df, seed, viz_per_class, viz_polygon_cap):
    """Per-class viz subsample: per-polygon cap then class cap. Applied
    uniformly to every class present, including bland and junk."""
    parts = []
    for cls in CLASSES + ["bland_dust", "bland_reject", "junk"]:
        sub = full_df[full_df["class"] == cls]
        if sub.empty:
            continue
        sub = _per_polygon_cap(sub, viz_polygon_cap, seed)
        sub = _subsample(sub, viz_per_class, seed)
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLS)
    return pd.concat(parts, ignore_index=True)[OUTPUT_COLS]


def _suppress_index(ndviz_raw):
    """MultiIndex of (tile_id, pixel_row, pixel_col) for every ndviz row
    (any decision type), used to anti-join lower-precedence sources. int32
    coords to match the corpus frames."""
    keys = ndviz_raw[["tile_id", "pixel_row", "pixel_col"]].copy()
    keys["pixel_row"] = keys["pixel_row"].astype(np.int32)
    keys["pixel_col"] = keys["pixel_col"].astype(np.int32)
    return pd.MultiIndex.from_frame(keys).unique()


def _anti_join(combined, suppress_idx):
    """Drop rows of ``combined`` whose pixel is in the ndviz suppression set."""
    if combined.empty:
        return combined
    key = pd.MultiIndex.from_arrays([
        combined["tile_id"].to_numpy(),
        combined["pixel_row"].to_numpy(),
        combined["pixel_col"].to_numpy(),
    ])
    return combined.loc[~key.isin(suppress_idx)]


def assemble(hand_path, confirmed_dirs, reassigned_dirs,
             out_path=None, viz_out_path=None, bland_path="__hand__",
             tag_dirs="__reassigned__", ndviz_dir="__default__",
             seed=42, viz_per_class=5000, viz_polygon_cap=200, write=True):
    """Assemble the labeled-spectra corpus and viz subsample.

    Returns (full_df, viz_df). Base bland rows (other>0.5) are read from
    ``hand_path`` and included in the full corpus unless ``bland_path`` is None
    (which skips only the base-bland source; reject->bland reassignments from
    ``reassigned_dirs`` are always included). The tag sources (alteration,
    junk) read from ``tag_dirs``, which defaults to ``reassigned_dirs``.

    ``ndviz_dir`` is the interactive-relabel session (review-format rows). Its
    decisions supersede at PIXEL level: every ndviz pixel is removed from all
    other sources (anti-join) before the precedence dedupe, then ndviz's own
    positive rows (reassignments + alteration/ambiguous tags; discards add
    nothing) are ingested with source='ndviz'. Absent dir -> no-op.
    """
    if tag_dirs == "__reassigned__":
        tag_dirs = reassigned_dirs
    if ndviz_dir == "__default__":
        ndviz_dir = _default_paths()["ndviz"]
    include_base_bland = bland_path is not None

    # ndviz session first: it drives pixel-level suppression of other sources.
    ndviz_raw = _read_ndviz(ndviz_dir)
    suppress_idx = (_suppress_index(ndviz_raw) if not ndviz_raw.empty
                    else None)

    # Build each source's output-shaped frame, freeing raw reads immediately;
    # the reject->bland source alone is ~8.6M rows on a 15GB box.
    frames = []
    hand_df, base_bland_df = _read_mrral_pixels(
        hand_path, include_bland=include_base_bland)
    frames.append(_explode_classes(hand_df, "hand"))
    del hand_df
    frames.append(_fixed_class_rows(base_bland_df, "bland_dust", "hand",
                                    force_weight=1.0))
    del base_bland_df
    frames.append(_explode_classes(_read_confirmed(confirmed_dirs),
                                   "confirmed"))
    frames.append(_explode_classes(_read_reassigned_minerals(reassigned_dirs),
                                   "reassigned"))
    frames.append(_fixed_class_rows(_read_tag_rows(tag_dirs, "alteration"),
                                    "alteration", "tag"))
    frames.append(_fixed_class_rows(_read_tag_rows(tag_dirs, "ambiguous"),
                                    "junk", "tag"))
    frames.append(_fixed_class_rows(_read_reject_bland(reassigned_dirs),
                                    "bland_reject", "reassigned"))
    frames = [f for f in frames if not f.empty]
    gc.collect()

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OUTPUT_COLS)
    del frames
    gc.collect()

    if suppress_idx is not None:
        # Remove every superseded pixel from the other sources, then ingest
        # ndviz's positive rows (discards deliberately add none).
        combined = _anti_join(combined, suppress_idx)
        ndviz_frames = _build_ndviz_positives(ndviz_raw)
        if ndviz_frames:
            combined = pd.concat([combined] + ndviz_frames, ignore_index=True)
        gc.collect()
    del ndviz_raw
    gc.collect()

    full_df = _dedupe(combined)
    del combined
    gc.collect()

    viz_df = _build_viz(full_df, seed, viz_per_class, viz_polygon_cap)

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
    ap.add_argument("--ndviz_dir", default=d["ndviz"],
                    help="interactive-relabel session dir (absent -> no-op)")
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
        ndviz_dir=args.ndviz_dir,
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
