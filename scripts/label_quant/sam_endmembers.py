"""Component 2 — SAM endmember analysis + polygon purity report.

Quantifies the mineral-label corpus produced by
``assemble_labeled_spectra.py`` using the spectral angle metric over the
57-band analysis window (m2..m58, 534-2457 nm, raw reflectance v1):

    theta(a, b) = arccos( a.b / (|a| |b|) )

All class-level analysis runs on **polygon mean spectra**, single-label rows
only (multi == False). A polygon is keyed by (class, source, tile_id,
polygon_id) — the same physical polygon can appear under two sources after the
dedup pixel split, so each key is treated independently. Polygons with fewer
than ``min_px`` pixels are skipped from endmember/medoid math but retained in
the purity report (flagged ``degenerate``).

Per class:
  * Medoid endmember   — polygon mean with min mean angle to same-class means
                         (top-5 candidates reported).
  * Discriminative     — polygon mean maximizing
                         (min angle to other-class medoids
                          - mean angle to own-class polygon means).
  * Intra-class spread — angles of all class polygon means to the medoid
                         (mean / p50 / p90).

Corpus level:
  * Inter-class medoid angle matrix (deg).
  * Polygon purity: every single-label polygon's angle to its own medoid, the
    nearest other-class medoid + angle, and margin (nearest-other - own).
    Negative margin => suspect.
  * Per-source purity breakdown per class (count, median own-angle, % suspect).
  * Cross-source coherence per class (angle between per-source sub-medoids).

Outputs -> ``reports/label_quantification/``:
  endmembers.csv, class_angle_matrix.csv, polygon_purity.csv, summary.md.
Angles are reported in DEGREES everywhere user-facing.

Spec: docs/superpowers/specs/2026-07-09-label-quantification-design.md
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --- Analysis window: m2..m58 (57 bands, 534-2457 nm). --------------------- #
BAND_COLS = [f"m{i}" for i in range(2, 59)]

CLASSES = ["olivine", "lcp", "hcp", "plagioclase", "alteration"]

# Polygon identity key.
KEY_COLS = ["class", "source", "tile_id", "polygon_id"]


# --------------------------------------------------------------------------- #
# Core spectral-angle math
# --------------------------------------------------------------------------- #
def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle (radians) between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return float("nan")
    cos = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.arccos(cos))


def spectral_angle_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise spectral angles (radians) among the rows of X (n, b).

    L2-normalizes the rows then angles = arccos(clip(N @ N.T, -1, 1)).
    """
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    N = X / norms
    cos = np.clip(N @ N.T, -1.0, 1.0)
    return np.arccos(cos)


def _angles_to(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Spectral angles (radians) from every row of X to a single vector v."""
    X = np.asarray(X, dtype=float)
    v = np.asarray(v, dtype=float)
    xn = np.linalg.norm(X, axis=1)
    vn = np.linalg.norm(v)
    denom = xn * vn
    denom = np.where(denom == 0.0, 1.0, denom)
    cos = np.clip((X @ v) / denom, -1.0, 1.0)
    return np.arccos(cos)


def _medoid_index(B: np.ndarray):
    """Return (medoid_row, mean_off_diagonal_angles) for a (k, b) matrix.

    mean_off_diagonal_angles[i] = mean angle of row i to all other rows.
    For k == 1 the single row is the medoid with mean angle 0.
    """
    k = B.shape[0]
    if k == 1:
        return 0, np.zeros(1)
    M = spectral_angle_matrix(B)
    # exclude self (diagonal == 0) from the mean
    mean_ang = M.sum(axis=1) / (k - 1)
    return int(np.argmin(mean_ang)), mean_ang


# --------------------------------------------------------------------------- #
# Polygon aggregation
# --------------------------------------------------------------------------- #
def polygon_means(df: pd.DataFrame, band_cols=BAND_COLS) -> pd.DataFrame:
    """Collapse pixel rows to one mean spectrum per polygon key.

    Returns KEY_COLS + n_px + confidence_weight (mean) + band means.
    """
    agg = {c: "mean" for c in band_cols}
    agg["confidence_weight"] = "mean"
    g = df.groupby(KEY_COLS, observed=True, sort=False)
    means = g.agg(agg)
    means["n_px"] = g.size()
    means = means.reset_index()
    means["n_px"] = means["n_px"].astype(int)
    return means


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze(df: pd.DataFrame, min_px: int = 10, band_cols=BAND_COLS) -> dict:
    """Run the full SAM endmember / purity analysis.

    Returns a dict with keys: polygon_means, medoids, medoid_keys, endmembers,
    angle_matrix, purity, per_source, cross_source, intra_spread.
    """
    single = df[df["multi"] == False]  # noqa: E712 (explicit boolean column)
    pm = polygon_means(single, band_cols)
    B = pm[band_cols].to_numpy(dtype=float)
    pm_class = pm["class"].to_numpy()
    pm_npx = pm["n_px"].to_numpy()

    classes = [c for c in CLASSES if c in set(pm_class)]
    # keep any unexpected class labels too, appended after the canonical order
    for c in pd.unique(pm_class):
        if c not in classes:
            classes.append(c)

    medoids: dict[str, np.ndarray] = {}
    medoid_pm_idx: dict[str, int] = {}
    qualifying_idx: dict[str, np.ndarray] = {}
    own_mean_angle: dict[str, np.ndarray] = {}  # aligned to qualifying_idx

    for cls in classes:
        idx = np.where((pm_class == cls) & (pm_npx >= min_px))[0]
        if idx.size == 0:
            continue
        rel, mean_ang = _medoid_index(B[idx])
        medoids[cls] = B[idx[rel]]
        medoid_pm_idx[cls] = int(idx[rel])
        qualifying_idx[cls] = idx
        own_mean_angle[cls] = mean_ang

    # ----- endmembers (medoid, candidate2..5, discriminative) per class ----- #
    em_rows = []
    for cls in classes:
        if cls not in medoids:
            continue
        idx = qualifying_idx[cls]
        mean_ang = own_mean_angle[cls]
        order = np.argsort(mean_ang)  # ascending mean angle
        # medoid + candidate2..5
        for rank, rel in enumerate(order[:5]):
            kind = "medoid" if rank == 0 else f"candidate{rank + 1}"
            em_rows.append(_endmember_row(pm, idx[rel], cls, kind, band_cols))

        # discriminative: maximize (min angle to other medoids) - own_mean_angle
        other = [medoids[c] for c in classes if c != cls and c in medoids]
        if other:
            other_mat = np.vstack(other)
            scores = np.empty(idx.size)
            for j, gi in enumerate(idx):
                min_other = _angles_to(other_mat, B[gi]).min()
                scores[j] = min_other - mean_ang[j]
            disc_rel = int(np.argmax(scores))
            em_rows.append(
                _endmember_row(pm, idx[disc_rel], cls, "discriminative", band_cols))

    endmembers = pd.DataFrame(em_rows)

    # ----- inter-class medoid angle matrix (deg) ---------------------------- #
    med_classes = [c for c in classes if c in medoids]
    mat = pd.DataFrame(index=med_classes, columns=med_classes, dtype=float)
    for c1 in med_classes:
        for c2 in med_classes:
            mat.loc[c1, c2] = np.degrees(angle_between(medoids[c1], medoids[c2]))

    # ----- polygon purity (every single-label polygon) ---------------------- #
    purity = _purity(pm, B, pm_class, pm_npx, medoids, med_classes, min_px, band_cols)

    # ----- per-source purity breakdown -------------------------------------- #
    per_source = _per_source(purity)

    # ----- cross-source coherence per class --------------------------------- #
    cross_source = _cross_source(pm, B, pm_class, pm_npx, classes, min_px)

    # ----- intra-class spread ----------------------------------------------- #
    intra_rows = []
    for cls in med_classes:
        idx = qualifying_idx[cls]
        angs = np.degrees(_angles_to(B[idx], medoids[cls]))
        intra_rows.append({
            "class": cls,
            "n_polygons": int(idx.size),
            "mean_deg": float(np.mean(angs)),
            "p50_deg": float(np.percentile(angs, 50)),
            "p90_deg": float(np.percentile(angs, 90)),
        })
    intra_spread = pd.DataFrame(intra_rows)

    return {
        "polygon_means": pm,
        "medoids": medoids,
        "medoid_keys": {c: tuple(pm.loc[medoid_pm_idx[c], KEY_COLS]) for c in medoids},
        "endmembers": endmembers,
        "angle_matrix": mat,
        "purity": purity,
        "per_source": per_source,
        "cross_source": cross_source,
        "intra_spread": intra_spread,
    }


def _endmember_row(pm, i, cls, kind, band_cols):
    r = pm.iloc[i]
    row = {
        "class": cls,
        "kind": kind,
        "source": r["source"],
        "tile_id": r["tile_id"],
        "polygon_id": r["polygon_id"],
        "n_px": int(r["n_px"]),
    }
    for c in band_cols:
        row[c] = float(r[c])
    return row


def _purity(pm, B, pm_class, pm_npx, medoids, med_classes, min_px, band_cols):
    rows = []
    med_mat = {c: medoids[c] for c in med_classes}
    for i in range(len(pm)):
        cls = pm_class[i]
        if cls not in med_mat:
            continue
        v = B[i]
        own = np.degrees(angle_between(v, med_mat[cls]))
        best_other, best_ang = None, np.inf
        for c2 in med_classes:
            if c2 == cls:
                continue
            a = np.degrees(angle_between(v, med_mat[c2]))
            if a < best_ang:
                best_ang, best_other = a, c2
        margin = best_ang - own if best_other is not None else np.nan
        r = pm.iloc[i]
        rows.append({
            "class": cls,
            "source": r["source"],
            "tile_id": r["tile_id"],
            "polygon_id": r["polygon_id"],
            "n_px": int(r["n_px"]),
            "confidence_weight": float(r["confidence_weight"]),
            "degenerate": bool(pm_npx[i] < min_px),
            "own_angle_deg": own,
            "nearest_other_class": best_other,
            "nearest_other_angle_deg": best_ang if best_other is not None else np.nan,
            "margin_deg": margin,
            "suspect": bool(margin < 0) if best_other is not None else False,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("margin_deg").reset_index(drop=True)
    return out


def _per_source(purity):
    if purity.empty:
        return purity
    rows = []
    for (cls, src), g in purity.groupby(["class", "source"], observed=True):
        n = len(g)
        rows.append({
            "class": cls,
            "source": src,
            "n_polygons": n,
            "median_own_angle_deg": float(g["own_angle_deg"].median()),
            "pct_suspect": float(100.0 * g["suspect"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["class", "source"]).reset_index(drop=True)


def _cross_source(pm, B, pm_class, pm_npx, classes, min_px):
    rows = []
    src_arr = pm["source"].to_numpy()
    for cls in classes:
        sources = sorted(set(src_arr[pm_class == cls]))
        submed = {}
        for src in sources:
            idx = np.where((pm_class == cls) & (src_arr == src) & (pm_npx >= min_px))[0]
            if idx.size == 0:
                continue
            rel, _ = _medoid_index(B[idx])
            submed[src] = (B[idx[rel]], int(idx.size))
        for sa, sb in combinations(sorted(submed), 2):
            rows.append({
                "class": cls,
                "source_a": sa,
                "source_b": sb,
                "n_a": submed[sa][1],
                "n_b": submed[sb][1],
                "angle_deg": np.degrees(angle_between(submed[sa][0], submed[sb][0])),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# I/O + report
# --------------------------------------------------------------------------- #
def _read_corpus(path: str) -> pd.DataFrame:
    cols = KEY_COLS + ["confidence_weight", "multi"] + BAND_COLS
    return pq.read_table(path, columns=cols).to_pandas()


def _md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not pd.isna(v):
                cells.append(floatfmt.format(v))
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_reports(res: dict, out_dir: str, min_px: int, runtime_s: float | None = None):
    os.makedirs(out_dir, exist_ok=True)

    # endmembers.csv
    res["endmembers"].to_csv(os.path.join(out_dir, "endmembers.csv"), index=False)

    # class_angle_matrix.csv
    res["angle_matrix"].to_csv(os.path.join(out_dir, "class_angle_matrix.csv"))

    # polygon_purity.csv
    res["purity"].to_csv(os.path.join(out_dir, "polygon_purity.csv"), index=False)

    # summary.md
    pur = res["purity"]
    suspects = pur[pur["suspect"]] if not pur.empty else pur
    worst = pur.head(20) if not pur.empty else pur
    worst_cols = ["class", "source", "tile_id", "polygon_id", "n_px",
                  "confidence_weight", "own_angle_deg", "nearest_other_class",
                  "nearest_other_angle_deg", "margin_deg"]

    n_by_class = (pur.groupby("class", observed=True)["suspect"].agg(["sum", "count"])
                  if not pur.empty else pd.DataFrame())

    lines = []
    lines.append("# Label Quantification — SAM Endmember Analysis\n")
    lines.append(f"Analysis window: 57 bands (m2..m58, 534-2457 nm), raw reflectance.\n")
    lines.append(f"min_px = {min_px}. Angles in DEGREES. "
                 "Class-level math on polygon mean spectra, single-label only.\n")
    if runtime_s is not None:
        lines.append(f"Runtime: {runtime_s:.1f} s.\n")

    lines.append("\n## Interpretation caveat\n")
    lines.append(
        "Raw-reflectance spectral angles are continuum/albedo-dominated (all\n"
        "inter-class medoid angles come out <3 deg), so absolute suspect counts\n"
        "are structurally inflated. Read margins as a RELATIVE worst-offenders\n"
        "ranking, not a mislabel census. Continuum removal is the planned v2\n"
        "that would make these angles mineralogical.\n")

    lines.append("\n## Per-source purity (headline)\n")
    lines.append("Count of polygons, median angle to own-class medoid, and "
                 "% with negative margin (closer to another class).\n\n")
    lines.append(_md_table(res["per_source"]))

    lines.append("\n## Cross-source coherence per class\n")
    lines.append("Angle between per-source sub-medoids — quantifies source "
                 "disagreement within a class.\n\n")
    lines.append(_md_table(res["cross_source"].sort_values(["class", "angle_deg"],
                                                            ascending=[True, False])
                           if not res["cross_source"].empty else res["cross_source"]))

    lines.append("\n## Inter-class medoid angle matrix (deg)\n\n")
    mat = res["angle_matrix"].copy()
    mat.insert(0, "class", mat.index)
    lines.append(_md_table(mat))

    lines.append("\n## Per-class intra-class spread (angle to own medoid, deg)\n\n")
    lines.append(_md_table(res["intra_spread"]))

    if not pur.empty:
        lines.append("\n## Suspect polygons (negative margin)\n")
        lines.append(f"Total suspects: {int(pur['suspect'].sum())} / {len(pur)} "
                     f"polygons ({100.0 * pur['suspect'].mean():.1f}%).\n\n")
        if not n_by_class.empty:
            byc = n_by_class.reset_index()
            byc.columns = ["class", "n_suspect", "n_polygons"]
            byc["n_suspect"] = byc["n_suspect"].astype(int)
            lines.append("Suspects by class:\n\n")
            lines.append(_md_table(byc))
        lines.append("\n### Top-20 worst-margin suspects (with provenance)\n\n")
        lines.append(_md_table(worst[worst_cols]))

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description="SAM endmember analysis + polygon purity")
    ap.add_argument("--corpus", default="data/labeled_spectra.parquet")
    ap.add_argument("--out_dir", default="reports/label_quantification")
    ap.add_argument("--min_px", type=int, default=10)
    args = ap.parse_args(argv)

    t0 = time.time()
    df = _read_corpus(args.corpus)
    res = analyze(df, min_px=args.min_px)
    runtime = time.time() - t0
    write_reports(res, args.out_dir, args.min_px, runtime_s=runtime)
    print(f"Wrote reports to {args.out_dir} in {runtime:.1f} s "
          f"({len(res['polygon_means'])} polygons, "
          f"{int(res['purity']['suspect'].sum())} suspects).")


if __name__ == "__main__":
    main()
