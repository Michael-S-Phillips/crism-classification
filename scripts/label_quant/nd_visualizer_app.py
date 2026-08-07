"""Component 3 — N-D Visualizer (ENVI-style) for the labeled-spectra corpus.

Streamlit + plotly app that renders the labeled mineral spectra as points in
band-space over the 450-2500 nm analysis window (m2..m58, 57 bands). Provides
three 3-D projection modes (PCA / raw bands / random orthonormal), several
colour modes (class / source / confidence / spectral-angle-to-endmember), a
filter sidebar, a stratified point budget, and a mean +/-1 sigma spectra panel.

Launch: ``bash scripts/label_quant/run_nd_visualizer.sh 8502``

Spec: docs/superpowers/specs/2026-07-09-label-quantification-design.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Make ``scripts.label_quant.*`` importable when launched via the streamlit CLI
# (cwd is the project root, but that is not on sys.path by default).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.label_quant.sam_endmembers import (  # noqa: E402
    _angles_to, BAND_COLS, BAD_BAND_RANGES_NM, good_band_mask,
    set_band_exclusion, apply_band_mask)

# --------------------------------------------------------------------------- #
# Module constants (overridable via env vars so tests can point at fixtures).
# --------------------------------------------------------------------------- #
DEFAULT_VIZ_PARQUET = os.environ.get(
    "NDVIZ_PARQUET", "data/labeled_spectra_viz.parquet")
DEFAULT_ENDMEMBERS = os.environ.get(
    "NDVIZ_ENDMEMBERS", "reports/label_quantification/endmembers.csv")
DEFAULT_WAVELENGTHS = os.environ.get(
    "NDVIZ_WAVELENGTHS",
    "reports/floor_tests/v3b_lrscale001/nili/vector_nili_6cls_wavelengths.json")
# Full pixel corpus (now with real pixel_row/pixel_col) + relabel session dir.
DEFAULT_CORPUS = os.environ.get("NDVIZ_CORPUS", "data/labeled_spectra.parquet")
DEFAULT_RELABELS_DIR = os.environ.get("NDVIZ_RELABELS_DIR", "data/ndviz_relabels")

# Relabel targets exposed in the visualizer's "relabel as:" selectbox.
RELABEL_OPTIONS = ["olivine", "lcp", "hcp", "plagioclase", "alteration",
                   "bland", "ambiguous", "discard"]

# Source-tile glob for back-filling m0/m1 (raster bands 1-2) at real pixel
# coords. {tile_id} is substituted; overridable so tests can point at a fixture.
DEFAULT_TILE_GLOB = os.environ.get(
    "NDVIZ_TILE_GLOB", "/Volumes/Mars_GIS/CRISM/MRDR/mc*/{tile_id}_mrral_*_0327_4.img")

# Review-app class palette (fixed).
CLASS_PALETTE = {
    "olivine": "#e6194b",
    "lcp": "#3cb44b",
    "hcp": "#4363d8",
    "plagioclase": "#f58231",
    "bland": "#aaaaaa",
    "alteration": "#cc8899",
    "junk": "#808080",
}
MINERALS = ["olivine", "lcp", "hcp", "plagioclase", "alteration"]
# Reference clouds (loadable but off by default in the class filter).
REFERENCE_CLASSES = ["bland", "junk"]

# Categorical palette for the "source" colour mode.
SOURCE_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

GREY = "#cccccc"

# Max principal components computed in PCA mode (bounded by data size at fit).
N_PCA = 10


# --------------------------------------------------------------------------- #
# Pure helpers (importable / testable without Streamlit)
# --------------------------------------------------------------------------- #
def spectral_angles_deg(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Spectral angles (degrees) from every row of X to a single vector v.

    Reuses ``sam_endmembers._angles_to``, which internally calls
    ``apply_band_mask`` on both inputs — so 1 um overlap-band exclusion is
    ALREADY applied here (respecting the module-level toggle). Callers pass the
    full 57-band window and must NOT pre-mask, or the columns would be dropped
    twice.
    """
    return np.degrees(_angles_to(X, v))


def load_wavelengths(path: str):
    """Return (wavelengths_nm[57], is_fallback).

    Maps m2..m58 to wavelengths_nm[2:59]. Falls back to a linspace over the
    analysis window when the JSON is missing.
    """
    n = len(BAND_COLS)
    if os.path.exists(path):
        try:
            data = json.load(open(path))
            w = np.asarray(data["wavelengths_nm"], dtype=float)
            sub = w[2:2 + n]
            if sub.size == n:
                return sub, False
        except Exception:
            pass
    return np.linspace(534.0, 2457.0, n), True


def _l2_normalize(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return M / norms


def resolve_clicked_points(event, df: pd.DataFrame) -> pd.DataFrame:
    """Resolve a plotly selection event to rows of ``df``.

    Each scatter point carries ``customdata`` whose first element is the
    positional row-id into ``df`` (the displayed frame). Using that row-id is
    unambiguous across multiple traces/classes, unlike per-trace point_number.
    ``event`` may be the Streamlit ``on_select`` return (dict-like with a
    ``selection.points`` list) or None. Returns the matching rows of ``df``
    (empty frame if nothing resolvable).
    """
    if event is None:
        return df.iloc[0:0]
    sel = event.get("selection") if isinstance(event, dict) else getattr(
        event, "selection", None)
    if sel is None:
        return df.iloc[0:0]
    points = sel.get("points", []) if isinstance(sel, dict) else getattr(
        sel, "points", [])
    row_ids = []
    for p in points or []:
        cd = p.get("customdata") if isinstance(p, dict) else getattr(
            p, "customdata", None)
        if not cd:
            continue
        try:
            rid = int(cd[0])
        except (TypeError, ValueError):
            continue
        if 0 <= rid < len(df) and rid not in row_ids:
            row_ids.append(rid)
    return df.iloc[row_ids]


def polygon_uid(orig_class, source, tile_id, polygon_id) -> str:
    """Stable, self-describing polygon identity for the relabel session."""
    return f"ndviz::{orig_class}::{source}::{tile_id}::{polygon_id}"


def read_polygon_rows(corpus_path: str, cls, source, tile_id, polygon_id
                      ) -> pd.DataFrame:
    """Read every pixel row for one polygon from the corpus via predicate
    pushdown on (class, source, tile_id, polygon_id). Includes the real
    pixel_row/pixel_col coords now carried by the corpus."""
    import pyarrow.parquet as pq
    cols = (["class", "source", "tile_id", "polygon_id",
             "pixel_row", "pixel_col"] + BAND_COLS)
    flt = [("class", "=", cls), ("source", "=", source),
           ("tile_id", "=", tile_id), ("polygon_id", "=", int(polygon_id))]
    return pq.read_table(corpus_path, columns=cols, filters=flt).to_pandas()


def find_tile_img(tile_id: str, tile_glob: str = DEFAULT_TILE_GLOB):
    """Return the source-tile .img path for ``tile_id`` (first glob match) or
    None if not found."""
    import glob
    matches = sorted(glob.glob(tile_glob.format(tile_id=tile_id)))
    return matches[0] if matches else None


def read_m0_m1(tile_id: str, rows: np.ndarray, cols: np.ndarray,
               tile_glob: str = DEFAULT_TILE_GLOB):
    """Read raster bands 1-2 (m0, m1) at the given pixel coords from the
    source tile. Returns (m0, m1, ok): ok is False (arrays zero-filled) when
    the tile img is missing or the read fails."""
    n = len(rows)
    path = find_tile_img(tile_id, tile_glob)
    if path is None:
        return np.zeros(n), np.zeros(n), False
    try:
        import rasterio
        with rasterio.open(path) as ds:
            b0 = ds.read(1)
            b1 = ds.read(2)
        r = np.clip(rows.astype(int), 0, b0.shape[0] - 1)
        c = np.clip(cols.astype(int), 0, b0.shape[1] - 1)
        return b0[r, c].astype(float), b1[r, c].astype(float), True
    except Exception:
        return np.zeros(n), np.zeros(n), False


def relabel_session_count(relabels_dir: str) -> int:
    """Number of decisions logged this session (decisions.csv rows, minus
    header)."""
    csv_path = os.path.join(relabels_dir, "decisions.csv")
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path) as fp:
        n = sum(1 for _ in fp)
    return max(0, n - 1)


def apply_relabel(polygons, new_label: str, confidence: str,
                  corpus_path: str, relabels_dir: str,
                  tile_glob: str = DEFAULT_TILE_GLOB, warn=None) -> list:
    """Write a whole-polygon relabel to a review-format session dir.

    ``polygons`` is a list of dicts with keys class, source, tile_id,
    polygon_id (as carried in the scatter customdata). ``new_label`` is one of
    RELABEL_OPTIONS. Reuses ``scripts.review.persistence`` (DecisionLog +
    HardNegativesWriter) so the output is ingestible by the 7-class build.

    Rows carry genuine 59-band spectra: m2..m58 from the corpus, and m0/m1
    read from raster bands 1-2 of the source tile at the real pixel coords.
    When the tile img is missing/unreadable, m0/m1 fall back to 0.0, ``warn``
    (if given, e.g. st.warning) is called, and ':nom01' is appended to the
    decision log's ``layer`` value to mark the degraded rows.
    """
    from scripts.review.persistence import DecisionLog, HardNegativesWriter

    dlog = DecisionLog(os.path.join(relabels_dir, "decisions.csv"))
    hn = HardNegativesWriter(os.path.join(relabels_dir, "hard_negatives"))

    written = []
    for p in polygons:
        cls, source = p["class"], p["source"]
        tile_id, pid = p["tile_id"], p["polygon_id"]
        sub = read_polygon_rows(corpus_path, cls, source, tile_id, pid)
        n = len(sub)
        if n == 0:
            continue

        rows_idx = sub["pixel_row"].to_numpy(dtype=np.int64)
        cols_idx = sub["pixel_col"].to_numpy(dtype=np.int64)

        spectra = np.zeros((n, 59), dtype=float)
        for i in range(2, 59):
            spectra[:, i] = sub[f"m{i}"].to_numpy(dtype=float)
        m0, m1, ok = read_m0_m1(tile_id, rows_idx, cols_idx, tile_glob)
        spectra[:, 0] = m0
        spectra[:, 1] = m1
        layer = source if ok else f"{source}:nom01"
        if not ok and warn is not None:
            warn(f"Tile img for {tile_id} not found/readable — m0/m1 set to "
                 f"0.0 for polygon {tile_id}/{pid}.")

        uid = polygon_uid(cls, source, tile_id, pid)
        if new_label == "discard":
            corrected, decision = None, "reject"
        else:
            corrected, decision = new_label, "reassign"

        hn.append_polygon(
            tile_id=str(tile_id), polygon_uid=uid, rows=rows_idx, cols=cols_idx,
            spectra=spectra, predicted_class=cls, corrected_class=corrected,
            confidence=confidence)

        dlog.append({
            "source_gpkg": "ndviz", "layer": layer, "polygon_uid": uid,
            "tile_id": str(tile_id), "predicted_class": cls,
            "decision": decision, "corrected_class": (corrected or ""),
            "n_pixels": n, "area_m2": 0, "confidence": confidence,
        })
        written.append({"uid": uid, "tile_id": tile_id, "polygon_id": pid,
                        "orig_class": cls, "new_label": new_label, "n_px": n,
                        "m01_ok": ok})
    return written


def stratified_subsample(df: pd.DataFrame, budget: int, seed: int = 0) -> pd.DataFrame:
    """Cap total rows at ``budget`` via a seeded per-class subsample."""
    if len(df) <= budget:
        return df
    classes = df["class"].unique()
    per = max(1, budget // max(1, len(classes)))
    parts = []
    rng = np.random.RandomState(seed)
    for cls, g in df.groupby("class", observed=True, sort=False):
        if len(g) <= per:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=per, replace=False)
            parts.append(g.loc[idx])
    out = pd.concat(parts)
    if len(out) > budget:
        out = out.sample(n=budget, random_state=seed)
    return out


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def main():
    import streamlit as st

    st.set_page_config(page_title="CRISM N-D Visualizer", layout="wide")
    st.title("CRISM label N-D Visualizer")

    @st.cache_data(show_spinner=False)
    def _load_viz(path):
        return pd.read_parquet(path)

    @st.cache_data(show_spinner=False)
    def _load_endmembers(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    @st.cache_data(show_spinner=False)
    def _load_wl(path):
        return load_wavelengths(path)

    # ---- sidebar: data sources ------------------------------------------- #
    st.sidebar.header("Data")
    viz_path = st.sidebar.text_input("viz parquet", DEFAULT_VIZ_PARQUET)
    em_path = st.sidebar.text_input("endmembers csv", DEFAULT_ENDMEMBERS)

    try:
        df_all = _load_viz(viz_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load viz parquet '{viz_path}': {exc}")
        st.stop()
        return

    endmembers = _load_endmembers(em_path)
    wl, wl_fallback = _load_wl(DEFAULT_WAVELENGTHS)
    if wl_fallback:
        st.sidebar.warning("Wavelength JSON missing — using linspace fallback.")
    if len(wl) != len(BAND_COLS):
        st.error(f"Wavelength/band length mismatch: {len(wl)} wavelengths vs "
                 f"{len(BAND_COLS)} bands — spectra x/y would be misaligned.")
        st.stop()
        return

    # ---- sidebar: filters ------------------------------------------------- #
    st.sidebar.header("Filters")
    all_classes = [c for c in CLASS_PALETTE if c in set(df_all["class"])]
    # Minerals on by default; reference clouds (bland, junk) off by default.
    default_classes = [c for c in all_classes if c not in REFERENCE_CLASSES]
    sel_classes = st.sidebar.multiselect(
        "classes", all_classes, default=default_classes, key="f_classes")

    all_sources = sorted(df_all["source"].unique().tolist())
    sel_sources = st.sidebar.multiselect(
        "sources", all_sources, default=all_sources, key="f_sources")

    include_multi = st.sidebar.checkbox("include multi-label", value=True,
                                        key="f_multi")
    conf_floor = st.sidebar.slider("confidence floor", 0.0, 1.0, 0.0, 0.05,
                                   key="f_conf")
    budget = st.sidebar.slider("max points", 1000, 30000, 15000, 1000,
                               key="f_budget")

    # ---- apply filters ---------------------------------------------------- #
    mask = df_all["class"].isin(sel_classes) & df_all["source"].isin(sel_sources)
    mask &= df_all["confidence_weight"] >= conf_floor
    if not include_multi:
        mask &= ~df_all["multi"].astype(bool)
    filt = df_all[mask]

    if filt.empty:
        st.warning("No points match the current filters.")
        st.stop()
        return

    shown = stratified_subsample(filt, budget, seed=0).reset_index(drop=True)
    M = shown[BAND_COLS].to_numpy(dtype=float)

    # customdata carrying row identity for click resolution (first col = row-id)
    row_id = np.arange(len(shown))
    cd_all = np.column_stack([
        row_id.astype(object),
        shown["tile_id"].to_numpy(dtype=object),
        shown["polygon_id"].to_numpy(dtype=object),
        shown["class"].to_numpy(dtype=object),
        shown["source"].to_numpy(dtype=object),
        shown["confidence_weight"].to_numpy(dtype=object),
    ])

    # ---- sidebar: band exclusion ----------------------------------------- #
    # 1 um detector-overlap bands (BAD_BAND_RANGES_NM → m16-m19) are excluded
    # from the angle + PCA/random math, matching sam_endmembers. Toggle drives
    # the module-level exclusion (so spectral_angles_deg follows automatically)
    # AND the PCA/random input masking below (via apply_band_mask).
    st.sidebar.header("Band exclusion")
    exclude_bands = st.sidebar.checkbox(
        "exclude 1 um overlap bands (angle + PCA)", value=True,
        key="exclude_bands")
    set_band_exclusion(exclude_bands)
    good_mask = good_band_mask(wl)  # for raw-band labelling / spectra shading
    # Good-band submatrix, masked BEFORE L2 (same order as sam_endmembers).
    Mg = apply_band_mask(M)
    Mgn = _l2_normalize(Mg)

    # ---- sidebar: projection --------------------------------------------- #
    st.sidebar.header("Projection")
    proj_mode = st.sidebar.radio(
        "mode", ["PCA", "Raw bands", "Random projection"], key="proj_mode")
    # 3-D plotly has NO selection support (on_select never fires for scatter3d),
    # so clicking/lassoing points requires a 2-D chart of the first two axes.
    pick2d = st.sidebar.checkbox(
        "2-D projection (click/lasso enabled)", value=False, key="pick2d")

    axis_titles = ["x", "y", "z"]
    if proj_mode == "PCA":
        from sklearn.decomposition import PCA
        ncomp = min(N_PCA, Mg.shape[0], Mg.shape[1])
        pca = PCA(n_components=ncomp)
        T = pca.fit_transform(Mgn)
        # Pad the transform to >=3 cols so the axis selectors always have 3.
        if T.shape[1] < 3:
            T = np.pad(T, ((0, 0), (0, 3 - T.shape[1])))
        evr = list(pca.explained_variance_ratio_) + [0.0] * 3
        pc_labels = [f"PC{i + 1} ({evr[i] * 100:.1f}%)" for i in range(T.shape[1])]
        pc_opts = list(range(T.shape[1]))
        pc_fmt = lambda i: pc_labels[i]
        # The component CHOICE just indexes columns of the (already fit)
        # transform — changing it does NOT refit the PCA. Defaults PC1/2/3,
        # clamped when fewer components are available.
        sel_pcs = []
        for axis, default, key in zip("XYZ", (0, 1, 2),
                                      ("pca_x", "pca_y", "pca_z")):
            sel_pcs.append(st.sidebar.selectbox(
                f"PC for {axis} axis", pc_opts,
                index=min(default, len(pc_opts) - 1),
                format_func=pc_fmt, key=key))
        coords = T[:, sel_pcs]
        axis_titles = [pc_labels[i] for i in sel_pcs]
    elif proj_mode == "Raw bands":
        opts = list(range(len(BAND_COLS)))
        # all bands selectable; overlap bands flagged so the user knows.
        fmt = lambda i: (f"{BAND_COLS[i]} ({wl[i]:.0f} nm)"
                         + ("" if good_mask[i] else " (excluded band)"))
        cx = st.sidebar.selectbox("band X", opts, index=0, format_func=fmt,
                                  key="raw_x")
        cy = st.sidebar.selectbox("band Y", opts, index=min(20, len(opts) - 1),
                                  format_func=fmt, key="raw_y")
        cz = st.sidebar.selectbox("band Z", opts, index=len(opts) - 1,
                                  format_func=fmt, key="raw_z")
        coords = np.column_stack([M[:, cx], M[:, cy], M[:, cz]])
        axis_titles = [fmt(cx), fmt(cy), fmt(cz)]
    else:  # Random projection
        if "rand_seed" not in st.session_state:
            st.session_state["rand_seed"] = 42
        if st.sidebar.button("🔀 Shuffle projection"):
            st.session_state["rand_seed"] = int(
                np.random.SeedSequence().generate_state(1)[0])
        seed = st.session_state["rand_seed"]
        rng = np.random.default_rng(seed)
        # random orthonormal projection over the GOOD bands only.
        G = rng.standard_normal((Mg.shape[1], 3))
        Q, _ = np.linalg.qr(G)
        coords = Mgn @ Q
        axis_titles = [f"rand{i + 1} (seed {seed})" for i in range(3)]

    # ---- sidebar: colour -------------------------------------------------- #
    st.sidebar.header("Colour")
    color_mode = st.sidebar.radio(
        "colour by",
        ["class", "source", "confidence_weight", "angle to endmember"],
        key="color_mode")

    # endmember options
    em_labels, em_lookup = [], {}
    if endmembers is not None and not endmembers.empty:
        for _, r in endmembers.iterrows():
            lbl = f"{r['class']}/{r['kind']}"
            # disambiguate duplicates
            base = lbl
            k = 1
            while lbl in em_lookup:
                k += 1
                lbl = f"{base}#{k}"
            em_labels.append(lbl)
            em_lookup[lbl] = r[BAND_COLS].to_numpy(dtype=float)

    # ---- header metrics --------------------------------------------------- #
    c1, c2, c3 = st.columns(3)
    c1.metric("points shown / total", f"{len(shown):,} / {len(filt):,}")
    c2.metric("classes", str(shown["class"].nunique()))
    c3.metric("sources", str(shown["source"].nunique()))

    # ---- build scatter (2-D pick mode, or 3-D exploration) --------------- #
    if pick2d:
        st.caption("2-D pick mode: click a point, or drag a box / lasso, to "
                   "plot those pixels' spectra in the panel below.")
    else:
        st.caption("3-D exploration view — plotly has no 3-D click support, so "
                   "switch to 2-D pick mode (sidebar) to click/lasso points. "
                   "Hover shows each point's identity.")

    fig = go.Figure()
    marker_base = dict(size=5 if pick2d else 3, opacity=0.7)
    active_em_label, active_em_vec = None, None
    hovertemplate = ("tile %{customdata[1]} / poly %{customdata[2]}<br>"
                     "%{customdata[3]} (%{customdata[4]})<br>"
                     "conf %{customdata[5]:.2f}<extra></extra>")

    def add_pts(sel, name, marker, cd, showlegend=True):
        """Append a trace (2-D Scattergl or 3-D Scatter3d) for a point subset."""
        x, y = coords[sel, 0], coords[sel, 1]
        if pick2d:
            fig.add_trace(go.Scattergl(
                x=x, y=y, mode="markers", name=name, customdata=cd,
                marker=marker, hovertemplate=hovertemplate,
                showlegend=showlegend))
        else:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=coords[sel, 2], mode="markers", name=name,
                customdata=cd, marker=marker, hovertemplate=hovertemplate,
                showlegend=showlegend))

    all_sel = np.ones(len(shown), dtype=bool)
    if color_mode == "class":
        for cls in shown["class"].unique():
            m = (shown["class"] == cls).to_numpy()
            add_pts(m, str(cls), dict(color=CLASS_PALETTE.get(cls, "#000000"),
                                      **marker_base), cd_all[m])
    elif color_mode == "source":
        srcs = list(shown["source"].unique())
        for i, src in enumerate(srcs):
            m = (shown["source"] == src).to_numpy()
            add_pts(m, str(src), dict(color=SOURCE_PALETTE[i % len(SOURCE_PALETTE)],
                                      **marker_base), cd_all[m])
    elif color_mode == "confidence_weight":
        add_pts(all_sel, "confidence",
                dict(color=shown["confidence_weight"].to_numpy(),
                     colorscale="Cividis", colorbar=dict(title="conf"),
                     cmin=0.0, cmax=1.0, **marker_base), cd_all)
    else:  # angle to endmember
        if not em_labels:
            st.info("No endmembers.csv found — angle colouring unavailable.")
            add_pts(all_sel, "points", dict(color="#4363d8", **marker_base), cd_all)
        else:
            em_sel = st.sidebar.selectbox("endmember", em_labels, key="em_color")
            active_em_label, active_em_vec = em_sel, em_lookup[em_sel]
            angles = spectral_angles_deg(M, em_lookup[em_sel])
            amax = float(np.nanmax(angles)) if np.isfinite(angles).any() else 90.0
            thresh = st.sidebar.slider("grey-out angle >= (deg)", 0.0,
                                       max(1.0, round(amax, 1)),
                                       max(1.0, round(amax, 1)), 0.5,
                                       key="em_thresh")
            keep = angles < thresh
            if keep.any():
                add_pts(keep, f"angle to {em_sel}",
                        dict(color=angles[keep], colorscale="Viridis",
                             colorbar=dict(title="angle (deg)"),
                             cmin=0.0, cmax=thresh, **marker_base), cd_all[keep])
            if (~keep).any():
                add_pts(~keep, "above threshold",
                        dict(color=GREY, **marker_base), cd_all[~keep])

    if pick2d:
        fig.update_layout(
            height=650, showlegend=True, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
            legend=dict(itemsizing="constant"), dragmode="lasso")
        event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun",
            selection_mode=("points", "box", "lasso"), key="scatter2d")
    else:
        fig.update_layout(
            height=650, showlegend=True, margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
                       zaxis_title=axis_titles[2]),
            legend=dict(itemsizing="constant"))
        st.plotly_chart(fig, use_container_width=True, key="scatter3d")
        event = None

    # ---- resolve + accumulate clicked pixels ----------------------------- #
    if "clicked" not in st.session_state:
        st.session_state["clicked"] = []  # list of row-dict records
    clicked_now = resolve_clicked_points(event, shown)
    if not clicked_now.empty:
        for _, r in clicked_now.iterrows():
            st.session_state["clicked"].append(r.to_dict())
        # dedupe by identity keeping last occurrence, cap at last 8
        # (a lasso can contribute many points at once).
        seen, deduped = set(), []
        for rec in reversed(st.session_state["clicked"]):
            key = (rec.get("tile_id"), rec.get("polygon_id"),
                   rec.get("class"), rec.get("source"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        st.session_state["clicked"] = list(reversed(deduped))[-8:]
    clicked_recs = st.session_state["clicked"]
    if clicked_recs and st.button("Clear selection"):
        st.session_state["clicked"] = []
        clicked_recs = []

    # ---- spectra panel ---------------------------------------------------- #
    st.subheader("Spectra (mean +/-1 sigma of current filter)")
    em_overlay = []
    if em_labels:
        em_overlay = st.multiselect("overlay endmembers", em_labels,
                                    default=[], key="em_overlay")

    sfig = go.Figure()
    for cls in filt["class"].unique():
        g = filt[filt["class"] == cls][BAND_COLS].to_numpy(dtype=float)
        mean = np.nanmean(g, axis=0)
        std = np.nanstd(g, axis=0)
        col = CLASS_PALETTE.get(cls, "#000000")
        # +/-1 sigma band
        sfig.add_trace(go.Scatter(
            x=np.concatenate([wl, wl[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself", fillcolor=_rgba(col, 0.15),
            line=dict(width=0), hoverinfo="skip", showlegend=False))
        sfig.add_trace(go.Scatter(
            x=wl, y=mean, mode="lines", name=str(cls),
            line=dict(color=col, width=2)))

    for lbl in em_overlay:
        sfig.add_trace(go.Scatter(
            x=wl, y=em_lookup[lbl], mode="lines", name=lbl,
            line=dict(dash="dash", width=2)))

    # clicked-pixel spectra (solid, on top)
    caption_lines = []
    for rec in clicked_recs:
        spec = np.array([rec[c] for c in BAND_COLS], dtype=float)
        cls = rec.get("class")
        lbl = f"{rec.get('tile_id')}/p{rec.get('polygon_id')} {cls} ({rec.get('source')})"
        sfig.add_trace(go.Scatter(
            x=wl, y=spec, mode="lines", name=lbl,
            line=dict(color=CLASS_PALETTE.get(cls, "#000000"), width=2.5)))
        meta = (f"**{lbl}** — weight {float(rec.get('confidence_weight', 0)):.2f}")
        if active_em_vec is not None:
            ang = spectral_angles_deg(spec[None, :], active_em_vec)[0]
            meta += f", angle to {active_em_label} = {ang:.1f} deg"
        caption_lines.append(meta)

    # Shade the detector-overlap region(s) — data is always plotted, but the
    # band is flagged as excluded from the angle/PCA math.
    for k, (lo, hi) in enumerate(BAD_BAND_RANGES_NM):
        sfig.add_vrect(
            x0=lo, x1=hi, fillcolor="grey", opacity=0.18, line_width=0,
            annotation_text=("detector overlap (excluded from angle/PCA math)"
                             if k == 0 else ""),
            annotation_position="top left",
            annotation=dict(font_size=10))

    sfig.update_layout(
        height=380, xaxis_title="wavelength (nm)", yaxis_title="reflectance",
        xaxis=dict(range=[450, 2500]), legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(sfig, use_container_width=True)

    if caption_lines:
        st.markdown("**Clicked pixels:**  \n" + "  \n".join(caption_lines))

    # ---- relabel panel (whole polygon) ----------------------------------- #
    corpus_path = DEFAULT_CORPUS
    relabels_dir = DEFAULT_RELABELS_DIR

    @st.cache_data(show_spinner=False)
    def _poly_count(cpath, cls, source, tile_id, pid):
        try:
            return len(read_polygon_rows(cpath, cls, source, tile_id, pid))
        except Exception:
            return 0

    if clicked_recs:
        st.subheader("Relabel")
        st.caption("Whole polygons of the clicked pixels. Deselect any you "
                   "don't want to relabel, choose a new label + confidence, "
                   "then Apply.")
        # dedupe clicked pixels to unique source polygons
        polys = {}
        for rec in clicked_recs:
            key = (rec.get("class"), rec.get("source"), rec.get("tile_id"),
                   rec.get("polygon_id"))
            polys[key] = rec

        chosen = []
        for (cls, source, tile_id, pid) in polys:
            n = _poly_count(corpus_path, cls, source, tile_id, pid)
            cb_key = f"relabel_pick_{tile_id}_{pid}_{cls}_{source}"
            if st.checkbox(f"{tile_id}/{pid} {cls} ({source}, {n} px in corpus)",
                           value=True, key=cb_key):
                chosen.append({"class": cls, "source": source,
                               "tile_id": tile_id, "polygon_id": pid})

        new_label = st.selectbox("relabel as:", RELABEL_OPTIONS,
                                 key="relabel_as")
        conf = st.radio("confidence", ["High", "Moderate", "Low"], index=0,
                        horizontal=True, key="relabel_conf")

        if st.button("Apply relabel"):
            if not chosen:
                st.warning("No polygons selected.")
            else:
                written = apply_relabel(chosen, new_label, conf, corpus_path,
                                        relabels_dir, warn=st.warning)
                if written:
                    summary = ", ".join(
                        f"{w['tile_id']}/{w['polygon_id']} {w['orig_class']}"
                        f"→{w['new_label']} ({w['n_px']} px)" for w in written)
                    st.success(f"Relabeled {len(written)} polygon(s) → "
                               f"{relabels_dir}: {summary}")
                    st.session_state["clicked"] = []
                else:
                    st.warning("Selected polygons had no rows in the corpus.")

    st.sidebar.metric("relabels this session",
                      relabel_session_count(DEFAULT_RELABELS_DIR))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


if __name__ == "__main__":
    main()
