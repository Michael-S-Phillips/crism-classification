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

from scripts.label_quant.sam_endmembers import _angles_to, BAND_COLS  # noqa: E402

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

# Review-app class palette (fixed).
CLASS_PALETTE = {
    "olivine": "#e6194b",
    "lcp": "#3cb44b",
    "hcp": "#4363d8",
    "plagioclase": "#f58231",
    "bland": "#aaaaaa",
    "alteration": "#cc8899",
}
MINERALS = ["olivine", "lcp", "hcp", "plagioclase", "alteration"]

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

    Reuses the normalized-dot-arccos implementation from ``sam_endmembers``.
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

    # ---- sidebar: filters ------------------------------------------------- #
    st.sidebar.header("Filters")
    all_classes = [c for c in CLASS_PALETTE if c in set(df_all["class"])]
    default_classes = [c for c in MINERALS if c in all_classes]
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
    Mn = _l2_normalize(M)

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

    # ---- sidebar: projection --------------------------------------------- #
    st.sidebar.header("Projection")
    proj_mode = st.sidebar.radio(
        "mode", ["PCA", "Raw bands", "Random projection"], key="proj_mode")

    axis_titles = ["x", "y", "z"]
    if proj_mode == "PCA":
        from sklearn.decomposition import PCA
        ncomp = min(N_PCA, M.shape[0], M.shape[1])
        pca = PCA(n_components=ncomp)
        T = pca.fit_transform(Mn)
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
        fmt = lambda i: f"{BAND_COLS[i]} ({wl[i]:.0f} nm)"
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
        G = rng.standard_normal((len(BAND_COLS), 3))
        Q, _ = np.linalg.qr(G)
        coords = Mn @ Q
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

    # ---- build 3-D scatter ----------------------------------------------- #
    st.caption("Tip: click a point in the scatter to plot that pixel's "
               "spectrum in the panel below.")
    fig = go.Figure()
    marker_base = dict(size=3, opacity=0.7)
    active_em_label, active_em_vec = None, None

    if color_mode == "class":
        for cls in shown["class"].unique():
            m = (shown["class"] == cls).to_numpy()
            fig.add_trace(go.Scatter3d(
                x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
                mode="markers", name=str(cls), customdata=cd_all[m],
                marker=dict(color=CLASS_PALETTE.get(cls, "#000000"), **marker_base)))
    elif color_mode == "source":
        srcs = list(shown["source"].unique())
        for i, src in enumerate(srcs):
            m = (shown["source"] == src).to_numpy()
            fig.add_trace(go.Scatter3d(
                x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
                mode="markers", name=str(src), customdata=cd_all[m],
                marker=dict(color=SOURCE_PALETTE[i % len(SOURCE_PALETTE)],
                            **marker_base)))
    elif color_mode == "confidence_weight":
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
            name="confidence", customdata=cd_all,
            marker=dict(color=shown["confidence_weight"].to_numpy(),
                        colorscale="Cividis", colorbar=dict(title="conf"),
                        cmin=0.0, cmax=1.0, **marker_base)))
    else:  # angle to endmember
        if not em_labels:
            st.info("No endmembers.csv found — angle colouring unavailable.")
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
                customdata=cd_all,
                marker=dict(color="#4363d8", **marker_base)))
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
                fig.add_trace(go.Scatter3d(
                    x=coords[keep, 0], y=coords[keep, 1], z=coords[keep, 2],
                    mode="markers", name=f"angle to {em_sel}",
                    customdata=cd_all[keep],
                    marker=dict(color=angles[keep], colorscale="Viridis",
                                colorbar=dict(title="angle (deg)"),
                                cmin=0.0, cmax=thresh, **marker_base)))
            if (~keep).any():
                fig.add_trace(go.Scatter3d(
                    x=coords[~keep, 0], y=coords[~keep, 1], z=coords[~keep, 2],
                    mode="markers", name="above threshold",
                    customdata=cd_all[~keep],
                    marker=dict(color=GREY, **marker_base)))

    fig.update_layout(
        height=650, showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
                   zaxis_title=axis_titles[2]),
        legend=dict(itemsizing="constant"))
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                            selection_mode="points", key="scatter3d")

    # ---- resolve + accumulate clicked pixels ----------------------------- #
    if "clicked" not in st.session_state:
        st.session_state["clicked"] = []  # list of row-dict records
    clicked_now = resolve_clicked_points(event, shown)
    if not clicked_now.empty:
        for _, r in clicked_now.iterrows():
            st.session_state["clicked"].append(r.to_dict())
        # dedupe by identity keeping last occurrence, cap at last 5
        seen, deduped = set(), []
        for rec in reversed(st.session_state["clicked"]):
            key = (rec.get("tile_id"), rec.get("polygon_id"),
                   rec.get("class"), rec.get("source"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        st.session_state["clicked"] = list(reversed(deduped))[-5:]
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

    sfig.update_layout(
        height=380, xaxis_title="wavelength (nm)", yaxis_title="reflectance",
        xaxis=dict(range=[450, 2500]), legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(sfig, use_container_width=True)

    if caption_lines:
        st.markdown("**Clicked pixels:**  \n" + "  \n".join(caption_lines))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


if __name__ == "__main__":
    main()
