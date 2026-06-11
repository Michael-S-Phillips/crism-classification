"""Streamlit app — browse polygons, compare to endmember library, curate.

Launch with:
    cd /mnt/mrdr/crism_classification
    bash scripts/launch_endmember_curator.sh
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from endmember_curator.library import (
    CLASSES,
    angles_to_library,
    load_decisions,
    load_library,
    load_polygon_spectra,
    log_decision,
    promote,
    save_library,
)


# Mineral colors — match project palette
COLORS = {
    "olivine": "#e53935",     # red
    "lcp": "#00bcd4",         # cyan
    "hcp": "#e91e63",         # magenta
    "plagioclase": "#ffd54f", # gold
}


# ---- Helpers --------------------------------------------------------

def primary_mineral(minerals: list[str]) -> str | None:
    """First of the four core classes present in the minerals list, else None."""
    if not isinstance(minerals, (list, tuple, np.ndarray)):
        return None
    for m in minerals:
        if m in CLASSES:
            return m
    return None


@st.cache_data(show_spinner="Loading polygon spectra…")
def cached_polygons() -> pd.DataFrame:
    df = load_polygon_spectra()
    df = df.copy()
    df["primary"] = df["minerals"].apply(primary_mineral)
    return df


def angle_bar_chart(angles: dict[str, float], current_class: str) -> go.Figure:
    """Horizontal bar chart of SAM angle (radians) per class endmember."""
    classes = list(angles.keys())
    vals = [angles[c] for c in classes]
    colors = [COLORS[c] if c != current_class else "#fff176" for c in classes]
    fig = go.Figure(go.Bar(
        x=vals, y=classes, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=200, margin=dict(l=8, r=8, t=18, b=8),
        title=dict(text="SAM angle to each class endmember (rad, lower = closer)", x=0.02, font=dict(size=12)),
        xaxis_title="angle (rad)",
        showlegend=False,
        plot_bgcolor="#1a1a1a", paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
    )
    return fig


def spectrum_figure(
    wvl: list[float],
    candidate: list[float],
    library: dict,
    show_classes: list[str],
    candidate_label: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wvl, y=candidate,
        mode="lines+markers",
        line=dict(color="#fff176", width=3),
        marker=dict(size=4),
        name=f"candidate — {candidate_label}",
    ))
    for cls in show_classes:
        em = library[cls]["mean"]
        fig.add_trace(go.Scatter(
            x=wvl, y=em,
            mode="lines",
            line=dict(color=COLORS[cls], width=1.8, dash="dash"),
            name=f"endmember — {cls}",
        ))
    fig.update_layout(
        height=420, margin=dict(l=8, r=8, t=8, b=42),
        xaxis_title="Wavelength (nm)", yaxis_title="Reflectance",
        plot_bgcolor="#1a1a1a", paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0.0),
    )
    return fig


# ---- App ------------------------------------------------------------

def main():
    st.set_page_config(page_title="Endmember Curator", layout="wide", initial_sidebar_state="expanded")
    st.title("Endmember curator")
    st.caption(
        "Browse labeled polygons, compare to the current endmember library, decide whether each is "
        "correctly labeled and whether it should replace the current platonic-ideal endmember."
    )

    polys = cached_polygons()
    library = load_library()
    decisions = load_decisions()

    # ---- Sidebar filters ----
    with st.sidebar:
        st.header("Filter candidates")
        cls_choice = st.selectbox(
            "Putative class",
            options=list(CLASSES),
            index=0,
            help="Show polygons whose category includes this mineral.",
        )
        pure_only = st.checkbox(
            "Pure (single-mineral) polygons only",
            value=True,
            help="Polygons whose category contains exactly one of the 4 mineral classes.",
        )
        tier_filter = st.multiselect(
            "Confidence tier(s)",
            options=["High", "Moderate", "Low", ""],
            default=["High"],
            help="Tier comes from the gpkg Category, e.g. 'plagioclase (High)'.",
        )
        min_pixels = st.number_input("Min polygon pixels", min_value=0, value=50, step=10)

        st.divider()
        sort_mode = st.radio(
            "Sort by",
            options=[
                "SAM angle to putative class (ascending)",
                "SAM angle to putative class (descending)",
                "Polygon size (largest first)",
                "Tile, then polygon number",
            ],
            index=0,
        )

        st.divider()
        skip_decided = st.checkbox(
            "Hide polygons I've already decided on", value=False,
        )

    # ---- Apply filters ----
    df = polys.copy()
    df = df[df["primary"] == cls_choice]
    if pure_only:
        df = df[df["is_pure"]]
    if tier_filter:
        df = df[df["confidence_tier"].isin(tier_filter)]
    df = df[df["n_pixels"] >= min_pixels]

    if skip_decided and len(decisions):
        df = df[~df["polygon_uid"].isin(decisions["polygon_uid"])]

    if len(df) == 0:
        st.warning("No polygons match the current filters. Loosen one of them.")
        st.stop()

    # Compute SAM angles to library for all filtered candidates
    angle_records = []
    for _, r in df.iterrows():
        spec = r["spectrum_mean"]
        a = angles_to_library(spec, library)
        angle_records.append(a)
    ang_df = pd.DataFrame(angle_records, index=df.index)
    df = df.join(ang_df.rename(columns={c: f"sam_{c}" for c in CLASSES}))

    # Sort
    putative_col = f"sam_{cls_choice}"
    if sort_mode.startswith("SAM angle to putative class (ascending)"):
        df = df.sort_values(putative_col, ascending=True, na_position="last")
    elif sort_mode.startswith("SAM angle to putative class (descending)"):
        df = df.sort_values(putative_col, ascending=False, na_position="last")
    elif sort_mode.startswith("Polygon size"):
        df = df.sort_values("n_pixels", ascending=False)
    else:
        df = df.sort_values(["tile", "polygon_number"])
    df = df.reset_index(drop=True)

    # ---- Top status bar ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("candidates after filter", f"{len(df):,}")
    c2.metric(
        f"current {cls_choice} endmember",
        library[cls_choice]["source"],
    )
    c3.metric("decisions logged", f"{len(decisions):,}")
    c4.write(
        "<small>library snapshots saved to "
        "<code>data/endmember_curator/versions/</code></small>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Pagination ----
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if st.session_state.idx >= len(df):
        st.session_state.idx = 0
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 3])
    if nav1.button("← prev", use_container_width=True, disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.rerun()
    if nav2.button("next →", use_container_width=True, disabled=st.session_state.idx >= len(df) - 1):
        st.session_state.idx += 1
        st.rerun()
    jump = nav3.number_input(
        "jump to row", min_value=0, max_value=max(0, len(df) - 1),
        value=st.session_state.idx, step=1, label_visibility="collapsed",
    )
    if jump != st.session_state.idx:
        st.session_state.idx = int(jump)
        st.rerun()
    nav4.caption(f"viewing row **{st.session_state.idx + 1} / {len(df):,}**")

    row = df.iloc[st.session_state.idx]
    wvl = row["wavelengths"]
    spec = row["spectrum_mean"]
    angles = {c: row[f"sam_{c}"] for c in CLASSES}
    closest_cls = min(angles.items(), key=lambda kv: kv[1] if not math.isnan(kv[1]) else math.inf)[0]
    matches_label = (closest_cls == cls_choice)

    # ---- Header for current polygon ----
    lab1, lab2, lab3 = st.columns(3)
    lab1.metric("polygon UID", row["polygon_uid"])
    lab2.metric("category", row["category"])
    lab3.metric(
        "closest class (by SAM)", closest_cls,
        delta=("✓ matches putative" if matches_label else "✗ differs from putative"),
        delta_color=("normal" if matches_label else "inverse"),
    )

    # ---- Spectra + angle bars side by side ----
    left, right = st.columns([3, 2])
    with left:
        show_classes = st.multiselect(
            "Overlay endmembers",
            options=list(CLASSES),
            default=[cls_choice],
            help="Choose which class endmembers to overlay on the spectrum plot.",
        )
        st.plotly_chart(
            spectrum_figure(wvl, spec, library, show_classes, candidate_label=row["polygon_uid"]),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(angle_bar_chart(angles, cls_choice), use_container_width=True)
        st.write(f"**n pixels:** {row['n_pixels']:,}  ·  **tier:** {row['confidence_tier'] or '—'}")
        st.write(f"**Closest endmember:** `{closest_cls}` (angle = {angles[closest_cls]:.4f} rad)")
        st.write(f"**Angle to putative ({cls_choice}):** {angles[cls_choice]:.4f} rad")

    st.divider()

    # ---- Decision row ----
    st.subheader("Decision")
    dec1, dec2, dec3 = st.columns([2, 2, 3])
    with dec1:
        label_correct = st.radio(
            "Label correctness",
            options=["correct", "incorrect", "uncertain"],
            index=0 if matches_label else 1,
            horizontal=True,
            key=f"correct_{row['polygon_uid']}",
        )
    with dec2:
        promote_as = st.selectbox(
            "Promote as endmember for class",
            options=["(do not promote)"] + list(CLASSES),
            index=0,
            key=f"promote_{row['polygon_uid']}",
        )
    with dec3:
        notes = st.text_area(
            "Notes", value="", height=80,
            placeholder="Free-form rationale (will be saved with the decision).",
            key=f"notes_{row['polygon_uid']}",
        )

    save_col, advance_col, info_col = st.columns([1, 1, 3])
    with save_col:
        if st.button("Save decision", type="primary", use_container_width=True):
            rec = {
                "polygon_uid": row["polygon_uid"],
                "decision": label_correct,
                "class": cls_choice,
                "category": row["category"],
                "angle_putative": float(angles[cls_choice]),
                "closest_class_by_sam": closest_cls,
                "promote_as": None if promote_as == "(do not promote)" else promote_as,
                "n_pixels": int(row["n_pixels"]),
                "notes": notes,
            }
            log_decision(rec)
            if promote_as != "(do not promote)":
                library = promote(
                    library, promote_as, row["polygon_uid"],
                    list(spec), int(row["n_pixels"]),
                )
                st.success(f"Promoted polygon {row['polygon_uid']} as new {promote_as} endmember.")
            else:
                st.success("Decision logged.")
            st.cache_data.clear()
            st.session_state.idx = min(st.session_state.idx + 1, len(df) - 1)
            st.rerun()
    with advance_col:
        if st.button("Skip", use_container_width=True):
            st.session_state.idx = min(st.session_state.idx + 1, len(df) - 1)
            st.rerun()
    info_col.caption(
        "Saving a decision writes one line to `data/endmember_curator/decisions.jsonl`. "
        "Promoting also snapshots the prior endmember to `data/endmember_curator/versions/`."
    )

    # ---- Library state browser ----
    with st.expander("Current endmember library state"):
        rows = []
        for c in CLASSES:
            e = library[c]
            rows.append({
                "class": c,
                "source": e["source"],
                "n_pixels": e.get("n_pixels", -1),
                "promoted_at": e.get("promoted_at", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
