"""Streamlit app for reviewing + relabeling olivine-only polygons suspected of
pyroxene (usually HCP) contamination.

Reads spectra/labels READ-ONLY from the categorized GeoPackages; writes new
labels to a SEPARATE CSV (data/olivine_relabels.csv). Original gpkgs are never
modified. Resumable — reloads prior entries on start.

Suspect set: all olivine-only polygons, ranked by 2 um pyroxene band depth
(descending), so the most-likely-contaminated come first.

Run:
  conda run -n crism streamlit run scripts/relabel_app.py
"""
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

GPKG_DIR = '/mnt/mrdr/categorized_mineral_units'
RELABEL_CSV = '/mnt/mrdr/crism_classification/data/olivine_relabels.csv'

# 2 um pyroxene band: continuum 1.81 -> 2.46 um, absorption minimum ~2.32 um
WL_LEFT, WL_MIN, WL_RIGHT = 1809.0, 2318.0, 2457.0


def parse_arr(s):
    """Parse a gpkg spectrum/wavelength string (comma- or space-separated)."""
    if not isinstance(s, str):
        return np.array([])
    s = s.strip().strip('[]').replace('\n', ' ')
    for sep in (',', ' '):
        try:
            a = np.fromstring(s, sep=sep)
            if a.size > 5:
                return a
        except Exception:
            pass
    return np.array([])


def is_olivine_only(cat):
    c = (cat or '').lower()
    return ('olivine' in c) and not any(
        k in c for k in ['lcp', 'hcp', 'pyrox', 'plag'])


def band_depth_2um(wvl, spec):
    if len(wvl) != len(spec) or len(wvl) < 10:
        return np.nan
    def nearest(w):
        return int(np.argmin(np.abs(wvl - w)))
    li, ci, ri = nearest(WL_LEFT), nearest(WL_MIN), nearest(WL_RIGHT)
    if spec[li] <= 0 or spec[ri] <= 0:
        return np.nan
    frac = (wvl[ci] - wvl[li]) / (wvl[ri] - wvl[li])
    cont = spec[li] + frac * (spec[ri] - spec[li])
    return float(1.0 - spec[ci] / cont) if cont > 0 else np.nan


@st.cache_data(show_spinner='Scanning categorized GeoPackages…')
def load_suspects():
    rows = []
    for gp in sorted(glob.glob(os.path.join(GPKG_DIR, 'T*.gpkg'))):
        tile = os.path.basename(gp)[:-5]
        try:
            g = gpd.read_file(gp)
        except Exception:
            continue
        for _, r in g.iterrows():
            cat = r.get('Category', '') or ''
            if not is_olivine_only(cat):
                continue
            wvl = parse_arr(r.get('wvl'))
            num = parse_arr(r.get('Spectrum Mean'))
            ratio = parse_arr(r.get('Ratio Spectrum'))
            rows.append({
                'tile': tile,
                'polygon': str(r.get('Polygon Number')),
                'category': cat,
                'd2um': band_depth_2um(wvl, num),
                'wvl': wvl, 'num': num, 'ratio': ratio,
            })
    df = pd.DataFrame(rows)
    return df.sort_values('d2um', ascending=False, na_position='last').reset_index(drop=True)


def load_relabels():
    if os.path.exists(RELABEL_CSV):
        return pd.read_csv(RELABEL_CSV, dtype={'polygon': str})
    return pd.DataFrame(columns=['tile', 'polygon', 'old_category', 'new_label', 'd2um'])


def save_relabel(tile, polygon, old_cat, new_label, d2um):
    df = load_relabels()
    df = df[~((df.tile == tile) & (df.polygon == str(polygon)))]
    df = pd.concat([df, pd.DataFrame([{
        'tile': tile, 'polygon': str(polygon), 'old_category': old_cat,
        'new_label': new_label, 'd2um': d2um}])], ignore_index=True)
    os.makedirs(os.path.dirname(RELABEL_CSV), exist_ok=True)
    df.to_csv(RELABEL_CSV, index=False)


def main():
    st.set_page_config(page_title='Olivine relabeling', layout='wide')
    st.title('Olivine-only polygon relabeling — pyroxene contamination review')

    suspects = load_suspects()
    relabels = load_relabels()
    done_keys = set(zip(relabels.tile, relabels.polygon.astype(str)))
    n = len(suspects)

    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    # --- Sidebar: navigation + progress ---
    with st.sidebar:
        st.metric('Suspect polygons', n)
        st.metric('Relabeled so far', len(relabels))
        only_todo = st.checkbox('Hide already-relabeled', value=False)
        st.caption(f'Writing to {RELABEL_CSV} (originals untouched)')
        if st.button('⟵ Prev'):
            st.session_state.idx = max(0, st.session_state.idx - 1)
        if st.button('Next ⟶'):
            st.session_state.idx = min(n - 1, st.session_state.idx + 1)
        jump = st.number_input('Jump to #', 0, max(0, n - 1), st.session_state.idx)
        if jump != st.session_state.idx:
            st.session_state.idx = int(jump)

    # Optionally skip already-done
    if only_todo:
        todo = [i for i in range(n)
                if (suspects.iloc[i].tile, suspects.iloc[i].polygon) not in done_keys]
        if not todo:
            st.success('All suspect polygons have been relabeled.'); return
        if st.session_state.idx not in todo:
            st.session_state.idx = todo[0]

    i = st.session_state.idx
    row = suspects.iloc[i]
    key = (row.tile, row.polygon)
    prior = relabels[(relabels.tile == row.tile) & (relabels.polygon == row.polygon)]

    c1, c2 = st.columns([3, 2])
    with c1:
        mode = st.radio('Spectrum', ['Ratio', 'Numerator'], horizontal=True)
        spec = row.ratio if mode == 'Ratio' else row.num
        wvl = row.wvl
        fig, ax = plt.subplots(figsize=(8, 4))
        if len(wvl) == len(spec) and len(wvl) > 0:
            ax.plot(wvl, spec, lw=1.5)
            ax.axvspan(1809, 2457, color='orange', alpha=0.08, label='2µm pyroxene region')
            ax.axvline(2318, color='red', ls=':', alpha=0.6, label='~2.3µm min')
            ax.set_xlabel('wavelength (nm)')
            ax.set_ylabel('ratio' if mode == 'Ratio' else 'reflectance')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'spectrum unavailable', ha='center')
        ax.set_title(f'{row.tile}  polygon {row.polygon}')
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader(f'Polygon {i+1} / {n}')
        st.write(f'**Tile:** {row.tile}  |  **Polygon #:** {row.polygon}')
        st.write(f'**Current label:** `{row.category}`')
        st.write(f'**2µm pyroxene depth:** {row.d2um:.4f}'
                 if row.d2um == row.d2um else '**2µm depth:** n/a')
        if len(prior):
            st.info(f"Already relabeled → `{prior.iloc[0].new_label}`")
        default = prior.iloc[0].new_label if len(prior) else row.category
        new_label = st.text_input('New label', value=str(default), key=f'lbl_{i}')
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button('💾 Save & Next', type='primary'):
                save_relabel(row.tile, row.polygon, row.category, new_label,
                             row.d2um)
                st.session_state.idx = min(n - 1, i + 1)
                st.rerun()
        with sc2:
            if st.button('Skip ⟶'):
                st.session_state.idx = min(n - 1, i + 1)
                st.rerun()
        st.caption('Common labels: olivine, olivine+hcp, hcp, lcp, '
                   'pyroxene, mixed, uncertain')


if __name__ == '__main__':
    main()
