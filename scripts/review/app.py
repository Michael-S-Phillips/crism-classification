"""Streamlit app for reviewing MC13 polygon predictions and harvesting
confirmed/hard-negative training pixels.

Run:
  conda run -n crism streamlit run scripts/review/app.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.review.polygon_queue import PolygonQueue
from scripts.review.loader import load_polygon_pixels
from scripts.review.persistence import (
    DecisionLog, ConfirmedPixelsWriter, HardNegativesWriter,
)

DEFAULT_GPKG_DIR = '/mnt/mrdr/crism_classification/data/vector_mc13_relabeled'
DEFAULT_MRRAL_DIR = '/mnt/mrdr/mc13'
DEFAULT_OUT_DIR = '/mnt/mrdr/crism_classification/data/mc13_review'
# Wavelengths are tile-invariant across mc13 — the contrastive run wrote the
# sidecar, the relabeled run did not. Both gpkg sources point at the same
# 59-band mrral cubes, so this file is the right reference regardless of which
# vector dir we're reviewing.
DEFAULT_WAVELENGTHS = '/mnt/mrdr/crism_classification/data/vector_mc13_contrastive/vector_mc13_contrastive_wavelengths.json'
TARGET_PIXELS_PER_CLASS = 30000

MINERALS = ['olivine', 'lcp', 'hcp']


# ---- pure helpers (covered by tests) ---------------------------------------

def compute_progress(decisions_csv: str, mineral: str,
                     target_pixels: int) -> dict:
    """Aggregate decisions.csv for one mineral."""
    if not os.path.exists(decisions_csv):
        return dict(confirmed_pixels=0, reviewed=0, confirm_count=0,
                    reject_count=0, skip_count=0,
                    target_pixels=target_pixels, fraction=0.0,
                    target_reached=False)
    df = pd.read_csv(decisions_csv)
    df = df[df['predicted_class'] == mineral]
    conf = df[df['decision'] == 'confirm']
    rej = df[df['decision'] == 'reject']
    skip = df[df['decision'] == 'skip']
    pixels = int(conf['n_pixels'].fillna(0).sum())
    return dict(
        confirmed_pixels=pixels,
        reviewed=len(df),
        confirm_count=len(conf),
        reject_count=len(rej),
        skip_count=len(skip),
        target_pixels=target_pixels,
        fraction=pixels / target_pixels if target_pixels else 0.0,
        target_reached=pixels >= target_pixels,
    )


def make_spectrum_figure(spectra: np.ndarray,
                          wavelengths_nm: np.ndarray) -> go.Figure:
    """Mean + ±1σ envelope. No band markers."""
    fig = go.Figure()
    if spectra.shape[0] == 0:
        fig.update_layout(title='no interior pixels', height=350)
        return fig
    mean = spectra.mean(axis=0)
    std = spectra.std(axis=0)
    upper = mean + std
    lower = mean - std
    fig.add_trace(go.Scatter(
        x=wavelengths_nm, y=upper, mode='lines',
        line=dict(width=0), name='envelope_upper',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=wavelengths_nm, y=lower, mode='lines',
        line=dict(width=0), name='envelope_lower',
        fill='tonexty', fillcolor='rgba(100,100,200,0.18)',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=wavelengths_nm, y=mean, mode='lines',
        line=dict(width=2, color='royalblue'), name='mean',
    ))
    fig.update_layout(
        xaxis_title='wavelength (nm)', yaxis_title='reflectance',
        height=400, margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
    )
    return fig


# ---- streamlit glue --------------------------------------------------------

def _load_wavelengths(path: str) -> np.ndarray:
    with open(path) as fp:
        d = json.load(fp)
    arr = np.asarray(d['wavelengths_nm'], dtype=float)
    if arr.size < 59:
        raise ValueError(
            f'{path} has {arr.size} wavelengths; mrral cubes have 59 bands')
    return arr[:59]


def main():
    # Import streamlit lazily so pytest imports of helpers don't pull it in.
    import streamlit as st

    st.set_page_config(page_title='MC13 polygon review', layout='wide')
    st.title('MC13 polygon review')

    # Sidebar config
    gpkg_dir = st.sidebar.text_input('gpkg dir', DEFAULT_GPKG_DIR)
    mrral_dir = st.sidebar.text_input('mrral tile dir', DEFAULT_MRRAL_DIR)
    out_dir = st.sidebar.text_input('output dir', DEFAULT_OUT_DIR)
    decisions_csv = os.path.join(out_dir, 'decisions.csv')
    confirmed_pq = os.path.join(out_dir, 'confirmed_pixels.parquet')
    hardneg_pq = os.path.join(out_dir, 'hard_negatives.parquet')

    # Mineral selector
    mineral = st.radio('mineral', MINERALS, horizontal=True,
                        index=MINERALS.index(st.session_state.get('mineral', 'hcp')))
    st.session_state['mineral'] = mineral

    # Progress bar
    prog = compute_progress(decisions_csv, mineral, TARGET_PIXELS_PER_CLASS)
    col1, col2 = st.columns([3, 1])
    col1.progress(min(1.0, prog['fraction']),
                  text=f"{prog['confirmed_pixels']:,} / {TARGET_PIXELS_PER_CLASS:,} confirmed {mineral} pixels")
    col2.metric('reviewed', prog['reviewed'],
                f"+{prog['confirm_count']} -{prog['reject_count']} ~{prog['skip_count']}")

    if prog['target_reached']:
        st.success(f"30k reached for {mineral}. Switch mineral above or keep reviewing for more headroom.")

    # Queue
    gpkg_path = os.path.join(gpkg_dir, f'{mineral}.gpkg')
    queue_key = f'queue::{gpkg_path}::{mineral}'
    if st.session_state.get('queue_key') != queue_key:
        st.session_state['queue_key'] = queue_key
        st.session_state['queue_iter'] = iter(PolygonQueue(
            gpkg_path=gpkg_path, mineral=mineral, decisions_csv=decisions_csv,
        ))
        st.session_state['current_item'] = None
        st.session_state['current_bundle'] = None

    # Advance to next polygon
    def _advance():
        try:
            st.session_state['current_item'] = next(st.session_state['queue_iter'])
            item = st.session_state['current_item']
            st.session_state['current_bundle'] = load_polygon_pixels(
                geometry=item.geometry, tile_id=item.tile_id,
                mrral_dir=mrral_dir,
            )
        except StopIteration:
            st.session_state['current_item'] = None
            st.session_state['current_bundle'] = None

    if st.session_state.get('current_item') is None:
        _advance()

    item = st.session_state.get('current_item')
    bundle = st.session_state.get('current_bundle')

    if item is None:
        st.info('No more polygons in this queue.')
        return

    # Card
    n_px = bundle.spectra.shape[0] if bundle is not None else 0
    st.markdown(
        f"**tile** `{item.tile_id}` · **layer** `{item.layer}` · "
        f"**polygon_uid** `{item.polygon_uid}` · **n_pixels** {n_px} · "
        f"**pred_prob** {item.pred_prob:.2f}"
    )
    wavelengths = _load_wavelengths(DEFAULT_WAVELENGTHS)
    st.plotly_chart(make_spectrum_figure(bundle.spectra, wavelengths),
                     use_container_width=True)

    # Decision buttons + corrected-class dropdown
    corrected = st.selectbox(
        'if rejected, actually:',
        options=['', 'olivine', 'lcp', 'hcp', 'other'],
        index=0,
    )
    b1, b2, b3 = st.columns(3)
    log = DecisionLog(decisions_csv)
    confirmed_writer = ConfirmedPixelsWriter(confirmed_pq)
    hardneg_writer = HardNegativesWriter(hardneg_pq)

    def _record(decision: str):
        log.append(dict(
            source_gpkg=item.source_gpkg, layer=item.layer,
            polygon_uid=item.polygon_uid, tile_id=item.tile_id,
            predicted_class=mineral, decision=decision,
            corrected_class=(corrected if decision == 'reject' else ''),
            n_pixels=n_px, area_m2=item.area_m2,
        ))
        if decision == 'confirm' and bundle is not None and n_px > 0:
            confirmed_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                label_class=mineral,
            )
            confirmed_writer.flush()
        elif decision == 'reject' and bundle is not None and n_px > 0:
            hardneg_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                predicted_class=mineral,
                corrected_class=(corrected or None),
            )
            hardneg_writer.flush()
        _advance()
        st.rerun()

    # if/elif so an accidental removal of st.rerun() inside _record can't
    # double-fire a Confirm + Reject in the same script run.
    if b1.button('Confirm', type='primary', use_container_width=True):
        _record('confirm')
    elif b2.button('Reject', use_container_width=True):
        _record('reject')
    elif b3.button('Skip', use_container_width=True):
        _record('skip')


if __name__ == '__main__':
    main()
