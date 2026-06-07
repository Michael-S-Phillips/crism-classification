"""Streamlit app for reviewing MC13 polygon predictions and harvesting
confirmed/hard-negative training pixels.

Run:
  conda run -n crism streamlit run scripts/review/app.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

# When launched via ``streamlit run scripts/review/app.py``, only the script's
# directory is on sys.path — not the project root. Add the project root so the
# ``scripts.review.*`` imports below resolve. (pytest adds the root for us, so
# this is only needed for the streamlit entrypoint path.)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.review.polygon_queue import PolygonQueue
from scripts.review.loader import load_polygon_pixels, load_thumbnail
from scripts.review.persistence import (
    DecisionLog, ConfirmedPixelsWriter, HardNegativesWriter,
)

DEFAULT_GPKG_DIR = '/mnt/mrdr/crism_classification/data/vector_mc13_relabeled'
DEFAULT_MRRAL_DIR = '/mnt/mrdr/mc13'
DEFAULT_OUT_DIR = '/mnt/mrdr/crism_classification/data/mc13_review'
# How many recent decisions to pull into the Previous-button history on
# startup. Each rehydrated polygon is metadata-only; its spectrum is
# loaded on demand when the user actually navigates back to it.
HISTORY_REHYDRATE_N = 30
# Wavelengths are tile-invariant across mc13 — the contrastive run wrote the
# sidecar, the relabeled run did not. Both gpkg sources point at the same
# 59-band mrral cubes, so this file is the right reference regardless of which
# vector dir we're reviewing.
DEFAULT_WAVELENGTHS = '/mnt/mrdr/crism_classification/data/vector_mc13_contrastive/vector_mc13_contrastive_wavelengths.json'
TARGET_PIXELS_PER_CLASS = 30000

MINERALS = ['olivine', 'lcp', 'hcp']


# ---- pure helpers (covered by tests) ---------------------------------------

def _last_n_decided_uids(decisions_csv: str, mineral: str, n: int,
                          source_gpkg: Optional[str] = None) -> list[str]:
    """Most recent N decisions for ``mineral`` (and optionally ``source_gpkg``).

    Source-gpkg filtering matters because the same uid format can collide
    across different vector products (e.g. `vector_mc13_relabeled/hcp.gpkg`
    vs `vector_mc13_v3_denoising/hcp.gpkg` — same tile_id+layer+idx pattern,
    completely different polygons). Used to rehydrate the Previous-button
    history. Multiple decisions for the same uid collapse to the latest.
    """
    if not os.path.exists(decisions_csv):
        return []
    df = pd.read_csv(decisions_csv)
    if 'predicted_class' not in df.columns or 'polygon_uid' not in df.columns:
        return []
    df = df[df['predicted_class'] == mineral]
    if source_gpkg and 'source_gpkg' in df.columns:
        df = df[df['source_gpkg'] == source_gpkg]
    if df.empty:
        return []
    df = df.drop_duplicates(subset='polygon_uid', keep='last')
    return df['polygon_uid'].astype(str).tail(n).tolist()


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

    # Queue + per-mineral history (so Previous walks back through visited polys)
    gpkg_path = os.path.join(gpkg_dir, f'{mineral}.gpkg')
    queue_key = f'queue::{gpkg_path}::{mineral}'
    if st.session_state.get('queue_key') != queue_key:
        st.session_state['queue_key'] = queue_key
        queue = PolygonQueue(
            gpkg_path=gpkg_path, mineral=mineral, decisions_csv=decisions_csv,
        )
        st.session_state['queue_iter'] = iter(queue)
        # history[i] = polygon_uid in visit order; cache[uid] = (item, bundle, thumb)
        st.session_state['history'] = []
        st.session_state['cache'] = {}
        st.session_state['cursor'] = -1
        st.session_state['current_item'] = None
        st.session_state['current_bundle'] = None
        st.session_state['current_thumb'] = None

        # Rehydrate Previous-button history from decisions.csv. Items are
        # metadata-only here; their spectra (and thumbnails) are loaded the
        # first time the user actually navigates to them via Previous.
        current_source_gpkg = (
            f'{os.path.basename(os.path.dirname(os.path.abspath(gpkg_path)))}/'
            f'{os.path.basename(gpkg_path)}')
        last_uids = _last_n_decided_uids(
            decisions_csv, mineral, HISTORY_REHYDRATE_N,
            source_gpkg=current_source_gpkg)
        if last_uids:
            items_map = queue.lookup_items(last_uids)
            for uid in last_uids:
                item = items_map.get(uid)
                if item is None:
                    continue
                st.session_state['cache'][uid] = (item, None, None)
                st.session_state['history'].append(uid)
            # Cursor points one past the end so the next _advance pulls a
            # fresh polygon from the queue (not a rehydrated one).
            st.session_state['cursor'] = len(st.session_state['history'])

    def _set_current(uid: str):
        item, bundle, thumb = st.session_state['cache'][uid]
        if bundle is None:  # rehydrated history entry — load spectra now
            try:
                bundle = load_polygon_pixels(
                    geometry=item.geometry, tile_id=item.tile_id,
                    mrral_dir=mrral_dir, source_crs=item.source_crs,
                )
                st.session_state['cache'][uid] = (item, bundle, thumb)
            except Exception as e:
                st.warning(f'failed to load spectra for {uid}: {e}')
        st.session_state['current_item'] = item
        st.session_state['current_bundle'] = bundle
        st.session_state['current_thumb'] = thumb

    def _advance():
        cur = st.session_state['cursor']
        hist = st.session_state['history']
        # If we're sitting in the middle of history (after a Previous), step
        # forward through it instead of pulling a fresh polygon from the queue.
        if cur < len(hist) - 1:
            st.session_state['cursor'] = cur + 1
            _set_current(hist[st.session_state['cursor']])
            return
        # At the live edge — pull next from the queue.
        try:
            item = next(st.session_state['queue_iter'])
            bundle = load_polygon_pixels(
                geometry=item.geometry, tile_id=item.tile_id,
                mrral_dir=mrral_dir, source_crs=item.source_crs,
            )
            # Thumbnail is lazy — only loaded when user clicks "Show context".
            uid = item.polygon_uid
            st.session_state['cache'][uid] = (item, bundle, None)
            hist.append(uid)
            st.session_state['cursor'] = len(hist) - 1
            _set_current(uid)
        except StopIteration:
            st.session_state['current_item'] = None
            st.session_state['current_bundle'] = None
            st.session_state['current_thumb'] = None

    def _load_thumb_for_current():
        item = st.session_state['current_item']
        if item is None:
            return
        uid = item.polygon_uid
        try:
            thumb = load_thumbnail(
                geometry=item.geometry, tile_id=item.tile_id,
                mrral_dir=mrral_dir, source_crs=item.source_crs,
            )
        except Exception as e:
            st.warning(f'thumbnail unavailable: {e}')
            return
        # Update both the cache and current_thumb in place. If the cache was
        # reset (e.g. mineral switch) the uid may be missing — still display
        # the thumbnail this turn, just skip the cache update.
        st.session_state['current_thumb'] = thumb
        cached = st.session_state['cache'].get(uid)
        if cached is not None:
            cached_item, cached_bundle, _ = cached
            st.session_state['cache'][uid] = (cached_item, cached_bundle, thumb)

    def _go_previous():
        if st.session_state['cursor'] > 0:
            st.session_state['cursor'] -= 1
            _set_current(st.session_state['history'][st.session_state['cursor']])

    if st.session_state.get('current_item') is None:
        _advance()

    item = st.session_state.get('current_item')
    bundle = st.session_state.get('current_bundle')

    if item is None:
        st.info('No more polygons in this queue.')
        return

    # If this polygon already has a recorded decision (re-viewing), surface it.
    log = DecisionLog(decisions_csv)
    confirmed_writer = ConfirmedPixelsWriter(confirmed_pq)
    hardneg_writer = HardNegativesWriter(hardneg_pq)
    prev_decision = log.most_recent_for(item.polygon_uid)
    is_reviewing = prev_decision is not None

    # Card header (include navigation position)
    n_px = bundle.spectra.shape[0] if bundle is not None else 0
    cursor = st.session_state['cursor']
    hist_len = len(st.session_state['history'])
    st.markdown(
        f"**tile** `{item.tile_id}` · **layer** `{item.layer}` · "
        f"**polygon_uid** `{item.polygon_uid}` · **n_pixels** {n_px} · "
        f"**pred_prob** {item.pred_prob:.2f}  ·  "
        f"_session position {cursor + 1}/{hist_len}_"
    )
    if is_reviewing:
        corr = prev_decision.get('corrected_class') or ''
        corr_part = f" (corrected: {corr})" if corr else ''
        st.warning(
            f"Already decided: **{prev_decision['decision']}**{corr_part} "
            f"at {prev_decision['ts']}. Clicking a button below will supersede this."
        )

    wavelengths = _load_wavelengths(DEFAULT_WAVELENGTHS)
    thumb = st.session_state.get('current_thumb')
    if thumb is not None:
        c_thumb, c_spec = st.columns([1, 2])
        c_thumb.image(
            thumb.rgb,
            caption='context (false-color RGB ~2.2/1.5/0.8 µm, polygon outlined red)',
            use_container_width=True,
        )
        c_spec.plotly_chart(make_spectrum_figure(bundle.spectra, wavelengths),
                             use_container_width=True)
    else:
        st.plotly_chart(make_spectrum_figure(bundle.spectra, wavelengths),
                         use_container_width=True)
        if st.button('Show context image', key='show_thumb'):
            _load_thumb_for_current()
            st.rerun()

    # Decision buttons + corrected-class dropdown
    corrected = st.selectbox(
        'if rejected, actually:',
        options=['', 'olivine', 'lcp', 'hcp', 'other'],
        index=0,
    )
    p1, b1, b2, b3 = st.columns([1, 1, 1, 1])

    def _record(decision: str):
        # Supersede any prior decision for this polygon by removing rows from
        # whichever derived parquet the old decision wrote to.
        if prev_decision is not None:
            prev = prev_decision.get('decision')
            if prev == 'confirm':
                confirmed_writer.drop_polygon(item.polygon_uid)
            elif prev == 'reject':
                hardneg_writer.drop_polygon(item.polygon_uid)
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

    # if/elif so an accidental removal of st.rerun() can't double-fire.
    if p1.button('← Previous', use_container_width=True,
                  disabled=(st.session_state['cursor'] <= 0)):
        _go_previous()
        st.rerun()
    elif b1.button('Confirm', type='primary', use_container_width=True):
        _record('confirm')
    elif b2.button('Reject', use_container_width=True):
        _record('reject')
    elif b3.button('Skip', use_container_width=True):
        _record('skip')


if __name__ == '__main__':
    main()
