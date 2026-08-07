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

DEFAULT_GPKG_DIR = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/vector_mc13_7cls_v3_lrscale001'
DEFAULT_MRRAL_DIR = '/Volumes/Mars_GIS/CRISM/MRDR/mc13'
DEFAULT_OUT_DIR = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/mc13_review_7cls_v3'

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gpkg_dir_choices() -> list:
    """Auto-discover dirs that actually contain per-mineral gpkgs (data/vector_*
    and reports/floor_tests/*/*), so the dropdown stays current as new vector
    products/floor tests appear — no hardcoded list to maintain."""
    import glob
    cands = {DEFAULT_GPKG_DIR}
    for pat in (os.path.join(_PROJ, 'data', 'vector_*'),
                os.path.join(_PROJ, 'reports', 'floor_tests', '*', '*')):
        for d in glob.glob(pat):
            if os.path.isdir(d) and glob.glob(os.path.join(d, '*.gpkg')):
                cands.add(d)
    return sorted(cands)


def _mrral_dir_choices() -> list:
    """The Mars-chart tile dirs under /Volumes/Mars_GIS/CRISM/MRDR (mc02..mc30)."""
    import glob
    return sorted(glob.glob('/Volumes/Mars_GIS/CRISM/MRDR/mc[0-9][0-9]')) or [DEFAULT_MRRAL_DIR]


def _out_dir_choices() -> list:
    """Known review output dirs (data/*review*), plus the standard defaults."""
    import glob
    cands = {DEFAULT_OUT_DIR,
             os.path.join(_PROJ, 'data', 'mc13_review'),
             os.path.join(_PROJ, 'data', 'cr_review')}
    for d in glob.glob(os.path.join(_PROJ, 'data', '*review*')):
        if os.path.isdir(d):
            cands.add(d)
    return sorted(cands)
# For MC11 review: gpkg dir data/vector_mc11_7cls_v3_lrscale001,
# mrral dir /Volumes/Mars_GIS/CRISM/MRDR/mc11, out dir data/mc11_review_7cls_v3.
# Default spectrum-plot wavelength window (nm). The mrral cube's first band
# (~410 nm) is frequently noisy; it stays out of the display AND out of the
# y-range computation.
PLOT_XRANGE_NM = (450.0, 2500.0)
# How many recent decisions to pull into the Previous-button history on
# startup. Each rehydrated polygon is metadata-only; its spectrum is
# loaded on demand when the user actually navigates back to it.
HISTORY_REHYDRATE_N = 30
# Cache eviction budgets. The cache holds loaded spectra + thumbnails for
# visited polygons. Per-pixel cost: rows/cols (16 B) + spectra (59 × 4 B =
# 236 B) ≈ 252 B/pixel. A 1M-pixel LCP polygon is ~250 MB; even a small
# entry-count cap (e.g. 40) can blow past 14 GB on big polygons, which is
# what was OOM-killing the app. Use a total-pixel budget instead.
MAX_CACHE_PIXELS = 1_500_000  # ~350 MB — enough for one giant LCP polygon
# Note: entry COUNT isn't capped anymore; eviction now drops only the bundle's
# pixel arrays (keeping the small item+thumb tuple), so a 1000-entry history
# with all-evicted bundles is still only a few MB of metadata.
MAX_CACHE_ENTRIES = 9999
# Wavelengths are tile-invariant across mc13 — the contrastive run wrote the
# sidecar, the relabeled run did not. Both gpkg sources point at the same
# 59-band mrral cubes, so this file is the right reference regardless of which
# vector dir we're reviewing.
DEFAULT_WAVELENGTHS = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/vector_mc13_6cls/vector_mc13_6cls_wavelengths.json'
TARGET_PIXELS_PER_CLASS = 30000

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']


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


def build_queue_dataframe(gpkg_path: str, mineral: str,
                           decisions_csv: str) -> pd.DataFrame:
    """Enumerate every polygon for the (gpkg, mineral) pair in queue order,
    annotating each with its current decision status from decisions.csv.

    Unlike the regular PolygonQueue, this does NOT apply the skip-decided
    filter — the table needs to surface ALL polygons (including ones already
    confirmed/rejected/skipped) so the user can jump back to revisit them.
    """
    # Pull current decision status for each uid first (cheap)
    decided: dict[str, dict] = {}
    if os.path.exists(decisions_csv):
        ddf = pd.read_csv(decisions_csv)
        if 'polygon_uid' in ddf.columns:
            ddf_last = ddf.drop_duplicates(subset='polygon_uid', keep='last')
            for _, row in ddf_last.iterrows():
                decided[str(row['polygon_uid'])] = {
                    'decision': str(row.get('decision', '') or ''),
                    'corrected': str(row.get('corrected_class', '') or ''),
                }

    # Stream the queue rather than materializing all PolygonItems at once.
    # On large gpkgs (e.g. MC11 LCP with ~80k polygons) the full list of
    # shapely geometries would otherwise blow up to hundreds of MB; we only
    # need scalar metadata for the table.
    q_all = PolygonQueue(gpkg_path=gpkg_path, mineral=mineral,
                          decisions_csv=None)
    rows = []
    for item in q_all:
        d = decided.get(item.polygon_uid, {})
        rows.append({
            'polygon_uid': item.polygon_uid,
            'tile_id': item.tile_id,
            'layer': item.layer,
            'area_m2': item.area_m2,
            'decision': d.get('decision', ''),
            'corrected': d.get('corrected', ''),
        })
        # item (with its shapely geometry) goes out of scope each iteration
        # so the GC can reclaim it; no persistent reference is held.

    if not rows:
        return pd.DataFrame(columns=['polygon_uid', 'tile_id', 'layer',
                                       'area_m2', 'decision', 'corrected'])
    return pd.DataFrame(rows)


def make_spectrum_figure(spectra: np.ndarray,
                          wavelengths_nm: np.ndarray,
                          continuum_removed: bool = False) -> go.Figure:
    """Mean + ±1σ envelope, with robust handling of out-of-range pixels.

    A single NODATA-sentinel or gain-stage artifact pixel can otherwise drag
    the mean/std hundreds of units off (-600 to +400 type blowouts). We:

      1. Drop pixels whose ANY band is outside the physical reflectance
         range [-0.5, 1.5] before computing aggregates — this kills the
         outlier influence at the source.
      2. Use the actual trace extremes (the mean and envelope we're about
         to draw) to decide whether to autoscale or fall back to [0, 1].

    When ``continuum_removed`` is set, the mean and envelope are divided by
    an upper-hull continuum (data.continuum_removal, the same 59-band m0..m58
    transform the CR classifier uses) so absorption band depths are legible;
    the 1 µm detector-overlap bands (m16-19) read flat (=1.0) by design.
    """
    fig = go.Figure()
    if spectra.shape[0] == 0:
        fig.update_layout(title='no interior pixels', height=350,
                          yaxis=dict(range=[0.0, 1.0]))
        return fig

    # Step 1: drop wildly out-of-range pixels (cleans the aggregates).
    valid_mask = ~((spectra > 1.5) | (spectra < -0.5)).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if not valid_mask.any():
        fig.update_layout(title='no in-range pixels', height=350,
                          yaxis=dict(range=[0.0, 1.0]))
        return fig
    if n_dropped:
        spectra = spectra[valid_mask]

    if continuum_removed and spectra.shape[1] == 59:
        # Continuum-remove PER PIXEL (median-filtered to tame noise so the
        # upper hull isn't dragged onto spikes), then aggregate in CR space —
        # this gives a coherent mean±σ envelope. Cap the pixel count so the
        # per-spectrum hull stays fast on huge polygons.
        from scipy.signal import medfilt
        from data.continuum_removal import continuum_removed as _cr, good_band_mask_59
        sp = spectra
        if sp.shape[0] > 4000:
            idx = np.random.default_rng(0).choice(sp.shape[0], 4000, replace=False)
            sp = sp[idx]
        sp = medfilt(sp, kernel_size=(1, 3))          # light band-wise denoise
        cr = _cr(sp)                                   # (n, 59), CR per pixel
        # Envelope as per-band 16-84 percentiles, NOT mean±σ. CR is bounded
        # above by 1.0 per pixel (each pixel hits its own hull anchors at 1.0),
        # so mean+σ pokes UNPHYSICALLY above 1.0 near the anchors, and a few
        # noisy pixels in the low-SNR visible bands blow σ into a sharp spike.
        # Percentiles are bounded by the data (≤1.0) and robust to those
        # outliers; the median centre always sits inside the band.
        center = np.nanmedian(cr, axis=0)
        lower = np.nanpercentile(cr, 16, axis=0)
        upper = np.nanpercentile(cr, 84, axis=0)
        center_name = 'median (16–84%)'
        # Gap the 1 µm detector-overlap bands (m16-19) so they don't render as
        # a flat notch/spike — plotly breaks the line at NaN.
        excl = ~good_band_mask_59()
        for arr in (center, upper, lower):
            arr[excl] = np.nan
    else:
        if continuum_removed:
            # CR needs the 59-band m0..m58 window; fall back to raw otherwise.
            continuum_removed = False
        center = spectra.mean(axis=0)
        std = spectra.std(axis=0)
        upper = center + std
        lower = center - std
        center_name = 'mean'

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
        x=wavelengths_nm, y=center, mode='lines',
        line=dict(width=2, color='royalblue'), name=center_name,
    ))

    # Step 2: robust explicit y-range. Compute it ONLY from bands inside the
    # displayed x-window (the ~410 nm band sits outside it and is often
    # noisy), and trim the top/bottom percentiles of the envelope so a single
    # spurious band that survived the pixel filter can't stretch the axis.
    # Always set an explicit range — plotly autoscale on the raw envelope is
    # what let outliers squash the spectral features (~1 in 5 polygons).
    win = ((wavelengths_nm >= PLOT_XRANGE_NM[0]) &
           (wavelengths_nm <= PLOT_XRANGE_NM[1]))
    if not win.any():
        win = np.ones(len(wavelengths_nm), dtype=bool)
    y_lo = float(np.nanpercentile(lower[win], 2))
    y_hi = float(np.nanpercentile(upper[win], 98))
    if not (np.isfinite(y_lo) and np.isfinite(y_hi)) or y_hi <= y_lo:
        y_lo, y_hi = 0.0, 1.0
    else:
        pad = 0.05 * max(y_hi - y_lo, 0.05)
        y_lo = max(y_lo - pad, -0.05)   # clamp to physical reflectance range
        y_hi = min(y_hi + pad, 1.05)
    yaxis_args = {'range': [y_lo, y_hi]}

    title = None
    if n_dropped:
        title = f'({n_dropped:,} of {n_dropped + len(spectra):,} pixels dropped: out-of-range values)'

    fig.update_layout(
        xaxis_title='wavelength (nm)',
        yaxis_title='continuum-removed reflectance' if continuum_removed else 'reflectance',
        xaxis=dict(range=list(PLOT_XRANGE_NM)),
        yaxis=yaxis_args,
        title=title, title_font_size=10,
        height=400, margin=dict(l=40, r=20, t=30, b=40), showlegend=False,
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

    # Sidebar config — dropdowns of known locations, with an "Other" escape
    # hatch that reveals a free-text box for a custom path.
    _OTHER = '⟨ other — type a path ⟩'

    def _pick(label, choices, default):
        opts = ([default] if default not in choices else []) + list(choices) + [_OTHER]
        sel = st.sidebar.selectbox(label, opts, index=opts.index(default))
        if sel == _OTHER:
            return st.sidebar.text_input(f'{label} — custom path', default)
        return sel

    gpkg_dir = _pick('gpkg dir', _gpkg_dir_choices(), DEFAULT_GPKG_DIR)
    mrral_dir = _pick('mrral tile dir', _mrral_dir_choices(), DEFAULT_MRRAL_DIR)
    out_dir = _pick('output dir', _out_dir_choices(), DEFAULT_OUT_DIR)
    continuum_removed = st.sidebar.checkbox(
        'continuum removed (band depths)', value=False,
        help='Divide the spectrum by an upper-hull continuum so absorption '
             'band depths are legible; 1 µm overlap bands (m16-19) read flat.')
    decisions_csv = os.path.join(out_dir, 'decisions.csv')
    # Per-polygon parquet datasets (one file per decided polygon under each
    # directory). The previous single-file design did a multi-GB
    # read-modify-write on every decision, OOM-killing the app. If a legacy
    # single-file parquet exists, the writers migrate it into the directory
    # as legacy.parquet on first init.
    confirmed_pq = os.path.join(out_dir, 'confirmed_pixels')
    hardneg_pq = os.path.join(out_dir, 'hard_negatives')

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

    def _evict_old_cache_entries():
        """Drop cached spectra+thumbnails for the oldest history UIDs once
        we exceed EITHER the total-pixel budget OR the entry-count cap.
        The history list is preserved (UIDs still navigable); _set_current
        will lazily reload spectra when revisited."""
        cache = st.session_state['cache']

        def _total_pixels() -> int:
            total = 0
            for _item, b, _t in cache.values():
                if b is not None:
                    total += int(b.spectra.shape[0])
            return total

        over_pixels = _total_pixels() > MAX_CACHE_PIXELS
        over_count = len(cache) > MAX_CACHE_ENTRIES
        if not over_pixels and not over_count:
            return

        hist = st.session_state['history']
        cursor = st.session_state['cursor']
        # Protect only the currently-displayed polygon. ±1 protection turned
        # out to cost real money on large-LCP reviews — better to reload one
        # neighbor on demand than carry it in RAM forever.
        protected = set()
        if 0 <= cursor < len(hist):
            protected.add(hist[cursor])

        evicted = 0
        for uid in hist:
            if (_total_pixels() <= MAX_CACHE_PIXELS and
                    len(cache) <= MAX_CACHE_ENTRIES):
                break
            if uid in cache and uid not in protected:
                # Drop the big spectra/rows/cols arrays but keep the item
                # metadata so we can reload the polygon on revisit (via
                # _set_current's bundle-is-None code path). Also drop the
                # thumbnail — small but adds up.
                ci, _, _ = cache[uid]
                cache[uid] = (ci, None, None)
                evicted += 1
        if evicted:
            # Force a cycle so numpy arrays actually return memory to the
            # allocator instead of sitting in pymalloc's free pool.
            import gc
            gc.collect()

    def _set_current(uid: str):
        item, bundle, thumb = st.session_state['cache'][uid]
        if bundle is None:  # rehydrated or evicted entry — load spectra now
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
        if thumb is None:
            # Context image shows by default. On failure _load_thumb_for_current
            # warns and leaves current_thumb None, and the 'Show context image'
            # button remains as a manual retry.
            _load_thumb_for_current()
        _evict_old_cache_entries()

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
            # Thumbnail is loaded by _set_current (context image on by default).
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
        coocc = str(prev_decision.get('co_occurring_classes') or '').strip()
        coocc_part = f" + co-occurring: {coocc}" if coocc and coocc != 'nan' else ''
        st.warning(
            f"Already decided: **{prev_decision['decision']}**{corr_part}"
            f"{coocc_part} at {prev_decision['ts']}. Clicking a button below "
            f"will supersede this."
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
        c_spec.plotly_chart(
            make_spectrum_figure(bundle.spectra, wavelengths,
                                 continuum_removed=continuum_removed),
            use_container_width=True)
    else:
        st.plotly_chart(
            make_spectrum_figure(bundle.spectra, wavelengths,
                                 continuum_removed=continuum_removed),
            use_container_width=True)
        if st.button('Show context image', key='show_thumb'):
            _load_thumb_for_current()
            st.rerun()

    # Co-occurring classes: for polygons that ARE the predicted class but ALSO
    # show another primary mineral. Multi-label loss treats each class
    # independently, so confirming "olivine only" on a polygon that has both
    # olivine and hcp would push the HCP logit DOWN — actively damaging HCP
    # training. Selecting hcp here writes hcp=1.0 alongside olivine=1.0,
    # which is what the loss actually wants for mixed-mineralogy polygons.
    # Options exclude the current mineral (it's implicitly included) and
    # exclude bland/ambiguous (those are only meaningful as the sole label).
    # alteration IS allowed as co-occurring: mixed mafic+alteration polygons
    # are real and the multi-label loss wants both classes positive.
    cooccur_options = [c for c in
                       ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']
                       if c != mineral]
    # NOTE: all three decision inputs below are keyed by (mineral, polygon_uid).
    # Without an explicit key, Streamlit derives widget identity from the
    # construction params, which don't change across polygons — so selections
    # (e.g. confidence=Low, co-occurring=hcp) silently STICK from one polygon
    # to the next. Per-polygon keys reset each input to its default on every
    # new polygon; Streamlit garbage-collects keyed state for widgets that are
    # no longer rendered, so the per-uid keys don't accumulate.
    co_occurring = st.multiselect(
        'also present (co-occurring minerals):',
        options=cooccur_options,
        default=[],
        key=f'cooccur::{mineral}::{item.polygon_uid}',
        help='Applies to confirms AND reject→mineral reassignments. Use when '
             'the polygon shows more than one primary mineral: a confirm '
             'writes predicted+co-occurring, a reject with "actually: X" '
             'writes X+co-occurring (all classes = 1.0) instead of '
             'single-class. Ignored for tag rejects (bland/alteration/'
             'ambiguous).',
    )

    # Decision buttons + corrected-class dropdown.
    # 'bland' is the UI name for the schema's 'other' label column (the
    # featureless / dust-dominated spectra harvested from known dust regions).
    # 'alteration' and 'ambiguous' are non-mineral tags — recorded as hard
    # negatives with no positive label. Use 'alteration' for spectra that
    # are clearly NOT the predicted primary mineral but show alteration-
    # mineral signatures (clays / sulfates / opal / prehnite / chlorite /
    # other 2.3-2.5 µm features often mistaken for HCP). Use 'ambiguous'
    # when the spectrum is rejected but you can't categorize what it is.
    corrected = st.selectbox(
        'if rejected, actually:',
        options=['', 'olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'ambiguous'],
        index=0,
        key=f'corrected::{mineral}::{item.polygon_uid}',
    )
    confidence = st.radio(
        'confidence', ['High', 'Moderate', 'Low'], horizontal=True, index=0,
        key=f'confidence::{mineral}::{item.polygon_uid}',
        help='Per-polygon training weight: High=1.0, Moderate=0.75, Low=0.5. '
             'Applied to confirms and reject→mineral reassignments.',
    )
    p1, b1, b2, b3, n1 = st.columns([1, 1, 1, 1, 1])

    def _record(decision: str):
        # Supersede any prior decision for this polygon by removing rows from
        # whichever derived parquet the old decision wrote to.
        if prev_decision is not None:
            prev = prev_decision.get('decision')
            if prev == 'confirm':
                confirmed_writer.drop_polygon(item.polygon_uid)
            elif prev == 'reject':
                hardneg_writer.drop_polygon(item.polygon_uid)
        # co_occurring applies to confirms and to reject→mineral reassignments
        # (the writer ignores it on tag/pure-reject branches; keep the csv in
        # step with what actually lands in the parquet).
        _cooccur_applies = (
            decision == 'confirm'
            or (decision == 'reject'
                and corrected in ('olivine', 'lcp', 'hcp', 'plagioclase')))
        log.append(dict(
            source_gpkg=item.source_gpkg, layer=item.layer,
            polygon_uid=item.polygon_uid, tile_id=item.tile_id,
            predicted_class=mineral, decision=decision,
            corrected_class=(corrected if decision == 'reject' else ''),
            n_pixels=n_px, area_m2=item.area_m2,
            co_occurring_classes=(';'.join(co_occurring)
                                   if _cooccur_applies else ''),
            confidence=confidence,
        ))
        # Patch the cached polygon-list table so the decision column stays
        # fresh without a full rebuild (which would be ~5 sec for large gpkgs).
        cached_qdf = st.session_state.get('queue_df')
        if cached_qdf is not None and not cached_qdf.empty:
            mask = cached_qdf['polygon_uid'] == item.polygon_uid
            if mask.any():
                cached_qdf.loc[mask, 'decision'] = decision
                cached_qdf.loc[mask, 'corrected'] = (
                    corrected if decision == 'reject' else '')
        if decision == 'confirm' and bundle is not None and n_px > 0:
            confirmed_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                label_class=mineral,
                extra_classes=co_occurring or None,
                confidence=confidence,
            )
            confirmed_writer.flush()
        elif decision == 'reject' and bundle is not None and n_px > 0:
            hardneg_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                predicted_class=mineral,
                corrected_class=(corrected or None),
                extra_classes=co_occurring or None,
                confidence=confidence,
            )
            hardneg_writer.flush()
        _advance()
        st.rerun()

    # if/elif so an accidental removal of st.rerun() can't double-fire.
    # Next → advances through history (or pulls fresh from queue at the
    # live edge) WITHOUT recording a new decision — use it to scroll
    # forward past polygons you've already decided.
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
    elif n1.button('Next →', use_container_width=True):
        _advance()
        st.rerun()

    # ── Polygon list table (jump-to navigation) ────────────────────────────
    # Cached per (gpkg, mineral) — rebuilt only when the queue changes. For
    # large gpkgs (e.g. MC11 LCP with 80k+ polygons) the build is a few
    # seconds; the cache makes subsequent navigations within the same
    # mineral instant.
    table_key = f'queue_df::{gpkg_path}::{mineral}'
    if st.session_state.get('queue_df_key') != table_key:
        with st.spinner(f'building polygon list for {mineral}...'):
            st.session_state['queue_df_key'] = table_key
            st.session_state['queue_df'] = build_queue_dataframe(
                gpkg_path=gpkg_path, mineral=mineral,
                decisions_csv=decisions_csv,
            )
    qdf = st.session_state['queue_df']

    def _jump_to_uid(target_uid: str):
        # Already in cache → just move the cursor (or append if not in history yet)
        cache = st.session_state['cache']
        hist = st.session_state['history']
        if target_uid in cache:
            if target_uid in hist:
                st.session_state['cursor'] = hist.index(target_uid)
            else:
                hist.append(target_uid)
                st.session_state['cursor'] = len(hist) - 1
            _set_current(target_uid)
            return
        # Not in cache → look up via gpkg + load spectra
        q = PolygonQueue(gpkg_path=gpkg_path, mineral=mineral)
        found = q.lookup_items([target_uid])
        new_item = found.get(target_uid)
        if new_item is None:
            st.error(f'Polygon {target_uid} not found in gpkg.')
            return
        try:
            new_bundle = load_polygon_pixels(
                geometry=new_item.geometry, tile_id=new_item.tile_id,
                mrral_dir=mrral_dir, source_crs=new_item.source_crs,
            )
        except Exception as e:
            st.error(f'Failed to load {target_uid}: {e}')
            return
        cache[target_uid] = (new_item, new_bundle, None)
        hist.append(target_uid)
        st.session_state['cursor'] = len(hist) - 1
        _set_current(target_uid)

    with st.expander(
        f'Polygon list — {len(qdf):,} polygons in {mineral} '
        f'(click a row to jump)', expanded=False,
    ):
        if qdf.empty:
            st.info('No polygons in this gpkg/mineral combination.')
        else:
            # The table key rotates (nonce) each time a selection is consumed.
            # Popping the server-side widget state is NOT enough: the browser
            # keeps the row highlighted and re-sends its selection with every
            # interaction, so the jump re-fired after each navigation (the
            # "hopping"). A new key = a new frontend component = selection
            # truly cleared on both sides.
            _tbl_nonce = st.session_state.setdefault('polygon_table_nonce', 0)
            sel = st.dataframe(
                qdf,
                selection_mode='single-row',
                on_select='rerun',
                hide_index=True,
                use_container_width=True,
                height=420,
                column_config={
                    'polygon_uid': st.column_config.TextColumn(
                        'polygon_uid', width='medium'),
                    'tile_id': st.column_config.TextColumn('tile', width='small'),
                    'layer': st.column_config.TextColumn('layer', width='small'),
                    'area_m2': st.column_config.NumberColumn(
                        'area (m²)', format='%.0f', width='small'),
                    'decision': st.column_config.TextColumn(
                        'decision', width='small',
                        help="confirm / reject / skip — blank = not yet decided"),
                    'corrected': st.column_config.TextColumn(
                        'corrected', width='small',
                        help="if rejected, the corrected class or tag"),
                },
                key=f'polygon_list_table_{_tbl_nonce}',
            )
            if sel and sel.selection and sel.selection.rows:
                row_idx = sel.selection.rows[0]
                target_uid = qdf.iloc[row_idx]['polygon_uid']
                current_uid = item.polygon_uid if item else None
                # Consume the selection by retiring this widget instance —
                # see nonce comment above.
                st.session_state['polygon_table_nonce'] = _tbl_nonce + 1
                if target_uid != current_uid:
                    _jump_to_uid(target_uid)
                st.rerun()


if __name__ == '__main__':
    main()
