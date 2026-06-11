"""Polygon-level evaluation harness for CRISM mineral classifiers (Phase 0).

Walks every labeled gpkg under ``--gpkg_dir`` (default
``/mnt/mrdr/categorized_mineral_units``), rasterizes each polygon onto its
source ``mrral`` tile, classifies every interior pixel with the supplied
checkpoint, and aggregates per-polygon by **mean softmax probability**. Emits:

  reports/polygon_eval_<ckpt_stem>/
    summary.md
    confusion_matrix.csv
    confusion_matrix.png
    per_polygon.parquet
    spatial_coherence.csv

Three checkpoint formats are auto-detected from the state dict:

  * ``SpatialSpectralClassifier`` — keys begin with ``encoder.`` plus
    ``head.weight``. Loaded straight into the classifier.
  * ``ContrastiveEncoder`` (encoder-only refinement) — top-level
    ``encoder_state`` or ``model_state`` with ``cls_token`` / ``band_embed.*``
    in the inner state. A linear probe ``Linear(embed_dim, 5)`` is trained
    inline (default 5 epochs) on the standard parquet train split + the
    1.8K extra plag ROI patches, mirroring ``eval_contrastive.run_linear_probe``.
  * mrrsu-aux variants — currently raise NotImplementedError; the spec
    documents the 2-channel aux concatenation but no aux variant exists in
    ``checkpoints/`` at the time this harness was authored. The hook
    ``--mrrsu_aux_npy`` is present for future use.

Polygon true class semantics:

  * Single-mineral polygons (e.g. ``plagioclase (Moderate)``): one true class.
    These are the only polygons used in the confusion matrix.
  * Multi-mineral polygons (e.g. ``hcp + olivine (High)``): polygon is counted
    *correct* if the argmax of polygon-mean-prob matches ANY of the labeled
    minerals. This is the "map-quality" view — if a pixel is plausibly any of
    the labeled minerals, getting any of them is fine.
  * Polygons whose category resolves to ``alteration`` / ``spinel`` only
    (i.e. zero overlap with the 5-class output space) are dropped from
    accuracy stats (counted under ``n_skipped_oob_class``).

Usage:

  conda run -n crism python scripts/eval_polygon_accuracy.py \\
      --ckpt checkpoints/ft_plag_aware_real_only_best.pt \\
      --region_filter T0433                                  # smoke test (1 tile)

  conda run -n crism python scripts/eval_polygon_accuracy.py \\
      --ckpt checkpoints/ft_plag_aware_real_only_best.pt     # full run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
from data.dataset import (CRISMSpectralPatchDataset, LABEL_COLS,
                          _collapse_labels, apply_olivine_relabels)
from models.contrastive_encoder import ContrastiveEncoder
from models.spatial_spectral_transformer import SpatialSpectralClassifier
from training.losses import AsymmetricLoss

# ---------------------------------------------------------------- constants
N_BANDS = 59
PATCH_SIZE = 7
HALF = PATCH_SIZE // 2
NODATA = 65535.0
CLIP_MAX = 0.5  # must match data.dataset.CRISMSpectralPatchDataset
N_CLASSES = 5   # LABEL_COLS == ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

# Region heuristic. Tiles fall into one of:
#   - argyre: T0431..T0437, T0505..T0507, T0540..T0542 (canonical argyre set)
#   - nili:   T1249..T1322 (Nili Fossae / Nili Patera area)
#   - hellas: T0150..T0250 (north-Hellas + Hellas rim, picked from existing
#             vector_argyre_v3 / build_hellas_dataset references)
#   - cmu:    everything else (most of the catalog, named for the source
#             gpkg dir /mnt/mrdr/categorized_mineral_units)
# Configurable via the --regions argument (overrides anything passed in).
DEFAULT_REGIONS = {
    'argyre': {'t0431', 't0432', 't0433', 't0434', 't0435', 't0436', 't0437',
               't0505', 't0506', 't0507', 't0540', 't0541', 't0542'},
    'nili':   {'t1249', 't1250', 't1321', 't1322'},
    'hellas': {'t0183', 't0184', 't0185', 't0235', 't0236', 't0237', 't0238',
               't0239', 't0240'},
}
# Anything not in these explicit sets is bucketed as 'cmu'.

# Category parsing — copied from endmember_curator.precompute.parse_category
# so this script is self-contained.
_MINERAL_KEYWORDS = {
    'olivine':     ['olivine'],
    'lcp':         ['lcp'],
    'hcp':         ['hcp'],
    'plagioclase': ['plagioclase', 'plagiolcase'],   # typo seen in some gpkgs
    'spinel':      ['spinel'],
    'alteration':  ['alteration', 'alter'],
}
_TIER_PATTERN = re.compile(r'\((high|moderate|low)\)', flags=re.IGNORECASE)


def parse_category(cat: str) -> tuple[list[str], str]:
    """Return (minerals_present, confidence_tier_or_empty). Mirrors
    endmember_curator.precompute.parse_category."""
    if not isinstance(cat, str) or not cat:
        return [], ''
    c = cat.lower()
    mins: list[str] = []
    for key, kws in _MINERAL_KEYWORDS.items():
        if any(k in c for k in kws):
            mins.append(key)
    if not mins and 'other' in c:
        mins = ['other']
    tier_m = _TIER_PATTERN.search(cat)
    tier = tier_m.group(1).capitalize() if tier_m else ''
    return mins, tier


def minerals_to_label_indices(mins: list[str]) -> set[int]:
    """Map a parsed mineral list onto a set of 5-class label indices.
    ``alteration`` and ``spinel`` are out-of-vocab and contribute nothing."""
    idx = set()
    for m in mins:
        if m in LABEL_COLS:
            idx.add(LABEL_COLS.index(m))
    return idx


def is_pure_in_label_space(mins: list[str]) -> bool:
    """Pure = exactly one in-vocabulary mineral. ``alteration``/``spinel`` are
    OOV, so a category like 'alteration + olivine' is treated as pure-olivine
    when scoring (only one in-vocab mineral)."""
    in_vocab = [m for m in mins if m in LABEL_COLS]
    return len(in_vocab) == 1


def tile_region(tile_id: str, regions: dict[str, set[str]]) -> str:
    t = tile_id.lower()
    for name, members in regions.items():
        if t in members:
            return name
    return 'cmu'


# ---------------------------------------------------------- mrral discovery
def build_mrral_map(cfg) -> dict[str, str]:
    data_root = cfg.get('data_root', '/mnt/mrdr')
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
    return {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


# ---------------------------------------------------------- checkpoint loading
def detect_ckpt_kind(ckpt_path: str) -> str:
    """Return one of 'classifier', 'contrastive_encoder', or 'mrrsu_aux'."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state') if isinstance(ck, dict) else None
    if state is None and isinstance(ck, dict):
        state = ck.get('encoder_state', ck)
    if not isinstance(state, dict):
        state = ck
    keys = list(state.keys())
    # ContrastiveEncoder has a projection head (`proj.0.weight` etc) and no
    # classification head. Check this first — both formats share `encoder.`
    # prefixed weights from the underlying SpatialSpectralTransformer.
    if any(k.startswith('proj.') for k in keys) and not any(k == 'head.weight' for k in keys):
        return 'contrastive_encoder'
    if any(k.startswith('encoder.') for k in keys) and any(k == 'head.weight' for k in keys):
        # SpatialSpectralClassifier
        # Quick mrrsu-aux check: heads might have a different input size if it's
        # the aux variant (embed_dim + 2 -> n_classes). Read head.weight shape.
        head_w = state.get('head.weight')
        if head_w is not None and head_w.shape[1] != 128:
            return 'mrrsu_aux'
        return 'classifier'
    if any(k.startswith('cls_token') or k.startswith('band_embed') for k in keys):
        return 'contrastive_encoder'
    raise ValueError(
        f'Unrecognised checkpoint format: {ckpt_path}. '
        f'First few keys: {keys[:5]}')


def load_classifier(ckpt_path: str, device) -> SpatialSpectralClassifier:
    model = SpatialSpectralClassifier(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=N_CLASSES,
        embed_dim=128, n_heads=4, n_layers=6,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    model.load_state_dict(state)
    model.eval()
    return model


class _LinearProbeModel(nn.Module):
    """Frozen contrastive encoder + Linear head, identical surface to
    ``SpatialSpectralClassifier`` (``model(patch) -> logits``)."""

    def __init__(self, encoder: ContrastiveEncoder, n_classes: int = N_CLASSES):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(encoder.embed_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder.encode(x)
        return self.head(h)


def _load_contrastive_encoder(ckpt_path: str, device) -> ContrastiveEncoder:
    enc = ContrastiveEncoder(
        n_bands=N_BANDS, patch_size=PATCH_SIZE, embed_dim=128,
        n_heads=4, n_layers=6, dropout=0.1, proj_dim=64,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get('model_state') if isinstance(ck, dict) else None
    if state is None and isinstance(ck, dict):
        state = ck.get('encoder_state', ck)
    if not isinstance(state, dict):
        state = ck
    if 'proj.0.weight' in state:
        enc.load_state_dict(state, strict=False)
    else:
        enc.load_encoder_state_dict(state)
    enc.eval()
    return enc


def _train_linear_probe(probe: _LinearProbeModel, cfg, device, *,
                        probe_epochs: int = 5,
                        batch_size: int = 256,
                        probe_lr: float = 1e-3,
                        extra_plag_dir: str | None = None,
                        debug_rows: int | None = None) -> None:
    """Train the linear head on parquet train split + (optional) ROI patches.

    Mirrors ``eval_contrastive.run_linear_probe`` (same loss, same loader
    pattern) but skips wandb and only runs the probe phase. Mutates ``probe``
    in place.
    """
    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    df = _collapse_labels(df)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    if debug_rows is not None:
        train_df = train_df.head(debug_rows).reset_index(drop=True)
    mrral_map = build_mrral_map(cfg)
    cache_dir = None if debug_rows else cfg.get('patch_cache_dir')
    base = CRISMSpectralPatchDataset(
        train_df, mrral_map, patch_size=PATCH_SIZE,
        cache_dir=cache_dir, split='train',
    )
    datasets = [base]
    if extra_plag_dir and os.path.exists(os.path.join(extra_plag_dir, 'patches.npy')):
        from data.contrastive_dataset import ExtraPositivesDataset
        try:
            extras = ExtraPositivesDataset(extra_plag_dir, positive_class='plagioclase')
            datasets.append(extras)
            print(f'    linear probe: +{len(extras)} extra plag ROI patches '
                  f'from {extra_plag_dir}')
        except FileNotFoundError as e:
            print(f'    linear probe: no extra plag patches ({e})')
    if len(datasets) > 1:
        from torch.utils.data import ConcatDataset
        ds = ConcatDataset(datasets)
    else:
        ds = base
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    optim = torch.optim.AdamW(probe.head.parameters(), lr=probe_lr,
                              weight_decay=1e-4)
    probe.head.train()
    for epoch in range(1, probe_epochs + 1):
        running = 0.0; n = 0
        for feats, labels, weights in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = probe(feats)
            loss = asl(logits, labels, weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item()) * feats.shape[0]
            n += feats.shape[0]
        print(f'    linear probe: epoch {epoch}/{probe_epochs}  '
              f'train_loss={running / max(n, 1):.4f}')
    probe.eval()


def load_model_auto(args, cfg, device) -> tuple[nn.Module, str]:
    """Returns (model_in_eval_mode, kind). For the contrastive encoder it
    will either load a probe head from --probe_head_path or train one inline.
    """
    kind = detect_ckpt_kind(args.ckpt)
    if kind == 'classifier':
        return load_classifier(args.ckpt, device), kind
    if kind == 'mrrsu_aux':
        raise NotImplementedError(
            'mrrsu_aux variant detected; this harness has not been wired up '
            'for the aux-input model yet. Add the SpatialSpectralClassifierAux '
            'load path here when needed.')
    # contrastive_encoder
    encoder = _load_contrastive_encoder(args.ckpt, device)
    probe = _LinearProbeModel(encoder, n_classes=N_CLASSES).to(device)
    if args.probe_head_path:
        sd = torch.load(args.probe_head_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'head_state' in sd:
            sd = sd['head_state']
        elif isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        probe.head.load_state_dict(sd)
        print(f'    loaded probe head from {args.probe_head_path}')
    else:
        if args.no_probe_train:
            raise ValueError('--no_probe_train requires --probe_head_path')
        print(f'    training linear probe ({args.probe_epochs} epochs) on top '
              f'of contrastive encoder…')
        _train_linear_probe(
            probe, cfg, device,
            probe_epochs=args.probe_epochs,
            batch_size=args.batch_size,
            probe_lr=args.probe_lr,
            extra_plag_dir=args.extra_plag_dir,
            debug_rows=args.probe_debug_rows,
        )
    probe.eval()
    return probe, kind


# ---------------------------------------------------------- polygon iteration
def gather_polygons(gpkg_dir: str, region_filter: list[str] | None = None,
                    relabels_path: str | None = None) -> list[dict]:
    """Walk *.gpkg files, parse Category, keep polygons whose mineral list has
    at least one in-vocabulary mineral (olivine/lcp/hcp/plagioclase/other).

    Each returned dict carries: tile_id (lower), polygon_id (int), category,
    minerals, confidence_tier, true_indices, is_pure, geometry. The relabel
    mapping is applied at the polygon level by overriding minerals for any
    (tile_id_lower, polygon_id) that appears in the CSV. The relabels CSV
    schema is the same one ``apply_olivine_relabels`` consumes:
    columns ``tile``, ``polygon``, ``new_label``.
    """
    import geopandas as gpd
    paths = sorted(glob.glob(os.path.join(gpkg_dir, 'T*.gpkg')))
    if region_filter:
        wanted = {r.lower() for r in region_filter}
        paths = [p for p in paths
                 if os.path.basename(p)[:-5].lower() in wanted]

    relabel_map: dict[tuple[str, str], list[str]] = {}
    if relabels_path:
        rel = pd.read_csv(relabels_path, dtype={'polygon': str})
        for _, r in rel.iterrows():
            mins, _ = parse_category(str(r['new_label']))
            relabel_map[(str(r['tile']).lower(), str(r['polygon']))] = mins

    rows: list[dict] = []
    for gp in paths:
        tile = os.path.basename(gp)[:-5]
        tid_lower = tile.lower()
        try:
            g = gpd.read_file(gp)
        except Exception as e:
            print(f'    skip {tile}: {e}')
            continue
        for _, r in g.iterrows():
            cat_raw = r.get('Category')
            if cat_raw is None or (isinstance(cat_raw, float) and np.isnan(cat_raw)):
                continue
            cat = str(cat_raw)
            mins, tier = parse_category(cat)
            poly_num = str(r.get('Polygon Number'))
            key = (tid_lower, poly_num)
            if key in relabel_map:
                # Override mineral set (keep tier from original category)
                mins = relabel_map[key]
            true_idx = minerals_to_label_indices(mins)
            if not true_idx:
                continue
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            rows.append({
                'polygon_uid': f'{tile}/{poly_num}',
                'tile_id': tid_lower,
                'polygon_id': poly_num,
                'category': cat,
                'minerals': mins,
                'confidence_tier': tier,
                'true_indices': sorted(true_idx),
                'is_pure': is_pure_in_label_space(mins),
                'geometry': geom,
                'gpkg_path': gp,
            })
    return rows


# ---------------------------------------------------------- per-tile inference
def patch_from_padded(padded: np.ndarray, pr: int, pc: int) -> np.ndarray:
    """Pull a ``(P, P, N_BANDS)`` patch from a ``(H+2*HALF, W+2*HALF, N_BANDS)``
    pre-padded cube. Identical layout to what CRISMSpectralPatchDataset emits."""
    return padded[pr:pr + PATCH_SIZE, pc:pc + PATCH_SIZE, :]


def infer_polygon_probs(polygons: list[dict],
                        model: nn.Module,
                        device,
                        mrral_map: dict[str, str],
                        batch_size: int = 256,
                        max_pixels_per_poly: int = 4000) -> dict:
    """Run inference for every polygon. Returns dict with arrays:

      polygon_uid (list[str])
      polygon_mean_probs (N, 5) float32 — softmax averaged over interior pixels
      n_pixels (N,) int32
      pred_class (N,) int32
      per_pixel_predictions (dict tile_id -> list of (poly_idx, pred_5))
        — used for spatial coherence at threshold 0.5
    """
    n = len(polygons)
    polygon_mean_probs = np.full((n, N_CLASSES), np.nan, dtype=np.float32)
    n_pixels = np.zeros(n, dtype=np.int32)

    # Group polygons by tile to amortise cube reads
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(polygons):
        groups[p['tile_id']].append(i)

    # Per-tile in-polygon pixel predictions for spatial coherence —
    # value: list of (poly_idx, pixel_row, pixel_col, pred_probs[5])
    per_tile_pixel_preds: dict[str, list] = {}

    rng = np.random.default_rng(0)
    n_no_tile = 0
    n_no_pixel = 0

    for tile_idx, (tid, idxs) in enumerate(sorted(groups.items())):
        if tid not in mrral_map:
            n_no_tile += len(idxs)
            continue
        path = mrral_map[tid]
        try:
            src = rasterio.open(path)
        except Exception as e:
            print(f'    [warn] cannot open {path}: {e}')
            n_no_tile += len(idxs)
            continue
        try:
            data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
            nodata_mask = (data == NODATA) | ~np.isfinite(data)
            data[nodata_mask] = 0.0
            np.clip(data, 0.0, CLIP_MAX, out=data)
            cube = data.transpose(1, 2, 0)  # (H, W, 59)
            valid_2d = ~nodata_mask.any(axis=0)  # (H, W)
            H, W = valid_2d.shape
            padded = np.pad(cube, ((HALF, HALF), (HALF, HALF), (0, 0)),
                            mode='constant')
            del data, nodata_mask, cube
        except Exception as e:
            print(f'    [warn] failed to read {path}: {e}')
            src.close()
            n_no_tile += len(idxs)
            continue
        tile_pixel_records = []
        try:
            for i in idxs:
                row = polygons[i]
                geom = row['geometry']
                try:
                    mask = rasterio.features.rasterize(
                        [(geom, 1)], out_shape=(H, W),
                        transform=src.transform, fill=0, dtype=np.uint8,
                    ).astype(bool)
                except Exception:
                    n_no_pixel += 1
                    continue
                mask = mask & valid_2d
                rs, cs = np.where(mask)
                if rs.size == 0:
                    n_no_pixel += 1
                    continue
                if rs.size > max_pixels_per_poly:
                    sel = rng.choice(rs.size, size=max_pixels_per_poly,
                                      replace=False)
                    rs = rs[sel]; cs = cs[sel]
                acc = np.zeros(N_CLASSES, dtype=np.float64)
                with torch.no_grad():
                    for s in range(0, rs.size, batch_size):
                        e = min(s + batch_size, rs.size)
                        batch = np.stack([
                            patch_from_padded(padded, int(r), int(c))
                            for r, c in zip(rs[s:e], cs[s:e])
                        ]).astype(np.float32)
                        x = torch.from_numpy(batch).to(device)
                        logits = model(x)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        acc += probs.sum(axis=0)
                        # Stash per-pixel predictions for spatial coherence
                        for k in range(probs.shape[0]):
                            tile_pixel_records.append(
                                (i, int(rs[s + k]), int(cs[s + k]),
                                 probs[k].astype(np.float32))
                            )
                polygon_mean_probs[i] = (acc / rs.size).astype(np.float32)
                n_pixels[i] = int(rs.size)
            per_tile_pixel_preds[tid] = {
                'shape': (H, W),
                'records': tile_pixel_records,
            }
            print(f'    [{tile_idx + 1}/{len(groups)}] tile {tid}: '
                  f'{len(idxs)} polygons → {sum(1 for j in idxs if n_pixels[j] > 0)} '
                  f'with pixels  ({len(tile_pixel_records)} total pixels)')
        finally:
            src.close()
            del padded, valid_2d

    print(f'  inference summary: {n_no_tile} polygons with no tile, '
          f'{n_no_pixel} polygons with no interior pixels.')

    pred_class = np.full(n, -1, dtype=np.int32)
    valid_mask = ~np.isnan(polygon_mean_probs).any(axis=1)
    pred_class[valid_mask] = polygon_mean_probs[valid_mask].argmax(axis=1)

    return {
        'polygon_mean_probs': polygon_mean_probs,
        'n_pixels': n_pixels,
        'pred_class': pred_class,
        'per_tile_pixel_preds': per_tile_pixel_preds,
        'n_no_tile': n_no_tile,
        'n_no_pixel': n_no_pixel,
    }


# ---------------------------------------------------------- spatial coherence
def compute_spatial_coherence(per_tile_pixel_preds: dict,
                              threshold: float = 0.5) -> pd.DataFrame:
    """For each class, build a binary map per tile at the given threshold and
    record the median connected-component size across all tiles. Larger ⇒
    more map-like / less salt-and-pepper.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        print('    spatial coherence skipped: scipy not available')
        return pd.DataFrame()

    per_class_sizes: dict[int, list[int]] = {c: [] for c in range(N_CLASSES)}
    for tid, payload in per_tile_pixel_preds.items():
        H, W = payload['shape']
        records = payload['records']
        if not records:
            continue
        # Build per-class binary maps from the in-polygon pixel predictions.
        # NB: we only have predictions for in-polygon pixels — the rest of
        # the tile is "unknown" (we never scored them). That's fine for this
        # spatial-coherence metric since it answers "for the pixels we DID
        # score, are class-c hot pixels clustered?" rather than the
        # tile-wide map question.
        bin_maps = np.zeros((N_CLASSES, H, W), dtype=np.uint8)
        for (_poly_idx, r, c, probs) in records:
            for cls in range(N_CLASSES):
                if probs[cls] >= threshold:
                    bin_maps[cls, r, c] = 1
        for cls in range(N_CLASSES):
            if bin_maps[cls].sum() == 0:
                continue
            _, n_components = label(bin_maps[cls])
            if n_components == 0:
                continue
            # Component sizes via bincount on labelled output
            labelled, _ = label(bin_maps[cls])
            sizes = np.bincount(labelled.ravel())[1:]   # skip background
            per_class_sizes[cls].extend(sizes.tolist())

    rows = []
    for cls in range(N_CLASSES):
        sizes = per_class_sizes[cls]
        if not sizes:
            rows.append({'class': LABEL_COLS[cls], 'n_components': 0,
                         'median_size': 0, 'p90_size': 0, 'max_size': 0})
            continue
        rows.append({
            'class': LABEL_COLS[cls],
            'n_components': len(sizes),
            'median_size': float(np.median(sizes)),
            'p90_size': float(np.percentile(sizes, 90)),
            'max_size': int(np.max(sizes)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- aggregation
def aggregate_results(polygons: list[dict], inference: dict,
                      regions: dict[str, set[str]]) -> dict:
    """Build the per-polygon report frame + region/tier accuracy summaries."""
    n = len(polygons)
    pred = inference['pred_class']
    probs = inference['polygon_mean_probs']
    n_pix = inference['n_pixels']

    rows = []
    for i, p in enumerate(polygons):
        if pred[i] < 0:
            # No prediction — skip in summary but keep in per_polygon parquet
            correct = None
        else:
            correct = int(pred[i]) in set(p['true_indices'])
        rows.append({
            'polygon_uid': p['polygon_uid'],
            'tile_id': p['tile_id'],
            'category': p['category'],
            'minerals': '|'.join(p['minerals']),
            'true_indices': '|'.join(LABEL_COLS[i] for i in p['true_indices']),
            'confidence_tier': p['confidence_tier'] or 'NA',
            'is_pure': p['is_pure'],
            'region': tile_region(p['tile_id'], regions),
            'n_pixels': int(n_pix[i]),
            'predicted_class': LABEL_COLS[pred[i]] if pred[i] >= 0 else 'NONE',
            'prob_olivine':     float(probs[i, 0]) if not np.isnan(probs[i, 0]) else float('nan'),
            'prob_lcp':         float(probs[i, 1]) if not np.isnan(probs[i, 1]) else float('nan'),
            'prob_hcp':         float(probs[i, 2]) if not np.isnan(probs[i, 2]) else float('nan'),
            'prob_plagioclase': float(probs[i, 3]) if not np.isnan(probs[i, 3]) else float('nan'),
            'prob_other':       float(probs[i, 4]) if not np.isnan(probs[i, 4]) else float('nan'),
            'correct': correct,
        })
    df = pd.DataFrame(rows)
    return df


def confusion_matrix_single_mineral(df: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix from single-mineral polygons only. Rows = true class,
    columns = predicted class. Index labels carry total support per row."""
    sub = df[df['is_pure'] & (df['predicted_class'] != 'NONE')].copy()
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
    for _, r in sub.iterrows():
        # is_pure ⇒ exactly one in-vocab mineral; we recover it from true_indices
        true_idx_lst = [LABEL_COLS.index(x) for x in r['true_indices'].split('|') if x]
        if len(true_idx_lst) != 1:
            continue
        t = true_idx_lst[0]
        p = LABEL_COLS.index(r['predicted_class'])
        cm[t, p] += 1
    df_cm = pd.DataFrame(cm,
                         index=[f'true_{c}' for c in LABEL_COLS],
                         columns=[f'pred_{c}' for c in LABEL_COLS])
    return df_cm


def plot_confusion_matrix(cm: pd.DataFrame, out_path: str, title: str = ''):
    fig, ax = plt.subplots(figsize=(7, 6))
    arr = cm.values
    # Row-normalise for visibility, keep counts annotated
    row_sums = arr.sum(axis=1, keepdims=True)
    norm = np.divide(arr, np.where(row_sums == 0, 1, row_sums))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(LABEL_COLS, rotation=45, ha='right')
    ax.set_yticklabels(LABEL_COLS)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title or 'Confusion matrix (single-mineral polygons)')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f'{arr[i, j]}\n({norm[i, j]:.2f})',
                    ha='center', va='center',
                    color='white' if norm[i, j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, label='row-normalised count')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def per_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df['predicted_class'] != 'NONE']
    rows = []
    for region, group in sub.groupby('region'):
        n_total = len(group)
        n_correct = int(group['correct'].sum())
        rows.append({
            'region': region,
            'n_polygons': n_total,
            'n_correct': n_correct,
            'accuracy': n_correct / n_total if n_total else float('nan'),
        })
    # Overall row
    n_total = len(sub); n_correct = int(sub['correct'].sum())
    rows.append({'region': 'ALL', 'n_polygons': n_total,
                 'n_correct': n_correct,
                 'accuracy': n_correct / n_total if n_total else float('nan')})
    return pd.DataFrame(rows).sort_values('region').reset_index(drop=True)


def per_tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df['predicted_class'] != 'NONE']
    rows = []
    for tier, group in sub.groupby('confidence_tier'):
        n_total = len(group); n_correct = int(group['correct'].sum())
        rows.append({
            'confidence_tier': tier,
            'n_polygons': n_total,
            'n_correct': n_correct,
            'accuracy': n_correct / n_total if n_total else float('nan'),
        })
    return pd.DataFrame(rows).sort_values('confidence_tier').reset_index(drop=True)


def per_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per true-class accuracy on single-mineral polygons (precision-of-truth view)."""
    sub = df[df['is_pure'] & (df['predicted_class'] != 'NONE')]
    rows = []
    for cls in LABEL_COLS:
        cls_mask = sub['true_indices'] == cls
        n_total = int(cls_mask.sum())
        if n_total == 0:
            continue
        n_correct = int(sub.loc[cls_mask, 'correct'].sum())
        rows.append({
            'true_class': cls,
            'n_polygons': n_total,
            'n_correct': n_correct,
            'recall': n_correct / n_total if n_total else float('nan'),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- summary writing
def _df_to_markdown(df: pd.DataFrame, floatfmt: str = '.4f',
                    index: bool = False) -> str:
    """Hand-rolled markdown table. ``pandas.to_markdown`` requires the
    optional ``tabulate`` dep, which isn't in the crism env. This is a tiny
    no-deps fallback that produces a GitHub-style markdown table."""
    if df is None or len(df) == 0:
        return '_(empty)_'

    def fmt(v):
        if v is None:
            return ''
        if isinstance(v, float):
            if np.isnan(v):
                return 'NaN'
            return f'{v:{floatfmt[1:] if floatfmt.startswith(".") else floatfmt}}'
        return str(v)

    rows = df.copy()
    if index:
        rows.insert(0, '', rows.index)
    headers = [str(c) for c in rows.columns]
    body = [[fmt(v) for v in row] for row in rows.itertuples(index=False, name=None)]
    widths = [max(len(h), max((len(r[i]) for r in body), default=0))
              for i, h in enumerate(headers)]
    def line(cells):
        return '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'
    out = [line(headers), '|' + '|'.join('-' * (w + 2) for w in widths) + '|']
    out.extend(line(r) for r in body)
    return '\n'.join(out)


def write_summary_md(out_dir: str, args, kind: str,
                     n_polygons: int,
                     region_df: pd.DataFrame,
                     tier_df: pd.DataFrame,
                     class_df: pd.DataFrame,
                     cm: pd.DataFrame,
                     coh_df: pd.DataFrame,
                     elapsed_sec: float,
                     n_no_tile: int,
                     n_no_pixel: int) -> str:
    lines = []
    lines.append(f'# Polygon-level evaluation')
    lines.append('')
    lines.append(f'- **Checkpoint:** `{args.ckpt}`')
    lines.append(f'- **Detected kind:** `{kind}`')
    lines.append(f'- **GPKG dir:** `{args.gpkg_dir}`')
    if args.region_filter:
        # args.region_filter is a comma-separated string; show as-is.
        lines.append(f'- **Region filter:** `{args.region_filter}`')
    if args.apply_relabels:
        lines.append(f'- **Relabels applied:** `{args.apply_relabels}` (corrected polygons)')
    else:
        lines.append('- **Relabels applied:** none (raw gpkg categories)')
    lines.append(f'- **Polygons evaluated:** {n_polygons:,}')
    lines.append(f'- **Polygons skipped (no source tile):** {n_no_tile:,}')
    lines.append(f'- **Polygons skipped (no interior pixels):** {n_no_pixel:,}')
    lines.append(f'- **Elapsed:** {elapsed_sec:.1f} sec')
    lines.append('')
    lines.append('## Polygon accuracy by region')
    lines.append('')
    lines.append(_df_to_markdown(region_df, floatfmt='.4f'))
    lines.append('')
    lines.append('## Polygon accuracy by confidence tier')
    lines.append('')
    lines.append(_df_to_markdown(tier_df, floatfmt='.4f'))
    lines.append('')
    lines.append('## Per-true-class recall (single-mineral polygons only)')
    lines.append('')
    if len(class_df):
        lines.append(_df_to_markdown(class_df, floatfmt='.4f'))
    else:
        lines.append('_(no single-mineral polygons with predictions)_')
    lines.append('')
    lines.append('## Confusion matrix (single-mineral polygons; counts)')
    lines.append('')
    lines.append(_df_to_markdown(cm, floatfmt='.0f', index=True))
    lines.append('')
    if len(coh_df):
        lines.append('## Spatial coherence (threshold = 0.5)')
        lines.append('')
        lines.append(_df_to_markdown(coh_df, floatfmt='.2f'))
        lines.append('')
    out_path = os.path.join(out_dir, 'summary.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    return out_path


# ---------------------------------------------------------- entry point
def parse_regions_arg(s: str | None) -> dict[str, set[str]]:
    if not s:
        return DEFAULT_REGIONS
    # Format: "argyre=t0433,t0434;nili=t1249"
    out: dict[str, set[str]] = {}
    for chunk in s.split(';'):
        if not chunk:
            continue
        name, members = chunk.split('=', 1)
        out[name.strip()] = {m.strip().lower() for m in members.split(',') if m.strip()}
    return out or DEFAULT_REGIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpkg_dir', default='/mnt/mrdr/categorized_mineral_units')
    ap.add_argument('--apply_relabels', default=None,
                    help='Path to a polygon-level relabels CSV '
                         '(schema: tile,polygon,new_label).')
    ap.add_argument('--out_dir', default=None,
                    help='Default: reports/polygon_eval_<ckpt_stem>')
    ap.add_argument('--region_filter', default=None,
                    help='Comma-separated tile prefixes (e.g. T0433,T0434) — '
                         'restrict to those gpkgs only. Useful for smoke tests.')
    ap.add_argument('--regions', default=None,
                    help='Override region heuristic; format '
                         '"argyre=t0433,t0434;nili=t1249".')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--max_pixels_per_poly', type=int, default=4000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--config', default='config.yaml')
    # Contrastive-encoder-only paths
    ap.add_argument('--probe_head_path', default=None,
                    help='Pre-trained Linear(128, 5) head for contrastive '
                         'encoder ckpts. If unset, train inline.')
    ap.add_argument('--probe_epochs', type=int, default=5)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--probe_debug_rows', type=int, default=None,
                    help='Limit probe-training rows for fast smoke tests.')
    ap.add_argument('--no_probe_train', action='store_true',
                    help='Refuse to auto-train the probe; require --probe_head_path.')
    ap.add_argument('--extra_plag_dir',
                    default='/mnt/mrdr/crism_classification/data/contrastive/extra_plag_roi',
                    help='Directory with patches.npy + meta.parquet for plag '
                         'ROI augmentation during probe training.')
    # mrrsu_aux passthrough — placeholder
    ap.add_argument('--mrrsu_aux_npy', default=None)
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    cfg = load_config(cfg_path)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device}')

    region_filter = None
    if args.region_filter:
        region_filter = [r.strip() for r in args.region_filter.split(',') if r.strip()]
    regions = parse_regions_arg(args.regions)

    ckpt_stem = Path(args.ckpt).stem
    out_dir = args.out_dir or os.path.join(
        cfg.get('reports_dir',
                os.path.join(os.path.dirname(__file__), '..', 'reports')),
        f'polygon_eval_{ckpt_stem}',
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f'output dir: {out_dir}')

    print('loading model…')
    model, kind = load_model_auto(args, cfg, device)
    print(f'  detected kind: {kind}')

    print('walking gpkg polygons…')
    polygons = gather_polygons(args.gpkg_dir, region_filter=region_filter,
                                relabels_path=args.apply_relabels)
    print(f'  collected {len(polygons):,} polygons (post-filter, in-vocab labels)')
    if not polygons:
        print('  nothing to score — exiting.')
        return

    print('building mrral tile map…')
    mrral_map = build_mrral_map(cfg)
    print(f'  found {len(mrral_map):,} mrral tiles')

    print('running per-polygon inference…')
    t0 = time.time()
    inference = infer_polygon_probs(
        polygons, model, device, mrral_map,
        batch_size=args.batch_size,
        max_pixels_per_poly=args.max_pixels_per_poly,
    )
    elapsed = time.time() - t0

    print('aggregating…')
    df = aggregate_results(polygons, inference, regions)
    region_df = per_region_summary(df)
    tier_df = per_tier_summary(df)
    class_df = per_class_summary(df)
    cm = confusion_matrix_single_mineral(df)

    print('writing outputs…')
    df_out = df.copy()
    df.to_parquet(os.path.join(out_dir, 'per_polygon.parquet'), index=False)
    cm.to_csv(os.path.join(out_dir, 'confusion_matrix.csv'))
    plot_confusion_matrix(cm, os.path.join(out_dir, 'confusion_matrix.png'),
                          title=f'Polygon confusion (single-mineral) — {ckpt_stem}')
    coh_df = compute_spatial_coherence(inference['per_tile_pixel_preds'])
    if len(coh_df):
        coh_df.to_csv(os.path.join(out_dir, 'spatial_coherence.csv'), index=False)
    summary_path = write_summary_md(
        out_dir, args, kind, len(polygons),
        region_df, tier_df, class_df, cm, coh_df,
        elapsed,
        n_no_tile=inference['n_no_tile'],
        n_no_pixel=inference['n_no_pixel'],
    )
    print(f'wrote {summary_path}')

    # Also dump a compact JSON for downstream tooling
    overall_acc = float(df.loc[df['predicted_class'] != 'NONE', 'correct'].mean()
                        if (df['predicted_class'] != 'NONE').any() else float('nan'))
    summary_payload = {
        'ckpt': args.ckpt,
        'kind': kind,
        'n_polygons': len(polygons),
        'overall_accuracy': overall_acc,
        'by_region': region_df.to_dict(orient='records'),
        'by_tier': tier_df.to_dict(orient='records'),
        'by_class': class_df.to_dict(orient='records'),
        'spatial_coherence': coh_df.to_dict(orient='records') if len(coh_df) else [],
        'elapsed_sec': elapsed,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary_payload, f, indent=2)
    print(f'overall polygon accuracy: {overall_acc:.4f}')


if __name__ == '__main__':
    main()
