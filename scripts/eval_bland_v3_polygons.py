"""
Per-polygon training-data evaluation for the bland-v3 classifier.

For every polygon in the labeled training set (across categorized_mineral_units,
the supplemental argyre-gis dir, and the north-Hellas GPKG), compute the mean
spectrum and run it through bland-v3 to compare hand-labeled mineral class
vs predicted argmax class.

Sources:
  1. /mnt/mrdr/categorized_mineral_units/T*.gpkg  (spectra inline)
  2. /mnt/gigas/massif-themis-analysis/argyre-gis/*.gpkg  (spectra inline)
  3. /mnt/mrdr/categorized_mineral_units/north_hellas_mafics_geometries_fixed.gpkg
     (no inline spectra; extract by rasterizing each polygon onto its mrral tile)

Outputs:
  reports/bland_v3_polygon_eval/
    summary.txt           — accuracy stats + per-source breakdown + confusion matrix
    wrong_*.png           — one per misclassified polygon (numerator + ratio
                            spectra, both labels in title)

Usage:
    conda run -n crism python scripts/eval_bland_v3_polygons.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import torch

PROJ = '/mnt/mrdr/crism_classification'
sys.path.insert(0, PROJ)
from data.label_parser import parse_category, CLASSES   # 6-class space
from models.spatial_spectral_transformer import SpatialSpectralClassifier

CKPT       = os.path.join(PROJ, 'checkpoints/ft_bland_v3_lrscale0001_best.pt')
OUT_DIR    = os.path.join(PROJ, 'reports/bland_v3_polygon_eval')

CMU_DIR    = '/mnt/mrdr/categorized_mineral_units'
ARGYRE_DIR = '/mnt/gigas/massif-themis-analysis/argyre-gis'
HELLAS_GPKG = os.path.join(CMU_DIR, 'north_hellas_mafics_geometries_fixed.gpkg')
MC_GLOB    = '/mnt/mrdr/mc*/t*_mrral_*_0327_4.img'

N_BANDS   = 59
NODATA    = 65535.0
CLIP_MAX  = 0.5
PATCH     = 7
N_CLS_OUT = 5
CLASS_OUT_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
# 6-class CLASSES index → 5-class output index (olivine_t1 + olivine_t2 → olivine)
COLLAPSE_6_TO_5 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
HELLAS_LABEL_MAP = {
    'Olivine': 'olivine', 'LCP': 'lcp', 'HCP': 'hcp', 'Plagioclase': 'plagioclase',
}
# 5-class output index for each name (use OUT_NAMES order)
CLASS_OUT_IDX = {n: i for i, n in enumerate(CLASS_OUT_NAMES)}


# ── parsers ──────────────────────────────────────────────────────────────────

def parse_floatlist(s):
    if not isinstance(s, str):
        return np.asarray(s, dtype=np.float64) if s is not None else None
    s2 = s.strip().strip('[]')
    s2 = re.sub(r',', ' ', s2)
    parts = []
    for tok in s2.split():
        t = tok.strip()
        if not t:
            continue
        if t.lower() == 'nan':
            parts.append(np.nan)
            continue
        try:
            parts.append(float(t))
        except ValueError:
            pass
    return np.array(parts, dtype=np.float64) if parts else None


def gt_collapsed(label6: np.ndarray) -> np.ndarray:
    """6-class hand-label vector → 5-class soft label (olivine_t1+olivine_t2 → olivine)."""
    collapsed = np.zeros(N_CLS_OUT, dtype=np.float64)
    for src_idx, dst_idx in COLLAPSE_6_TO_5.items():
        collapsed[dst_idx] += float(label6[src_idx])
    return collapsed


def gt_5class_argmax(label6: np.ndarray) -> int:
    """Primary GT class — used for confusion-matrix rows only."""
    return int(np.argmax(gt_collapsed(label6)))


def gt_5class_positives(label6: np.ndarray) -> set[int]:
    """Set of 5-class indices that are active in the hand label.
    A model prediction is counted CORRECT iff its argmax is in this set —
    this handles multi-mineral labels like 'hcp + lcp' where either head
    firing is a valid prediction.
    """
    coll = gt_collapsed(label6)
    return {i for i in range(N_CLS_OUT) if coll[i] > 0}


def gt_label_string(label6: np.ndarray) -> str:
    """Human-readable summary of which 5-class minerals fire above 0."""
    coll = gt_collapsed(label6)
    active = [f'{CLASS_OUT_NAMES[i]}({coll[i]:.2f})'
              for i in range(N_CLS_OUT) if coll[i] > 0]
    return ' + '.join(active) if active else '(none)'


# ── source GPKG loaders ──────────────────────────────────────────────────────

def load_inline_polygons(gpkg_path: str, source_tag: str) -> list[dict]:
    """For categorized_mineral_units + argyre-gis: parse spectra from inline cols."""
    try:
        g = gpd.read_file(gpkg_path)
    except Exception as e:
        print(f'  [skip] {gpkg_path}: {e}')
        return []
    rows = []
    tid = re.match(r'(T\d+)', os.path.basename(gpkg_path))
    tile_id = tid.group(1).lower() if tid else None
    for idx, row in g.iterrows():
        cat = row.get('Category')
        if cat is None or (isinstance(cat, float) and np.isnan(cat)):
            continue
        try:
            label6, conf = parse_category(str(cat))
        except ValueError:
            continue
        # All-zero labels happen for categories with no mineral assigned (e.g. "alteration")
        if float(np.sum(label6)) == 0:
            continue
        spec_mean = parse_floatlist(row.get('Spectrum Mean'))
        spec_ratio = parse_floatlist(row.get('Ratio Spectrum'))
        wvl = parse_floatlist(row.get('wvl'))
        if spec_mean is None or len(spec_mean) < N_BANDS:
            continue
        rows.append({
            'source': source_tag,
            'gpkg': gpkg_path,
            'tile_id': tile_id,
            'polygon_id': int(idx),
            'category': str(cat),
            'label6': label6,
            'conf': conf,
            'spec_mean_72': spec_mean,
            'spec_ratio_72': spec_ratio,
            'wvl_72': wvl,
            'geometry': row.geometry,    # in the GPKG's per-tile CRS
        })
    return rows


def build_mrral_index() -> list[dict]:
    """For each mrral tile, store metadata + a rasterio dataset handle. Used
    for Hellas spectrum extraction. Returns list of dicts."""
    paths = sorted(glob.glob(MC_GLOB))
    out = []
    for p in paths:
        m = re.match(r't(\d+)_mrral', os.path.basename(p))
        if not m:
            continue
        try:
            ds = rasterio.open(p)
        except Exception:
            continue
        # Compute Mars geographic (lat/lon) bbox for cheap point-in-tile checks.
        # IAU_2015:49900 is the registered Mars sphere geographic CRS.
        try:
            from rasterio.warp import transform_bounds
            b = transform_bounds(ds.crs, 'IAU_2015:49900', *ds.bounds)
        except Exception:
            ds.close(); continue
        out.append({
            'path': p,
            'tile_id': 't' + m.group(1),
            'geog_bounds': b,        # (minlon, minlat, maxlon, maxlat)
            'src_crs': ds.crs,
            'src_transform': ds.transform,
            'shape': (ds.height, ds.width),
        })
        ds.close()
    return out


def extract_mean_spectrum_for_polygon(geom_latlon, mrral_meta):
    """Given a polygon in lat/lon, find its mrral tile and return the mean
    spectrum across band 0..58 over the polygon's pixels. Returns None if no
    tile contains the polygon or if the polygon has 0 valid pixels.
    """
    from rasterio.warp import transform_geom
    cent = geom_latlon.centroid
    candidate = None
    for m in mrral_meta:
        lon0, lat0, lon1, lat1 = m['geog_bounds']
        if lon0 <= cent.x <= lon1 and lat0 <= cent.y <= lat1:
            candidate = m
            break
    if candidate is None:
        return None, None
    src = rasterio.open(candidate['path'])
    try:
        # Reproject geom (in Mars lat/lon == IAU_2015:49900) to the tile's
        # per-tile equirectangular meter CRS
        geom_mars = transform_geom('IAU_2015:49900', src.crs, geom_latlon.__geo_interface__)
        mask = rasterio.features.rasterize(
            [(geom_mars, 1)], out_shape=(src.height, src.width),
            transform=src.transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        if not mask.any():
            return None, None
        pixel_rows, pixel_cols = np.where(mask)
        r0, r1 = int(pixel_rows.min()), int(pixel_rows.max()) + 1
        c0, c1 = int(pixel_cols.min()), int(pixel_cols.max()) + 1
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        chunk = src.read(list(range(1, N_BANDS + 1)), window=window).astype(np.float32)
        # Valid pixels (no nodata)
        nodata_mask = (chunk == NODATA) | ~np.isfinite(chunk)
        # Crop the polygon mask to the same window
        sub_mask = mask[r0:r1, c0:c1]
        valid_pixel = sub_mask & ~nodata_mask.any(axis=0)
        if not valid_pixel.any():
            return None, candidate['tile_id']
        spec = chunk[:, valid_pixel].mean(axis=1)
        return spec.astype(np.float64), candidate['tile_id']
    finally:
        src.close()


def load_hellas_polygons() -> list[dict]:
    g = gpd.read_file(HELLAS_GPKG)
    print(f'  Hellas GPKG has {len(g)} polygons')
    # The GPKG declares an Earth-WGS84 CRS by accident, but the coordinates are
    # Mars lat/lon. Override to IAU_2015:49900 so subsequent transforms succeed.
    g = g.set_crs('IAU_2015:49900', allow_override=True)
    print(f'  building mrral tile index across {MC_GLOB} …')
    mrral_meta = build_mrral_index()
    print(f'  found {len(mrral_meta)} mrral tiles')

    rows = []
    n_skip_no_tile, n_skip_no_label = 0, 0
    for idx, row in g.iterrows():
        label_str = row.get('Interpreta')
        if label_str is None or label_str not in HELLAS_LABEL_MAP:
            n_skip_no_label += 1
            continue
        canonical = HELLAS_LABEL_MAP[label_str]
        # Build 6-class label vector
        label6 = np.zeros(len(CLASSES), dtype=np.float64)
        # collapse: olivine_t1 + olivine_t2 split 50/50 (we don't know subtype)
        if canonical == 'olivine':
            label6[0] = 0.5; label6[1] = 0.5
        else:
            label6[CLASSES.index(canonical)] = 1.0

        geom = row.geometry
        spec, tile_id = extract_mean_spectrum_for_polygon(geom, mrral_meta)
        if spec is None:
            n_skip_no_tile += 1
            continue
        rows.append({
            'source': 'hellas',
            'gpkg': HELLAS_GPKG,
            'tile_id': tile_id,
            'polygon_id': int(idx),
            'category': f'{label_str} (Hellas)',
            'label6': label6,
            'conf': 1.0,
            'spec_mean_72': spec,      # only first 59 bands populated
            'spec_ratio_72': None,
            'wvl_72': None,
            'geometry': geom,           # in IAU_2015:49900 lat/lon
        })
        if (idx + 1) % 100 == 0:
            print(f'    processed {idx + 1}/{len(g)} (kept {len(rows)}, skip_label={n_skip_no_label}, skip_tile={n_skip_no_tile})')
    print(f'  Hellas done: kept {len(rows)} / {len(g)} '
          f'(skip_label={n_skip_no_label}, skip_tile={n_skip_no_tile})')
    return rows


# ── classifier ───────────────────────────────────────────────────────────────

def load_classifier(device):
    model = SpatialSpectralClassifier(
        n_bands=N_BANDS, patch_size=PATCH, n_classes=N_CLS_OUT,
        embed_dim=128, n_heads=4, n_layers=6,
    ).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_patches(patches):
    B = patches.shape[0]
    flat = patches.reshape(B, -1).astype(np.float32)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((flat - mu) / sd).reshape(patches.shape).astype(np.float32)


def predict_polygon_probs(polygons, model, device, mrral_meta,
                           batch_size=2048, max_pixels_per_poly=100):
    """For each polygon, classify EVERY in-polygon pixel using its real 7×7
    spatial neighborhood from the mrral tile, then aggregate per-polygon by
    averaging the per-pixel probabilities.

    Patches use the same zero-padding + per-patch normalization as
    classify_tile_supervised, so each pixel's prediction is exactly what the
    production tile-inference pipeline would have produced for it. Big
    polygons get many votes; small polygons get few.

    Polygons are grouped by mrral tile so each cube is loaded only once.
    `max_pixels_per_poly` caps work for huge polygons (subsamples interior
    pixels uniformly above the cap).
    """
    from rasterio.warp import transform_geom
    from shapely.geometry import shape as _shape
    PAD = PATCH // 2

    tile_map = {m['tile_id']: m for m in mrral_meta}

    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(polygons):
        tid = row.get('tile_id')
        if tid is None or tid not in tile_map:
            continue
        groups[tid].append(i)

    print(f'  {sum(len(v) for v in groups.values())} polygons mapped to '
          f'{len(groups)} tiles '
          f'(skipped {len(polygons) - sum(len(v) for v in groups.values())} '
          f'without tile)')

    probs = np.full((len(polygons), N_CLS_OUT), np.nan, dtype=np.float32)
    n_pixels = np.zeros(len(polygons), dtype=np.int32)
    skipped_no_pixel = 0

    rng = np.random.default_rng(0)

    for tile_idx, (tid, idxs) in enumerate(sorted(groups.items())):
        meta = tile_map[tid]
        path = meta['path']
        src = rasterio.open(path)
        try:
            data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
            nodata_mask = (data == NODATA) | ~np.isfinite(data)
            np.clip(data, 0.0, CLIP_MAX, out=data)
            data[nodata_mask] = 0.0
            cube = data.transpose(1, 2, 0)
            valid_mask = ~nodata_mask.any(axis=0)
            H, W, _ = cube.shape
            padded = np.pad(cube, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')

            # Process polygons within the tile one at a time and run inference
            # in small chunks to avoid holding all patches in memory.
            n_polys_this_tile = 0
            n_patches_this_tile = 0
            for i in idxs:
                row = polygons[i]
                geom = row.get('geometry')
                if geom is None:
                    skipped_no_pixel += 1; continue
                if row['source'] == 'hellas':
                    g = _shape(transform_geom(
                        'IAU_2015:49900', src.crs, geom.__geo_interface__))
                else:
                    g = geom
                mask = rasterio.features.rasterize(
                    [(g, 1)], out_shape=(H, W),
                    transform=src.transform, fill=0, dtype=np.uint8,
                ).astype(bool)
                mask = mask & valid_mask
                interior_rs, interior_cs = np.where(mask)
                if len(interior_rs) == 0:
                    skipped_no_pixel += 1; continue
                if len(interior_rs) > max_pixels_per_poly:
                    sel = rng.choice(len(interior_rs), size=max_pixels_per_poly,
                                      replace=False)
                    interior_rs = interior_rs[sel]; interior_cs = interior_cs[sel]

                # Build patch batch for this single polygon and aggregate
                # immediately (mean over its pixels). No large flat array.
                n_pix = len(interior_rs)
                acc = np.zeros(N_CLS_OUT, dtype=np.float64)
                with torch.no_grad():
                    for s in range(0, n_pix, batch_size):
                        e = min(s + batch_size, n_pix)
                        chunk = np.stack([
                            padded[rr:rr + PATCH, cc:cc + PATCH, :]
                            for rr, cc in zip(interior_rs[s:e], interior_cs[s:e])
                        ]).astype(np.float32)
                        batch_norm = normalize_patches(chunk)
                        x = torch.from_numpy(batch_norm).to(device)
                        p = torch.sigmoid(model(x)).cpu().numpy()
                        acc += p.sum(axis=0)
                probs[i] = (acc / n_pix).astype(np.float32)
                n_pixels[i] = n_pix
                n_polys_this_tile += 1
                n_patches_this_tile += n_pix

            # Free large arrays before next tile
            del padded, cube, data, nodata_mask, valid_mask
            import gc; gc.collect()

            if (tile_idx + 1) % 10 == 0 or n_polys_this_tile > 0:
                print(f'    [{tile_idx + 1}/{len(groups)}] tile {tid}: '
                      f'{n_polys_this_tile} polys, {n_patches_this_tile} patches')
        finally:
            src.close()

    print(f'  per-pixel inference: skipped {skipped_no_pixel} polygons '
          f'(no in-polygon pixels found).')
    print(f'  pixel-count summary: median {int(np.median(n_pixels[n_pixels > 0])):,}, '
          f'p90 {int(np.percentile(n_pixels[n_pixels > 0], 90)):,}, '
          f'max {int(n_pixels.max()):,}')
    return probs


# ── per-polygon figure renderer ──────────────────────────────────────────────

WAVELENGTHS_NM = None    # filled below from one of the inline GPKG entries


def render_wrong_polygon(row, probs_row, out_path):
    """Render numerator + ratio spectra for one wrong polygon. Spectra are
    truncated to the first N_BANDS wavelengths (the classifier's input range,
    410–2457 nm) — the upper 13 bands (2530–3923 nm) are too noisy to use
    and were excluded from training.
    """
    spec_mean = np.asarray(row['spec_mean_72'])
    spec_ratio = row['spec_ratio_72']
    wvl = row['wvl_72']
    if wvl is None:
        wvl = np.array(WAVELENGTHS_NM)
    wvl = np.asarray(wvl)

    # Truncate everything to the classifier's input range (first N_BANDS)
    n = min(len(spec_mean), len(wvl), N_BANDS)
    if n < 3:
        # Malformed spectrum — skip rather than crash
        return False
    wvl_p = wvl[:n]
    spec_mean_p = spec_mean[:n]
    spec_ratio_p = np.asarray(spec_ratio)[:n] if spec_ratio is not None else None
    if spec_ratio_p is not None and len(spec_ratio_p) != n:
        spec_ratio_p = None

    gt_idx = gt_5class_argmax(row['label6'])
    pred_idx = int(np.argmax(probs_row))
    pred_name = CLASS_OUT_NAMES[pred_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    # Numerator (raw mean) spectrum — first N_BANDS only
    axes[0].plot(wvl_p, spec_mean_p, color='#1f77b4', linewidth=1.2)
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Reflectance (I/F)')
    axes[0].set_title(f'Numerator spectrum (Spectrum Mean), {N_BANDS} bands')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(wvl_p[0], wvl_p[-1])

    # Ratio spectrum
    if spec_ratio_p is not None:
        axes[1].plot(wvl_p, spec_ratio_p, color='#ff7f0e', linewidth=1.2)
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Ratio (I/F / denom)')
        axes[1].set_title(f'Ratio spectrum, {N_BANDS} bands')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(wvl_p[0], wvl_p[-1])
    else:
        axes[1].text(0.5, 0.5,
                     '(no ratio spectrum stored in GPKG)',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=10, color='gray')
        axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle(
        f'{row["source"]} | tile={row["tile_id"]} | poly={row["polygon_id"]}\n'
        f'Hand-labeled: {row["category"]}  →  {gt_label_string(row["label6"])}\n'
        f'Predicted (argmax): {pred_name} (p={probs_row[pred_idx]:.3f})  |  '
        + ' '.join(f'{nm}={probs_row[i]:.2f}' for i, nm in enumerate(CLASS_OUT_NAMES)),
        fontsize=9,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading inline-spectrum polygons …')
    polys = []
    # categorized_mineral_units — Tile-prefixed GPKGs (NOT north_hellas; that's separate)
    for gpkg in sorted(glob.glob(os.path.join(CMU_DIR, 'T*.gpkg'))):
        polys.extend(load_inline_polygons(gpkg, 'cmu'))
    # argyre-gis supplemental
    for gpkg in sorted(glob.glob(os.path.join(ARGYRE_DIR, 'T*.gpkg'))):
        polys.extend(load_inline_polygons(gpkg, 'argyre-gis'))
    print(f'  inline polygons: {len(polys)}')

    # Capture wavelengths from the first inline source for use by Hellas figures
    global WAVELENGTHS_NM
    for r in polys:
        if r['wvl_72'] is not None and len(r['wvl_72']) >= N_BANDS:
            WAVELENGTHS_NM = r['wvl_72'][:N_BANDS].tolist()
            break

    print()
    print('Extracting Hellas spectra (rasterize each polygon onto its mrral tile) …')
    polys.extend(load_hellas_polygons())
    print(f'  total polygons: {len(polys)}')

    print()
    print('Loading bland-v3 classifier …')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_classifier(device)

    # Build mrral tile index ONCE; reused by Hellas loader (above) and also
    # required by patch-based inference.
    if 'mrral_meta' not in dir():
        print('Building mrral tile index for patch extraction …')
        mrral_meta = build_mrral_index()
        print(f'  {len(mrral_meta)} mrral tiles')

    print('Predicting on all polygons (extracting real 7×7 patches centred on each polygon) …')
    probs = predict_polygon_probs(polys, model, device, mrral_meta)

    # Polygons with NaN probs (skipped patch extraction) are excluded from scoring
    valid_probs = ~np.isnan(probs).any(axis=1)
    n_total = len(polys)
    n_valid = int(valid_probs.sum())
    print(f'  {n_valid}/{n_total} polygons scored '
          f'({n_total - n_valid} skipped — no patch extractable)')

    # Score: multi-label-aware. A polygon is "correct" if the predicted argmax
    # class is in the SET of hand-label-active classes (so 'hcp + lcp' is right
    # when the model predicts either hcp or lcp). Confusion-matrix rows still
    # use the GT-argmax for readability.
    correct = 0
    by_source = defaultdict(lambda: {'n': 0, 'correct': 0})
    by_class_gt = defaultdict(lambda: {'n': 0, 'correct': 0})
    conf = np.zeros((N_CLS_OUT, N_CLS_OUT), dtype=int)   # rows=gt-argmax, cols=pred-argmax
    wrong = []   # (row, probs_row, gt_argmax, pred_idx)

    for row, p, ok in zip(polys, probs, valid_probs):
        if not ok:
            continue
        gt_set = gt_5class_positives(row['label6'])
        gt_primary = gt_5class_argmax(row['label6'])
        pr = int(np.argmax(p))
        conf[gt_primary, pr] += 1
        is_correct = (pr in gt_set)
        if is_correct:
            correct += 1
        else:
            wrong.append((row, p, gt_primary, pr))
        by_source[row['source']]['n'] += 1
        by_source[row['source']]['correct'] += int(is_correct)
        by_class_gt[gt_primary]['n'] += 1
        by_class_gt[gt_primary]['correct'] += int(is_correct)

    # Write summary
    summary_lines = []
    sp = summary_lines.append
    sp(f'Bland-v3 per-polygon mean-spectrum eval — {len(polys)} polygons')
    sp(f'  checkpoint: {CKPT}')
    sp(f'  accuracy:   {correct}/{len(polys)} = {100 * correct / len(polys):.2f}%')
    sp('')
    sp('Per-source:')
    for src, d in sorted(by_source.items()):
        pct = 100 * d['correct'] / max(1, d['n'])
        sp(f'  {src:<14}  {d["correct"]}/{d["n"]} = {pct:.2f}%')
    sp('')
    sp('Per-class (ground-truth argmax):')
    for c in range(N_CLS_OUT):
        d = by_class_gt[c]
        if d['n'] == 0:
            continue
        pct = 100 * d['correct'] / max(1, d['n'])
        sp(f'  {CLASS_OUT_NAMES[c]:<12}  {d["correct"]}/{d["n"]} = {pct:.2f}%')
    sp('')
    sp('Confusion matrix (rows=ground truth, cols=predicted; argmax of 5-class):')
    header = '  ' + ' ' * 14 + ' '.join(f'{n:>10}' for n in CLASS_OUT_NAMES)
    sp(header)
    for c in range(N_CLS_OUT):
        cells = ' '.join(f'{conf[c, j]:>10,}' for j in range(N_CLS_OUT))
        sp(f'  GT {CLASS_OUT_NAMES[c]:<12} {cells}')
    summary = '\n'.join(summary_lines)
    print()
    print(summary)
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f'\nSummary saved → {os.path.join(OUT_DIR, "summary.txt")}')

    # Render figures for misclassified polygons; guard so one bad polygon
    # doesn't stop the loop.
    print()
    print(f'Rendering {len(wrong)} wrong-polygon figures …')
    n_rendered, n_skipped = 0, 0
    for row, p, gt, pr in wrong:
        gt_name = CLASS_OUT_NAMES[gt]
        pred_name = CLASS_OUT_NAMES[pr]
        tag = f'{row["source"]}_{row["tile_id"]}_{row["polygon_id"]:04d}'
        fname = f'wrong_gt-{gt_name}_pred-{pred_name}_{tag}.png'
        out_path = os.path.join(OUT_DIR, fname)
        try:
            ok = render_wrong_polygon(row, p, out_path)
            if ok:
                n_rendered += 1
            else:
                n_skipped += 1
        except Exception as e:
            n_skipped += 1
            if n_skipped < 5:
                print(f'    [warn] skipped {tag}: {type(e).__name__}: {str(e)[:80]}')

    print(f'Done. {n_rendered} figures in {OUT_DIR}/  ({n_skipped} skipped)')


if __name__ == '__main__':
    main()
