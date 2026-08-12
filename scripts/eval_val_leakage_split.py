"""Is a checkpoint's val advantage real generalisation, or scene memorisation?

Motivating question (2026-08-12): the dual-CR arm leads the hull-CR arm by ~0.12
val_mAP_core. Is that the 118-channel representation genuinely generalising, or
is it exploiting a weaker invariance to fingerprint scenes it has already seen?

The mechanism to worry about is specific. Linear-CR divides by a fitted LINE, so
it removes level and slope but preserves broad curvature. Hull-CR divides by the
upper convex hull, which removes broad curvature too. Scene-specific nuisance --
atmospheric residual, dust opacity, calibration state, observation geometry --
lives largely in that broad shape. So the dual representation hands the model
more scene-fingerprinting material than hull-CR does, by construction.

Whether that matters depends on whether train and val share scenes. They do, in
part. `scripts/split_units.py` assigns whole GEOGRAPHIC UNITS (polygons
single-linked at 0.25 deg ~ 15 km, unioned with any polygons sharing literal
pixels) to train/val/test, so there is no pixel-level or polygon-level leakage --
that part is sound. But a unit is a cluster, not a tile, and an MRDR tile spans
5 deg, so one tile can contribute units to BOTH splits. Measured on
mrral_pixels.parquet: 8 of 83 tiles straddle train and val, and those account for
40.5% of val rows.

That gives a clean natural experiment, run here:

    val_disjoint  val rows whose tile appears NOWHERE in train.
                  No scene the model has seen. Honest generalisation.
    val_shared    val rows whose tile ALSO appears in train, >0.25 deg away.
                  Same scene, unseen location. Scene fingerprinting is available.

Read the RESULT, not either number alone:

  * advantage holds on BOTH subsets      -> the gain is real; memorisation is
                                            largely ruled out
  * advantage only on val_shared         -> scene fingerprinting; the val lead is
                                            an artifact and the floor test will
                                            not reproduce it
  * advantage larger on val_disjoint     -> stronger result than val_mAP implies

This is a post-hoc evaluation on finished checkpoints. It does not retrain
anything, and it is not a substitute for the floor test -- that remains the
arbiter, because its tiles are outside the training parquet entirely. This is the
diagnostic that says WHY the floor test came out the way it did.

Usage
    # one checkpoint
    python scripts/eval_val_leakage_split.py --ckpt <ft_best.pt> \
        --parquet <mrral_pixels_7cls_handcore.parquet>

    # both arms side by side (the comparison that answers the question)
    python scripts/eval_val_leakage_split.py \
        --ckpt   <dual_best.pt>   --patch_cache_dir <...patch_cache_handcore_dualcr> \
        --ckpt_b <hull_best.pt>   --patch_cache_dir_b <...patch_cache_handcore_cr> \
        --parquet <mrral_pixels_7cls_handcore.parquet>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config  # noqa: E402
from device import get_device  # noqa: E402
from evaluation.metrics import compute_map  # noqa: E402

N_BANDS_HULL = 59
N_BANDS_DUAL = 118
CKPT_CHANNEL_KEY = 'encoder.band_embed.weight'


def build_mrral_map(data_root: str) -> dict[str, str]:
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
    return {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def load_classifier(ckpt_path: str, device):
    """Return (model, label_cols, n_bands, brightness_aux).

    The checkpoint is authoritative for BOTH the channel count and the vocab --
    guessing either would silently produce a wrong map rather than an error.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state', ck.get('state_dict', ck))

    if CKPT_CHANNEL_KEY not in state:
        raise SystemExit(
            f'{ckpt_path}: no {CKPT_CHANNEL_KEY}. Cannot determine the channel '
            f'count, and assuming one would produce a silently wrong result. '
            f'Keys: {sorted(state)[:6]}')
    # band_embed is nn.Linear(n_bands, embed_dim) -> weight (embed_dim, n_bands).
    n_bands = int(state[CKPT_CHANNEL_KEY].shape[-1])
    embed_dim = int(state[CKPT_CHANNEL_KEY].shape[0])
    if n_bands not in (N_BANDS_HULL, N_BANDS_DUAL):
        raise SystemExit(f'{ckpt_path}: unexpected channel count {n_bands}')

    from data.dataset import label_cols_for_ckpt
    label_cols = label_cols_for_ckpt(state)
    brightness_aux = any(k.startswith('aux_mlp.') or k.startswith('aux.')
                         for k in state)
    n_layers = 1 + max(
        (int(k.split('layers.')[1].split('.')[0])
         for k in state if 'encoder.encoder.layers.' in k), default=5)

    if brightness_aux:
        from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
        model = SpatialSpectralClassifierAux(
            n_bands=n_bands, patch_size=7, n_classes=len(label_cols),
            embed_dim=embed_dim, n_heads=4, n_layers=n_layers, aux_dim=1)
    else:
        from models.spatial_spectral_transformer import SpatialSpectralClassifier
        model = SpatialSpectralClassifier(
            n_bands=n_bands, patch_size=7, n_classes=len(label_cols),
            embed_dim=embed_dim, n_heads=4, n_layers=n_layers)
    model.load_state_dict(state)
    return model.eval().to(device), label_cols, n_bands, brightness_aux


def score(model, ds, brightness_aux, device, batch_size, workers):
    """Return (y_true, y_score) as (N, C) arrays, in dataset order."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            if brightness_aux:
                x, aux, y, _w = batch
                out = model(x.to(device), aux.to(device))
            else:
                x, y, _w = batch
                out = model(x.to(device))
            ps.append(torch.sigmoid(out).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate(ckpt, cache_dir, df_val, is_disjoint, mrral_map, device, args, tag):
    model, label_cols, n_bands, baux = load_classifier(ckpt, device)
    import data.dataset
    data.dataset.LABEL_COLS = list(label_cols)
    from data.dataset import CRISMSpectralPatchDataset

    dual = n_bands == N_BANDS_DUAL
    print(f'\n=== {tag}: {os.path.basename(ckpt)}')
    print(f'    {n_bands} channels ({"dual-CR" if dual else "hull-CR"}), '
          f'{len(label_cols)}-class {label_cols}, brightness_aux={baux}')
    print(f'    cache: {cache_dir}')

    ds = CRISMSpectralPatchDataset(
        df_val, mrral_map, patch_size=7, cache_dir=cache_dir, split='val',
        continuum_removed=True, cache_is_cr=True,
        return_brightness=baux, dual_cr=dual)
    y_true, y_score = score(model, ds, baux, device, args.batch_size, args.workers)

    out = {}
    for subset, m in (('val_disjoint', is_disjoint), ('val_shared', ~is_disjoint),
                      ('val_all', np.ones(len(is_disjoint), bool))):
        if m.sum() == 0:
            out[subset] = None
            continue
        yt, ys = y_true[m], y_score[m]
        out[subset] = {
            'n': int(m.sum()),
            'mAP': compute_map(yt, ys),
            'mAP_core': compute_map(yt, ys, exclude=('junk',)),
            # Per class, so a single class carrying the difference is visible.
            'per_class': {
                c: (float('nan') if yt[:, i].sum() == 0
                    else compute_map(yt[:, i:i + 1], ys[:, i:i + 1]))
                for i, c in enumerate(label_cols)},
            'pos': {c: int(yt[:, i].sum()) for i, c in enumerate(label_cols)},
        }
    return out, label_cols


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--patch_cache_dir', default=None)
    ap.add_argument('--ckpt_b', default=None, help='second arm, evaluated identically')
    ap.add_argument('--patch_cache_dir_b', default=None)
    ap.add_argument('--parquet', default=None)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--data_root', default=None)
    args = ap.parse_args()

    cfg = load_config()
    root = args.data_root or cfg['data_root']
    device = get_device()
    parquet = args.parquet or os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')

    df = pd.read_parquet(parquet)
    for col in ('split', 'tile_id'):
        if col not in df.columns:
            raise SystemExit(f'{parquet} has no {col!r} column')
    train_tiles = set(df.loc[df['split'] == 'train', 'tile_id'].unique())
    df_val = df[df['split'] == 'val'].reset_index(drop=True)
    if not len(df_val):
        raise SystemExit(f'{parquet} has no val rows')
    is_disjoint = ~df_val['tile_id'].isin(train_tiles).to_numpy()

    print(f'parquet: {parquet}')
    print(f'val rows: {len(df_val):,}   train tiles: {len(train_tiles)}')
    print(f'  val_disjoint (tile unseen in train): {is_disjoint.sum():,} '
          f'({100 * is_disjoint.mean():.1f}%)')
    print(f'  val_shared   (tile also in train):   {(~is_disjoint).sum():,} '
          f'({100 * (~is_disjoint).mean():.1f}%)')
    if is_disjoint.sum() == 0:
        raise SystemExit(
            'Every val tile also appears in train, so there is no disjoint '
            'subset and this diagnostic cannot separate the two hypotheses.')
    if (~is_disjoint).sum() == 0:
        print('  NOTE: no shared subset — val is already fully tile-disjoint, so '
              'scene fingerprinting was never available. Nothing to rule out.')

    mrral_map = build_mrral_map(root)
    cache_a = args.patch_cache_dir or cfg.get('patch_cache_dir')
    res_a, cols = evaluate(args.ckpt, cache_a, df_val, is_disjoint, mrral_map,
                           device, args, 'ARM A')
    res_b = None
    if args.ckpt_b:
        cache_b = args.patch_cache_dir_b or cfg.get('patch_cache_dir')
        res_b, _ = evaluate(args.ckpt_b, cache_b, df_val, is_disjoint, mrral_map,
                            device, args, 'ARM B')

    print('\n' + '=' * 70)
    print(f'{"subset":<16}{"n":>10}{"A mAP_core":>13}'
          + (f'{"B mAP_core":>13}{"A − B":>10}' if res_b else ''))
    print('-' * 70)
    gaps = {}
    for subset in ('val_all', 'val_disjoint', 'val_shared'):
        a = res_a.get(subset)
        if a is None:
            continue
        line = f'{subset:<16}{a["n"]:>10,}{a["mAP_core"]:>13.4f}'
        if res_b and res_b.get(subset):
            b = res_b[subset]
            gaps[subset] = a['mAP_core'] - b['mAP_core']
            line += f'{b["mAP_core"]:>13.4f}{gaps[subset]:>+10.4f}'
        print(line)

    print(f'\n{"class":<14}{"pos(dj)":>9}{"A dj":>8}{"A sh":>8}'
          + (f'{"B dj":>8}{"B sh":>8}{"gap dj":>9}{"gap sh":>9}' if res_b else ''))
    print('-' * (39 + (42 if res_b else 0)))
    for c in cols:
        a_dj, a_sh = res_a['val_disjoint'], res_a['val_shared']
        row = (f'{c:<14}{a_dj["pos"][c]:>9,}'
               f'{a_dj["per_class"][c]:>8.3f}{a_sh["per_class"][c]:>8.3f}')
        if res_b:
            b_dj, b_sh = res_b['val_disjoint'], res_b['val_shared']
            row += (f'{b_dj["per_class"][c]:>8.3f}{b_sh["per_class"][c]:>8.3f}'
                    f'{a_dj["per_class"][c] - b_dj["per_class"][c]:>+9.3f}'
                    f'{a_sh["per_class"][c] - b_sh["per_class"][c]:>+9.3f}')
        print(row)

    if res_b and 'val_disjoint' in gaps and 'val_shared' in gaps:
        dj, sh = gaps['val_disjoint'], gaps['val_shared']
        print('\n' + '=' * 70)
        print(f'A − B on tile-DISJOINT val: {dj:+.4f}')
        print(f'A − B on tile-SHARED   val: {sh:+.4f}')
        # Deliberately does not print a verdict. The subsets differ in class mix
        # and size as well as in scene overlap, so the ratio is evidence to weigh
        # against the floor test, not a test statistic with a threshold.
        if dj > 0 and sh > 0:
            print(f'  A leads on BOTH subsets (disjoint {dj:+.4f} / shared '
                  f'{sh:+.4f}). The lead is not solely scene overlap.')
            if dj < 0.5 * sh:
                print('  But it is >2x larger on shared — part of the val gap '
                      'is plausibly scene-driven. Weight the floor test heavily.')
        elif dj <= 0 < sh:
            print('  A leads ONLY where train and val share a scene. That is the '
                  'signature of scene fingerprinting; expect the floor test not '
                  'to reproduce the val gap.')
        elif dj > 0 >= sh:
            print('  A leads only on unseen scenes — stronger than val_mAP '
                  'suggests, not weaker.')
        print('\nPer-class positives differ between subsets, so a class with few '
              'disjoint positives (see pos(dj)) carries little weight above.')
        print('The floor test remains the arbiter: its tiles are outside the '
              'training parquet entirely.')


if __name__ == '__main__':
    main()
