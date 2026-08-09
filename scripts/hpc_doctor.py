"""Pre-flight environment check. Run this BEFORE submitting a build.

Every check here exists because it cost a submit-fail-diagnose cycle on
2026-08-08. Nothing speculative.

  A. Config & paths     — where does every input actually resolve, and is it there?
                          (the review sessions were in /groups, not the xdisk
                          layout the slurm assumed; the job died on line one)
  B. Tile inventory     — mc*/ subdirs or flat at data_root? how many tiles?
                          (HPC is flat, the workstation is mc*/; scripts that
                          globbed only mc*/ reported every tile as missing)
  C. Refresh watermark  — the newest tile mtime. ANY derived artifact older than
                          it was built from pre-refresh tiles and may have
                          zero-fill corruption baked in. This is the big one:
                          hpc_build_global_cache_cr.slurm already warned that
                          "building from truncated tiles bakes zero-fill
                          corruption into the CR cache and every downstream
                          pretrain", and both defects found on 2026-08-08 were
                          exactly this — artifacts extracted before the Jul 8
                          tile refresh and never regenerated.
  D. Review sessions    — located, fragment counts, and their age vs the watermark.
  E. Manifest           — --emit lets two machines be diffed. HPC's base parquet
                          was a month older than the workstation's and nobody
                          knew until a build failed.

Usage
    python scripts/hpc_doctor.py                      # full report
    python scripts/hpc_doctor.py --emit host.json     # write a manifest
    python scripts/hpc_doctor.py --compare other.json # diff against another host

Exit codes: 0 clean, 1 problems found, 2 bad invocation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATCH_ITEM_BYTES = 7 * 7 * 59 * 4     # one 7x7x59 float32 patch

# Derived artifacts worth checking, relative to the config's output_dir.
PARQUET_GLOBS = ['*.parquet']
CACHE_GLOBS = ['patch_cache*']
REVIEW_SESSIONS = ['mc13_review', 'mc13_review_7cls_v3', 'ndviz_relabels']


def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _stamp(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else '-'


def _human(n: int) -> str:
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if abs(n) < 1024 or unit == 'T':
            return f'{n:.0f}{unit}' if unit == 'B' else f'{n:.1f}{unit}'
        n /= 1024.0
    return str(n)


def tile_inventory(data_root: str) -> dict:
    """Detect the tile layout and find the newest tile mtime (the watermark)."""
    nested = glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.img'))
    flat = glob.glob(os.path.join(data_root, 't*mrral*.img'))
    layout = ('mc*/ subdirs' if len(nested) >= len(flat) and nested
              else 'flat at data_root' if flat else 'NONE FOUND')
    tiles = nested if len(nested) >= len(flat) else flat
    newest_t, newest_f = 0.0, None
    for f in tiles:
        m = _mtime(f)
        if m > newest_t:
            newest_t, newest_f = m, f
    return {'layout': layout, 'n_tiles': len(tiles),
            'n_nested': len(nested), 'n_flat': len(flat),
            'watermark': newest_t, 'watermark_file': newest_f}


def parquet_rows(path: str) -> int | None:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def survey_artifacts(output_dir: str) -> list[dict]:
    out = []
    for g in PARQUET_GLOBS:
        for p in sorted(glob.glob(os.path.join(output_dir, g))):
            out.append({'kind': 'parquet', 'path': p,
                        'name': os.path.basename(p),
                        'bytes': os.path.getsize(p), 'mtime': _mtime(p),
                        'rows': parquet_rows(p)})
    for g in CACHE_GLOBS:
        for d in sorted(glob.glob(os.path.join(output_dir, g))):
            if not os.path.isdir(d):
                continue
            files = sorted(glob.glob(os.path.join(d, '*.npy')))
            newest = max((_mtime(f) for f in files), default=0.0)
            total = sum(os.path.getsize(f) for f in files)
            splits = sorted({os.path.basename(f).split('_')[1]
                             for f in files if len(os.path.basename(f).split('_')) > 1})
            out.append({'kind': 'cache', 'path': d, 'name': os.path.basename(d),
                        'bytes': total, 'mtime': newest, 'rows': None,
                        'n_files': len(files), 'splits': splits})
    return out


def survey_reviews(search_dirs: list[str]) -> list[dict]:
    found = []
    for name in REVIEW_SESSIONS:
        for base in search_dirs:
            cand = os.path.join(base, name)
            hn = os.path.join(cand, 'hard_negatives')
            cf = os.path.join(cand, 'confirmed_pixels')
            if os.path.isdir(hn) or os.path.isdir(cf):
                frags = glob.glob(os.path.join(hn, '*.parquet')) + \
                        glob.glob(os.path.join(cf, '*.parquet'))
                found.append({'name': name, 'path': cand,
                              'n_fragments': len(frags),
                              'mtime': max((_mtime(f) for f in frags), default=0.0),
                              'bytes': sum(os.path.getsize(f) for f in frags)})
                break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emit', metavar='FILE',
                    help='write a machine manifest for cross-host comparison')
    ap.add_argument('--compare', metavar='FILE',
                    help='diff this host against a manifest from another host')
    ap.add_argument('--grace_hours', type=float, default=1.0,
                    help='artifacts younger than watermark-minus-this are OK '
                         '(default 1.0, absorbs clock skew and long builds)')
    args = ap.parse_args()

    from config_loader import load_config
    cfg = load_config()
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems: list[str] = []   # definitely wrong -> exit 1
    warnings: list[str] = []   # worth verifying -> reported, exit 0

    print('=' * 78)
    print('A. CONFIG & PATHS')
    print('=' * 78)
    keys = ('data_root', 'output_dir', 'checkpoints_dir', 'patch_cache_dir',
            'gpkg_dir', 'predictions_dir', 'reports_dir')
    for k in keys:
        v = cfg.get(k, '(unset)')
        ok = os.path.exists(v) if v != '(unset)' else False
        print(f'  {k:<17} {v:<58} {"OK" if ok else "MISSING"}')
        if not ok and k in ('data_root', 'output_dir'):
            problems.append(f'{k} does not exist: {v}')
    print(f'  {"repo":<17} {proj}')

    print()
    print('=' * 78)
    print('B. TILE INVENTORY')
    print('=' * 78)
    inv = tile_inventory(cfg['data_root'])
    print(f'  layout            {inv["layout"]}   (mc*/: {inv["n_nested"]}, '
          f'flat: {inv["n_flat"]})')
    print(f'  tiles found       {inv["n_tiles"]:,}')
    if inv['n_tiles'] == 0:
        problems.append('no mrral tiles found under data_root')
    print(f'  newest tile       {_stamp(inv["watermark"])}'
          f'   {os.path.basename(inv["watermark_file"] or "-")}')
    print('  ^ the REFRESH WATERMARK. Artifacts older than this were built from')
    print('    pre-refresh tiles and may carry zero-fill corruption.')

    print()
    print('=' * 78)
    print('C. DERIVED ARTIFACTS vs WATERMARK')
    print('=' * 78)
    arts = survey_artifacts(cfg['output_dir'])
    cutoff = inv['watermark'] - args.grace_hours * 3600
    print(f'  {"name":<44}{"size":>9}{"rows":>12}  {"built":<17} verdict')
    print('  ' + '-' * 90)
    stale = []
    for a in sorted(arts, key=lambda x: x['mtime']):
        rows = f'{a["rows"]:,}' if a.get('rows') else (
            f'{a["n_files"]}f' if a['kind'] == 'cache' else '-')
        old = inv['watermark'] > 0 and a['mtime'] < cutoff
        verdict = 'PRE-REFRESH' if old else 'ok'
        if old:
            stale.append(a['name'])
        print(f'  {a["name"][:43]:<44}{_human(a["bytes"]):>9}{rows:>12}  '
              f'{_stamp(a["mtime"]):<17} {verdict}')
    if stale:
        # A WARNING, not a problem. Pre-refresh means "extracted before the
        # tiles were repaired, so verify it" -- not "corrupt". The legacy review
        # session is pre-refresh and passed the tile cross-check cleanly. A
        # check that fires on every run is a check people learn to ignore.
        warnings.append(
            f'{len(stale)} artifact(s) predate the tile refresh — verify before '
            f'use: {", ".join(stale[:6])}' + (' …' if len(stale) > 6 else ''))
        print()
        print('  Artifacts built before the tile refresh may contain zero-filled')
        print('  spectra. Check with:  python scripts/audit_spectra_quality.py'
              ' <path> --fail_tile_over 5')

    print()
    print('=' * 78)
    print('D. REVIEW SESSIONS')
    print('=' * 78)
    revs = survey_reviews([cfg['output_dir'], os.path.join(proj, 'data')])
    if not revs:
        print('  none found')
    for r in revs:
        old = inv['watermark'] > 0 and r['mtime'] < cutoff
        print(f'  {r["name"]:<24}{r["n_fragments"]:>6} frags{_human(r["bytes"]):>9}  '
              f'{_stamp(r["mtime"]):<17} {"PRE-REFRESH" if old else "ok"}')
        print(f'      {r["path"]}')
        if old:
            warnings.append(f'review session {r["name"]} predates the tile '
                            f'refresh — verify with --verify_against_tiles')

    manifest = {
        'host': os.uname().nodename,
        'data_root': cfg['data_root'],
        'output_dir': cfg['output_dir'],
        'tile_layout': inv['layout'],
        'n_tiles': inv['n_tiles'],
        'watermark': inv['watermark'],
        'artifacts': {a['name']: {'bytes': a['bytes'], 'rows': a.get('rows'),
                                  'mtime': a['mtime']} for a in arts},
        'reviews': {r['name']: {'bytes': r['bytes'], 'n': r['n_fragments'],
                                'mtime': r['mtime']} for r in revs},
    }
    if args.emit:
        with open(args.emit, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'\nmanifest written to {args.emit}')

    if args.compare:
        print()
        print('=' * 78)
        print('E. CROSS-HOST DIVERGENCE')
        print('=' * 78)
        with open(args.compare) as f:
            other = json.load(f)
        print(f'  this host: {manifest["host"]}   vs   {other.get("host","?")}')
        if other.get('tile_layout') != manifest['tile_layout']:
            print(f'  tile layout DIFFERS: {manifest["tile_layout"]} vs '
                  f'{other.get("tile_layout")}  (expected — layouts are per-machine)')
        shared = set(manifest['artifacts']) & set(other.get('artifacts', {}))
        diffs = 0
        for name in sorted(shared):
            a, b = manifest['artifacts'][name], other['artifacts'][name]
            if a['bytes'] != b['bytes'] or a.get('rows') != b.get('rows'):
                diffs += 1
                print(f'  DIVERGED  {name}')
                print(f'      here : {_human(a["bytes"]):>9} '
                      f'{a.get("rows") or "-"} rows  {_stamp(a["mtime"])}')
                print(f'      there: {_human(b["bytes"]):>9} '
                      f'{b.get("rows") or "-"} rows  {_stamp(b["mtime"])}')
        only_here = sorted(set(manifest['artifacts']) - set(other.get('artifacts', {})))
        only_there = sorted(set(other.get('artifacts', {})) - set(manifest['artifacts']))
        if only_here:
            print(f'  only here : {", ".join(only_here[:8])}'
                  + (' …' if len(only_here) > 8 else ''))
        if only_there:
            print(f'  only there: {", ".join(only_there[:8])}'
                  + (' …' if len(only_there) > 8 else ''))
        if diffs:
            problems.append(f'{diffs} shared artifact(s) differ between hosts')
        elif not only_here and not only_there:
            print('  no divergence in shared artifacts.')

    print()
    print('=' * 78)
    if warnings:
        print(f'{len(warnings)} WARNING(S) — verify, but not blocking:')
        for w in warnings:
            print(f'  ~ {w}')
    if problems:
        print(f'RESULT: {len(problems)} PROBLEM(S)')
        for p in problems:
            print(f'  - {p}')
    else:
        print('RESULT: OK' + ('  (with warnings above)' if warnings else ''))
    print('=' * 78)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
