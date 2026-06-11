"""Run Phase 0 polygon + calibration eval over the three baseline checkpoints.

Wraps ``eval_polygon_accuracy.py`` and ``eval_calibration.py`` for:

  * ``ft_plag_aware_real_only_best.pt``    (current supervised champion)
  * ``ft_plag_aware_relabeled_best.pt``    (relabel-trained supervised)
  * ``cont1_encoder_only.pt`` or any contrastive encoder ckpt (linear probe)

Then concatenates each polygon-eval ``summary.md`` plus a one-line calibration
header into a single comparison report at
``reports/phase0_baselines_summary.md``.

Why this exists: Phase 0 of the plag improvement roadmap requires the three
baseline polygon-accuracy numbers in one place so subsequent interventions
have a real target. Re-running all three manually is tedious and easy to
forget; this script does it in one pass and keeps every output under
``reports/`` for diffing.

Usage:

  conda run -n crism python scripts/eval_baselines_phase0.py
  conda run -n crism python scripts/eval_baselines_phase0.py --skip contrastive
  conda run -n crism python scripts/eval_baselines_phase0.py --region_filter T0433  # smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(PROJ, 'checkpoints')

DEFAULT_BASELINES = [
    {
        'name': 'ft_plag_aware_real_only',
        'ckpt': os.path.join(CKPT_DIR, 'ft_plag_aware_real_only_best.pt'),
        'kind': 'classifier',
    },
    {
        'name': 'ft_plag_aware_relabeled',
        'ckpt': os.path.join(CKPT_DIR, 'ft_plag_aware_relabeled_best.pt'),
        'kind': 'classifier',
    },
    {
        # No contrastive_plag_v1_best.pt yet — fall back to the earlier cont1
        # encoder-only ckpt if that's what's on disk. The polygon harness
        # detects the contrastive format and trains an inline linear probe.
        'name': 'contrastive_plag_v1',
        'ckpt': os.path.join(CKPT_DIR, 'contrastive_plag_v1_best.pt'),
        'fallback_ckpt': os.path.join(CKPT_DIR, 'cont1_encoder_only.pt'),
        'kind': 'contrastive_encoder',
    },
]


def _resolve_ckpt(b: dict) -> str | None:
    if os.path.exists(b['ckpt']):
        return b['ckpt']
    fb = b.get('fallback_ckpt')
    if fb and os.path.exists(fb):
        print(f"  note: {b['ckpt']} missing; using fallback {fb}")
        return fb
    return None


def run_polygon(ckpt_path: str, *, region_filter: str | None,
                apply_relabels: str | None, extra: list[str]) -> tuple[int, str]:
    cmd = [
        sys.executable, os.path.join(PROJ, 'scripts/eval_polygon_accuracy.py'),
        '--ckpt', ckpt_path,
    ]
    if region_filter:
        cmd += ['--region_filter', region_filter]
    if apply_relabels:
        cmd += ['--apply_relabels', apply_relabels]
    cmd += extra
    print('  $', ' '.join(cmd))
    res = subprocess.run(cmd, cwd=PROJ)
    return res.returncode, ckpt_path


def run_calibration(ckpt_path: str, *, apply_relabels: str | None,
                    extra: list[str]) -> tuple[int, str]:
    cmd = [
        sys.executable, os.path.join(PROJ, 'scripts/eval_calibration.py'),
        '--ckpt', ckpt_path,
    ]
    if apply_relabels:
        cmd += ['--apply_relabels', apply_relabels]
    cmd += extra
    print('  $', ' '.join(cmd))
    res = subprocess.run(cmd, cwd=PROJ)
    return res.returncode, ckpt_path


def _read_or(p: str, label: str) -> str:
    if not os.path.exists(p):
        return f'_(missing: {label})_'
    with open(p) as f:
        return f.read()


def concat_summary(reports_dir: str, baselines: list[dict],
                   out_path: str, region_filter: str | None,
                   apply_relabels: str | None) -> str:
    lines = []
    lines.append('# Phase 0 baselines — polygon accuracy + calibration')
    lines.append('')
    if region_filter:
        lines.append(f'_Region filter:_ `{region_filter}`')
    if apply_relabels:
        lines.append(f'_Relabels applied:_ `{apply_relabels}`')
    lines.append('')
    lines.append('## Quick comparison (overall polygon accuracy)')
    lines.append('')
    lines.append('| Baseline | Detected kind | n_polygons | Overall acc | Mean Brier | Mean ECE |')
    lines.append('|---|---|---|---|---|---|')

    payloads = []
    for b in baselines:
        ckpt_path = b.get('_resolved_ckpt') or b['ckpt']
        stem = Path(ckpt_path).stem
        poly_json = os.path.join(reports_dir, f'polygon_eval_{stem}', 'summary.json')
        cal_json = os.path.join(reports_dir, f'calibration_{stem}', 'summary.json')
        poly = _safe_json(poly_json)
        cal = _safe_json(cal_json)
        payloads.append({'name': b['name'], 'poly': poly, 'cal': cal,
                         'stem': stem})
        row = [
            b['name'],
            poly.get('kind', '—') if poly else '—',
            f'{poly["n_polygons"]:,}' if poly else '—',
            f'{poly["overall_accuracy"]:.4f}' if poly and poly.get('overall_accuracy') is not None else '—',
            f'{cal["mean_brier"]:.4f}' if cal else '—',
            f'{cal["mean_ece"]:.4f}' if cal else '—',
        ]
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

    # Per-baseline detailed sections
    for p in payloads:
        lines.append(f'## {p["name"]}')
        lines.append('')
        lines.append(f'### Polygon eval')
        lines.append('')
        lines.append(_read_or(os.path.join(reports_dir,
                                           f'polygon_eval_{p["stem"]}',
                                           'summary.md'),
                              f'polygon_eval_{p["stem"]}/summary.md'))
        lines.append('')
        lines.append(f'### Calibration')
        lines.append('')
        lines.append(_read_or(os.path.join(reports_dir,
                                           f'calibration_{p["stem"]}',
                                           'summary.md'),
                              f'calibration_{p["stem"]}/summary.md'))
        lines.append('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    return out_path


def _safe_json(path: str):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region_filter', default=None,
                    help='Forward to both eval scripts (smoke test).')
    ap.add_argument('--apply_relabels', default=None)
    ap.add_argument('--skip', nargs='*', default=[],
                    help='Skip baselines by name (e.g. "contrastive_plag_v1").')
    ap.add_argument('--reports_dir', default=None,
                    help='Where individual reports live. '
                         'Defaults to <project>/reports.')
    ap.add_argument('--out_path', default=None)
    ap.add_argument('--skip_polygon', action='store_true')
    ap.add_argument('--skip_calibration', action='store_true')
    ap.add_argument('--probe_debug_rows', type=int, default=None,
                    help='Forward to the contrastive baseline for fast smoke tests.')
    ap.add_argument('--polygon_args', default='',
                    help='Extra args forwarded to eval_polygon_accuracy.py, '
                         'as a single shell-quoted string e.g. '
                         '"--device cpu --batch_size 128".')
    ap.add_argument('--calibration_args', default='',
                    help='Extra args forwarded to eval_calibration.py, as '
                         'a single shell-quoted string.')
    ap.add_argument('--device', default=None,
                    help='Convenience: shortcut for adding `--device <x>` to '
                         'both wrappers without quoting.')
    ap.add_argument('--debug_rows', type=int, default=None,
                    help='Convenience: shortcut for adding `--debug_rows N` to '
                         'eval_calibration only.')
    args = ap.parse_args()

    reports_dir = args.reports_dir or os.path.join(PROJ, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    baselines = [b for b in DEFAULT_BASELINES if b['name'] not in args.skip]
    print(f'running {len(baselines)} baseline(s)…')

    import shlex
    poly_base_extra = shlex.split(args.polygon_args) if args.polygon_args else []
    cal_base_extra = shlex.split(args.calibration_args) if args.calibration_args else []
    if args.device:
        if '--device' not in poly_base_extra:
            poly_base_extra += ['--device', args.device]
        if '--device' not in cal_base_extra:
            cal_base_extra += ['--device', args.device]
    if args.debug_rows is not None:
        if '--debug_rows' not in cal_base_extra:
            cal_base_extra += ['--debug_rows', str(args.debug_rows)]

    for b in baselines:
        ckpt = _resolve_ckpt(b)
        if ckpt is None:
            print(f'  [skip] {b["name"]}: no checkpoint at {b["ckpt"]}')
            b['_resolved_ckpt'] = None
            continue
        b['_resolved_ckpt'] = ckpt
        poly_extra = list(poly_base_extra)
        cal_extra = list(cal_base_extra)
        if b['kind'] == 'contrastive_encoder' and args.probe_debug_rows is not None:
            poly_extra += ['--probe_debug_rows', str(args.probe_debug_rows)]
            cal_extra += ['--probe_debug_rows', str(args.probe_debug_rows)]
        if not args.skip_polygon:
            print(f'\n=== polygon eval: {b["name"]} ===')
            rc, _ = run_polygon(ckpt, region_filter=args.region_filter,
                                 apply_relabels=args.apply_relabels,
                                 extra=poly_extra)
            if rc != 0:
                print(f'  [warn] polygon eval returned {rc} for {b["name"]}')
        if not args.skip_calibration:
            print(f'\n=== calibration: {b["name"]} ===')
            rc, _ = run_calibration(ckpt, apply_relabels=args.apply_relabels,
                                     extra=cal_extra)
            if rc != 0:
                print(f'  [warn] calibration returned {rc} for {b["name"]}')

    baselines_with_ckpt = [b for b in baselines if b.get('_resolved_ckpt')]
    out_path = args.out_path or os.path.join(reports_dir,
                                              'phase0_baselines_summary.md')
    concat_summary(reports_dir, baselines_with_ckpt, out_path,
                   region_filter=args.region_filter,
                   apply_relabels=args.apply_relabels)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
