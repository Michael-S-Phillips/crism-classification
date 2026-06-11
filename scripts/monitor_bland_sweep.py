"""
Quick status check of the `ft_bland_*` finetune sweep on wandb.

Queries space-imagery-center/crism-mineral-classification for runs matching
`^ft_bland`, sorts by val_mAP, and prints a snapshot.

Usage:
    conda run -n crism python scripts/monitor_bland_sweep.py
    conda run -n crism python scripts/monitor_bland_sweep.py --sort created
"""
from __future__ import annotations

import argparse

import wandb

ENTITY  = 'space-imagery-center'
PROJECT = 'crism-mineral-classification'
PRIOR_CHAMPION_VAL_MAP = 0.5981   # ft_v3_denoising_lrscale001 (from ckpt summary)


def fmt(val, digits=4, width=7):
    if isinstance(val, (int, float)):
        return f'{val:.{digits}f}'.rjust(width)
    return 'n/a'.rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort', choices=['val_map', 'test_map', 'created'],
                        default='val_map')
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(api.runs(
        f'{ENTITY}/{PROJECT}',
        filters={'display_name': {'$regex': '^ft_bland'}},
    ))

    rows = []
    for r in runs:
        s = r.summary
        rows.append({
            'name':     r.name,
            'state':    r.state,
            'epoch':    s.get('epoch'),
            'val_map':  s.get('val_mAP'),
            'test_map': s.get('test_mAP'),
            'created':  str(r.created_at),
        })

    if args.sort == 'val_map':
        rows.sort(key=lambda x: -(x['val_map'] if isinstance(x['val_map'], (int, float)) else -1))
    elif args.sort == 'test_map':
        rows.sort(key=lambda x: -(x['test_map'] if isinstance(x['test_map'], (int, float)) else -1))
    else:
        rows.sort(key=lambda x: x['created'])

    print(f'Sweep status: {len(rows)} / 9 ft_bland_* runs in {ENTITY}/{PROJECT}')
    print(f'(prior champion ft_v3_denoising_lrscale001 had val_mAP = {PRIOR_CHAMPION_VAL_MAP:.4f})')
    print()

    print(f'  {"run":<35} {"state":<10} {"ep":>4}   val_mAP   test_mAP   Δ_vs_prior')
    print(f'  {"-"*35} {"-"*10} {"-"*4}   {"-"*7}   {"-"*8}   {"-"*10}')
    for x in rows:
        vmap = x['val_map'] if isinstance(x['val_map'], (int, float)) else None
        delta = (f'{vmap - PRIOR_CHAMPION_VAL_MAP:+.4f}' if vmap is not None else '   n/a   ')
        print(f'  {x["name"]:<35} {x["state"]:<10} {str(x["epoch"]):>4}   '
              f'{fmt(x["val_map"])}   {fmt(x["test_map"])}   {delta}')

    finished = [r for r in rows if r['state'] in ('finished', 'crashed')]
    running  = [r for r in rows if r['state'] == 'running']
    print()
    print(f'  finished: {len(finished)} / 9   running: {len(running)}   '
          f'remaining: {9 - len(finished) - len(running)}')

    valid = [r for r in rows if isinstance(r['val_map'], (int, float))]
    if valid:
        best = max(valid, key=lambda x: x['val_map'])
        print(f'  best so far: {best["name"]}  val_mAP={best["val_map"]:.4f}  '
              f'(Δ = {best["val_map"] - PRIOR_CHAMPION_VAL_MAP:+.4f} vs prior champion)')


if __name__ == '__main__':
    main()
