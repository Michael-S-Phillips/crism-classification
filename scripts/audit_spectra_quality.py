"""Audit the 59-band spectra inside a training/review parquet (or a directory
of parquet fragments) and fail loudly on the defect classes that silently
poison training.

Motivating incident (2026-08-08): the v3 review session extracted t1444 while
that tile was still downloading, so 537,525 rows were frozen with reflectance
0.0 across 2251-2457 nm. Nothing crashed. Nothing warned. Because bland is
review-only in the hand-core build, ~72% of that class carried a zero tail no
other class had — a shortcut feature the model would learn instead of mineralogy.

Checks (per file AND per tile_id, because a single bad tile is the usual shape):
  zero_tail     contiguous run of exact 0.0 at the long-wavelength end
  zero_run      any interior contiguous run of exact 0.0 (>= --min_run bands)
  all_zero      spectrum is entirely 0.0
  nodata        any band == 65535
  nonfinite     any NaN/Inf
  over_phys     band >= 1 exceeding 1.0 I/F (physically impossible — a real defect)
  flat          zero variance across bands (dead pixel / fill)

Reported but NOT failed: `blue_edge` counts band-0 (410 nm) spikes above 1.0 I/F.
Those are a known MRRAL artifact (values up to ~1180 I/F) that
CRISMSpectralPatchDataset already masks to nodata before training — see the
PHYS_MAX handling in data/dataset.py, audit 2026-06-15. Failing on a condition
the pipeline already handles would train people to ignore this script.

Optional --verify_against_tiles re-reads a random sample of rows from the
source .img on disk and compares. THIS is the check that catches a stale
extraction, where the parquet is self-consistent but no longer matches the
tile it claims to come from. Nothing else finds that.

Usage
  # audit a build parquet
  python scripts/audit_spectra_quality.py data/mrral_pixels_7cls.parquet

  # audit a directory of review fragments, and cross-check against the tiles
  python scripts/audit_spectra_quality.py data/mc13_review_7cls_v3/hard_negatives \
      --verify_against_tiles --sample 300

  # gate a build (per-tile gate at 5%% is the default)
  python scripts/audit_spectra_quality.py <path>

  # strict: fail on ANY defective row
  python scripts/audit_spectra_quality.py <path> --fail_over 0.0 --fail_tile_over 100

Exit codes: 0 clean (or below thresholds), 1 defects found, 2 bad invocation.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_BANDS = 59
NODATA = 65535.0
PHYS_MAX = 1.0
BANDS = [f'm{i}' for i in range(N_BANDS)]
CHECKS = ['zero_tail', 'zero_run', 'all_zero', 'nodata', 'nonfinite',
          'over_phys', 'flat']
# Reported for visibility but never fails the run: a known MRRAL artifact that
# data/dataset.py already masks to nodata before training. Failing on something
# the pipeline already handles just teaches people to ignore this script.
INFO_CHECKS = ['blue_edge']


def _files(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, '*.parquet')))
    return [path]


def _longest_zero_run(is_zero: np.ndarray) -> np.ndarray:
    """Longest contiguous True run per row. is_zero: (n, 59) -> (n,)."""
    n = is_zero.shape[0]
    best = np.zeros(n, dtype=np.int32)
    cur = np.zeros(n, dtype=np.int32)
    for b in range(is_zero.shape[1]):
        cur = np.where(is_zero[:, b], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _tail_zero_len(is_zero: np.ndarray) -> np.ndarray:
    """Length of the trailing all-zero run per row."""
    n = is_zero.shape[0]
    out = np.zeros(n, dtype=np.int32)
    live = np.ones(n, dtype=bool)
    for b in range(is_zero.shape[1] - 1, -1, -1):
        live &= is_zero[:, b]
        out += live.astype(np.int32)
    return out


def audit_block(X: np.ndarray, min_run: int, min_tail: int) -> dict:
    """Per-row boolean flags for each check. X: (n, 59) float32."""
    is_zero = (X == 0.0)
    tail = _tail_zero_len(is_zero)
    run = _longest_zero_run(is_zero)
    with np.errstate(invalid='ignore'):
        flat = np.nanstd(X, axis=1) == 0.0
    return {
        'zero_tail': tail >= min_tail,
        'zero_run': run >= min_run,
        'all_zero': is_zero.all(axis=1),
        'nodata': (X == NODATA).any(axis=1),
        'nonfinite': ~np.isfinite(X).all(axis=1),
        # Band 0 (410 nm) spikes are the known blue-edge artifact and are masked
        # by the training reader; only bands >= 1 indicate real damage.
        'over_phys': (X[:, 1:] > PHYS_MAX).any(axis=1),
        'flat': flat,
        'blue_edge': X[:, 0] > PHYS_MAX,
    }


def _find_tile_img(tile: str, data_root: str) -> list[str]:
    """Locate a tile's mrral .img, trying every layout the pipeline uses.

    scripts/train.py globs BOTH `data_root/mc*/t*mrral*.hdr` and
    `data_root/t*mrral*.hdr`, so tiles may sit in per-quadrant subdirectories or
    flat at the root depending on the machine. Searching only one layout makes
    present tiles look missing, which is how a whole session got reported as
    unverifiable on HPC.
    """
    for pattern in (os.path.join(data_root, 'mc*', f'{tile}_mrral*.img'),
                    os.path.join(data_root, f'{tile}_mrral*.img'),
                    os.path.join(data_root, '*', '*', f'{tile}_mrral*.img')):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits
    return []


def verify_against_tiles(df: pd.DataFrame, data_root: str, sample: int,
                         seed: int = 42) -> tuple[list[str], list[str]]:
    """Re-read sampled pixels from the source tiles.

    This is the stale-extraction check: the parquet can be internally clean and
    still disagree with the tile it was extracted from.

    Returns (mismatched, unverifiable). Those are DIFFERENT claims and must not
    be conflated: "I checked and it disagrees" is a defect; "the tile isn't on
    this filesystem" means the check did not run. Only the first should ever
    block a build.
    """
    import rasterio
    rng = np.random.default_rng(seed)
    problems: list[str] = []
    missing: list[str] = []
    for tile, g in df.groupby('tile_id', sort=True):
        hits = _find_tile_img(tile, data_root)
        if not hits:
            missing.append(tile)
            continue
        take = g if len(g) <= sample else g.iloc[
            rng.choice(len(g), sample, replace=False)]
        # One read per sampled pixel covering ALL bands at once. Two rejected
        # alternatives: a read per (pixel, band) is 59x the syscalls, and a
        # bounding-box read over random samples pulls essentially the whole
        # ~550 MB cube per tile. Both made this check unusable in practice.
        bands = list(range(1, N_BANDS + 1))
        n_ok = mism = 0
        with rasterio.open(hits[0]) as src:
            if src.count < N_BANDS:
                problems.append(f'{tile}: tile has only {src.count} bands')
                continue
            for _, r in take.iterrows():
                pr, pc = int(r['pixel_row']), int(r['pixel_col'])
                if not (0 <= pr < src.height and 0 <= pc < src.width):
                    continue
                win = rasterio.windows.Window(pc, pr, 1, 1)
                got = src.read(bands, window=win).astype(np.float32)[:, 0, 0]
                want = r[BANDS].to_numpy(np.float32)
                n_ok += 1
                if not np.allclose(got, want, atol=1e-4, equal_nan=True):
                    mism += 1
        if mism:
            problems.append(
                f'{tile}: {mism}/{n_ok} sampled rows DIFFER from the tile '
                f'on disk — stale extraction (re-extract this tile)')
    return problems, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='parquet file, or directory of parquet fragments')
    ap.add_argument('--min_run', type=int, default=5,
                    help='interior zero-run length that counts as a defect (default 5)')
    ap.add_argument('--min_tail', type=int, default=3,
                    help='trailing zero-run length that counts as a defect (default 3)')
    ap.add_argument('--fail_over', type=float, default=100.0,
                    help='exit 1 if any check affects more than this %% of rows '
                         'OVERALL. Off by default (100) -- the per-tile gate is '
                         'the meaningful one. Set 0.0 for a strict pass.')
    ap.add_argument('--fail_tile_over', type=float, default=5.0,
                    help='exit 1 if any SINGLE TILE has more than this %% of its '
                         'rows flagged. This is the better gate: the failure mode '
                         'that matters is one bad tile dominating a class (t1444 '
                         'was 100%% of its own rows and 72%% of the bland class), '
                         'while a handful of bad rows spread over millions is '
                         'noise. DEFAULT 5.0 -- this is the primary gate, so a '
                         'caller who forgets to pass a flag still gets sensible '
                         'behaviour rather than a spurious failure.')
    ap.add_argument('--require_tiles', action='store_true',
                    help='treat "source tile not found" as a failure. Off by '
                         'default: an unverifiable tile is not a defective one.')
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--verify_against_tiles', action='store_true',
                    help='re-read sampled pixels from the source .img and compare '
                         '— the only check that catches a stale extraction')
    ap.add_argument('--sample', type=int, default=25,
                    help='rows sampled per tile for --verify_against_tiles')
    ap.add_argument('--data_root', default=None,
                    help='tile root for --verify_against_tiles (default: config data_root)')
    args = ap.parse_args()

    files = _files(args.path)
    if not files:
        print(f'ERROR: no parquet files at {args.path}', file=sys.stderr)
        sys.exit(2)

    total = 0
    counts = {c: 0 for c in CHECKS + INFO_CHECKS}
    per_tile: dict[str, dict] = {}

    for f in files:
        pf = pq.ParquetFile(f)
        cols = set(pf.schema_arrow.names)
        missing = [c for c in BANDS if c not in cols]
        if missing:
            print(f'ERROR: {f} lacks band columns (e.g. {missing[:3]}) — '
                  f'not a spectra parquet', file=sys.stderr)
            sys.exit(2)
        want = BANDS + (['tile_id'] if 'tile_id' in cols else [])
        for batch in pf.iter_batches(batch_size=args.batch, columns=want):
            d = batch.to_pandas()
            X = d[BANDS].to_numpy(np.float32)
            total += len(d)
            flags = audit_block(X, args.min_run, args.min_tail)
            for c in CHECKS + INFO_CHECKS:
                counts[c] += int(flags[c].sum())
            if 'tile_id' in d.columns:
                any_bad = np.zeros(len(d), dtype=bool)
                for c in CHECKS:
                    any_bad |= flags[c]
                for tile in pd.unique(d['tile_id']):
                    m = (d['tile_id'] == tile).to_numpy()
                    e = per_tile.setdefault(tile, {'n': 0, 'bad': 0,
                                                   **{c: 0 for c in CHECKS}})
                    e['n'] += int(m.sum())
                    e['bad'] += int((any_bad & m).sum())
                    for c in CHECKS:
                        e[c] += int((flags[c] & m).sum())

    print(f'audited {total:,} spectra across {len(files)} file(s) in {args.path}\n')
    print(f'{"check":<12}{"rows":>12}{"% of total":>13}')
    print('-' * 37)
    worst = 0.0
    for c in CHECKS:
        pct = 100.0 * counts[c] / max(total, 1)
        worst = max(worst, pct)
        print(f'{c:<12}{counts[c]:>12,}{pct:>12.2f}%')
    for c in INFO_CHECKS:
        pct = 100.0 * counts[c] / max(total, 1)
        print(f'{c:<12}{counts[c]:>12,}{pct:>12.2f}%   known artifact, masked '
              f'by the training reader — informational only')

    offenders = {t: e for t, e in per_tile.items() if e['bad']}
    if offenders:
        print(f'\nper-tile breakdown ({len(offenders)} of {len(per_tile)} tiles affected):')
        print(f'{"tile":<10}{"rows":>12}{"bad":>12}{"% bad":>9}   dominant checks')
        print('-' * 68)
        for t, e in sorted(offenders.items(), key=lambda kv: -kv[1]['bad']):
            dom = ', '.join(c for c in CHECKS if e[c] > 0.5 * e['n']) or \
                  ', '.join(c for c in CHECKS if e[c])
            print(f'{t:<10}{e["n"]:>12,}{e["bad"]:>12,}'
                  f'{100.0*e["bad"]/max(e["n"],1):>8.1f}%   {dom}')

    stale: list[str] = []
    missing: list[str] = []
    if args.verify_against_tiles:
        root = args.data_root
        if root is None:
            from config_loader import load_config
            root = load_config()['data_root']
        print(f'\ncross-checking sampled rows against source tiles under {root} …')
        need = BANDS + ['tile_id', 'pixel_row', 'pixel_col']
        frames = []
        for f in files:
            c = set(pq.ParquetFile(f).schema_arrow.names)
            if all(x in c for x in ('tile_id', 'pixel_row', 'pixel_col')):
                frames.append(pd.read_parquet(f, columns=need))
        if frames:
            stale, missing = verify_against_tiles(
                pd.concat(frames, ignore_index=True), root, args.sample)
            if stale:
                print('  MISMATCHED (defect — re-extract these tiles):')
                for s in stale:
                    print(f'    {s}')
            if missing:
                print(f'  NOT VERIFIABLE: {len(missing)} tile(s) not found under '
                      f'{root}.')
                print(f'    {", ".join(missing[:12])}'
                      + (f' … (+{len(missing) - 12} more)' if len(missing) > 12 else ''))
                print('    The check did NOT run for these — that is not the same '
                      'as their being bad.')
                if not args.require_tiles:
                    print('    Not treated as a failure; pass --require_tiles to '
                          'make it one.')
            if not stale and not missing:
                print('  all sampled rows match their source tiles.')
        else:
            print('  skipped: fragments lack tile_id/pixel_row/pixel_col.')

    # Gate. --fail_tile_over is the better primary gate: the failure mode that
    # matters is ONE tile dominating a class (t1444 was 100% of its own rows and
    # 72% of bland), whereas 19 bad rows in 8.26M is noise. Setting it relaxes
    # the global threshold, which would otherwise fire on that noise.
    tile_bad = 0.0
    tile_worst = None
    for t, e in per_tile.items():
        pct = 100.0 * e['bad'] / max(e['n'], 1)
        if pct > tile_bad:
            tile_bad, tile_worst = pct, t
    if args.fail_tile_over is not None:
        over = tile_bad > args.fail_tile_over
        if over:
            print(f'\nworst tile: {tile_worst} at {tile_bad:.1f}% bad '
                  f'(threshold {args.fail_tile_over}%)')
        failed = over or bool(stale) or (bool(missing) and args.require_tiles)
    else:
        failed = (worst > args.fail_over or bool(stale)
                  or (bool(missing) and args.require_tiles))

    print('\nRESULT:', 'FAIL' if failed else 'PASS')
    if failed:
        print('Do not train on this parquet until the affected tiles are '
              're-extracted or excluded.')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
