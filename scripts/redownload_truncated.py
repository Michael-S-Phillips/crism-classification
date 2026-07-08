"""Re-download truncated PDS MRDR .img files listed in the integrity audit.

Reads reports/img_integrity_audit.csv (from the size-vs-header audit), and for
each TRUNCATED file:
  1. Spot-checks 3 random byte ranges of the local prefix against the PDS copy
     (HTTP Range requests). Match -> resume from our bytes; mismatch -> fresh.
  2. Downloads to <file>.part with curl -C - (resumable).
  3. Verifies final size == header-expected, then atomically replaces the
     original.

Usage:
    python scripts/redownload_truncated.py [--audit reports/img_integrity_audit.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys

BASE_URL = ('https://pds-geosciences.wustl.edu/mro/'
            'mro-m-crism-5-rdr-multispectral-v1/mrocr_3201/mrdr')
CHUNK = 8192


def url_for(path: str) -> str:
    mc = path.rstrip('/').split('/')[-2]
    return f'{BASE_URL}/{mc}/{os.path.basename(path)}'


def fetch_range(url: str, start: int, n: int) -> bytes:
    out = subprocess.run(
        ['curl', '-sf', '-r', f'{start}-{start + n - 1}', url],
        capture_output=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f'range fetch failed rc={out.returncode}')
    return out.stdout


def prefix_matches(path: str, url: str, local_size: int, rng) -> bool:
    """Compare 3 random local ranges against the remote copy."""
    if local_size < CHUNK:
        return False
    offsets = [rng.randrange(0, local_size - CHUNK) for _ in range(3)]
    with open(path, 'rb') as fp:
        for off in offsets:
            fp.seek(off)
            local = fp.read(CHUNK)
            if fetch_range(url, off, CHUNK) != local:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', default='reports/img_integrity_audit.csv')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.audit) as fp:
        bad = [r for r in csv.DictReader(fp) if r['status'] == 'TRUNCATED']
    print(f'{len(bad)} truncated files to re-download', flush=True)

    failures = []
    for i, row in enumerate(bad, 1):
        path, expected = row['path'], int(row['expected'])
        url = url_for(path)
        part = path + '.part'
        tag = f'[{i}/{len(bad)}] {os.path.basename(path)}'
        try:
            local_size = os.path.getsize(path)
            if not os.path.exists(part):
                if prefix_matches(path, url, local_size, rng):
                    shutil.copyfile(path, part)   # resume from our prefix
                    print(f'{tag}: prefix OK, resuming from '
                          f'{local_size/expected:.0%}', flush=True)
                else:
                    print(f'{tag}: prefix MISMATCH, fetching fresh', flush=True)
            rc = subprocess.run(
                ['curl', '-sf', '-C', '-', '-o', part, url],
                timeout=7200).returncode
            if rc != 0:
                raise RuntimeError(f'curl rc={rc}')
            got = os.path.getsize(part)
            if got != expected:
                raise RuntimeError(f'size {got:,} != expected {expected:,}')
            os.replace(part, path)
            print(f'{tag}: DONE ({expected:,} bytes)', flush=True)
        except Exception as e:
            failures.append((path, str(e)))
            print(f'{tag}: FAILED — {e}', flush=True)

    print(f'\ncomplete: {len(bad) - len(failures)} ok, {len(failures)} failed',
          flush=True)
    for path, err in failures:
        print(f'  FAILED {path}: {err}', flush=True)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
