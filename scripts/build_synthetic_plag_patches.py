# scripts/build_synthetic_plag_patches.py
"""Build the synthetic plagioclase patch cache + parquet fragment.

Reads the two ENVI plag libraries, resamples each spectrum to the 59 mrral bands,
synthesizes augmented 7x7x59 patches (train-split only), and writes:
  <output_dir>/synth_plag_patches_p7.npy   (M, 7, 7, 59) float32
  <output_dir>/synth_plag_rows.parquet     M rows, mrral_pixels schema subset

Usage:
  conda run -n crism python scripts/build_synthetic_plag_patches.py \\
    --n_aug 300 \\
    --mrral_hdr /mnt/mrdr/mc26/t0505_mrral_35s313_0327_4.hdr
"""
import argparse
import glob
import os
import sys

import numpy as np
import spectral.io.envi as envi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_plag import build_synth_rows, interp_to_mrral_wavelengths

LIBS = [
    ("/mnt/mrdr/plagioclase-targeted/unratioed_plag_highconfidence.hdr", "High"),
    ("/mnt/mrdr/plagioclase-targeted/unratioed_plag_moderateconfidence_w_2micron.hdr",
     "Moderate"),
]


def load_target_wavelengths(mrral_hdr: str) -> np.ndarray:
    img = envi.open(mrral_hdr)
    return np.asarray(img.bands.centers, dtype=np.float64)[:59]


def load_library_resampled(hdr: str, target_wl: np.ndarray) -> dict:
    lib = envi.open(hdr)
    lib_wl = np.asarray(lib.bands.centers, dtype=np.float64)
    spectra = np.asarray(lib.spectra, dtype=np.float64)   # (n_spectra, n_bands)
    names = list(lib.names)
    out = {}
    for name, refl in zip(names, spectra):
        out[name] = interp_to_mrral_wavelengths(lib_wl, refl, target_wl).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_aug", type=int, default=300,
                    help="augmented patches per source spectrum")
    ap.add_argument("--mrral_hdr", type=str, default=None,
                    help="any mrral .hdr to read the 59 target wavelengths from")
    ap.add_argument("--output_dir", type=str,
                    default="/mnt/mrdr/crism_classification/data/patch_cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mrral_hdr = args.mrral_hdr or sorted(glob.glob("/mnt/mrdr/mc*/t*_mrral_*.hdr"))[0]
    target_wl = load_target_wavelengths(mrral_hdr)
    print(f"target wavelengths: {len(target_wl)} bands, "
          f"{target_wl[0]:.1f}-{target_wl[-1]:.1f} nm (from {mrral_hdr})")

    rng = np.random.default_rng(args.seed)
    all_patches, all_dfs = [], []
    for hdr, tier in LIBS:
        spectra = load_library_resampled(hdr, target_wl)
        patches, df = build_synth_rows(spectra, n_aug=args.n_aug, rng=rng,
                                       confidence_tier=tier)
        print(f"  {os.path.basename(hdr)}: {len(spectra)} spectra -> "
              f"{len(df)} rows ({tier})")
        all_patches.append(patches)
        all_dfs.append(df)

    import pandas as pd
    patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    df = pd.concat(all_dfs, ignore_index=True)
    assert len(df) == len(patches)

    os.makedirs(args.output_dir, exist_ok=True)
    npy_path = os.path.join(args.output_dir, "synth_plag_patches_p7.npy")
    pq_path = os.path.join(args.output_dir, "synth_plag_rows.parquet")
    # Save as a plain .npy (loadable via np.load mmap_mode='r')
    np.save(npy_path, patches)
    df.to_parquet(pq_path, index=False)
    print(f"wrote {npy_path}  shape={patches.shape}")
    print(f"wrote {pq_path}   rows={len(df)}")


if __name__ == "__main__":
    main()
