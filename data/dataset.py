"""
PyTorch Dataset classes and sklearn array loaders for the CRISM pixel dataset.
"""
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

LABEL_COLS_RAW = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                  'other', 'alteration']
# Default 5-class label set — preserved for backward compatibility with every
# pipeline trained before 2026-06-10. Training entrypoints that want the
# 6-class output replace this at module load time with the 6-class variant
# (train.py does this when --with_alteration is passed).
LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
LABEL_COLS_WITH_ALTERATION = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other',
                               'alteration']
# 7-class label set: bland replaces other; junk is the new spectrally-ambiguous
# catch-all. Train with --seven_class in train.py.
LABEL_COLS_7CLASS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland',
                     'alteration', 'junk']
# pyx (pyroxene) 6-class label set: lcp and hcp merged into a single 'pyx' class
# for tasks requiring a unified pyroxene mineral class.
LABEL_COLS_PYX = ['olivine', 'pyx', 'plagioclase', 'bland', 'alteration', 'junk']
# pyx 5-class label set for HAND-LABELED-ONLY training (no review data): the base
# parquet's native classes with lcp+hcp merged into pyx. 'bland'/'junk' are
# review-derived and absent here, so 'other' stays as the non-mineral catch-all.
# Train with --pyx_alt. All five columns are produced by _collapse_labels.
LABEL_COLS_PYX_ALT = ['olivine', 'pyx', 'plagioclase', 'other', 'alteration']

# Ordered list of all known label sets, indexed by head width.
_LABEL_COLS_BY_N = {
    5: LABEL_COLS,
    6: LABEL_COLS_WITH_ALTERATION,
    7: LABEL_COLS_7CLASS,
}


def label_cols_for_ckpt(state_dict) -> list:
    """Return the class-name list matching a classifier checkpoint's head.

    Eval/inference scripts should call this instead of importing LABEL_COLS
    at module level — a 6-class checkpoint (``--with_alteration``) has
    head.weight of shape (6, 128) and needs LABEL_COLS_WITH_ALTERATION,
    while every pre-2026-06-10 checkpoint is (5, 128). Accepts either a
    raw state_dict or a checkpoint dict with a 'model_state' key.
    """
    if 'model_state' in state_dict and 'head.weight' not in state_dict:
        state_dict = state_dict['model_state']
    head_w = state_dict.get('head.weight')
    if head_w is None:
        raise KeyError("checkpoint has no 'head.weight' — not a classifier "
                       "checkpoint (MAE/encoder-only ckpts have no head)")
    n = int(head_w.shape[0])
    if n == 1:
        return ['target']  # binary mode; caller knows the target class
    if n not in _LABEL_COLS_BY_N:
        raise ValueError(f'checkpoint head has {n} outputs; no known label '
                         f'set for that width (known: {sorted(_LABEL_COLS_BY_N)})')
    return list(_LABEL_COLS_BY_N[n])


# Tier → per-pixel sample weight. A scheme omits the graded 'reviewed-high/
# moderate/low' keys to let the per-polygon reviewer weight stamped by
# scripts/review/persistence.py pass through _collapse_labels verbatim (see
# tests/test_collapse_reviewed_tier.py). 'level' therefore reproduces the
# pre-2026-08-08 behaviour exactly.
#
# 'reviewed-legacy' is always present (in all three schemes below), because it
# is NOT a graded reviewer tier — it is scripts/build_7cls_dataset.py's stamp
# for human-reviewed-but-ungraded ("legacy") rows, distinguishing them from
# hand-labeled rows that used to share the same 'High' tier string (see
# _stamp_legacy_tier). Its per-scheme value is a plain lookup, not a
# stamped-weight passthrough.
WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    # 'reviewed-legacy' = 1.0: identical to what legacy resolved to TODAY (via
    # the 'high' key, since legacy rows used to be stamped tier='High') —
    # preserves default training behaviour exactly.
    'level':     {'high': 1.0, 'moderate': 0.85, 'low': 0.70,
                  'reviewed-legacy': 1.0},
    # 'reviewed-legacy' = 1.5: human-reviewed, so upweighted like graded
    # review, but below graded review's 2.0 because it was never graded.
    'review_up': {'high': 1.0, 'moderate': 0.85, 'low': 0.70,
                  'reviewed-high': 2.0, 'reviewed-moderate': 1.7,
                  'reviewed-low': 1.4, 'reviewed-legacy': 1.5},
    # 'reviewed-legacy' = 0.85: least-trusted source here — review data that
    # carries no reviewer grade — so it sits below both hand 'High' (1.5) and
    # graded review's floor.
    'hand_up':   {'high': 1.5, 'moderate': 1.3, 'low': 1.0,
                  'reviewed-high': 1.0, 'reviewed-moderate': 0.85,
                  'reviewed-low': 0.70, 'reviewed-legacy': 0.85},
}

_ACTIVE_SCHEME = 'level'
_TIER_WEIGHTS = WEIGHT_SCHEMES[_ACTIVE_SCHEME]


def set_weight_scheme(name: str) -> None:
    """Select the tier→weight table used by _collapse_labels.

    Must be called before any dataset is constructed. Schemes that omit the
    'reviewed-*' keys leave the stamped per-polygon reviewer weight intact.
    """
    global _ACTIVE_SCHEME, _TIER_WEIGHTS  # noqa: PLW0603
    if name not in WEIGHT_SCHEMES:
        raise ValueError(
            f'unknown weight scheme {name!r}; known: {sorted(WEIGHT_SCHEMES)}')
    _ACTIVE_SCHEME = name
    _TIER_WEIGHTS = WEIGHT_SCHEMES[name]


def active_weight_scheme() -> str:
    return _ACTIVE_SCHEME


def _collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with olivine_t1/t2 merged into a single 'olivine' column
    and confidence_weight derived from confidence_tier.

    Olivine collapse (hard label):
      olivine = 1.0 if either olivine_t1 or olivine_t2 is positive, else 0.0.
      This treats untyped "olivine" annotations (parser writes 0.5/0.5 for
      these because the type is unknown) as full olivine presence at training
      time — pixels the annotator called olivine should train the model to
      predict olivine confidently, regardless of whether subtype was specified.

    Confidence weights (per-pixel sample weight in the loss):
      Manually-annotated low-confidence detections were still confident enough
      to flag, so they keep most of their signal rather than being heavily
      down-weighted (1.0/0.5/0.25 → 1.0/0.85/0.70):
        High:     1.0
        Moderate: 0.85
        Low:      0.70
      Tiers outside this map (e.g. 'Reviewed', 'Ambiguous' from the MC13
      review pipeline) keep the confidence_weight already stamped in the
      parquet — build_review_augmented_train.py sets those deliberately
      (2.0 / 3.0) and they must not be clobbered. Rows with an unknown tier
      AND no stamped weight default to Moderate. (Before 2026-06-11 this
      function silently overwrote review weights to 0.85 — audit bug #1.)
    """
    out = df.copy()
    out['olivine'] = (
        out[['olivine_t1', 'olivine_t2']].max(axis=1) > 0
    ).astype(np.float32)
    # Flat columns added by the 6-class and 7-class pipelines. Default to 0
    # when missing so pre-7cls parquets load without schema changes.
    if 'alteration' not in out.columns:
        out['alteration'] = np.float32(0.0)
    else:
        out['alteration'] = (out['alteration'] > 0).astype(np.float32)
    if 'bland' not in out.columns:
        # 5/6-class parquets use 'other' for bland tiles; mirror it here so
        # LABEL_COLS_7CLASS['bland'] still resolves correctly.
        out['bland'] = (out['other'] > 0).astype(np.float32) if 'other' in out.columns else np.float32(0.0)
    else:
        out['bland'] = (out['bland'] > 0).astype(np.float32)
    if 'junk' not in out.columns:
        out['junk'] = np.float32(0.0)
    else:
        out['junk'] = (out['junk'] > 0).astype(np.float32)
    # pyx = pyroxene merge (LCP+HCP) for the 6-class pyx target; lcp/hcp are
    # left intact for the post-hoc ortho/clino overlay (Spec B).
    if 'lcp' not in out.columns:
        out['lcp'] = np.float32(0.0)
    if 'hcp' not in out.columns:
        out['hcp'] = np.float32(0.0)
    out['pyx'] = out[['lcp', 'hcp']].max(axis=1).astype(np.float32)
    tier_weights = WEIGHT_SCHEMES[_ACTIVE_SCHEME]
    if 'confidence_tier' in out.columns:
        mapped = (
            out['confidence_tier']
            .astype(str).str.lower()
            .map(tier_weights)
        )
        if 'confidence_weight' in out.columns:
            # Unknown tiers (Reviewed-*, Ambiguous) fall back to the stamped
            # weight; only rows lacking both get the Moderate default.
            mapped = mapped.fillna(
                pd.to_numeric(out['confidence_weight'], errors='coerce'))
        out['confidence_weight'] = (
            mapped.fillna(tier_weights['moderate']).astype(np.float32)
        )
    elif 'confidence_weight' not in out.columns:
        out['confidence_weight'] = np.float32(tier_weights['moderate'])
    return out
BAND_COLS = [f'b{i}' for i in range(60)]
MRRAL_BAND_COLS = [f'm{i}' for i in range(59)]  # 59 bands, 410-2457 nm (< 2500 nm cutoff)
NODATA_VALUE = 65535


class CRISMPixelDataset(Dataset):
    """Per-pixel dataset for MLP and linear models."""

    def __init__(self, df: pd.DataFrame):
        df = _collapse_labels(df)
        self.features = torch.tensor(df[BAND_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


class CRISMSpectralDataset(Dataset):
    """
    Per-pixel dataset using mrral 59-band reflectance spectra (m0..m58).
    Used by all mrral-based models: SpectralCNN1D, SpectralTransformer, MAE.

    Requires mrral_pixels.parquet — run scripts/build_mrral_dataset.py first.
    """

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in MRRAL_BAND_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"mrral parquet missing columns: {missing[:5]}... "
                "Run scripts/build_mrral_dataset.py first."
            )
        df = _collapse_labels(df)
        self.features = torch.tensor(df[MRRAL_BAND_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


class CRISMPatchDataset(Dataset):
    """
    Spatial patch dataset for CNN and ViT.

    Extracts a (bands × patch_size × patch_size) neighbourhood around each
    pixel from the corresponding mrrsu raster at runtime. Border pixels are
    zero-padded. File handles are cached per tile and re-opened safely after
    DataLoader fork (detected via os.getpid() comparison).

    Call close() when done, or use as a context manager.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mrrsu_map: Dict[str, str],
        patch_size: int = 7,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        df = _collapse_labels(df).reset_index(drop=True)
        self.mrrsu_map = mrrsu_map
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        # Extract hot-path columns as arrays to avoid per-item DataFrame overhead
        self._tile_ids = df['tile_id'].values
        self._pixel_rows = df['pixel_row'].values.astype(np.int64)
        self._pixel_cols = df['pixel_col'].values.astype(np.int64)
        self._n = len(df)
        # File handles cached per tile; cleared on DataLoader fork (pid check)
        self._handles: Dict[str, rasterio.DatasetReader] = {}
        self._pid = os.getpid()
        # Load memmap cache if available — bypasses rasterio reads at item time
        self._cache = None
        if cache_dir and split:
            cache_file = os.path.join(cache_dir, f'{split}_patches_p{patch_size}.npy')
            if os.path.exists(cache_file):
                self._cache = np.memmap(
                    cache_file, dtype='float32', mode='r',
                    shape=(self._n, len(BAND_COLS), patch_size, patch_size)
                )

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if self._cache is not None:
            patch = torch.from_numpy(self._cache[idx].copy())
            return patch, self.labels[idx], self.weights[idx]

        # Re-open handles if we've been forked into a DataLoader worker
        current_pid = os.getpid()
        if current_pid != self._pid:
            self._handles.clear()
            self._pid = current_pid

        tile_id = self._tile_ids[idx]
        pr = int(self._pixel_rows[idx])
        pc = int(self._pixel_cols[idx])

        if tile_id not in self._handles:
            if tile_id not in self.mrrsu_map:
                raise KeyError(
                    f"tile_id {tile_id!r} not found in mrrsu_map. "
                    f"Available tiles (first 5): {sorted(self.mrrsu_map)[:5]}"
                )
            try:
                self._handles[tile_id] = rasterio.open(self.mrrsu_map[tile_id])
            except Exception as e:
                raise OSError(f"Failed to open raster for tile {tile_id!r}: {e}") from e
        src = self._handles[tile_id]

        if src.count != len(BAND_COLS):
            raise ValueError(
                f"Tile {tile_id!r} has {src.count} bands, expected {len(BAND_COLS)}"
            )

        h = self.half
        r0 = max(0, pr - h)
        r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h)
        c1 = min(src.width, pc + h + 1)

        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        chunk = src.read(window=window).astype(np.float32)  # (bands, h, w)

        # Replace nodata and NaN with 0
        chunk[chunk >= NODATA_VALUE] = 0.0
        chunk = np.nan_to_num(chunk, nan=0.0)

        # Place chunk into zero-padded full patch
        full = np.zeros((len(BAND_COLS), self.patch_size, self.patch_size), dtype=np.float32)
        dst_r = h - (pr - r0)  # offset in output array (= max(0, h - pr) at borders)
        dst_c = h - (pc - c0)
        full[:, dst_r:dst_r + chunk.shape[1], dst_c:dst_c + chunk.shape[2]] = chunk

        return torch.from_numpy(full), self.labels[idx], self.weights[idx]

    def close(self):
        """Close all open rasterio file handles."""
        for src in getattr(self, '_handles', {}).values():
            try:
                src.close()
            except Exception:
                pass
        if hasattr(self, '_handles'):
            self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class CRISMCombinedDataset(Dataset):
    """
    Per-pixel dataset combining mrral 59-band reflectance with mrrsu 60-band
    summary parameters into a single 119-dim feature vector.

    Requires a merged DataFrame with both m0..m58 (mrral) and b0..b59 (mrrsu)
    columns present. Build via:

        mrral_df = pd.read_parquet('data/mrral_pixels.parquet')
        mrrsu_df = pd.read_parquet('data/pixels.parquet')
        MERGE_KEYS = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
        combined = mrral_df.merge(mrrsu_df[MERGE_KEYS + BAND_COLS], on=MERGE_KEYS, how='inner')

    Features layout: features[:59] = mrral bands, features[59:] = mrrsu bands.
    This layout matches SpectralHybridClassifier.forward() which splits on dim 59.
    """

    N_MRRAL = 59
    N_MRRSU = 60
    N_FEATURES = N_MRRAL + N_MRRSU  # 119

    def __init__(self, df: pd.DataFrame):
        missing_mrral = [c for c in MRRAL_BAND_COLS if c not in df.columns]
        if missing_mrral:
            raise ValueError(
                f"DataFrame missing mrral columns: {missing_mrral[:5]}... "
                "Merge mrral_pixels.parquet with pixels.parquet first."
            )
        missing_mrrsu = [c for c in BAND_COLS if c not in df.columns]
        if missing_mrrsu:
            raise ValueError(
                f"DataFrame missing mrrsu columns: {missing_mrrsu[:5]}... "
                "Merge mrral_pixels.parquet with pixels.parquet first."
            )
        df = _collapse_labels(df)
        mrral = df[MRRAL_BAND_COLS].values.astype('float32')
        mrrsu = df[BAND_COLS].values.astype('float32')
        self.features = torch.tensor(
            np.concatenate([mrral, mrrsu], axis=1), dtype=torch.float32
        )
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


def load_sklearn_arrays(parquet_path: str):
    """
    Load train/val/test arrays for sklearn models.

    Returns
    -------
    X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test
    All as numpy arrays. y arrays are shape (n, 5) float32.
    """
    df = _collapse_labels(pd.read_parquet(parquet_path))

    def _split(split_name):
        sub = df[df['split'] == split_name]
        X = sub[BAND_COLS].values.astype(np.float32)
        y = sub[LABEL_COLS].values.astype(np.float32)
        w = sub['confidence_weight'].values.astype(np.float32)
        return X, y, w

    return (*_split('train'), *_split('val'), *_split('test'))


class CRISMSpectralPatchDataset(Dataset):
    """
    Spatial patch dataset for SpatialSpectralClassifier fine-tuning.

    Reads 7×7×59 mrral reflectance patches around labeled pixel centers.
    Applies the same clipping (to [0, 0.5]) as the pretraining patch cache builder.
    Border pixels are zero-padded. File handles are cached per tile, pid-safe.
    """

    CLIP_MAX = 0.5
    NODATA = 65535.0
    PHYS_MAX = 1.0   # reflectance (I/F) above this is corrupt → treat as nodata

    def __init__(
        self,
        df: pd.DataFrame,
        mrral_map: Dict[str, str],
        patch_size: int = 7,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        continuum_removed: bool = False,
        return_brightness: bool = False,
        cache_is_cr: bool = False,
        dual_cr: bool = False,
    ):
        """Read 7x7x59 (or 7x7x118 with dual_cr) mrral patches around labeled
        centers.

        Continuum-removal options (all off by default → unchanged raw behaviour):
          continuum_removed: apply upper-hull CR (data.continuum_removal.cr_patch)
              to each patch. On the on-the-fly and CR-on-read cache paths CR is
              applied after the [0, CLIP_MAX] clip, identical to the Task-2 global
              cache builder. When cache_is_cr=True the cache is already CR and is
              returned as-is.
          return_brightness: also return the center-pixel brightness scalar (mean
              good-band reflectance, pre-CR) as a (1,) aux tensor, so __getitem__
              yields (patch, brightness, label, weight) — matching the aux-model
              batch contract. Requires continuum_removed=True.
          cache_is_cr: the memmap cache already holds CR patches (built by
              scripts/build_global_patch_cache.py --continuum_removed or the
              labeled equivalent). Brightness is then read from the parallel
              mrral_{split}_patches_p{P}_brightness.npy sidecar.
          dual_cr: serve the 118-channel hull-CR ⊕ linear-CR representation
              (data.continuum_removal.dual_continuum) instead of the 59-channel
              hull-only one. Requires continuum_removed=True. A memmap cache
              used with dual_cr=True must itself be 118-channel wide (built by
              the --dual patch-cache builders) — the byte-exact size check
              below is keyed on the actual channel count, so a 59-channel
              cache is rejected rather than silently read as half of a dual
              patch.
        """
        assert patch_size % 2 == 1, "patch_size must be odd"
        if return_brightness and not continuum_removed:
            raise ValueError('return_brightness requires continuum_removed=True')
        if dual_cr and not continuum_removed:
            raise ValueError('dual_cr requires continuum_removed=True')
        df = _collapse_labels(df).reset_index(drop=True)
        self.mrral_map = mrral_map
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.continuum_removed = continuum_removed
        self.return_brightness = return_brightness
        self.cache_is_cr = cache_is_cr
        self.dual_cr = dual_cr
        self.n_channels = 118 if dual_cr else 59
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        self._tile_ids = df['tile_id'].values
        self._pixel_rows = df['pixel_row'].values.astype(np.int64)
        self._pixel_cols = df['pixel_col'].values.astype(np.int64)
        self._n = len(df)
        self._handles: Dict[str, rasterio.DatasetReader] = {}
        self._pid = os.getpid()
        # Load memmap cache if available — bypasses rasterio reads at item time
        self._cache = None
        self._bright_cache = None
        if cache_dir and split:
            cache_file = os.path.join(cache_dir, f'mrral_{split}_patches_p{patch_size}.npy')
            if os.path.exists(cache_file):
                # The cache is a raw memmap (no npy header), so np.memmap
                # would silently read a prefix of a stale/oversized file —
                # exactly the failure mode when the parquet is rebuilt but
                # the cache isn't. Require an exact byte-size match.
                expected_bytes = self._n * patch_size * patch_size * self.n_channels * 4
                actual_bytes = os.path.getsize(cache_file)
                if actual_bytes != expected_bytes:
                    raise ValueError(
                        f'patch cache {cache_file} is {actual_bytes:,} bytes '
                        f'but the {split} dataframe ({self._n:,} rows) needs '
                        f'exactly {expected_bytes:,} ({self.n_channels} '
                        f'channels). The cache was built from different '
                        f'parquet rows or a different channel count '
                        f'(dual_cr={dual_cr}) — rebuild it '
                        f'(scripts/cache_mrral_patches.py) or fix '
                        f'--patch_cache_dir.')
                self._cache = np.memmap(
                    cache_file, dtype='float32', mode='r',
                    shape=(self._n, patch_size, patch_size, self.n_channels)
                )
                # A CR cache (cache_is_cr) has already discarded albedo, so the
                # brightness aux must come from the parallel sidecar written
                # alongside it. On-read CR (cache_is_cr=False) computes brightness
                # from the raw cache directly and needs no sidecar.
                if self.cache_is_cr and self.return_brightness:
                    bfile = os.path.join(
                        cache_dir, f'mrral_{split}_patches_p{patch_size}_brightness.npy')
                    if not os.path.exists(bfile):
                        raise FileNotFoundError(
                            f'cache_is_cr with return_brightness needs the '
                            f'brightness sidecar {bfile}; rebuild the CR cache so '
                            f'it is written alongside the patches.')
                    self._bright_cache = np.load(bfile, mmap_mode='r')
            elif self.cache_is_cr:
                # cache_is_cr asserts this dir holds CR patches for this split.
                # Falling back to the on-the-fly rasterio path here would serve
                # a DIFFERENT representation than the split we do cache, so an
                # absent file means an incomplete cache build, not "no cache".
                # Fail here rather than mid-epoch. (An incomplete
                # patch_cache_base_cr — train written, val never built — is what
                # killed the pyxalt_cr runs on 2026-08-07.)
                raise FileNotFoundError(
                    f"cache_is_cr was set but the '{split}' split is missing "
                    f'from the CR cache: {cache_file} does not exist. The CR '
                    f'cache build is incomplete — rebuild it for every split '
                    f'you train on:  python scripts/build_cr_labeled_cache.py '
                    f'--raw_dir <raw cache> --out_dir {cache_dir} '
                    f'--splits train val test')

    def __len__(self):
        return self._n

    def _finish(self, patch: np.ndarray, idx: int, brightness=None,
                from_cr_cache: bool = False):
        """Apply CR (if requested) and pack the return tuple for a raw patch.

        patch: (P, P, 59) raw, or already-CR when from_cr_cache.
        brightness: optional precomputed (P, P) brightness map (CR-cache path).
        from_cr_cache: this patch came out of a cache_is_cr memmap, so CR has
            already been applied and `brightness` came from the sidecar. Keyed
            on the actual read path rather than on self.cache_is_cr: the flag
            describes the cache, and a patch read off any other path still
            needs CR applied here.
        """
        if self.continuum_removed and not from_cr_cache:
            if self.dual_cr:
                from data.continuum_removal import (dual_continuum,
                                                    brightness_scalar)
                brightness = brightness_scalar(patch)     # from RAW, pre-transform
                patch = dual_continuum(patch)
            else:
                from data.continuum_removal import cr_patch
                patch, brightness = cr_patch(patch)
        patch_t = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))
        if self.return_brightness:
            b = float(brightness[self.half, self.half])
            bright_t = torch.tensor([b], dtype=torch.float32)
            return patch_t, bright_t, self.labels[idx], self.weights[idx]
        return patch_t, self.labels[idx], self.weights[idx]

    def __getitem__(self, idx):
        if self._cache is not None:
            patch = self._cache[idx].copy()
            brightness = None
            if self.cache_is_cr and self._bright_cache is not None:
                brightness = np.asarray(self._bright_cache[idx])
            return self._finish(patch, idx, brightness,
                                from_cr_cache=self.cache_is_cr)

        current_pid = os.getpid()
        if current_pid != self._pid:
            self._handles.clear()
            self._pid = current_pid

        tile_id = self._tile_ids[idx]
        pr = int(self._pixel_rows[idx])
        pc = int(self._pixel_cols[idx])

        if tile_id not in self._handles:
            if tile_id not in self.mrral_map:
                raise KeyError(f"tile_id {tile_id!r} not found in mrral_map")
            self._handles[tile_id] = rasterio.open(self.mrral_map[tile_id])
        src = self._handles[tile_id]

        h = self.half
        r0 = max(0, pr - h);  r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h);  c1 = min(src.width,  pc + h + 1)
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)

        # Read first 59 bands (1-indexed for rasterio)
        chunk = src.read(list(range(1, 60)), window=window).astype(np.float32)  # (59, h, w)

        # Zero-pad to (59, patch_size, patch_size)
        patch = np.zeros((59, self.patch_size, self.patch_size), dtype=np.float32)
        dr0 = pr - h - max(0, pr - h) + (max(0, pr - h) - (pr - h))
        dc0 = pc - h - max(0, pc - h) + (max(0, pc - h) - (pc - h))
        # Simplified: pad offset is how far the actual window start is from intended
        dr0 = max(0, h - pr)
        dc0 = max(0, h - pc)
        ph, pw = chunk.shape[1], chunk.shape[2]
        patch[:, dr0:dr0 + ph, dc0:dc0 + pw] = chunk

        # Handle nodata. Besides the 65535 sentinel and non-finite values,
        # treat physically-impossible reflectance (I/F > 1.0) as nodata: the
        # MRRAL blue edge (band 0, 410 nm) carries rare corrupt spikes up to
        # ~860 I/F that the [0, CLIP_MAX] clip below would otherwise cap to a
        # plausible-looking 0.5 rather than masking. Audit 2026-06-15.
        patch[(patch == self.NODATA) | ~np.isfinite(patch)
              | (patch > self.PHYS_MAX)] = 0.0

        # Normalize: clip to [0, CLIP_MAX]
        patch = np.clip(patch, 0.0, self.CLIP_MAX)

        # (59, 7, 7) → (7, 7, 59)
        patch = patch.transpose(1, 2, 0)

        return self._finish(patch.copy(), idx)

    def close(self):
        for src in self._handles.values():
            src.close()
        self._handles.clear()


class SyntheticPatchDataset(Dataset):
    """Serves pre-synthesized plagioclase patches from a .npy + parquet fragment.

    Mirrors CRISMSpectralPatchDataset's __getitem__ contract:
    returns (patch (7,7,59) float32 tensor, label (n_cls,) tensor in LABEL_COLS order,
    weight scalar tensor). Patches are read from a memmap'd .npy aligned row-for-row
    with the full parquet.

    If `split` is given (e.g. 'train', 'val', 'test'), only rows matching that split
    column are served; the original row indices into the npy are preserved so the
    alignment holds even after filtering.

    `expect_repr` declares the representation the RUN serves, and the cache is
    checked against it. Patches here are served VERBATIM -- no transform is
    applied -- so a raw cache in a --continuum_removed run injects one class at
    ~4x the level of every other class. That happened: ft_7cls_handcore_level
    ran `--continuum_removed --cache_is_cr` with the RAW MTRDR plag caches, so
    plagioclase was the only class carrying raw level and slope, making it
    trivially separable in validation and prone to over-firing on a CR tile.
    Legal values:
        'any'  (default, inert) -- no check, 59 channels, BASE behaviour
        'raw'  -- 59 channels, raw reflectance (median <= RAW_LEVEL_MAX)
        'hull' -- 59 channels, hull-CR (median > RAW_LEVEL_MAX, bounded <= 1)
        'dual' -- 118 channels, hull 0-58 ⊕ linear 59-117, both CR-level
    Convert a raw cache with scripts/convert_synth_cache_representation.py.
    """

    _REPRS = ('any', 'raw', 'hull', 'dual')

    def __init__(self, npy_path: str, parquet_path: str, patch_size: int = 7,
                 split: Optional[str] = None, expect_repr: str = 'any'):
        if expect_repr not in self._REPRS:
            raise ValueError(
                f'expect_repr={expect_repr!r}; expected one of {self._REPRS}')
        full_df = pd.read_parquet(parquet_path)
        full_n = len(full_df)
        if split is not None and 'split' in full_df.columns:
            mask = (full_df['split'] == split).values
            df = _collapse_labels(full_df[mask].reset_index(drop=True))
            self._indices = np.where(mask)[0]
        else:
            df = _collapse_labels(full_df.reset_index(drop=True))
            self._indices = np.arange(full_n)
        self._n = len(df)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        self._cache = np.load(npy_path, mmap_mode='r')
        assert self._cache.shape[0] == full_n, (
            f"patch count {self._cache.shape[0]} != parquet rows {full_n}")
        n_ch = 118 if expect_repr == 'dual' else 59
        assert self._cache.shape[1:] == (patch_size, patch_size, n_ch)
        if expect_repr != 'any':
            self._check_representation(npy_path, expect_repr)

    def _check_representation(self, npy_path: str, expect: str) -> None:
        """Fail loudly if the cache's statistics contradict `expect`.

        Channel count alone distinguishes 'dual'; raw-vs-hull are both 59
        channels and are separated by level (see continuum_removal.RAW_LEVEL_MAX).
        """
        from data.continuum_removal import (RAW_LEVEL_MAX, detect_representation,
                                            sample_level)
        fix = ('Convert it: python scripts/convert_synth_cache_representation.py '
               f'--input {npy_path} --output <dest> --mode '
               f'{"dual" if expect == "dual" else "hull"}')
        if expect == 'dual':
            hull_lvl = sample_level(self._cache, chan_slice=slice(0, 59))
            lin_lvl = sample_level(self._cache, chan_slice=slice(59, 118))
            if not (hull_lvl > RAW_LEVEL_MAX and lin_lvl > RAW_LEVEL_MAX):
                raise ValueError(
                    f'synth cache {npy_path} is 118-channel but does not look '
                    f'dual-CR: hull-block median {hull_lvl:.4f}, linear-block '
                    f'median {lin_lvl:.4f}; both must exceed '
                    f'{RAW_LEVEL_MAX} (standardized dual sits near 13.2 and 5.8, '
                    f'unstandardized near 0.93 and 1.0; raw would be ~0.23). '
                    f'{fix}')
            return
        got = detect_representation(self._cache)
        if got != expect:
            raise ValueError(
                f'synth cache {npy_path} looks {got.upper()} but this run serves '
                f'{expect.upper()} patches (median level '
                f'{sample_level(self._cache):.4f}, boundary {RAW_LEVEL_MAX}). '
                f'SyntheticPatchDataset applies NO transform, so the cache would '
                f'inject one class at the wrong level -- the bug that invalidated '
                f'plagioclase in ft_7cls_handcore_level. {fix}')

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        patch = torch.from_numpy(
            np.asarray(self._cache[self._indices[idx]], dtype=np.float32).copy()
        )
        return patch, self.labels[idx], self.weights[idx]


def _parse_relabel_soft(new_label):
    """Parse a manual relabel string -> {class: soft_target}.

    e.g. 'olivine + hcp (Moderate)' -> {'hcp': 0.6}; 'olivine' / 'alteration +
    olivine (Low)' -> {} (no pyroxene added). Tier maps High->1.0, Moderate->0.6,
    Low->0.3; a bare 'hcp'/'lcp' with no tier -> 1.0.
    """
    import re
    s = str(new_label).lower()
    tier_map = {'high': 1.0, 'moderate': 0.6, 'low': 0.3}
    m = re.search(r'\((high|moderate|low)\)', s)
    tier = tier_map[m.group(1)] if m else None
    out = {}
    for cls in ('hcp', 'lcp'):
        if cls in s:
            out[cls] = tier if tier is not None else 1.0
    return out


def apply_olivine_relabels(df, csv_path):
    """Apply manual olivine-polygon relabels (soft pyroxene targets) to a copy of df.

    Matches relabel rows to df by (tile_id lowercased, polygon_id) and sets the
    pyroxene class column(s) to the parsed soft target for those pixels, leaving
    olivine and all other labels untouched. Returns (df_copy, n_pixels_changed).
    Row order is preserved so an aligned patch cache stays valid.
    """
    rel = pd.read_csv(csv_path, dtype={'polygon': str})
    df = df.copy()
    tid = df['tile_id'].astype(str).str.lower()
    pid = df['polygon_id'].astype(str)
    n_changed = 0
    for _, r in rel.iterrows():
        soft = _parse_relabel_soft(r['new_label'])
        if not soft:
            continue
        mask = (tid == str(r['tile']).lower()) & (pid == str(r['polygon']))
        if not mask.any():
            continue
        for cls, val in soft.items():
            df.loc[mask, cls] = float(val)
        n_changed += int(mask.sum())
    return df, n_changed


class MrrsuAuxPatchDataset(Dataset):
    """Wraps a CRISMSpectralPatchDataset and appends a normalized mrrsu aux vector.

    Yields (patch (7,7,59), aux2 (2,), label (5,), weight). aux2 is the
    normalized [mean_7x7 RPEAK1, mean_7x7 BD1300] from the aligned
    mrrsu_aux_{split}.npy cache. NaN aux rows (tiles without a paired mrrsu,
    physically-implausible windows, or all-NODATA windows) map to 0.0
    post-normalization -- i.e. the sample mean, contributing no information.

    Supported normalization modes (read from ``mrrsu_aux_stats.json["mode"]``):
      - ``zscore``        : (x - global_mean) / global_std
      - ``minmax``        : (x - global_min) / (global_max - global_min), clipped to [0, 1]
      - ``pertile_zscore``: (x - tile_mean) / tile_std, computed per tile from
                            the physically-valid aux rows belonging to that tile
                            in the current split. Tiles with fewer than
                            ``min_valid_per_tile`` valid rows fall back to the
                            global ``fallback_mean`` / ``fallback_std``.

    The stats JSON must have ``version == 2`` -- legacy v1 JSONs (no ``version``
    field) raise ``ValueError`` to force a rebuild via ``scripts/build_mrrsu_aux.py``.
    """

    def __init__(self, df, mrral_map, patch_size, aux_npy, stats_json,
                 cache_dir=None, split='train'):
        self.inner = CRISMSpectralPatchDataset(
            df, mrral_map, patch_size=patch_size, cache_dir=cache_dir, split=split)
        aux = np.load(aux_npy).astype(np.float32)
        assert len(aux) == len(self.inner), (
            f"aux rows {len(aux)} != patch rows {len(self.inner)}")
        with open(stats_json) as f:
            st = json.load(f)
        version = st.get('version')
        if version != 2:
            raise ValueError(
                f"{stats_json} has version={version!r} but this code requires "
                "version=2. Regenerate the aux cache with "
                "`scripts/build_mrrsu_aux.py` (which now writes v2 stats)."
            )
        mode = st.get('mode')
        self.mode = mode
        if mode == 'zscore':
            z = self._apply_zscore(aux, st)
        elif mode == 'minmax':
            z = self._apply_minmax(aux, st)
        elif mode == 'pertile_zscore':
            # Reset the dataframe index so positional indexing aligns with `aux`
            # rows (build_mrrsu_aux.py uses ``reset_index(drop=True)`` per split).
            df_reset = df.reset_index(drop=True)
            if 'tile_id' not in df_reset.columns:
                raise ValueError(
                    "pertile_zscore mode requires df['tile_id']; got columns="
                    f"{list(df_reset.columns)}"
                )
            z = self._apply_pertile_zscore(aux, st, df_reset['tile_id'].to_numpy())
        else:
            raise ValueError(
                f"unsupported norm mode {mode!r} in {stats_json}; expected one of "
                "{'zscore', 'minmax', 'pertile_zscore'}"
            )
        z[~np.isfinite(z)] = 0.0          # NaN/inf -> 0 (== sample mean post-transform)
        self.aux = torch.from_numpy(z.astype(np.float32))

    # ------------------------------------------------------------------ transforms
    @staticmethod
    def _apply_zscore(aux: np.ndarray, st: dict) -> np.ndarray:
        mean = np.asarray(st['mean'], dtype=np.float32)
        std = np.asarray(st['std'], dtype=np.float32)
        return (aux - mean) / std

    @staticmethod
    def _apply_minmax(aux: np.ndarray, st: dict) -> np.ndarray:
        mn = np.asarray(st['min'], dtype=np.float32)
        mx = np.asarray(st['max'], dtype=np.float32)
        denom = np.where((mx - mn) < 1e-8, np.float32(1.0), (mx - mn))
        out = (aux - mn) / denom
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _apply_pertile_zscore(aux: np.ndarray, st: dict,
                              tile_ids: np.ndarray) -> np.ndarray:
        """Per-tile z-score with a global fallback for tiles below threshold.

        Computed at init time once: each tile-group's mean/std over rows whose
        aux is finite. Tiles with fewer than ``min_valid_per_tile`` finite rows
        use ``fallback_mean`` / ``fallback_std`` (== global zscore stats).
        """
        from data.mrrsu_aux import (
            physically_valid_mask as _pvm,
            AUX_BAND_ORDER as _ORDER,
        )

        fallback_mean = np.asarray(st['fallback_mean'], dtype=np.float32)
        fallback_std = np.asarray(st['fallback_std'], dtype=np.float32)
        min_valid = int(st.get('min_valid_per_tile', 1000))

        # Per-row validity mask (both bands physically valid AND finite)
        # NB: aux is post-pooling so we just need finite (NaN entries are
        # already the sentinel for invalid). Use ``_pvm`` for symmetry / safety.
        row_valid = np.ones(len(aux), dtype=bool)
        for j, b in enumerate(_ORDER):
            row_valid &= _pvm(aux[:, j], b)

        out = np.empty_like(aux, dtype=np.float32)
        tile_ids = np.asarray(tile_ids)
        # Group rows by tile_id; vectorize per-tile mean/std on valid subset.
        for tid in np.unique(tile_ids):
            tile_rows = np.where(tile_ids == tid)[0]
            tile_valid = tile_rows[row_valid[tile_rows]]
            if len(tile_valid) >= min_valid:
                tm = aux[tile_valid].mean(axis=0).astype(np.float32)
                ts = aux[tile_valid].std(axis=0).astype(np.float32) + np.float32(1e-8)
            else:
                tm = fallback_mean
                ts = fallback_std
            out[tile_rows] = (aux[tile_rows] - tm) / ts
        return out

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        patch, label, weight = self.inner[idx]
        return patch, self.aux[idx], label, weight
