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
    if n > len(LABEL_COLS_WITH_ALTERATION):
        raise ValueError(f'checkpoint head has {n} outputs; no known label '
                         f'set that large')
    return list(LABEL_COLS_WITH_ALTERATION[:n])


_TIER_WEIGHTS = {'high': 1.0, 'moderate': 0.85, 'low': 0.70}


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
    # alteration is a flat column (no t1/t2 split). For backward compat with
    # parquets predating the 6-class schema, default to 0.0 when missing.
    if 'alteration' not in out.columns:
        out['alteration'] = np.float32(0.0)
    else:
        out['alteration'] = (out['alteration'] > 0).astype(np.float32)
    if 'confidence_tier' in out.columns:
        mapped = (
            out['confidence_tier']
            .astype(str).str.lower()
            .map(_TIER_WEIGHTS)
        )
        if 'confidence_weight' in out.columns:
            # Unknown tiers (Reviewed/Ambiguous) fall back to the stamped
            # weight; only rows lacking both get the Moderate default.
            mapped = mapped.fillna(
                pd.to_numeric(out['confidence_weight'], errors='coerce'))
        out['confidence_weight'] = (
            mapped.fillna(_TIER_WEIGHTS['moderate']).astype(np.float32)
        )
    elif 'confidence_weight' not in out.columns:
        out['confidence_weight'] = np.float32(_TIER_WEIGHTS['moderate'])
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
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        df = _collapse_labels(df).reset_index(drop=True)
        self.mrral_map = mrral_map
        self.patch_size = patch_size
        self.half = patch_size // 2
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
        if cache_dir and split:
            cache_file = os.path.join(cache_dir, f'mrral_{split}_patches_p{patch_size}.npy')
            if os.path.exists(cache_file):
                # The cache is a raw memmap (no npy header), so np.memmap
                # would silently read a prefix of a stale/oversized file —
                # exactly the failure mode when the parquet is rebuilt but
                # the cache isn't. Require an exact byte-size match.
                expected_bytes = self._n * patch_size * patch_size * 59 * 4
                actual_bytes = os.path.getsize(cache_file)
                if actual_bytes != expected_bytes:
                    raise ValueError(
                        f'patch cache {cache_file} is {actual_bytes:,} bytes '
                        f'but the {split} dataframe ({self._n:,} rows) needs '
                        f'exactly {expected_bytes:,}. The cache was built '
                        f'from different parquet rows — rebuild it '
                        f'(scripts/cache_mrral_patches.py) or fix '
                        f'--patch_cache_dir.')
                self._cache = np.memmap(
                    cache_file, dtype='float32', mode='r',
                    shape=(self._n, patch_size, patch_size, 59)
                )

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if self._cache is not None:
            patch = torch.from_numpy(self._cache[idx].copy())
            return patch, self.labels[idx], self.weights[idx]

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

        return torch.from_numpy(patch.copy()), self.labels[idx], self.weights[idx]

    def close(self):
        for src in self._handles.values():
            src.close()
        self._handles.clear()


class SyntheticPatchDataset(Dataset):
    """Serves pre-synthesized plagioclase patches from a .npy + parquet fragment.

    Mirrors CRISMSpectralPatchDataset's __getitem__ contract:
    returns (patch (7,7,59) float32 tensor, label (5,) tensor in LABEL_COLS order,
    weight scalar tensor). Patches are read from a memmap'd .npy aligned row-for-row
    with the parquet fragment.
    """

    def __init__(self, npy_path: str, parquet_path: str, patch_size: int = 7):
        df = _collapse_labels(pd.read_parquet(parquet_path)).reset_index(drop=True)
        self._n = len(df)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)
        self._cache = np.load(npy_path, mmap_mode='r')
        assert self._cache.shape[0] == self._n, (
            f"patch count {self._cache.shape[0]} != parquet rows {self._n}")
        assert self._cache.shape[1:] == (patch_size, patch_size, 59)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        patch = torch.from_numpy(np.asarray(self._cache[idx], dtype=np.float32).copy())
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
