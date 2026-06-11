# Patch Pre-Cache Design

**Date:** 2026-03-07
**Status:** Approved

## Problem

`CRISMPatchDataset.__getitem__` extracts a 7×7 spatial patch via `rasterio.read(window=...)`
per sample at training time. With 726k training samples across 26 tiles on a SAMBA mount,
random small-window reads make each epoch take 30–60+ minutes.

## Solution

Pre-extract all patches once and save as numpy memmaps. At training time, load a patch with
a single array index instead of a rasterio read. SAMBA handles large sequential reads well,
making this ~10–100x faster than the current approach.

## Design

### Storage

Cache lives alongside the parquet in `{output_dir}/patch_cache/`:

```
data/patch_cache/
  train_patches_p7.npy   # shape (726033, 60, 7, 7) float32
  val_patches_p7.npy     # shape (98016,  60, 7, 7) float32
  test_patches_p7.npy    # shape (~100k,  60, 7, 7) float32
```

Filename encodes split and patch_size so a different `patch_size` won't silently reuse
stale cache. Total size ~10 GB.

### New script: `scripts/cache_patches.py`

- Reads `pixels.parquet` + `config.yaml`
- Calls `find_tile_pairs` to build `mrrsu_map`
- Iterates all rows per split using the same patch-extraction logic as `CRISMPatchDataset`
- Writes each split to a numpy memmap file
- Prints progress with tqdm; skips splits whose cache file already exists
- CLI: `python scripts/cache_patches.py [--patch_size 7] [--config config.yaml]`

### Modified: `data/dataset.py` — `CRISMPatchDataset`

Gains an optional `cache_dir: str = None` parameter. If `cache_dir` is set and the
corresponding `{split}_patches_p{patch_size}.npy` file exists:
- Load it as `np.memmap(..., dtype='float32', mode='r')`
- `__getitem__` returns `torch.from_numpy(self._cache[idx].copy())` instead of calling rasterio

If cache is absent, behavior is identical to today (rasterio live reads).

### Modified: `training/train_torch.py` — `train_torch_model`

Gains an optional `cache_dir: str = None` parameter, passed through to `make_dataset`.

### Modified: `scripts/train.py`

Reads `cfg.get('patch_cache_dir')` from config and passes it to `train_torch_model`.

### Modified: `config.yaml`

Adds:
```yaml
patch_cache_dir: /mnt/gigas/CRISM/MRDR/crism_classification/data/patch_cache
```

### Modified: `scripts/run_all_models.sh`

Before the CNN/ViT block, add a cache-generation step:
```bash
# Generate patch cache if not already done
if [[ ! -f "$PROJ_DIR/data/patch_cache/train_patches_p7.npy" ]]; then
    log "===== Generating patch cache ====="
    conda run -n crism python "$PROJ_DIR/scripts/cache_patches.py" 2>&1 | tee -a "$LOG_FILE"
fi
```

## Trade-offs

| | Cache | Live reads |
|---|---|---|
| First run | Slow (one-time build ~1–2 hrs) | Fast start |
| Subsequent epochs | Fast (memmap index) | Slow (~30–60 min/epoch) |
| Disk space | ~10 GB | 0 |
| Different patch_size | Rebuild cache | No rebuild needed |
| Machine portability | Cache on SAMBA, always available | Always works |

## Out of Scope

- Compression (HDF5/zarr) — not worth the complexity for this use case
- Per-tile cache files — single per-split file is simpler
- Automatic cache invalidation on parquet changes — user runs cache_patches.py manually if data changes
