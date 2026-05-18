# Pre-training Patch Cache + Dataset Replacement — Design Spec

**Date:** 2026-05-18
**Status:** Design (not yet implemented)
**Related:** I/O bottleneck investigation report at `/tmp/io_bottleneck_investigation.md`

## 1. Goal

Replace the streaming `CRISMGlobalPatchDataset` (which incurs ~109 ms/patch from BSQ-induced read amplification) with a pre-built mmap-backed cache of 5M patches stored as 50 sharded `.npy` files on HPC `/xdisk`. This brings the per-epoch wall time from ~135 min into the ~5–9 min budget originally assumed by the 32-hour SLURM plan.

## 2. Background and rationale

The two pre-training runs (v3 denoising MAE and v4 SPEND MAE) are running ~14× slower than budgeted: ~135 min/epoch instead of the planned 9.6 min/epoch. Both will time out at ~10% of intended training. Profiling pinpointed the root cause:

- CRISM mrral tiles are stored as **BSQ ENVI with `blockysize=1`**.
- Reading a 7×7×59 patch requires **59 separate disk seeks** spanning a 517 MB working region in a 631 MB file.
- Read amplification is **191×** per band: each band must fetch 7 full scanlines × 1340 pixels × 4 bytes = 5.36 KB to extract 196 useful bytes.
- Total per-patch I/O: **2.16 MB read to deliver 11.3 KB useful**.
- HPC RAM (48 GB) cannot effectively page-cache the 912 GB total working set.

Observed cost: ~252 ms per patch (worker time), of which:
- BSQ windowed read × 1.45× rejection overhead: ~158 ms (63%)
- Worker respawn + cold rasterio.open: ~37 ms (15%)
- OS page fault overhead: ~40 ms (16%)
- Python / NumPy / torch compute: 0.2 ms (<1%)

The investigation considered three families of fix: in-place streaming optimizations (BIP/BIL conversion, GDAL block tweaks), partial pre-computation (valid-center masks), and a full pre-built patch cache. Only the cache closes the gap by enough to fit within budget. The streaming optimizations top out at ~3–4× speedup and still leave the run 5–10× over budget.

## 3. Design summary

- **One-time HPC SLURM job** builds the cache by extracting 5M valid patches uniformly across the 1764 mrral tiles (~2834 patches/tile), filtered by pre-computed valid-center masks so the 31% rejection overhead is eliminated.
- **Cache format:** 50 sharded `.npy` files of 100K patches each, shape `(N, 7, 7, 59)` float32, stored as raw clipped I/F values (per-patch normalization happens on the dataloader read).
- **`CRISMGlobalPatchDataset` is deleted.** A new `CRISMCachedPatchDataset` (mmap-backed `IterableDataset`, single responsibility) replaces it.
- **Both pretraining scripts** (`pretrain_spatial_mae_denoising.py` and `pretrain_spatial_mae_spend.py`) are updated to use the new dataset. `persistent_workers=True` is added to the DataLoader at the same time.

## 4. Cache layout and build process

### 4.1 File layout

At `/xdisk/sbyrne/phillipsm/crism_patch_cache/`:

```
crism_patch_cache/
  global_patches_000.npy        # shape (100000, 7, 7, 59), float32 → 1.16 GB
  global_patches_001.npy
  ...
  global_patches_049.npy        # 50 shards × 100K patches = 5M total → 58 GB
  shard_index.json              # build metadata
```

### 4.2 Shard index schema

```json
{
  "n_shards": 50,
  "patches_per_shard": 100000,
  "patch_size": 7,
  "n_bands": 59,
  "min_valid_frac": 0.8,
  "clip_max": 0.5,
  "nodata_value": 65535,
  "seed": 42,
  "patches_per_tile_target": 2834,
  "tiles_used": 1764,
  "tiles_skipped": [],
  "shards": [
    {"id": 0, "n_patches": 100000, "tile_count": 36, "patches_per_tile_mean": 2778, "build_time_s": 1023.4}
  ],
  "total_build_time_s": 50621.0
}
```

### 4.3 Build process

Implemented in `scripts/build_global_patch_cache.py`. Architecture: `multiprocessing.Pool` with `imap_unordered` — workers produce per-tile patch arrays, main process buffers and writes shards.

**Worker function** (one task per tile, parallelism = `--workers`, default 16):

1. Open mrral via rasterio.
2. Load one full band (band 1) to compute a tile-wide **nodata mask** at native resolution.
3. **Vectorize the validity check** via `scipy.signal.convolve2d` on the nodata mask, producing a per-center `valid_frac[r, c]` array in one pass (~1s per tile). Centers where `valid_frac >= min_valid_frac` (default 0.8) are eligible.
4. **Sample `patches_per_tile_target=2834` centers uniformly without replacement** from the eligible set, using a tile-local RNG seeded as `seed * 1000003 + tile_index`. If fewer eligible centers exist, take all and record the shortfall.
5. For each sampled center, read the 7×7×59 patch via `src.read([1..59], window=...)`, cast to float32, clip to `[0, clip_max]` (default 0.5, matching `data/global_patch_dataset.py`).
6. Return `(tile_id, patches_array_of_shape_(n, 7, 7, 59), skipped_short_count)` to the main process.

**Main coordinator process:**

1. Submit all 1764 tiles to the pool via `pool.imap_unordered`.
2. Maintain a buffer (list of arrays). For each result:
   - Append `patches_array` to the buffer.
   - While the buffer holds ≥ `patches_per_shard=100000` patches: take the first 100000, `np.save` to `global_patches_NNN.npy`, increment shard counter.
3. After all results consumed, flush any remaining buffer as the final shard (may be < 100K). The final shard count is therefore approximately 50 but not guaranteed exactly 50.
4. Optionally call `np.random.shuffle` (with the main RNG) on each shard before writing to disk to interleave tile sources within a shard.
5. Write `shard_index.json`.

**Memory ceiling:** workers each produce up to 2834 patches per tile (~33 MB). Main buffer caps at ~200K patches (~232 MB) before flushing. Peak across all 16 workers + main: ~1 GB. Comfortably fits in the 64 GB SLURM allocation.

### 4.4 Random seed

Fixed at `42` by default. Configurable via `--seed`. Two distinct seeds produce distinct caches (different patch selections); the seed is recorded in `shard_index.json` for reproducibility.

### 4.5 SLURM build job

`scripts/hpc_build_global_cache.slurm`. Use `scripts/hpc_build_cache.slurm` as a structural template (same env activation, account, partition). Differences:

- `--cpus-per-task=16` (up from 4 — wider parallelism over 1764 tiles)
- `--mem=64gb` (up from 16gb — accommodates 16 concurrent tile-results + main buffer)
- `--time=24:00:00` (up from 2 hr — generous safety margin)
- `--partition=standard` (same as existing build job)
- Calls `${PYTHON} -u scripts/build_global_patch_cache.py --workers 16 --output /xdisk/sbyrne/phillipsm/crism_patch_cache/ --seed 42`

**Estimated build time:** ~5.5 hours on HPC `/xdisk`. Each tile requires 2834 patch reads × 109 ms cold + valid-mask compute (~1s) ≈ 310 s per tile sequentially. With 16 workers in parallel: 1764 tiles × 310 s / 16 workers = ~5.7 hours. The 24-hour SLURM budget includes safety margin for filesystem variability.

## 5. `CRISMCachedPatchDataset` class

### 5.1 Interface

`data/cached_patch_dataset.py`:

```python
class CRISMCachedPatchDataset(IterableDataset):
    """Yields (patch_size, patch_size, n_bands) float32 tensors from a pre-built shard cache.

    Each shard is mmap-loaded on demand; per-shard handles are held for the
    worker's lifetime. Per-patch normalization (zero-mean, unit-variance) is
    applied on read.
    """
    def __init__(
        self,
        shard_dir: str,
        normalize: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.shard_dir = shard_dir
        self.normalize = normalize
        self.shuffle = shuffle
        self.seed = seed
        self.shards = sorted(glob.glob(os.path.join(shard_dir, 'global_patches_*.npy')))
        if not self.shards:
            raise FileNotFoundError(f"No shards in {shard_dir}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        rng_seed = self.seed
        if rng_seed is not None and worker_info is not None:
            rng_seed = rng_seed + worker_info.id
        rng = np.random.default_rng(rng_seed)

        shard_order = list(range(len(shards)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for si in shard_order:
            arr = np.load(shards[si], mmap_mode='r')   # (N_si, 7, 7, 59)
            indices = np.arange(len(arr))
            if self.shuffle:
                rng.shuffle(indices)
            for i in indices:
                patch = arr[i].astype(np.float32, copy=True)   # detach from mmap
                if self.normalize:
                    mean = patch.mean()
                    std = patch.std() + 1e-8
                    patch = (patch - mean) / std
                yield torch.from_numpy(patch)
```

### 5.2 Properties

- **Single responsibility:** cache reading + per-patch normalization. No tile traversal, no validity check, no rasterio dependency.
- **Mmap-backed:** `np.load(..., mmap_mode='r')` lazily faults shard pages from disk. With 6 workers × 1 shard at a time = ~7 GB working set, fits comfortably in HPC node RAM after a warm-up epoch.
- **No `__len__`:** as an `IterableDataset`, the training loop already uses `patches_per_epoch / batch_size` to size each epoch — no change needed in pretrain scripts.
- **Worker sharding:** each DataLoader worker receives a non-overlapping slice of the shard list (`shards[worker_id :: num_workers]`). No two workers yield the same shard.
- **Per-worker reproducibility:** when `seed` is set, each worker's RNG is seeded with `seed + worker_id`. With a fixed seed and num_workers, the iteration order is deterministic.
- **Per-patch normalization on read:** zero-mean / unit-variance per patch using `(patch - patch.mean()) / (patch.std() + 1e-8)` — same as the deleted `CRISMGlobalPatchDataset`.

### 5.3 Deleted code

- `data/global_patch_dataset.py` — the entire `CRISMGlobalPatchDataset` class.
- `tests/test_global_patch_dataset.py` — the streaming dataset tests.

No code outside the pretraining scripts imports `CRISMGlobalPatchDataset`. Removal is clean.

## 6. Pretrain script integration

### 6.1 Code changes

In both `scripts/pretrain_spatial_mae_denoising.py` and `scripts/pretrain_spatial_mae_spend.py`:

**Remove:**
```python
data_root = cfg.get('data_root', '/mnt/crism/MRDR')
globs_to_try = [
    os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
    os.path.join(data_root, 't*mrral*.hdr'),
]
hdr_files = []
for g in globs_to_try:
    hdr_files = sorted(glob.glob(g))
    if hdr_files:
        break
if not hdr_files:
    raise FileNotFoundError(...)
log.info(f"Found {len(hdr_files)} mrral tiles")

from data.global_patch_dataset import CRISMGlobalPatchDataset
ds = CRISMGlobalPatchDataset(hdr_files, patch_size=7, min_valid_frac=0.8)
loader = DataLoader(
    ds, batch_size=args.batch_size, num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=4 if args.num_workers > 0 else None,
)
```

**Replace with:**
```python
shard_dir = cfg.get('patch_cache_dir')
if not shard_dir:
    raise KeyError("config.local.yaml must define patch_cache_dir")
log.info(f"Patch cache: {shard_dir}")

from data.cached_patch_dataset import CRISMCachedPatchDataset
ds = CRISMCachedPatchDataset(shard_dir=shard_dir, normalize=True, shuffle=True)
loader = DataLoader(
    ds, batch_size=args.batch_size, num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=4 if args.num_workers > 0 else None,
    persistent_workers=args.num_workers > 0,
)
```

`glob` and `os` imports stay (still used elsewhere in the scripts).

### 6.2 SLURM script changes

In both `scripts/hpc_pretrain_denoising.slurm` and `scripts/hpc_pretrain_spend.slurm`:

The auto-generated `config.local.yaml` already includes `patch_cache_dir`. Update it to point at the new global cache:

**Before:**
```yaml
patch_cache_dir: ${WORK_DIR}/data/patch_cache
```

**After:**
```yaml
patch_cache_dir: /xdisk/sbyrne/phillipsm/crism_patch_cache
```

(The supervised-classifier patch cache at `${WORK_DIR}/data/patch_cache` is named differently — `mrral_train_patches_p7.npy` etc. — and is read directly via its own path in the supervised training pipeline, not via `patch_cache_dir`. No collision.)

## 7. Testing strategy

Tests in `tests/test_cached_patch_dataset.py`:

1. **Shape & dtype:** yielded tensors have shape `(7, 7, 59)` and `dtype=torch.float32`.
2. **Determinism with seed:** two iterators with `seed=0` and identical worker config yield the same first K patches (element-wise equal).
3. **Different seeds → different orders:** `seed=0` and `seed=1` produce different first-patch tensors (or at minimum a different overall sequence).
4. **Normalization applied when enabled:** with `normalize=True`, sampled patches have absolute mean < 1e-5 and std in `(0.5, 1.5)` (loose bounds to allow for per-patch variation).
5. **Raw values preserved when disabled:** with `normalize=False`, all yielded values are in `[0, clip_max]` (per the build-time clip).
6. **Worker sharding:** with `num_workers=4`, the union of patches yielded across workers in a fixed-seed run equals the full cache, with no duplicates within an epoch.
7. **Mmap reuse:** with `np.load` instrumented (monkey-patch), an epoch over a 3-shard cache yields exactly `ceil(n_shards / n_workers)` `np.load` calls per worker (not one per patch). For 3 shards and 1 worker: 3 calls. For 3 shards and 3 workers: 1 call each.
8. **Missing shard dir:** constructing with a non-existent or empty directory raises `FileNotFoundError`.

Tests in `tests/test_build_global_patch_cache.py`:

9. **Builder produces a valid cache on a synthetic 3-tile fixture.** Use `tmp_path`. Synthesize three small mock mrral tiles (e.g. 100×100×59) with controlled nodata patterns. Run the builder with `patches_per_tile_target=20, patches_per_shard=30`. Confirm: 3 shard files exist with combined 60 patches; `shard_index.json` is well-formed; loading via `CRISMCachedPatchDataset` yields the expected total count.
10. **Validity filter respected:** in the synthetic fixture, tiles with no valid centers produce zero patches and appear in `tiles_skipped`.
11. **Determinism:** same seed → same patch tensors written to disk.

**Manual integration smoke** (documented, not automated):

12. After cache build completes on HPC, run a 2-epoch dry-run of `pretrain_spatial_mae_spend.py` against the real cache. Confirm: data loads, loss is finite and descending, anneal schedule fires correctly, no NaNs. Compare per-epoch wall time against the budget.

## 8. Success criteria

- **Performance:** per-epoch wall time drops from ~135 min to under 10 min on HPC. 200 epochs complete within the 32-hour wall time.
- **Numerical correctness:** the first epoch's loss on cached data is within 5% of a streaming-dataset run (both starting from the same seed) on a small dev cache. Verified once via the manual integration smoke.
- **Cache reproducibility:** building twice with the same seed produces byte-identical shard files. (Verified by hashing in test 11 above.)
- **No regressions in v3/v4:** the existing 23 SPEND tests and 13 denoising MAE tests continue to pass after the dataset swap.

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `/xdisk` quota exceeded by 58 GB cache | Low | Check `quota -s` before building. 58 GB is well within typical sbyrne allocations. |
| Cache build crashes partway through | Medium | Builder writes shards incrementally; partial caches are usable for smoke tests. Resume support is out of scope (just rerun from scratch). |
| Mmap performance worse than expected on /xdisk filesystem | Low | The investigation profiled NVMe-backed local disk. /xdisk is shared and may be slower; if epoch time is still >20 min after cache, investigate filesystem-level issues. |
| Cache becomes stale (new tiles added, patch_size changes) | Medium | The `shard_index.json` records the build parameters. Future code can compare current config against the index and rebuild if mismatched. Not implemented in this spec — manual rebuild on parameter change. |
| Mmap working set spills out of RAM | Low | 6 workers × 1 shard each = 7 GB peak. 48 GB HPC RAM is ample. |
| Per-patch normalization regression vs streaming dataset | Medium | Test 4 + manual integration smoke (test 12) confirm normalization output matches. |
| Worker sharding bug: some shards never visited or visited twice | Medium | Test 6 directly verifies the union/disjoint invariant. |

## 10. Out of scope

- **Cache for the labeled-patch supervised dataset** (`mrral_train_patches_p7.npy`, etc.) — already exists and is unrelated.
- **Augmentation pipeline.** Pretraining doesn't use augmentation today; if added later, augmentations happen at load time after normalization.
- **Multi-band reading optimizations on the streaming path.** The streaming dataset is being deleted.
- **Conversion of mrral tiles to BIP/BIL interleave.** Only the cache is needed.
- **Backward-compatibility shims for `CRISMGlobalPatchDataset` imports.** No code outside pretraining uses it.
- **Cache versioning / stale-cache detection.** The `shard_index.json` records build parameters but no code consumes them for validation in this spec.
- **Resume support for the cache builder.** Partial caches are usable as smoke fixtures; full rebuild is the recovery path.
- **Parallel sharded reads during training (e.g. multiple shards mmap'd per worker).** Single-shard-at-a-time is already fast enough per the projection.

## 11. Open questions resolved during brainstorming

1. **Cache size:** 5M patches (~58 GB). 2M was deemed too sparse; 10M would burn build time and storage for marginal diversity gain.
2. **Normalization timing:** at load time. Store raw clipped values; per-patch normalization happens in `__iter__`. Keeps flexibility for future normalization ablations.
3. **Dataset class shape:** replace `CRISMGlobalPatchDataset` entirely. No backward-compat fallback. The cache is required for pretraining going forward.
4. **Storage location:** HPC `/xdisk/sbyrne/phillipsm/crism_patch_cache/` only. No local copy needed.
