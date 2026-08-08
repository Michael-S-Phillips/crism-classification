# Multi-Machine Setup

This project runs on multiple machines with different mount points for the CRISM data.

## Config override pattern

`config.yaml` is committed with the canonical structure. Each machine has a **`config.local.yaml`** (gitignored) that overrides just what differs — typically only `data_root`.

**Setup on a new machine:**
```bash
cp config.local.yaml.template config.local.yaml
# Edit config.local.yaml to set your data_root
```

All derived paths (`output_dir`, `checkpoints_dir`, `gpkg_dir`, etc.) are recomputed automatically from `data_root`. You can also set `CRISM_DATA_ROOT` as an environment variable instead.

## Known mount points by machine

| Machine | `data_root` |
|---------|------------|
| Primary workstation (Linux/WSL) | `/mnt/mars-gis/CRISM/MRDR` |
| Primary workstation (older mount) | `/mnt/mrdr` |
| Older workstation | `/mnt/gigas/CRISM/MRDR` |
| Server | `/mnt/crism/MRDR` |
| Mac workstation (Apple Silicon) | `/Volumes/Mars_GIS/CRISM/MRDR` |
| HPC (xdisk) | `/xdisk/sbyrne/phillipsm/CRISM_MRDR` |

`config_loader.py` treats all four of these as recognized roots, so overriding only
`data_root` in `config.local.yaml` re-derives every other path. If you add a new
machine with a fresh mount, add its old root to `_KNOWN_OLD_ROOTS` in `config_loader.py`
so stale committed paths get rewritten.

## How it works

`config_loader.py` (project root) merges configs in priority order:
1. `CRISM_DATA_ROOT` env var (overrides `data_root` only)
2. `config.local.yaml` (overrides any key)
3. `config.yaml` (committed baseline)

All scripts use `from config_loader import load_config` instead of opening `config.yaml` directly.

## macOS (Apple Silicon) setup

The Mac workstation differs from the Linux machines in three ways beyond `data_root`:

1. **No CUDA — use MPS.** Scripts select their device via `device.get_device()`
   (project root, next to `config_loader.py`), which resolves `cuda → mps → cpu`,
   so on this Mac they run on the **Apple GPU (MPS)** by default. Force a specific
   backend for any run with the `CRISM_DEVICE` env var:
   ```bash
   CRISM_DEVICE=cpu conda run -n crism python scripts/classify_tile_supervised.py ...
   ```
   Heavy training still belongs on HPC (real CUDA GPUs), but inference/analysis run
   locally on MPS. Note: MPS has no float64 — keep on-device tensors float32.
2. **OpenMP double-load.** torch, xgboost, and lightgbm each ship their own OpenMP runtime;
   loaded in one process they cause an intermittent `Abort trap: 6` / segfault. Fixed by
   env vars baked into the `crism` env:
   ```bash
   conda env config vars set -n crism KMP_DUPLICATE_LIB_OK=TRUE PYTHONNOUSERSITE=1
   ```
   `PYTHONNOUSERSITE=1` also isolates the env from a broken `langsmith` pytest plugin in
   `~/.local` that otherwise breaks every `pytest` run.
3. **Extra conda packages** not in `environment.yml` (were silently borrowed from `~/.local`
   on the old machine; now installed into the env): `fiona`, `plotly`, `streamlit`,
   `scikit-image` — needed by the review/visualizer apps and some scripts.

Create the env with `conda env create -f environment.yml` (installs numpy<2.0, torch, etc.),
then apply the two steps above.

## Note for Claude Code sessions

If you're a Claude Code agent on a different machine: **create `config.local.yaml` before running any scripts.** The template is at `config.local.yaml.template`. Without it, paths may resolve incorrectly against the primary workstation's `/mnt/mrdr` mount.
