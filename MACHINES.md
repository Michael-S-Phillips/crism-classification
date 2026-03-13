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
| Primary workstation | `/mnt/mrdr` |
| Older workstation | `/mnt/gigas/CRISM/MRDR` |
| Server | `/mnt/crism/MRDR` |

## How it works

`config_loader.py` (project root) merges configs in priority order:
1. `CRISM_DATA_ROOT` env var (overrides `data_root` only)
2. `config.local.yaml` (overrides any key)
3. `config.yaml` (committed baseline)

All scripts use `from config_loader import load_config` instead of opening `config.yaml` directly.

## Note for Claude Code sessions

If you're a Claude Code agent on a different machine: **create `config.local.yaml` before running any scripts.** The template is at `config.local.yaml.template`. Without it, paths may resolve incorrectly against the primary workstation's `/mnt/mrdr` mount.
