# MPS-aware device selection — design

**Date:** 2026-08-06
**Status:** approved, implemented

## Problem

After migrating from a Linux/CUDA workstation to an Apple-Silicon Mac, all 27
torch scripts hard-code `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
On the Mac that always resolves to **CPU**, leaving the Apple GPU (MPS) idle even
though `torch.backends.mps.is_available()` is `True`. The machine is capable enough
to run real inference/analysis locally, so we want MPS used by default.

## Approach

One small helper module, swept across every device-selection site.

### `device.py` (project root, alongside `config_loader.py`)

- `get_device(prefer=None) -> torch.device`
- Resolution order: explicit `prefer` arg → `CRISM_DEVICE` env var → auto
  (`cuda` → `mps` → `cpu`). Unavailable accelerators downgrade automatically.
- `resolve_device_str()` exposes the string form.
- Imported the same way scripts already reach `config_loader`: a
  `sys.path.insert(project_root)` bootstrap + `from device import get_device`.

### Sweep (27 files under `scripts/`, `training/`, `sam_analysis/`)

- Replace the inline `torch.device(... cuda ... cpu)` and bare-string variants
  with `get_device()`. Verified there are **no** `device == 'cuda'` string
  comparisons anywhere, so returning a `torch.device` is safe at every call site.
- Inject the import bootstrap after the file's `import torch` (top-level, or
  indented where torch is imported lazily inside a function).

### Test behavior

- `tests/conftest.py` sets `CRISM_DEVICE=cpu` (via `setdefault`) so the suite is
  deterministic and backend-independent. Real scripts still auto-select MPS.

### Environment

- `PYTORCH_ENABLE_MPS_FALLBACK=1` baked into the `crism` conda env so the few
  ops not yet implemented on MPS fall back to CPU instead of raising mid-run.

## Constraints / caveats

- **MPS does not support float64.** The models are float32, so this is a
  non-issue in practice; documented in the `device.py` docstring.
- Per-run override: `CRISM_DEVICE=cpu conda run -n crism python scripts/...`.

## Out of scope

- No changes to the cuda/HPC training path (those machines have real GPUs).
- No per-op MPS optimization or autocast; fallback handles unsupported ops.
