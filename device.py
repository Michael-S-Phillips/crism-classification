"""Torch device selection for CRISM scripts.

Resolution order (highest priority first):
  1. explicit ``prefer`` argument to :func:`get_device`
  2. the ``CRISM_DEVICE`` environment variable (``cpu`` | ``cuda`` | ``mps``)
  3. auto-detect: CUDA -> MPS (Apple GPU) -> CPU

Usage
-----
    from device import get_device
    device = get_device()             # auto: uses MPS on Apple Silicon
    device = get_device(args.device)  # honor a --device CLI flag (None -> auto)

Force CPU for a single run without editing code::

    CRISM_DEVICE=cpu conda run -n crism python scripts/classify_tile_supervised.py ...

Notes
-----
* Apple's MPS backend does **not** support float64 — keep on-device tensors
  float32 (the models already are). ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is baked
  into the ``crism`` conda env so the few ops not yet implemented on MPS fall
  back to CPU instead of raising mid-run.
* The test suite pins ``CRISM_DEVICE=cpu`` (see ``tests/conftest.py``) for
  deterministic, backend-independent results.
"""
import os

import torch

_VALID = {"cpu", "cuda", "mps"}


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend) and backend.is_available()


def _auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_device_str(prefer: str = None) -> str:
    """Return a resolved device name string, downgrading unavailable accelerators."""
    name = (prefer or os.environ.get("CRISM_DEVICE") or _auto()).lower()
    if name not in _VALID:
        raise ValueError(f"Invalid device {name!r}; choose from {sorted(_VALID)}")
    if name == "cuda" and not torch.cuda.is_available():
        name = _auto()
    if name == "mps" and not _mps_available():
        name = "cpu"
    return name


def get_device(prefer: str = None) -> torch.device:
    """Return the torch.device to use. See module docstring for resolution order."""
    return torch.device(resolve_device_str(prefer))
