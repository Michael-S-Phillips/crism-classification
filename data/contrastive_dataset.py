"""ContrastiveTripletDataset for plagioclase-vs-olivine encoder refinement.

Three pre-harvested patch pools (each shape ``(N, 7, 7, 59)``):

* **positives** — confirmed plagioclase pixels from labeled gpkgs
  (``plagioclase (High|Moderate)``).
* **hard_negatives** — MC13 polygons the current classifier confidently labels
  plag but which are spectrally olivine (the bias we are trying to break).
* **soft_negatives** — confirmed Type 1/2 olivine (High) pixels — the standard
  "should be far away" negatives.

For each anchor (one positive), the dataset returns:

  (anchor, positive, hard_negatives, soft_negatives)

with shapes ``(7,7,59)``, ``(7,7,59)``, ``(n_hard, 7,7,59)``, ``(n_soft, 7,7,59)``.
``positive`` is a *different* positive patch (random sample without replacement
when ``len(positives) > 1``). Negatives are sampled with replacement (the pools
are typically large enough that this is fine and keeps the code simple).
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


_PathLike = Union[str, os.PathLike, np.ndarray]


def _load_patches(src: _PathLike) -> np.ndarray:
    """Load a patch pool from a .npy path or accept an in-memory array."""
    if isinstance(src, np.ndarray):
        arr = src
    else:
        arr = np.load(str(src), mmap_mode='r')
    if arr.ndim != 4 or arr.shape[-1] != 59:
        raise ValueError(
            f"expected patches of shape (N, P, P, 59); got {arr.shape}"
        )
    return arr


class ContrastiveTripletDataset(Dataset):
    """Yield (anchor, positive, hard_negatives, soft_negatives) triplets."""

    def __init__(
        self,
        positives: _PathLike,
        hard_negatives: _PathLike,
        soft_negatives: _PathLike,
        n_hard_per_batch: int = 8,
        n_soft_per_batch: int = 8,
        seed: Optional[int] = None,
    ):
        self.positives = _load_patches(positives)
        self.hard_negatives = _load_patches(hard_negatives)
        self.soft_negatives = _load_patches(soft_negatives)

        if len(self.positives) == 0:
            raise ValueError("positives pool is empty")
        if len(self.hard_negatives) == 0:
            raise ValueError("hard_negatives pool is empty")
        if len(self.soft_negatives) == 0:
            raise ValueError("soft_negatives pool is empty")

        if n_hard_per_batch < 1 or n_soft_per_batch < 1:
            raise ValueError("n_hard_per_batch and n_soft_per_batch must be >= 1")

        self.n_hard_per_batch = int(n_hard_per_batch)
        self.n_soft_per_batch = int(n_soft_per_batch)
        # Each worker should have a deterministic-but-distinct stream when seeded;
        # we don't try to be clever about that here (the consumer can re-seed in
        # a worker_init_fn if needed).
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.positives)

    def _sample_positive(self, idx: int) -> np.ndarray:
        n = len(self.positives)
        if n == 1:
            return np.asarray(self.positives[idx], dtype=np.float32).copy()
        # uniform over the n-1 other indices
        j = int(self.rng.integers(0, n - 1))
        if j >= idx:
            j += 1
        return np.asarray(self.positives[j], dtype=np.float32).copy()

    def _sample_negatives(self, pool: np.ndarray, k: int) -> np.ndarray:
        idx = self.rng.integers(0, len(pool), size=k)
        # memmap-safe copy
        return np.stack([np.asarray(pool[i], dtype=np.float32) for i in idx], axis=0)

    def __getitem__(self, idx: int):
        anchor = np.asarray(self.positives[idx], dtype=np.float32).copy()
        positive = self._sample_positive(idx)
        hard = self._sample_negatives(self.hard_negatives, self.n_hard_per_batch)
        soft = self._sample_negatives(self.soft_negatives, self.n_soft_per_batch)
        return (
            torch.from_numpy(anchor),
            torch.from_numpy(positive),
            torch.from_numpy(hard),
            torch.from_numpy(soft),
        )


class ExtraPositivesDataset(Dataset):
    """Wraps a pre-built positive patch pool for supervised use in eval.

    Yields ``(patch, label_vec, weight)`` matching the
    ``CRISMSpectralPatchDataset`` contract, so it can be concatenated with the
    standard supervised train loader (e.g. inside ``eval_contrastive.py``).

    ``label_vec`` is one-hot over ``LABEL_COLS`` with the configured class
    (default ``'plagioclase'``) set to 1.0. ``weight`` is a constant per-sample
    confidence weight (default 1.0; these patches are hand-vetted).
    """

    def __init__(
        self,
        pool_dir: str,
        positive_class: str = 'plagioclase',
        weight: float = 1.0,
        label_cols: Optional[tuple] = None,
    ):
        if label_cols is None:
            # Import lazily so this module doesn't pull rasterio at import time.
            from data.dataset import LABEL_COLS as _LBL
            label_cols = tuple(_LBL)
        if positive_class not in label_cols:
            raise ValueError(
                f"positive_class={positive_class!r} not in label_cols={label_cols!r}"
            )
        npy_path = os.path.join(pool_dir, 'patches.npy')
        self.patches = _load_patches(npy_path)
        self.n_classes = len(label_cols)
        self.plag_idx = label_cols.index(positive_class)
        self.weight = float(weight)
        # Pre-build the static label vector (every sample gets the same one)
        self._label_vec = np.zeros(self.n_classes, dtype=np.float32)
        self._label_vec[self.plag_idx] = 1.0

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = np.asarray(self.patches[idx], dtype=np.float32).copy()
        return (
            torch.from_numpy(patch),
            torch.from_numpy(self._label_vec.copy()),
            torch.tensor(self.weight, dtype=torch.float32),
        )
