# Contrastive plag-vs-olivine encoder refinement (Task D)

**Goal:** Fine-tune the MAE encoder with an InfoNCE objective that pulls confident plag pixels together and pushes them from MC13-classifier-predicted-plag pixels (which are spectrally olivine per user's qualitative call + the SAM diagnostic showing 55 % olivine misclassification in t0434). Then linear-probe + full-fine-tune the resulting encoder and compare plag AP to the current champion.

**Why now:** The mrrsu-aux normalisation sweep + plag-targeted long run confirmed plag is **encoder-bottlenecked**, not classifier-bottlenecked. Aux-head injection plateaued ~0.15 plag AP. Contrastive learning addresses the representation directly.

**Status of relevant infra (audit findings):**
- No contrastive code exists yet. `mae/contrastive_learning/` was never created — the task spec referenced it from memory.
- Reusable patterns:
  - `scripts/build_mtrdr_plag_patches.py` — walks gpkg, rasterizes polygons via `rasterio.features.rasterize`, extracts 7×7 patches. Direct template for harvesting MC13 plag-output polygons.
  - `models/spatial_spectral_transformer.py::SpatialSpectralTransformer` — the encoder we want to refine. It returns (B, n_tokens+1, embed_dim); center token at idx `n_tokens // 2 + 1`.
  - `data/synthetic_plag.py::interp_to_mrral_wavelengths` — not needed here (all input is mrral-native).
  - `data/vector_mc13_relabeled/plagioclase.gpkg` — 7 polygons at threshold ≥0.92, but ~80k+ pixels at the lower-threshold layers. Plenty of hard negatives.
- Encoder checkpoint to start from: `checkpoints/plag_aware_mae_128d_6l_best.pt` (same as ft_plag_aware_relabeled uses).

---

## Module layout

```
data/contrastive/                               (output dir for harvested patches)
  hard_negatives.npy / .parquet
  positives.npy / .parquet
  soft_negatives.npy / .parquet  (olivine, optional)
data/contrastive_dataset.py                     (new — Dataset class)
models/contrastive_encoder.py                   (new — encoder + projection head)
training/contrastive_train.py                   (new — training loop + InfoNCE)
scripts/build_contrastive_data.py               (new — gpkg → patch parquets)
scripts/train_contrastive.py                    (new — CLI driver)
scripts/hpc_contrastive.slurm                   (new — HPC launcher)
tests/test_contrastive_dataset.py               (new)
tests/test_contrastive_loss.py                  (new)
tests/test_contrastive_encoder.py               (new)
```

---

## Task 1 — Harvest the three pixel pools from gpkgs

**File:** `scripts/build_contrastive_data.py` (new)

**Three pools, all stored as 7×7×59 patches + metadata:**

| pool | source | label/intent | expected size |
|---|---|---|---|
| **hard_negatives** | `data/vector_mc13_relabeled/plagioclase.gpkg` rasterized onto each MC13 mrral tile | classifier said plag; we treat as NOT-plag (olivine) | ~50–80k pixels at the >=0.85 threshold layer |
| **positives** | `/mnt/mrdr/categorized_mineral_units/T*.gpkg` filtered to `category in {plagioclase (High), plagioclase (Moderate)}` | confirmed plag | ~few hundred polygons → maybe ~20–50k pixels |
| **soft_negatives** | same gpkgs, filtered to `Type 1 olivine (High)` ∪ `Type 2 olivine (High)` | confirmed olivine, standard contrast | thousands of polygons → many pixels — subsample |

For each pool, write a directory with:
- `patches.npy` shape `(N, 7, 7, 59)` float32, clipped [0, 0.5] same as training
- `meta.parquet` with columns `tile_id, row, col, source_polygon, source_gpkg`

**CLI:**
```bash
python scripts/build_contrastive_data.py \
    --mc13_plag_gpkg data/vector_mc13_relabeled/plagioclase.gpkg \
    --mc13_threshold_layer thresh_0.85 \
    --labeled_gpkg_dir /mnt/mrdr/categorized_mineral_units \
    --output_dir data/contrastive \
    --patch_size 7 \
    --max_per_polygon 200    # cap to avoid one huge polygon dominating
```

Skip polygons too small to extract a 7×7 (need `n_pixels >= patch_size**2`). Use the existing `rasterio.features.rasterize` pattern from `build_mtrdr_plag_patches.py`.

**Test:** `tests/test_build_contrastive_data.py` — small synthetic gpkg + raster fixture, assert correct count and patch values.

---

## Task 2 — Contrastive dataset

**File:** `data/contrastive_dataset.py` (new)

Class `ContrastiveTripletDataset(Dataset)`:
- `__init__(pos_patches, hard_neg_patches, soft_neg_patches, n_hard_per_batch=N_h, n_soft_per_batch=N_s)`
- `__getitem__(idx)` returns:
  - `anchor`: the idx-th positive patch
  - `positive`: a random *other* positive patch (could later add augmentation; v1: just sample another positive)
  - `hard_negatives`: `n_hard_per_batch` random hard-negative patches
  - `soft_negatives`: `n_soft_per_batch` random soft-negative patches
- `__len__`: `len(positives)`

**Test:** `tests/test_contrastive_dataset.py` — shapes match, no anchor == positive collisions when `len(pos) > 1`, fixed RNG produces reproducible negative samples.

---

## Task 3 — Contrastive encoder + InfoNCE loss

**File:** `models/contrastive_encoder.py` (new)

```python
class ContrastiveEncoder(nn.Module):
    """Wraps SpatialSpectralTransformer encoder + L2-normalised projection head.
    For contrastive training only; the projection head is discarded at eval time.
    """
    def __init__(self, n_bands=59, patch_size=7, embed_dim=128, n_heads=4, n_layers=6,
                 proj_dim=64):
        ...
        self.encoder = SpatialSpectralTransformer(...)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self._center_idx = ...

    def encode(self, x):
        """Return the center-token embedding (B, embed_dim). For downstream
        linear-probe / fine-tune."""
        out = self.encoder(x)
        return out[:, self._center_idx]

    def forward(self, x):
        """Return L2-normalised projected embedding (B, proj_dim). For contrastive loss."""
        z = self.proj(self.encode(x))
        return F.normalize(z, dim=-1)
```

**File:** `training/contrastive_train.py` (new)

InfoNCE loss with explicit hard-negative pool:

```python
def info_nce_loss(z_anchor, z_pos, z_hard_neg, z_soft_neg, tau=0.07,
                  hard_weight=2.0, soft_weight=1.0):
    """
    z_anchor:    (B, D)
    z_pos:       (B, D) — one positive per anchor
    z_hard_neg:  (B, N_h, D) — N_h hard negatives per anchor
    z_soft_neg:  (B, N_s, D) — N_s soft negatives per anchor

    Hard negatives get higher weight in the loss denominator via `hard_weight`.
    """
    # sim shapes:
    # anchor·pos: (B,)
    # anchor·hard: (B, N_h)
    # anchor·soft: (B, N_s)
    sim_pos = (z_anchor * z_pos).sum(-1) / tau
    sim_hard = torch.einsum('bd,bnd->bn', z_anchor, z_hard_neg) / tau
    sim_soft = torch.einsum('bd,bnd->bn', z_anchor, z_soft_neg) / tau

    # weighted log-sum-exp denominator
    log_denom = torch.logsumexp(
        torch.cat([
            sim_pos.unsqueeze(-1),
            sim_hard + torch.log(torch.tensor(hard_weight)),
            sim_soft + torch.log(torch.tensor(soft_weight)),
        ], dim=-1), dim=-1)
    return -(sim_pos - log_denom).mean()
```

**Test:** `tests/test_contrastive_loss.py`
- Identical anchors+positives, random orthogonal negatives → loss near 0
- Anchor = hard_neg → loss large
- Verify gradient flows through z_anchor, z_pos, z_hard_neg, z_soft_neg

---

## Task 4 — Training driver

**File:** `scripts/train_contrastive.py` (new)

CLI driver that:
- Loads the three patch pools via the dataset
- Loads `ContrastiveEncoder`, warm-starts encoder from `--pretrain_ckpt`
- Optimises with AdamW; `--encoder_lr_scale` (default 0.01) for slow encoder updates
- Logs to wandb under `contrastive_plag_v1`
- After every N epochs runs the linear-probe eval (Task 5) inline and logs `linear_probe_plag_AP`

**Test:** Smoke run on a 100-anchor subset; train 2 epochs; verify checkpoint saves and probe AP is finite.

---

## Task 5 — Linear-probe + full-fine-tune evaluation

**File:** `scripts/eval_contrastive.py` (new)

Two evaluation modes:

1. **Linear probe (fast)**: freeze encoder; train a single Linear(128 → 5) head on the standard val pixels for 5 epochs; report per-class AP. Quick signal for "did the encoder representation actually improve."

2. **Full fine-tune (slow)**: drop projection head, attach `SpatialSpectralClassifier` head, fine-tune end-to-end with the standard recipe (asl_loss, encoder_lr_scale 0.001, etc.) for N epochs. Compare to current champion `ft_plag_aware_real_only_best.pt`.

Linear probe is the fast feedback loop; full fine-tune is the bottom-line answer.

---

## Task 6 — HPC slurm

**File:** `scripts/hpc_contrastive.slurm` (new)

Same pattern as the mrrsu-aux slurm:
- `#SBATCH --mem=64gb`, `--gres=gpu:1`, `--time=18:00:00`
- Conda env activation
- Pre-build the three patch pools if missing (calls `scripts/build_contrastive_data.py`)
- Run contrastive training
- Run linear probe at the end
- Optionally: launch a follow-up sbatch dependency that does the full fine-tune

---

## Definition of Done

- Three patch pools harvested locally (`data/contrastive/*`).
- All unit tests pass: `pytest tests/test_contrastive_*.py -v`
- Local CPU smoke: 2-epoch contrastive training on a 100-anchor subset completes without errors and saves a checkpoint.
- Linear-probe runs on the resulting checkpoint and emits per-class AP.
- HPC slurm parses (`bash -n` clean).

## Out of Scope (Phase 1)

- Augmentation (random spectral noise, band drop). InfoNCE works without it for v1.
- Multi-class contrastive (extending to LCP, HCP simultaneously). v1 is plag-vs-olivine only.
- BYOL / SimSiam / SwAV alternatives. v1 is InfoNCE.
- Hyperparameter sweep over τ, hard_weight. v1 uses defaults (τ=0.07, hard_weight=2.0); a sweep can come later if the linear-probe gain is promising.

## Open questions captured

- After contrastive training, do we discard the projection head and reuse the encoder as-is for downstream classification fine-tune? **Yes — that's the standard SimCLR pattern and what the eval script assumes.**
- Patch size 7 same as classifier? **Yes** — keeps the encoder compatible with the existing fine-tune pipeline.
- Single positive per anchor, or augmented self-positives? **Single random other positive for v1**, simpler and reuses the existing patch pool.
