# Labels vs Predicted Comparison Figure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the global mineral color scheme and produce `reports/fig_labels_vs_predicted.png` — a 2×2 figure comparing human-annotated label polygons to the supervised classifier's vectroscopy output for T0435 and T0434, with color blending for mixed-mineral labels and a percentile-based unclassified threshold for the prototype classifier.

**Architecture:** Three changes in order: (1) update `fig_style.py` colors globally; (2) add `apply_min_similarity` helper + `--min_similarity` CLI flag to `classify_tile_prototype.py`; (3) create `scripts/plot_labels_vs_predicted.py` with pure helper functions (`parse_category`, `blend_mineral_color`) tested via TDD, then a figure renderer using the existing `plot_vector_mineral_maps.py` pattern (rasterio tile extent, geopandas polygon plotting, shared legend).

**Tech Stack:** matplotlib, geopandas, rasterio, numpy. All data from existing `data/vector/` GeoPackages and `/mnt/mrdr/categorized_mineral_units/` label GeoPackages — no model inference required.

---

## Chunk 1: Color update + min_similarity

## Task 1: Update MINERAL_COLORS in fig_style.py

**Files:**
- Modify: `scripts/fig_style.py`

- [ ] **Step 1: Apply color changes**

Open `scripts/fig_style.py` and replace `MINERAL_COLORS`:

```python
MINERAL_COLORS = {
    'olivine':     '#e53935',   # red
    'lcp':         '#00bcd4',   # cyan
    'hcp':         '#e91e63',   # magenta
    'plagioclase': '#ffeb3b',   # yellow
    'other':       '#9e9e9e',   # gray (unchanged)
}
```

- [ ] **Step 2: Run existing tests to verify no regressions**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -m pytest tests/test_build_prototypes.py tests/test_classify_tile_prototype.py tests/test_compute_global_thresholds.py tests/test_vectorize_tile_minerals.py tests/test_classify_tile_supervised_save_probs.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/fig_style.py
git commit -m "feat: update mineral colors (olivine=red, lcp=cyan, hcp=magenta, plagioclase=yellow)"
```

---

## Task 2: Add apply_min_similarity + --min_similarity to classify_tile_prototype.py

**Files:**
- Modify: `scripts/classify_tile_prototype.py`
- Modify: `tests/test_classify_tile_prototype.py`

### Key context

`classify_tile_prototype.py` currently has three pure helper functions: `cosine_similarity_classify`, `apply_valid_mask`, `load_prototype_npz`. The flow in `run_proto()` is:

```python
embs = embed_tile(tile, encoder, device, args.batch_size)   # (H*W, 128)
sims_flat = cosine_similarity_classify(embs, protos)         # (H*W, 5)
sims_hw5  = apply_valid_mask(sims_flat, valid_mask)          # (H, W, 5)
```

We add `apply_min_similarity` as a fourth pure helper called after `apply_valid_mask`. It zeros out all 5 class channels for pixels whose max similarity across classes falls below a threshold. When threshold is `None`, it computes the 10th percentile of max-similarity across valid pixels.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classify_tile_prototype.py`:

```python
def test_apply_min_similarity_fixed_threshold():
    """Pixels with max similarity below threshold are zeroed; above are unchanged."""
    from scripts.classify_tile_prototype import apply_min_similarity
    H, W = 2, 3
    # Row 0: max sim = 0.1 (below 0.3); row 1: max sim = 0.9 (above 0.3)
    sims = np.zeros((H, W, 5), dtype=np.float32)
    sims[0, :, :] = 0.1    # all channels = 0.1 → max = 0.1
    sims[1, :, 0] = 0.9    # first channel = 0.9 → max = 0.9
    valid_mask = np.ones((H, W), dtype=bool)
    result = apply_min_similarity(sims, valid_mask, min_similarity=0.3)
    # Row 0 all zeroed
    np.testing.assert_array_equal(result[0], np.zeros((W, 5), dtype=np.float32))
    # Row 1 unchanged
    assert result[1, 0, 0] == pytest.approx(0.9)


def test_apply_min_similarity_auto_percentile():
    """When min_similarity is None, uses 10th percentile of valid-pixel max sims."""
    from scripts.classify_tile_prototype import apply_min_similarity
    H, W = 1, 10
    sims = np.zeros((H, W, 5), dtype=np.float32)
    # Valid pixels with max sims: 0.05, 0.10, 0.15, ..., 0.50 (10 values)
    for i in range(10):
        sims[0, i, 0] = 0.05 * (i + 1)
    valid_mask = np.ones((H, W), dtype=bool)
    result = apply_min_similarity(sims, valid_mask, min_similarity=None)
    # 10th percentile of [0.05, 0.10, ..., 0.50] ≈ 0.095 → pixel 0 (max=0.05) zeroed
    assert result[0, 0].sum() == 0.0    # max=0.05 below ~0.095 threshold
    assert result[0, 1, 0] > 0.0       # max=0.10 at or above threshold


def test_apply_min_similarity_ignores_invalid_pixels():
    """Invalid pixels (already zeroed by apply_valid_mask) don't skew the percentile."""
    from scripts.classify_tile_prototype import apply_min_similarity
    H, W = 2, 2
    sims = np.zeros((H, W, 5), dtype=np.float32)
    sims[0, 0, 0] = 0.9   # valid, high
    sims[0, 1, 0] = 0.8   # valid, high
    sims[1, 0, 0] = 0.0   # invalid (masked), should not drag down percentile
    sims[1, 1, 0] = 0.7   # valid, high
    valid_mask = np.array([[True, True], [False, True]])
    # All valid pixels have max >= 0.7; 10th pctile ~= 0.72 → none zeroed
    result = apply_min_similarity(sims, valid_mask, min_similarity=None)
    assert result[0, 0, 0] == pytest.approx(0.9)
    assert result[0, 1, 0] == pytest.approx(0.8)
    assert result[1, 1, 0] == pytest.approx(0.7)
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_classify_tile_prototype.py::test_apply_min_similarity_fixed_threshold tests/test_classify_tile_prototype.py::test_apply_min_similarity_auto_percentile tests/test_classify_tile_prototype.py::test_apply_min_similarity_ignores_invalid_pixels -v 2>&1 | head -20
```

Expected: ImportError — `apply_min_similarity` does not exist yet.

- [ ] **Step 3: Implement apply_min_similarity in classify_tile_prototype.py**

Add after the `apply_valid_mask` function (before `load_prototype_npz`):

```python
def apply_min_similarity(
    sims_hw5: np.ndarray,
    valid_mask: np.ndarray,
    min_similarity: float | None,
) -> np.ndarray:
    """Zero out pixels whose max class similarity falls below min_similarity.

    Args:
        sims_hw5: (H, W, 5) float32, output of apply_valid_mask
        valid_mask: (H, W) bool — used to exclude invalid pixels from percentile
        min_similarity: explicit threshold in [0, 1], or None to auto-compute
            as the 10th percentile of max-similarity across valid pixels.

    Returns:
        (H, W, 5) float32 — low-confidence pixels zeroed on all channels.
    """
    result = sims_hw5.copy()
    if min_similarity is None:
        valid_max = sims_hw5[valid_mask].max(axis=-1)   # (N_valid,)
        threshold = float(np.percentile(valid_max, 10))
    else:
        threshold = min_similarity
    low_conf = result.max(axis=-1) < threshold           # (H, W) bool
    result[low_conf] = 0.0
    return result
```

- [ ] **Step 4: Add --min_similarity CLI argument and wire it into run_proto**

In `main()`, add argument after `--batch_size`:

```python
parser.add_argument(
    '--min_similarity', type=float, default=None,
    help='Minimum max cosine similarity to classify a pixel. '
         'Default: 10th percentile of tile max-similarity distribution. '
         'Pass 0.0 to disable.',
)
```

In the `run_proto()` inner function, add one line after `apply_valid_mask`:

```python
def run_proto(proto_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    protos, _, ckpt_path = load_prototype_npz(proto_path)
    tag = os.path.splitext(os.path.basename(proto_path))[0]
    print(f'Loading encoder for {tag}: {ckpt_path}')
    encoder, _ = load_encoder(ckpt_path, device)
    print(f'Embedding tile ({tag})...')
    embs = embed_tile(tile, encoder, device, args.batch_size)
    sims_flat = cosine_similarity_classify(embs, protos)
    sims_hw5  = apply_valid_mask(sims_flat, valid_mask)
    sims_hw5  = apply_min_similarity(sims_hw5, valid_mask, args.min_similarity)
    return sims_hw5, embs, tag
```

- [ ] **Step 5: Run tests**

```bash
conda run -n crism python -m pytest tests/test_classify_tile_prototype.py -v 2>&1 | tail -20
```

Expected: all tests pass (previously passing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/classify_tile_prototype.py tests/test_classify_tile_prototype.py
git commit -m "feat: add apply_min_similarity with 10th-percentile default to classify_tile_prototype"
```

---

## Chunk 2: plot_labels_vs_predicted.py

## Task 3: Tests for parse_category and blend_mineral_color

**Files:**
- Create: `tests/test_plot_labels_vs_predicted.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plot_labels_vs_predicted.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_parse_category_single():
    """`"lcp (High)"` → (['lcp'], 'High')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('lcp (High)')
    assert minerals == ['lcp']
    assert tier == 'High'


def test_parse_category_mixed():
    """`"hcp + olivine (Moderate)"` → (['hcp', 'olivine'], 'Moderate')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('hcp + olivine (Moderate)')
    assert minerals == ['hcp', 'olivine']
    assert tier == 'Moderate'


def test_parse_category_three_way():
    """`"alteration + hcp + olivine (Low)"` → (['other', 'hcp', 'olivine'], 'Low')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('alteration + hcp + olivine (Low)')
    assert minerals == ['other', 'hcp', 'olivine']
    assert tier == 'Low'


def test_parse_category_type_olivine():
    """`"Type 2 olivine (High)"` → (['olivine'], 'High')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('Type 2 olivine (High)')
    assert minerals == ['olivine']
    assert tier == 'High'


def test_parse_category_other_uppercase():
    """`"Other (High)"` (capital O) → (['other'], 'High')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('Other (High)')
    assert minerals == ['other']
    assert tier == 'High'


def test_parse_category_red_slope():
    """`"red slope (Low)"` is treated as a single other token, not split further."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('red slope (Low)')
    assert minerals == ['other']
    assert tier == 'Low'


def test_blend_single():
    """Single mineral returns its exact MINERAL_COLORS RGB."""
    import matplotlib.colors as mc
    from scripts.plot_labels_vs_predicted import blend_mineral_color
    from scripts.fig_style import MINERAL_COLORS
    result = blend_mineral_color(['olivine'])
    expected = mc.to_rgb(MINERAL_COLORS['olivine'])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_blend_two():
    """Two minerals return the component-wise average of their RGB values."""
    import matplotlib.colors as mc
    from scripts.plot_labels_vs_predicted import blend_mineral_color
    from scripts.fig_style import MINERAL_COLORS
    r1, g1, b1 = mc.to_rgb(MINERAL_COLORS['hcp'])
    r2, g2, b2 = mc.to_rgb(MINERAL_COLORS['olivine'])
    result = blend_mineral_color(['hcp', 'olivine'])
    assert result[0] == pytest.approx((r1 + r2) / 2, abs=1e-6)
    assert result[1] == pytest.approx((g1 + g2) / 2, abs=1e-6)
    assert result[2] == pytest.approx((b1 + b2) / 2, abs=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_plot_labels_vs_predicted.py -v 2>&1 | head -20
```

Expected: ImportError — `plot_labels_vs_predicted` does not exist yet.

---

## Task 4: Implement plot_labels_vs_predicted.py

**Files:**
- Create: `scripts/plot_labels_vs_predicted.py`

- [ ] **Step 1: Create the script with pure helpers first**

```python
"""
Figure: Label polygons vs. supervised vectroscopy output for T0435 and T0434.

2×2 grid: rows = tiles (T0435, T0434), columns = Labels | Predicted.
Labels come from /mnt/mrdr/categorized_mineral_units/.
Predicted comes from data/vector/ vectroscopy GeoPackages.
Colors: olivine=red, lcp=cyan, hcp=magenta, plagioclase=yellow, other=gray.
Mixed-mineral labels use an RGB blend of constituent classes.

Output: reports/fig_labels_vs_predicted.png

Usage:
    conda run -n crism python scripts/plot_labels_vs_predicted.py
"""
import os
import re
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
import geopandas as gpd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, DPI

PROJ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = '/mnt/mrdr/categorized_mineral_units'
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')
REPORTS   = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS, exist_ok=True)

TILES = [
    {
        'label': 'T0435',
        'label_gpkg': os.path.join(LABEL_DIR, 'T0435.gpkg'),
        'label_layer': 'T0435',
        'pred_gpkg': os.path.join(VECTOR_DIR, 't0435_mrral_40s323_0327_4_mineral_map.gpkg'),
        'img': '/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img',
    },
    {
        'label': 'T0434',
        'label_gpkg': os.path.join(LABEL_DIR, 'T0434.gpkg'),
        'label_layer': 'T0434',
        'pred_gpkg': os.path.join(VECTOR_DIR, 't0434_mrral_40s318_0327_4_mineral_map.gpkg'),
        'img': '/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img',
    },
]

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
TIER_ALPHA = {1: 0.40, 2: 0.65, 3: 0.90}
LABEL_TIER_ALPHA = {'Low': 0.40, 'Moderate': 0.65, 'High': 0.90}

# Canonical token mapping (lowercase keys)
_CANONICAL = {
    'olivine': 'olivine',
    'type 1 olivine': 'olivine',
    'type 2 olivine': 'olivine',
    'lcp': 'lcp',
    'hcp': 'hcp',
    'plagioclase': 'plagioclase',
}
_OTHER_TOKENS = {'alteration', 'red slope', 'other', 'bland', 'denom', 'uncertain'}


# ── pure helpers ──────────────────────────────────────────────────────────────

def parse_category(category: str) -> Tuple[List[str], str]:
    """Parse a Category string into (canonical_mineral_names, tier).

    Examples:
        "lcp (High)"               → (['lcp'], 'High')
        "hcp + olivine (Moderate)" → (['hcp', 'olivine'], 'Moderate')
        "red slope (Low)"          → (['other'], 'Low')
        "Other (High)"             → (['other'], 'High')
    """
    tier_match = re.search(r'\((\w+)\)\s*$', category)
    tier = tier_match.group(1) if tier_match else 'Low'
    mineral_str = category[:tier_match.start()].strip() if tier_match else category.strip()

    tokens = [t.strip() for t in mineral_str.split(' + ')]
    minerals = []
    for token in tokens:
        t_lower = token.lower()
        if t_lower in _CANONICAL:
            minerals.append(_CANONICAL[t_lower])
        elif t_lower in _OTHER_TOKENS:
            minerals.append('other')
        else:
            minerals.append('other')   # fail-safe for unknown tokens

    return minerals, tier


def blend_mineral_color(mineral_names: List[str]) -> Tuple[float, float, float]:
    """Return RGB (0-1) blend of the given minerals' MINERAL_COLORS.

    Single mineral returns its exact color. Multiple minerals return the
    component-wise average of their RGB values.
    """
    rgbs = [mc.to_rgb(MINERAL_COLORS[m]) for m in mineral_names]
    r = sum(c[0] for c in rgbs) / len(rgbs)
    g = sum(c[1] for c in rgbs) / len(rgbs)
    b = sum(c[2] for c in rgbs) / len(rgbs)
    return (r, g, b)


# ── panel renderers ───────────────────────────────────────────────────────────

def setup_ax(ax, img_path: str) -> None:
    """Set axes extent from tile raster bounds; grey background; no ticks/spines."""
    with rasterio.open(img_path) as src:
        b = src.bounds
    ax.set_facecolor('#e0e0e0')
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_label_panel(ax, tile: dict) -> None:
    """Plot label polygons coloured by mineral class (blended for mixed labels)."""
    setup_ax(ax, tile['img'])
    try:
        gdf = gpd.read_file(tile['label_gpkg'], layer=tile['label_layer'])
    except Exception as e:
        ax.text(0.5, 0.5, f'no data\n{e}', transform=ax.transAxes,
                ha='center', va='center', color='#999', fontsize=8)
        return

    for _, row in gdf.iterrows():
        cat = row.get('Category', '')
        if not cat:
            continue
        minerals, tier = parse_category(str(cat))
        color = blend_mineral_color(minerals)
        alpha = LABEL_TIER_ALPHA.get(tier, 0.40)
        gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf.crs).plot(
            ax=ax, color=[color], edgecolor='none', alpha=alpha,
        )

    ax.text(0.02, 0.03, f'{len(gdf):,}', transform=ax.transAxes,
            fontsize=7, color='#333', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


def plot_predicted_panel(ax, tile: dict) -> None:
    """Plot predicted mineral polygons from vectroscopy GeoPackage."""
    setup_ax(ax, tile['img'])
    total = 0
    for mineral in MINERALS:   # render bottom-to-top: olivine first, other last
        try:
            gdf = gpd.read_file(tile['pred_gpkg'], layer=mineral)
        except Exception:
            continue
        if gdf.empty:
            continue
        for tier in [1, 2, 3]:
            subset = gdf[gdf['confidence'] == tier]
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                color=MINERAL_COLORS[mineral],
                edgecolor='none',
                alpha=TIER_ALPHA[tier],
            )
        total += len(gdf)

    ax.text(0.02, 0.03, f'{total:,}', transform=ax.transAxes,
            fontsize=7, color='#333', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


# ── figure assembly ───────────────────────────────────────────────────────────

def main():
    n_rows = len(TILES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows),
                             constrained_layout=True)

    for row, tile in enumerate(TILES):
        axes[row, 0].set_title(f"{tile['label']} — Labels", fontsize=11,
                               fontweight='bold', pad=4)
        axes[row, 1].set_title(f"{tile['label']} — Predicted", fontsize=11,
                               fontweight='bold', pad=4)
        plot_label_panel(axes[row, 0], tile)
        plot_predicted_panel(axes[row, 1], tile)

    # Mineral legend
    mineral_handles = [
        mpatches.Patch(facecolor=MINERAL_COLORS[m], label=m.capitalize(), alpha=0.85)
        for m in MINERALS
    ]
    # Confidence tier legend
    tier_handles = [
        mpatches.Patch(facecolor='#888888', alpha=TIER_ALPHA[1], label='Low / Tier 1'),
        mpatches.Patch(facecolor='#888888', alpha=TIER_ALPHA[2], label='Moderate / Tier 2'),
        mpatches.Patch(facecolor='#888888', alpha=TIER_ALPHA[3], label='High / Tier 3'),
    ]
    fig.legend(handles=mineral_handles + tier_handles,
               loc='lower center', ncol=8, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Label polygons vs. supervised vectroscopy — T0435 & T0434',
                 fontsize=12, y=1.01)

    out = os.path.join(REPORTS, 'fig_labels_vs_predicted.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the unit tests**

```bash
conda run -n crism python -m pytest tests/test_plot_labels_vs_predicted.py -v 2>&1 | tail -15
```

Expected: all 8 tests PASS.

- [ ] **Step 3: Smoke-test the figure**

```bash
conda run -n crism python scripts/plot_labels_vs_predicted.py 2>&1
```

Expected: `Saved → reports/fig_labels_vs_predicted.png`

Verify file exists and is non-empty:
```bash
ls -lh reports/fig_labels_vs_predicted.png
```

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_labels_vs_predicted.py tests/test_plot_labels_vs_predicted.py
git commit -m "feat: add plot_labels_vs_predicted.py with mixed-color label comparison"
```

- [ ] **Step 5: Add plot_labels_vs_predicted.py to README**

In `README.md`, add under the Vectroscopy Pipeline section:

```bash
# Compare label polygons to predicted mineral maps
conda run -n crism python scripts/plot_labels_vs_predicted.py
# → reports/fig_labels_vs_predicted.png
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add plot_labels_vs_predicted.py to README"
```
