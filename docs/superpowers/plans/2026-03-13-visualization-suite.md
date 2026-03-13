# Visualization Suite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create five standalone publication-ready figure scripts (`fig_style.py` + 5 plot scripts) that produce 300 DPI PNGs in `reports/`.

**Architecture:** A shared `scripts/fig_style.py` module holds constants and helpers. Five independent plot scripts each import from it, build a figure with hardcoded or parquet-loaded data, and save to `<project_root>/reports/`. Tests live in `tests/test_figures.py` as smoke tests that call each script's `main()` with mocked/minimal data and assert the output PNG was created.

**Tech Stack:** matplotlib, numpy, pandas, spectral (ENVI header parsing), pytest, unittest.mock

---

## File Structure

| File | Role |
|------|------|
| `scripts/fig_style.py` | Constants (`MINERAL_COLORS`, `LABEL_COLS`, figsize, DPI) + `apply_style()` + `despine(ax)` |
| `scripts/plot_model_progression.py` | Fig 1 — hardcoded data horizontal bar chart |
| `scripts/plot_per_class_heatmap.py` | Fig 2 — hardcoded data AP heatmap with mAP sidebar |
| `scripts/plot_class_spectra_v2.py` | Fig 3 — loads `mrral_pixels.parquet`, 2×3 spectral grid |
| `scripts/plot_dataset_stats.py` | Fig 4 — loads `pixels.parquet`, class prevalence + confidence tier |
| `scripts/plot_ablation_waterfall.py` | Fig 5 — hardcoded data waterfall bar chart |
| `tests/test_figures.py` | Smoke tests for all 6 scripts |

---

## Chunk 1: Shared Style, Fig 1 (Model Progression), Fig 2 (Heatmap)

### Task 1: fig_style.py

**Files:**
- Create: `scripts/fig_style.py`
- Test: `tests/test_figures.py` (new file, `TestFigStyle` class only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_figures.py`:

```python
"""Smoke tests for visualization figure scripts."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFigStyle:
    def test_mineral_colors_keys(self):
        from scripts.fig_style import MINERAL_COLORS, LABEL_COLS
        assert set(MINERAL_COLORS.keys()) == {'olivine', 'lcp', 'hcp', 'plagioclase', 'other'}

    def test_label_cols(self):
        from scripts.fig_style import LABEL_COLS
        assert LABEL_COLS == ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

    def test_dpi(self):
        from scripts.fig_style import DPI
        assert DPI == 300

    def test_apply_style_sets_font_size(self):
        import matplotlib
        import matplotlib.pyplot as plt
        from scripts.fig_style import apply_style
        original = dict(plt.rcParams)
        apply_style()
        assert plt.rcParams['font.size'] == 11
        matplotlib.rcdefaults()  # reset to avoid state leak into later tests

    def test_despine_hides_top_right(self):
        import matplotlib.pyplot as plt
        from scripts.fig_style import despine
        fig, ax = plt.subplots()
        despine(ax)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        plt.close(fig)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_figures.py::TestFigStyle -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.fig_style'`

- [ ] **Step 3: Implement `scripts/fig_style.py`**

```python
"""Shared style constants and helpers for visualization scripts."""
import matplotlib.pyplot as plt

FIGSIZE_SINGLE = (7, 4.5)
FIGSIZE_WIDE   = (10, 4.5)
FIGSIZE_GRID   = (10, 7)

DPI = 300

MINERAL_COLORS = {
    'olivine':     '#4caf50',
    'lcp':         '#2196f3',
    'hcp':         '#ff9800',
    'plagioclase': '#9c27b0',
    'other':       '#9e9e9e',
}

LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']


def apply_style():
    """Set global matplotlib rcParams for consistent figure style."""
    plt.rcParams.update({
        'font.size':       11,
        'axes.titlesize':  12,
        'axes.labelsize':  11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid':       True,
        'grid.alpha':      0.3,
    })


def despine(ax):
    """Remove top and right spines from axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n crism pytest tests/test_figures.py::TestFigStyle -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/fig_style.py tests/test_figures.py
git commit -m "feat: add fig_style shared module and test skeleton"
```

---

### Task 2: Figure 1 — Model Progression

**Files:**
- Create: `scripts/plot_model_progression.py`
- Modify: `tests/test_figures.py` (add `TestModelProgression` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_figures.py`:

```python
class TestModelProgression:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_model_progression as m
        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))
        m.main()
        assert (tmp_path / 'fig_model_progression.png').exists()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n crism pytest tests/test_figures.py::TestModelProgression -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.plot_model_progression'`

**Note on imports:** `scripts/` has no `__init__.py` and doesn't need one — pytest adds the project root to `sys.path` via rootdir discovery, so `import scripts.plot_model_progression as m` works in tests. This is consistent with existing tests like `test_predict_tile.py`.

- [ ] **Step 3: Implement `scripts/plot_model_progression.py`**

```python
"""
Figure 1: Model Progression horizontal bar chart.

Output: reports/fig_model_progression.png
"""
import os, sys
import matplotlib.pyplot as plt
import numpy as np

# Insert project root so 'scripts.fig_style' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import DPI, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hardcoded sweep results (stable; not loaded from files)
# Format: (run_name, val_mAP, group)
RUNS = [
    ('scnn_base',            0.554, 'v3'),
    ('scnn_focal',           0.615, 'v3'),
    ('svit_base',            0.522, 'v3'),
    ('svit_mae',             0.507, 'v3'),
    ('svit_base_v5',         0.561, 'v5'),
    ('svit_mae_v5',          0.520, 'v5'),
    ('scnn_asl_v6',          0.333, 'v6'),
    ('svit_asl_v6',          0.534, 'v6'),
    ('svit_asl_diffr_v6',    0.523, 'v6'),
    ('shybrid_asl_v6',       0.608, 'v6'),
    ('shybrid_asl_diffr_v6', 0.614, 'v6'),
]

GROUP_COLORS = {'v3': '#e3f2fd', 'v5': '#e8f5e9', 'v6': '#fff3e0'}
GROUP_LABELS = {'v3': 'v3\n(6-class)', 'v5': 'v5\n(5-class)', 'v6': 'v6\n(5-class)'}

# Annotation text placed just above the boundary after each group
GROUP_TRANSITION_TEXT = {
    'v3': '+mrral spectral input (ViT)',
    'v5': '+ASL loss',
}


def main():
    apply_style()

    names  = [r[0] for r in RUNS]
    maps   = [r[1] for r in RUNS]
    groups = [r[2] for r in RUNS]
    n = len(RUNS)

    # y_pos: 0=bottom, n-1=top. We want table-order top-to-bottom,
    # so table row 0 (scnn_base) = y = n-1 (top), table row n-1 = y = 0 (bottom).
    y_pos = [n - 1 - i for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Group shading (axhspan) ---
    for g in ['v3', 'v5', 'v6']:
        indices = [i for i, r in enumerate(RUNS) if r[2] == g]
        ymin = min(y_pos[i] for i in indices) - 0.4
        ymax = max(y_pos[i] for i in indices) + 0.4
        ax.axhspan(ymin, ymax, color=GROUP_COLORS[g], alpha=0.5, zorder=0)

    # --- Horizontal bars ---
    ax.barh(y_pos, maps, color='#1976d2', edgecolor='white', height=0.6, zorder=2)

    # --- Y-axis tick labels (run names) ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)

    # --- Group labels on far left via ax.text ---
    for g in ['v3', 'v5', 'v6']:
        indices = [i for i, r in enumerate(RUNS) if r[2] == g]
        mid = (min(y_pos[i] for i in indices) + max(y_pos[i] for i in indices)) / 2
        ax.text(-0.03, mid, GROUP_LABELS[g], fontsize=8, color='#444444',
                va='center', ha='right', transform=ax.get_yaxis_transform())

    # --- Between-group boundaries and intervention labels ---
    for i in range(len(RUNS) - 1):
        if RUNS[i][2] != RUNS[i + 1][2]:
            boundary_y = (y_pos[i] + y_pos[i + 1]) / 2
            ax.axhline(y=boundary_y, linestyle='--', color='#aaaaaa', linewidth=0.8, zorder=1)
            from_group = RUNS[i][2]
            if from_group in GROUP_TRANSITION_TEXT:
                ax.text(0.02, boundary_y + 0.15, GROUP_TRANSITION_TEXT[from_group],
                        fontsize=8, color='#555555', va='bottom')

    # --- Axes formatting ---
    ax.set_xlim(0, 0.72)
    ax.set_xlabel('val mAP')
    ax.set_title('Model Progression by Sweep Family')
    despine(ax)

    # --- Footnote ---
    fig.text(
        0.01, 0.01,
        '† v3 mAP computed over 6 classes (olivine_t1/t2 separate); '
        'v5/v6 over 5 classes (olivine collapsed). Values not directly comparable.',
        fontsize=7, color='#666666', va='bottom',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(REPORTS_DIR, 'fig_model_progression.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n crism pytest tests/test_figures.py::TestModelProgression -v
```

Expected: 1 passed.

- [ ] **Step 5: Smoke-run the script and visually verify output**

```bash
conda run -n crism python scripts/plot_model_progression.py
```

Expected: `Saved .../reports/fig_model_progression.png`. Open the PNG and confirm 11 bars, group shading, between-group dashed lines with labels, footnote.

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_model_progression.py tests/test_figures.py
git commit -m "feat: add Fig 1 model progression bar chart"
```

---

### Task 3: Figure 2 — Per-Class AP Heatmap

**Files:**
- Create: `scripts/plot_per_class_heatmap.py`
- Modify: `tests/test_figures.py` (add `TestHeatmap` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_figures.py`:

```python
class TestHeatmap:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_per_class_heatmap as m
        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))
        m.main()
        assert (tmp_path / 'fig_per_class_heatmap.png').exists()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n crism pytest tests/test_figures.py::TestHeatmap -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.plot_per_class_heatmap'`

- [ ] **Step 3: Implement `scripts/plot_per_class_heatmap.py`**

```python
"""
Figure 2: Per-class AP heatmap for v6 models.

Output: reports/fig_per_class_heatmap.png
"""
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Insert project root so 'scripts.fig_style' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import LABEL_COLS, DPI, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hardcoded per-class AP values (from v6 sweep logs), sorted by mAP descending.
# Columns: name, olivine, lcp, hcp, plagioclase, other, mAP
MODELS = [
    ('shybrid_asl_diffr_v6', 0.97, 0.85, 0.51, 0.26, 0.48, 0.614),
    ('shybrid_asl_v6',       0.97, 0.86, 0.51, 0.22, 0.48, 0.608),
    ('svit_asl_v6',          0.90, 0.90, 0.55, 0.22, 0.48, 0.534),
    ('svit_asl_diffr_v6',    0.97, 0.65, 0.38, 0.29, 0.32, 0.523),
    ('scnn_asl_v6',          0.90, 0.10, 0.22, 0.20, 0.24, 0.333),
]


def main():
    apply_style()

    model_names = [m[0] for m in MODELS]
    # AP matrix: rows=models, cols=classes (olivine, lcp, hcp, plagioclase, other)
    ap_matrix = np.array([[m[1], m[2], m[3], m[4], m[5]] for m in MODELS])
    map_vals   = [m[6] for m in MODELS]

    fig = plt.figure(figsize=(9, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[5, 0.6], figure=fig, wspace=0.08)
    ax   = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    # --- Heatmap ---
    im = ax.imshow(ap_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

    # Cell text
    for i in range(5):
        for j in range(5):
            val = ap_matrix[i, j]
            color = 'white' if val >= 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color)

    ax.set_xticks(range(5))
    ax.set_xticklabels(LABEL_COLS, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title('Per-Class Average Precision (v6 models, ASL loss)')
    # No colorbar — the mAP sidebar and cell text provide sufficient value indication.

    # --- Right mAP column ---
    # Copy imshow's y-axis limits (inverted: top row = low y value in display)
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_xlim(0, 1)
    ax_r.axis('off')
    ax_r.set_title('mAP', fontsize=10)
    for i, mval in enumerate(map_vals):
        ax_r.text(0.1, i, f'{mval:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_per_class_heatmap.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n crism pytest tests/test_figures.py::TestHeatmap -v
```

Expected: 1 passed.

- [ ] **Step 5: Smoke-run and visually verify**

```bash
conda run -n crism python scripts/plot_per_class_heatmap.py
```

Expected: `Saved .../reports/fig_per_class_heatmap.png`. Open PNG and confirm 5×5 heatmap with cell values, YlOrRd colormap, mAP sidebar column.

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_per_class_heatmap.py tests/test_figures.py
git commit -m "feat: add Fig 2 per-class AP heatmap"
```

---

## Chunk 2: Fig 3 (Spectral Profiles), Fig 4 (Dataset Stats), Fig 5 (Ablation Waterfall)

### Task 4: Figure 3 — Class Spectral Profiles

**Files:**
- Create: `scripts/plot_class_spectra_v2.py`
- Modify: `tests/test_figures.py` (add `TestClassSpectra` class, `_make_mrral_df` fixture helper)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_figures.py`:

```python
def _make_mrral_df():
    """Minimal mrral_pixels.parquet fixture — 50 rows, 59 bands."""
    rng = np.random.default_rng(42)
    n = 50
    data = {f'm{i}': rng.random(n).astype('float32') for i in range(59)}
    for cls in ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']:
        # ~30% positive per class
        data[f'label_{cls}'] = np.where(rng.random(n) > 0.7, 1.0, 0.0).astype('float32')
    data['split'] = ['train'] * 30 + ['val'] * 10 + ['test'] * 10
    data['confidence_tier'] = ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 10
    data['tile_id']    = ['t001'] * n
    data['polygon_id'] = list(range(n))
    data['pixel_row']  = list(range(n))
    data['pixel_col']  = list(range(n))
    return pd.DataFrame(data)


class TestClassSpectra:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_class_spectra_v2 as m
        from unittest.mock import patch

        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))

        # Write fixture parquet to tmp_path
        df = _make_mrral_df()
        parquet_path = tmp_path / 'mrral_pixels.parquet'
        df.to_parquet(parquet_path)

        cfg = {'output_dir': str(tmp_path), 'data_root': str(tmp_path)}
        fixed_wavelengths = np.linspace(410, 2457, 59)

        with patch('scripts.plot_class_spectra_v2.load_config', return_value=cfg), \
             patch('scripts.plot_class_spectra_v2.get_wavelengths',
                   return_value=fixed_wavelengths):
            m.main()

        assert (tmp_path / 'fig_class_spectra_v2.png').exists()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n crism pytest tests/test_figures.py::TestClassSpectra -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.plot_class_spectra_v2'`

- [ ] **Step 3: Implement `scripts/plot_class_spectra_v2.py`**

```python
"""
Figure 3: Class spectral profiles — mean ± 1σ reflectance per mineral class.

Output: reports/fig_class_spectra_v2.png
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insert project root so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import FIGSIZE_GRID, DPI, MINERAL_COLORS, LABEL_COLS, apply_style, despine
from config_loader import load_config

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

BAND_COLS = [f'm{i}' for i in range(59)]


def get_wavelengths(data_root: str) -> np.ndarray:
    """Read wavelength array from first mrral .hdr found under data_root.

    Falls back to np.linspace(410, 2457, 59) if no .hdr found or parse fails.
    """
    hdrs = glob.glob(os.path.join(data_root, '**', '*mrral*.hdr'), recursive=True)
    if hdrs:
        try:
            import spectral.io.envi as envi
            hdr = envi.read_envi_header(hdrs[0])
            return np.array(hdr['wavelength'], dtype=float)[:59]
        except Exception:
            pass
    return np.linspace(410, 2457, 59)


def main():
    apply_style()
    cfg = load_config()
    parquet_path = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df = pd.read_parquet(parquet_path)
    wavelengths = get_wavelengths(cfg['data_root'])

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID, sharey=True)
    axes_flat = axes.flatten()  # indices 0–5, row-major

    # Panels 0–4: one class each
    for idx, cls in enumerate(LABEL_COLS):
        ax = axes_flat[idx]
        mask   = df[f'label_{cls}'] > 0.4
        subset = df.loc[mask, BAND_COLS].values.astype('float32')
        n      = len(subset)
        ax.set_title(f'{cls} (n={n:,})', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel('Reflectance', fontsize=9)
        despine(ax)
        if n == 0:
            continue
        color  = MINERAL_COLORS[cls]
        mean   = subset.mean(axis=0)
        std    = subset.std(axis=0)
        ax.plot(wavelengths, mean, color=color, linewidth=1.5)
        ax.fill_between(wavelengths, mean - std, mean + std, color=color, alpha=0.25)

    # Panel 5: all-class overlay
    ax5 = axes_flat[5]
    for cls in LABEL_COLS:
        mask   = df[f'label_{cls}'] > 0.4
        subset = df.loc[mask, BAND_COLS].values.astype('float32')
        mean   = subset.mean(axis=0)
        ax5.plot(wavelengths, mean, color=MINERAL_COLORS[cls], label=cls, linewidth=1.5)
    ax5.set_title('All classes', fontsize=10)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_xlabel('Wavelength (nm)', fontsize=9)
    despine(ax5)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_class_spectra_v2.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n crism pytest tests/test_figures.py::TestClassSpectra -v
```

Expected: 1 passed.

- [ ] **Step 5: Smoke-run against real data (if available)**

```bash
conda run -n crism python scripts/plot_class_spectra_v2.py
```

Expected: `Saved .../reports/fig_class_spectra_v2.png`. Verify 6-panel grid, labeled axes, shaded bands.

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_class_spectra_v2.py tests/test_figures.py
git commit -m "feat: add Fig 3 class spectral profiles"
```

---

### Task 5: Figure 4 — Dataset Statistics

**Files:**
- Create: `scripts/plot_dataset_stats.py`
- Modify: `tests/test_figures.py` (add `TestDatasetStats` class, `_make_pixels_df` fixture helper)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_figures.py`:

```python
def _make_pixels_df():
    """Minimal pixels.parquet fixture — 60 rows with 6-class labels before collapse."""
    rng = np.random.default_rng(42)
    n = 60
    data = {}
    for cls in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[f'label_{cls}'] = np.where(rng.random(n) > 0.6, 1.0, 0.0).astype('float32')
    data['split']           = ['train'] * 40 + ['val'] * 10 + ['test'] * 10
    data['confidence_tier'] = ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 20
    return pd.DataFrame(data)


class TestDatasetStats:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_dataset_stats as m
        from unittest.mock import patch

        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))

        df = _make_pixels_df()
        parquet_path = tmp_path / 'pixels.parquet'
        df.to_parquet(parquet_path)

        cfg = {'output_dir': str(tmp_path)}
        with patch('scripts.plot_dataset_stats.load_config', return_value=cfg):
            m.main()

        assert (tmp_path / 'fig_dataset_stats.png').exists()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n crism pytest tests/test_figures.py::TestDatasetStats -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.plot_dataset_stats'`

- [ ] **Step 3: Implement `scripts/plot_dataset_stats.py`**

```python
"""
Figure 4: Dataset statistics — class prevalence and confidence tier breakdown.

Output: reports/fig_dataset_stats.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insert project root so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import FIGSIZE_WIDE, DPI, MINERAL_COLORS, apply_style, despine
from config_loader import load_config
from data.dataset import _collapse_labels, LABEL_COLS

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

TIER_COLORS = {'High': '#43a047', 'Moderate': '#ffa726', 'Low': '#ef5350'}


def main():
    apply_style()
    cfg = load_config()
    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    df = pd.read_parquet(parquet_path)
    df = _collapse_labels(df)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # --- Left panel: class prevalence in train split ---
    train_df  = df[df['split'] == 'train']
    n_train   = len(train_df)
    percentages = []
    counts      = []
    for cls in LABEL_COLS:
        n_pos = int((train_df[f'label_{cls}'] > 0.4).sum())
        percentages.append(100.0 * n_pos / n_train if n_train > 0 else 0.0)
        counts.append(n_pos)

    y_left  = range(len(LABEL_COLS))
    colors  = [MINERAL_COLORS[cls] for cls in LABEL_COLS]
    ax_left.barh(list(y_left), percentages, color=colors, edgecolor='white', height=0.6)
    for i, (pct, cnt) in enumerate(zip(percentages, counts)):
        ax_left.text(pct + 0.5, i, f'n={cnt:,}', va='center', fontsize=9)
    ax_left.set_yticks(list(y_left))
    ax_left.set_yticklabels(LABEL_COLS)
    ax_left.set_xlabel('% of train pixels')
    ax_left.set_xlim(0, max(percentages) * 1.3 if max(percentages) > 0 else 100)
    ax_left.set_title('Class Prevalence (train split)')
    despine(ax_left)

    # --- Right panel: confidence tier breakdown per split ---
    splits     = ['train', 'val', 'test']
    tier_order = ['High', 'Moderate', 'Low']
    y_right    = range(len(splits))
    lefts      = np.zeros(len(splits))

    for tier in tier_order:
        vals = np.array(
            [(df[df['split'] == s]['confidence_tier'] == tier).sum() for s in splits],
            dtype=float,
        )
        ax_right.barh(list(y_right), vals, left=lefts,
                      color=TIER_COLORS[tier], label=tier, edgecolor='white', height=0.6)
        lefts += vals

    ax_right.set_yticks(list(y_right))
    ax_right.set_yticklabels(splits)
    ax_right.set_xlabel('Pixel count')
    ax_right.set_title('Confidence Tier by Split')
    ax_right.legend(loc='upper right', fontsize=9)
    despine(ax_right)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_dataset_stats.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n crism pytest tests/test_figures.py::TestDatasetStats -v
```

Expected: 1 passed.

- [ ] **Step 5: Smoke-run against real data**

```bash
conda run -n crism python scripts/plot_dataset_stats.py
```

Expected: `Saved .../reports/fig_dataset_stats.png`. Verify two panels, class bars with counts, stacked tier bars.

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_dataset_stats.py tests/test_figures.py
git commit -m "feat: add Fig 4 dataset statistics"
```

---

### Task 6: Figure 5 — Ablation Waterfall

**Files:**
- Create: `scripts/plot_ablation_waterfall.py`
- Modify: `tests/test_figures.py` (add `TestAblationWaterfall` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_figures.py`:

```python
class TestAblationWaterfall:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_ablation_waterfall as m
        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))
        m.main()
        assert (tmp_path / 'fig_ablation_waterfall.png').exists()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n crism pytest tests/test_figures.py::TestAblationWaterfall -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.plot_ablation_waterfall'`

- [ ] **Step 3: Implement `scripts/plot_ablation_waterfall.py`**

```python
"""
Figure 5: Ablation waterfall — cumulative val mAP by training intervention.

Output: reports/fig_ablation_waterfall.png
"""
import os, sys
import matplotlib.pyplot as plt

# Insert project root so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import FIGSIZE_SINGLE, DPI, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hardcoded intervention sequence: (step_index, x_label, val_mAP)
STEPS = [
    (0, 'Baseline\n(scnn, BCE)',      0.554),
    (1, '+ Focal loss',               0.615),
    (2, '+ mrral\nViT',               0.561),
    (3, '+ MAE\npretrain',            0.562),
    (4, '+ ASL loss',                 0.534),
    (5, '+ Hybrid\n(mrrsu feats)',    0.608),
    (6, '+ Hybrid\n+ diff LR',        0.614),
]

COLOR_BASELINE = '#9e9e9e'
COLOR_GAIN     = '#4caf50'
COLOR_REGRESS  = '#ef5350'


def main():
    apply_style()

    step_indices = [s[0] for s in STEPS]
    labels       = [s[1] for s in STEPS]
    maps         = [s[2] for s in STEPS]

    colors = [COLOR_BASELINE]
    for i in range(1, len(maps)):
        colors.append(COLOR_GAIN if maps[i] >= maps[i - 1] else COLOR_REGRESS)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.bar(step_indices, maps, color=colors, edgecolor='white', width=0.6)

    # Delta annotations for steps 1–6
    for i in range(1, len(maps)):
        delta = round(maps[i] - maps[i - 1], 2)
        delta_str = f'+{delta:.2f}' if delta >= 0 else f'{delta:.2f}'
        ax.text(step_indices[i], maps[i] + 0.01, delta_str,
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(step_indices)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel('val mAP')
    ax.set_title('Cumulative val mAP by training intervention')
    despine(ax)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_ablation_waterfall.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n crism pytest tests/test_figures.py::TestAblationWaterfall -v
```

Expected: 1 passed.

- [ ] **Step 5: Run full test suite**

```bash
conda run -n crism pytest tests/test_figures.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 6: Smoke-run all five scripts**

```bash
conda run -n crism python scripts/plot_model_progression.py
conda run -n crism python scripts/plot_per_class_heatmap.py
conda run -n crism python scripts/plot_class_spectra_v2.py
conda run -n crism python scripts/plot_dataset_stats.py
conda run -n crism python scripts/plot_ablation_waterfall.py
```

Expected: 5 PNG files in `reports/`.

- [ ] **Step 7: Commit and finish**

```bash
git add scripts/plot_ablation_waterfall.py tests/test_figures.py
git commit -m "feat: add Fig 5 ablation waterfall; all 5 figures complete"
```
