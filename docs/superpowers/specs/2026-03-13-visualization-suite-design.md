# Visualization Suite Design

**Date:** 2026-03-13
**Goal:** Five standalone publication-ready figures documenting the CRISM mineral classification project — data, models, and results.

---

## Audience & Quality Bar

Internal/working use now; designed to upgrade cleanly to paper/slides later. All figures: 300 DPI PNG, consistent typography and color palette, axes fully labeled with units, legends included.

---

## Shared Style Module

**`scripts/fig_style.py`** — constants and two small helper functions:
- `FIGSIZE_SINGLE = (7, 4.5)` — single-panel figures
- `FIGSIZE_WIDE = (10, 4.5)` — two-panel figures
- `FIGSIZE_GRID = (10, 7)` — 2×3 grid figures
- `DPI = 300`
- `MINERAL_COLORS` — dict, fixed hex color per class: olivine=`#4caf50`, lcp=`#2196f3`, hcp=`#ff9800`, plagioclase=`#9c27b0`, other=`#9e9e9e`
- `LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']` (collapsed 5-class)
- `apply_style()` — calls `matplotlib.rcParams.update(...)` with: `{'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.grid': True, 'grid.alpha': 0.3}`. Note: spine visibility cannot be set via rcParams; use `despine(ax)` below.
- `despine(ax)` — sets `ax.spines['top'].set_visible(False)` and `ax.spines['right'].set_visible(False)` on the given axes object. Called by each script after creating each axes.

**Reports output path:** Each script defines:
```python
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)
```
This ensures output lands in `<project_root>/reports/` regardless of invocation directory.

**sys.path setup** (all scripts that import from project root):
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

All five scripts import from `fig_style`.

---

## Figure 1: Model Progression

**Script:** `scripts/plot_model_progression.py`
**Output:** `reports/fig_model_progression.png`
**Figsize:** `FIGSIZE_SINGLE = (7, 4.5)` with `figsize` overridden to `(8, 6)` to accommodate 11 bars and footnote.

Horizontal bar chart. Each bar is a named run; val_mAP on x-axis (range 0.0–0.7). Bars grouped by sweep family (v3, v5, v6).

**Group background shading:** For each group, draw an `axhspan` rectangle spanning the y-range of that group's bars (with 0.3 padding on each side), color=`#f5f5f5` (light gray), alpha=0.5, zorder=0. Drawn before bars.

**Between-group annotations:** Vertical dashed lines (not arrows) at x=0.0 between groups, with a text label to the right. Specifically: after the last v3 bar and before the first v5 bar, add `ax.text(0.02, midpoint_y, '+mrral spectral input', fontsize=9, color='#555555', va='center')`. Same pattern between v5 and v6 with text `'+ASL loss'`. Between v6 scnn/svit and v6 hybrid rows: `'+hybrid mrrsu features'`. The vertical dashed line is `ax.axhline(y=boundary_y, linestyle='--', color='#aaaaaa', linewidth=0.8)` placed at the y boundary between groups.

**Runs to include:**

| run | val_mAP | group |
|-----|---------|-------|
| scnn_base | 0.554 | v3 |
| scnn_focal | 0.615 | v3 |
| svit_base | 0.522 | v3 |
| svit_mae | 0.507 | v3 |
| svit_base_v5 | 0.561 | v5 |
| svit_mae_v5 | 0.520 | v5 |
| scnn_asl_v6 | 0.333 | v6 |
| svit_asl_v6 | 0.534 | v6 |
| svit_asl_diffr_v6 | 0.523 | v6 |
| shybrid_asl_v6 | 0.608 | v6 |
| shybrid_asl_diffr_v6 | 0.614 | v6 |

**Sort order within each group:** bars appear in the order listed in the table above. y-axis is built top-to-bottom: first run in each group = highest y-index within that group. Groups are ordered v3 (top), v5 (middle), v6 (bottom). All 11 runs are plotted in a single `barh` call using a `range(11)` index; group shading and annotations are overlaid after.

**Label-space mismatch:** v3 models used 6-class labels (olivine_t1 + olivine_t2 separate); v5/v6 use 5-class collapsed olivine. Figure handles this with:
- Group y-axis label (via group tick or annotation) reads "v3 (6-class)" and "v5/v6 (5-class)" respectively.
- Figure-level footnote via `fig.text(0.01, 0.01, '† v3 mAP computed over 6 classes (olivine_t1/t2 separate); v5/v6 over 5 classes (olivine collapsed). Values not directly comparable.', fontsize=8, color='#666666', va='bottom')`.

---

## Figure 2: Per-Class AP Heatmap

**Script:** `scripts/plot_per_class_heatmap.py`
**Output:** `reports/fig_per_class_heatmap.png`
**Figsize:** `(9, 4)` (custom; not one of the named sizes — heatmap needs square-ish cells).

Heatmap: rows = 5 models, columns = 5 mineral classes. Cell color = AP value, colormap = `YlOrRd` (vmin=0, vmax=1). Cell text shows AP value to 2 decimal places, color=black if AP < 0.6 else white.

**Models shown** (5-class runs only, hardcoded):

| model | olivine | lcp | hcp | plagioclase | other | mAP |
|-------|---------|-----|-----|-------------|-------|-----|
| scnn_asl_v6 | 0.90 | 0.10 | 0.22 | 0.20 | 0.24 | 0.333 |
| svit_asl_v6 | 0.90 | 0.90 | 0.55 | 0.22 | 0.48 | 0.534 |
| svit_asl_diffr_v6 | 0.97 | 0.65 | 0.38 | 0.29 | 0.32 | 0.523 |
| shybrid_asl_v6 | 0.97 | 0.86 | 0.51 | 0.22 | 0.48 | 0.608 |
| shybrid_asl_diffr_v6 | 0.97 | 0.85 | 0.51 | 0.26 | 0.48 | 0.614 |

Rows sorted by mAP descending (as shown).

**Right-side mAP column:** Implemented via `gridspec` with width_ratios `[5, 0.6]`. Left axes = heatmap. Right axes (width ratio 0.6) = plain bar or just text. In the right axes: `ax_r.set_xlim(0, 1)`, `ax_r.axis('off')`, then for each row i: `ax_r.text(0.1, i + 0.5, f'{mAP:.3f}', va='center', fontsize=10)`. Add a title `ax_r.set_title('mAP', fontsize=10)`. The right axes shares the y-axis limits of the heatmap axes (both span 0 to 5).

---

## Figure 3: Class Spectral Profiles

**Script:** `scripts/plot_class_spectra_v2.py`
**Output:** `reports/fig_class_spectra_v2.png`
**Figsize:** `FIGSIZE_GRID = (10, 7)`

Loads `mrral_pixels.parquet`. For each of the 5 collapsed mineral classes, plots mean reflectance ± 1σ as a shaded band over wavelength (nm). Layout: 2×3 grid (5 class panels + 1 overlay panel). X-axis: wavelength (nm). Y-axis: reflectance (0–1). Each per-class panel titled with class name and pixel count (n=X).

**6th panel (row=1, col=2 in 0-indexed 2×3 grid):** All-class overlay. All 5 class mean spectra (no shaded bands) plotted on the same axes, each in its `MINERAL_COLORS` color. Legend inside panel (`loc='upper right'`). Title: "All classes".

**Data loading:**
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
import pandas as pd
cfg = load_config()
parquet_path = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
df = pd.read_parquet(parquet_path)
```

**Parquet schema:** `mrral_pixels.parquet` columns: `m0..m58` (59 float32 reflectance bands), `label_olivine`, `label_lcp`, `label_hcp`, `label_plagioclase`, `label_other` (float32 confidence weights 0/0.5/1.0), `split`, `confidence_tier`, `tile_id`, `polygon_id`, `pixel_row`, `pixel_col`. Use all rows (no split filter) for spectral profiles.

**Band-to-wavelength mapping:** Read from the first `.hdr` file found via `glob.glob(os.path.join(cfg['data_root'], '**', '*mrral*.hdr'), recursive=True)`. Parse using the `spectral` library: `import spectral.io.envi as envi; hdr = envi.read_envi_header(hdr_path); wavelengths = np.array(hdr['wavelength'], dtype=float)`. Slice to first 59 values. Falls back to `np.linspace(410, 2457, 59)` if no `.hdr` found or if `'wavelength'` key is absent.

**Class membership:** Pixel belongs to class C if `label_C > 0.4`. Pixels where no label exceeds 0.4 are excluded silently (they contribute to neither class's mean calculation).

**Panel layout:** `fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID, sharey=True)`. Panels 0–4 (in row-major order) = olivine, lcp, hcp, plagioclase, other. Panel 5 = all-class overlay. `sharey=True` means y-axis 0–1 applies to all panels.

---

## Figure 4: Dataset Statistics

**Script:** `scripts/plot_dataset_stats.py`
**Output:** `reports/fig_dataset_stats.png`
**Figsize:** `FIGSIZE_WIDE = (10, 4.5)`

Two-panel figure (side by side).

**Data loading:**
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
import pandas as pd
from data.dataset import _collapse_labels, LABEL_COLS
cfg = load_config()
parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
df = pd.read_parquet(parquet_path)
df = _collapse_labels(df)
```

`pixels.parquet` schema (relevant columns): `label_olivine_t1`, `label_olivine_t2`, `label_lcp`, `label_hcp`, `label_plagioclase`, `label_other` (float32 confidence weights), `split` (train/val/test), `confidence_tier` (High/Moderate/Low). After `_collapse_labels()`, the DataFrame has `label_olivine` (merged from t1+t2) and the original t1/t2 columns are dropped. `LABEL_COLS` from `data.dataset` = `['olivine', 'lcp', 'hcp', 'plagioclase', 'other']` and matches `fig_style.LABEL_COLS`; use `from data.dataset import LABEL_COLS` (do not import from `fig_style`).

**Left panel — Class prevalence:** Horizontal bar chart. Train split only (`df[df['split']=='train']`). One bar per class in `LABEL_COLS` order (olivine top, other bottom). Length = % of train pixels positive for that class (positive = `label_<class> > 0.4`). Multi-label: percentages can sum >100%. Color = `MINERAL_COLORS[cls]`. Text annotation at bar end: `f'n={count:,}'`. X-axis: "% of train pixels", 0–100.

**Right panel — Confidence tier breakdown:** Stacked horizontal bar per split (order: train, val, test top to bottom). For each split, count pixels per `confidence_tier`. Stack segments left-to-right in order: High, Moderate, Low. Colors: High=`#43a047`, Moderate=`#ffa726`, Low=`#ef5350`. X-axis: "Pixel count". Legend above or to the right of panel.

---

## Figure 5: Ablation Waterfall

**Script:** `scripts/plot_ablation_waterfall.py`
**Output:** `reports/fig_ablation_waterfall.png`
**Figsize:** `FIGSIZE_SINGLE = (7, 4.5)`

Bar chart (not a true waterfall — each bar's absolute height = val_mAP at that step). X-axis: steps 0–6, tick labels = step labels rotated 30°. Y-axis: val_mAP 0.0–0.7, labeled "val mAP". Title: "Cumulative val mAP by training intervention".

**Bar colors:**
- Step 0 (baseline): gray (`#9e9e9e`)
- Steps where val_mAP ≥ previous step val_mAP: green (`#4caf50`)
- Steps where val_mAP < previous step val_mAP: red (`#ef5350`)

**Delta annotations:** For steps 1–6, draw `ax.text(x, val_mAP + 0.01, delta_str, ha='center', va='bottom', fontsize=9)`. The `+0.01` offset places the text just above the bar top in all cases (no conditional threshold needed — bars are all > 0.3 mAP so there is no risk of text going outside the axes with a 0.7 y-limit). Step 0 has no annotation.

**Delta computation:** `delta = round(val_mAP[i] - val_mAP[i-1], 2)`. Format: `f'+{delta:.2f}'` if delta >= 0 else `f'{delta:.2f}'`.

**Intervention sequence (hardcoded):**

| step | label | val_mAP |
|------|-------|---------|
| 0 | Baseline (scnn, BCE loss) | 0.554 |
| 1 | + Focal loss | 0.615 |
| 2 | + mrral spectral input (ViT) | 0.561 |
| 3 | + MAE pretraining | 0.562 |
| 4 | + ASL loss | 0.534 |
| 5 | + Hybrid (mrrsu features) | 0.608 |
| 6 | + Hybrid + diff LR | 0.614 |

---

## File Outputs

All outputs to `<project_root>/reports/` (created by each script using absolute path relative to script location):
- `fig_model_progression.png`
- `fig_per_class_heatmap.png`
- `fig_class_spectra_v2.png`
- `fig_dataset_stats.png`
- `fig_ablation_waterfall.png`

---

## Out of Scope

- Interactive plots (no plotly/bokeh)
- Test set evaluation (val metrics only for now)
- Learning curve figures (already generated per-run by wandb)
- PDF export (PNG only for now; can add later with `savefig(..., format='pdf')`)
