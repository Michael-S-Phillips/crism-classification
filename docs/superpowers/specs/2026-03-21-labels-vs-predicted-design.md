# Labels vs Predicted Comparison Figure Design

## Goal

Produce `reports/fig_labels_vs_predicted.png`: a 2×2 figure comparing human-annotated mineral label polygons to the fine-tuned supervised classifier's vectroscopy output, for tiles T0435 and T0434. Update `fig_style.py` globally with the new mineral color scheme.

## Color Scheme Update

Replace current `MINERAL_COLORS` in `scripts/fig_style.py`:

| Mineral | Old | New |
|---|---|---|
| olivine | `#4caf50` (green) | `#e53935` (red) |
| lcp | `#2196f3` (blue) | `#00bcd4` (cyan) |
| hcp | `#ff9800` (orange) | `#e91e63` (magenta) |
| plagioclase | `#9c27b0` (purple) | `#ffeb3b` (yellow) |
| other | `#9e9e9e` (gray) | `#9e9e9e` (gray — unchanged) |

All existing scripts that import `MINERAL_COLORS` from `fig_style.py` automatically pick up the new palette on next run.

## Figure Layout

`reports/fig_labels_vs_predicted.png` — 2 rows × 2 columns, ~5×4 inches per panel.

```
Col 0 (Labels)              Col 1 (Predicted)
┌──────────────────────┬──────────────────────┐  Row 0: T0435
│ T0435 — Labels       │ T0435 — Predicted    │
│ (categorized_mineral │ (data/vector/t0435   │
│  _units/T0435.gpkg)  │  _mineral_map.gpkg)  │
├──────────────────────┼──────────────────────┤  Row 1: T0434
│ T0434 — Labels       │ T0434 — Predicted    │
│                      │                      │
└──────────────────────┴──────────────────────┘
   [shared mineral legend + confidence tier legend at bottom]
```

Both panels share a grey background showing the tile valid-mask extent, axes ticks off, no spines.

## Label Panel

**Source:** `/mnt/mrdr/categorized_mineral_units/T0435.gpkg` and `T0434.gpkg`, one GeoPackage per tile, single layer each (layer name = tile ID).

**Color:** parsed from the `Category` column (e.g. `"hcp + olivine (Moderate)"`). Steps:

1. Extract tier from trailing parenthetical: `High`, `Moderate`, or `Low`.
2. Strip tier text to get mineral string; split on ` + ` to get components.
3. Normalise each component to lowercase, then map to one of the 5 canonical classes:
   - `"olivine"`, `"type 1 olivine"`, `"type 2 olivine"` → `olivine`
   - `"lcp"` → `lcp`
   - `"hcp"` → `hcp`
   - `"plagioclase"` → `plagioclase`
   - `"alteration"`, `"red slope"`, `"other"`, `"bland"`, `"denom"`, `"uncertain"` → `other`
   - Any unrecognised token → `other` (fail-safe)
4. **Single-class polygon:** use that mineral's color from `MINERAL_COLORS`.
5. **Multi-class polygon:** blend colors by averaging the RGB components of the constituent minerals (e.g. `hcp + olivine` → average of magenta and red ≈ deep pink; `alteration + hcp + olivine` → average of gray, magenta, red). The returned mineral list from `parse_category` preserves input token order (not canonical order); downstream code must not rely on ordering.
6. **Opacity:** `High`=0.9, `Moderate`=0.65, `Low`=0.4.

This logic lives in two helper functions in the script:

```python
def parse_category(category: str) -> tuple[list[str], str]:
    """Return (canonical_mineral_names, tier_string)."""

def blend_mineral_color(mineral_names: list[str]) -> tuple[float, float, float]:
    """Return RGB (0–1) blend of the given minerals' MINERAL_COLORS."""
```

## Predicted Panel

**Source:** `data/vector/t043X_mrral_*_mineral_map.gpkg` — one layer per mineral class.

**Color:** `MINERAL_COLORS[mineral_name]` from `fig_style.py`.

**Opacity:** confidence tier column → `{1: 0.4, 2: 0.65, 3: 0.9}` (same scale as labels).

**Rendering order (bottom to top):** olivine, lcp, hcp, plagioclase, other — so less common classes are not buried under the dominant olivine signal.

## Legends

Two legends placed below the figure:

1. **Mineral legend:** one patch per canonical mineral class using `MINERAL_COLORS` (solid, full opacity).
2. **Confidence tier legend:** three patches using neutral grey at the three opacity levels, labelled `Low / Tier 1`, `Moderate / Tier 2`, `High / Tier 3`.

## Script

**New file:** `scripts/plot_labels_vs_predicted.py` — standalone, reads from hardcoded paths for the two tiles, produces the figure. No changes to other scripts.

```
PROJ = project root
LABEL_DIR = /mnt/mrdr/categorized_mineral_units
VECTOR_DIR = {PROJ}/data/vector

TILES = [
    {
        'id':    't0435_mrral_40s323_0327_4',
        'label_gpkg': f'{LABEL_DIR}/T0435.gpkg',
        'label_layer': 'T0435',
        'pred_gpkg': f'{VECTOR_DIR}/t0435_mrral_40s323_0327_4_mineral_map.gpkg',
        'img': '/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img',
        'label': 'T0435',
    },
    { ... T0434 ... },
]
```

Output: `reports/fig_labels_vs_predicted.png` at `DPI` from `fig_style.DPI` (300).

## Tests

**`tests/test_plot_labels_vs_predicted.py`:**

- `test_parse_category_single` — `"lcp (High)"` → `(['lcp'], 'High')`
- `test_parse_category_mixed` — `"hcp + olivine (Moderate)"` → `(['hcp', 'olivine'], 'Moderate')`
- `test_parse_category_three_way` — `"alteration + hcp + olivine (Low)"` → `(['other', 'hcp', 'olivine'], 'Low')`
- `test_parse_category_type_olivine` — `"Type 2 olivine (High)"` → `(['olivine'], 'High')`
- `test_parse_category_other_uppercase` — `"Other (High)"` → `(['other'], 'High')`
- `test_parse_category_red_slope` — `"red slope (Low)"` → `(['other'], 'Low')` (two-word token, not split further)
- `test_blend_single` — single mineral returns its exact MINERAL_COLORS RGB
- `test_blend_two` — two minerals return the component-wise average of their RGB values

## classify_tile_prototype.py: --min_similarity flag

Add a `--min_similarity` argument to `scripts/classify_tile_prototype.py`. Pixels whose maximum cosine similarity across all classes falls below this value are left unclassified (similarity map set to 0.0 for all classes at that pixel, on top of the existing `valid_mask` zeroing).

**Default behaviour:** compute the 10th percentile of `max(similarity, axis=-1)` across all valid pixels in the tile, and use that as the threshold. This means the least-confident 10% of pixels go unclassified. Pass `--min_similarity 0.0` to disable (backward-compatible with current behaviour).

The predicted panel in the comparison figure already shows unclassified pixels as grey background because the vectroscopy GeoPackages only contain polygons above the tier-1 threshold — no additional change needed to the figure script.

## Not in Scope

- Re-running `vectorize_tile_minerals.py` — existing `data/vector/` GeoPackages already contain all 5 mineral classes and are used as-is.
- Modifying `plot_vector_mineral_maps.py` — that script stays unchanged (colors update automatically via `fig_style.py`).
- Prototype classifier comparison — separate figure (`fig_prototype_*.png`).
