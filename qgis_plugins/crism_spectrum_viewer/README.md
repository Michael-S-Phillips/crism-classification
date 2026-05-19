# CRISM Spectrum Viewer — QGIS 3 Plugin

Click a polygon feature that has `band_00`..`band_58` attribute columns to
view its CRISM MRDR reflectance spectrum in a dock widget.

## Requirements

- QGIS 3.0+
- matplotlib (bundled with QGIS on all platforms)
- No extra `pip install` steps needed

## Installation (symlink — recommended)

```bash
ln -s /mnt/mrdr/crism_classification/qgis_plugins/crism_spectrum_viewer \
      ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/crism_spectrum_viewer
```

Then in QGIS: **Plugins → Manage and Install Plugins → Installed** — tick
**CRISM Spectrum Viewer** and click OK.

## Usage

1. Load the GeoPackage layer (e.g. `nili_v3_denoising_categories.gpkg`).
2. Make that layer the *active* layer in the Layers panel.
3. Click **View CRISM Spectrum** in the toolbar (or via
   **Plugins → CRISM Spectrum Viewer → View CRISM Spectrum**).
   The cursor changes to a crosshair picker.
4. Click any polygon.  A dock widget on the right shows the spectrum.
5. Click more polygons — up to 5 traces are shown overlaid (oldest drops off).
6. Click **Clear** in the dock to reset.
7. Click the toolbar button again to deactivate the picker.

## Wavelength sidecar file

The plugin looks for a JSON file in the **same directory** as the `.gpkg`:

```
<gpkg_stem>_wavelengths.json
```

For `nili_v3_denoising_categories.gpkg` the expected sidecar name would be
`nili_v3_denoising_categories_wavelengths.json`.  Because the actual file is
named `nili_v3_denoising_wavelengths.json` (stem mismatch), the plugin
automatically falls back to "any `*_wavelengths.json` in the same directory."

If no sidecar is found, the x-axis shows band indices 0–58.

## Data columns expected

Each polygon feature must have exactly these 59 float columns:
`band_00`, `band_01`, …, `band_58`

Optional columns used for the trace label: `tile_id`, `category`, `count_px`.

## Known limitations

- Feature identification uses a small rectangular search box around the click
  point.  If polygons are very small or the map is zoomed far out, the tolerance
  heuristic may miss or select the wrong feature.  Zoom in for precise picks.
- The tolerance calculation uses a simple heuristic; it may not be perfect for
  all CRS projections.  If clicks miss, zoom in closer.
- Only the *active* layer is queried.  Switch the active layer in the Layers
  panel before clicking if you have multiple band_NN layers loaded.
- Matplotlib must be importable inside the QGIS Python environment.  It is
  bundled with the official QGIS installers; custom/conda-based QGIS builds
  may need `pip install matplotlib` inside the QGIS Python.
