# CRISM Spectrum Viewer: continuum removal, x-range, bad bands — design

**Date:** 2026-08-15
**Status:** approved
**Touches:** `qgis_plugins/crism_spectrum_viewer/plugin.py`
**Creates:** `qgis_plugins/crism_spectrum_viewer/crism_cr.py`,
`tests/test_plugin_cr_parity.py`

---

## Problem

The viewer plots raw reflectance only. Three gaps:

1. Band depths are read off continuum-removed spectra, not raw ones. Judging a
   detection by eye currently means mentally removing the continuum.
2. The x-axis is fixed to the full 410–2457 nm range, so the 1–2.5 µm region
   where the diagnostic absorptions live occupies a fraction of the plot.
3. Band 0 (410.1 nm) carries the known blue-edge artefact reaching ~1180 I/F.
   It survives into the polygon means and compresses every autoscaled y-axis.

## Constraints that shape the design

**The plugin runs in QGIS's Python, not the `crism` conda env.** It cannot import
`data/continuum_removal.py`. Continuum removal must therefore be reimplemented
inside the plugin, which creates the central risk: a divergent implementation
would show spectra that differ from what the model consumed, while looking
entirely plausible.

**The pipeline's CR is hardcoded to 59 bands.** `_cr_one` indexes a fixed
`_GOOD_IDX` and interpolates a hull over a fixed wavelength grid. The plugin
already supports variable band counts (its comment notes VRDR layers carry 87).

**`band_NN` attributes are polygon MEAN reflectances**, written by the vectorizer
as the mean over each polygon's pixels. Removing a continuum from that mean gives
CR-of-the-mean, which is not the mean of per-pixel CRs. This is the conventional
thing to do and is kept, but it must be stated on the axis so nobody reads it as
a per-pixel spectrum.

## Design

### A standalone CR module

`crism_cr.py` in the plugin directory: numpy only, no QGIS and no repo imports.
It exposes the wavelength grid, the good-band mask, and two transforms:

    WAVELENGTHS_59      (59,) float — the model's wavelength grid
    good_band_mask()    (59,) bool — False inside 1000–1065 nm
    hull_cr(spec59)     (59,) float — upper-hull CR, excluded bands -> 1.0
    linear_cr(spec59)   (59,) float — per-spectrum least-squares line, clipped [0, 2]

Both are ports of `data/continuum_removal.py`, including its degenerate-input
behaviour: a non-finite or effectively empty spectrum (max ≤ 1e-6) returns all
ones rather than raising, and the hull is floored at 1e-6 before division.

### Parity test — the load-bearing requirement

`tests/test_plugin_cr_parity.py` runs **outside QGIS**, in the `crism` env, and
asserts that `crism_cr.hull_cr` and `crism_cr.linear_cr` match
`data.continuum_removal.continuum_removed` and `linear_continuum_removed` to
float tolerance on real CRISM spectra drawn from a tile, plus the degenerate
cases (all-zero, non-finite, flat).

This is the only thing preventing silent divergence between what the viewer draws
and what the model sees. A test asserting merely that the plugin's output is
finite, or in [0, 1], would not catch a wrong hull.

### Three controls

A **view mode** selector — Reflectance, Hull-CR, Linear-CR. All traces share one
mode; per-trace modes are out of scope.

A **bad bands** checkbox, default **on**, masking band 0 (410.1 nm blue edge) and
indices 16–19 (the 1000–1065 nm detector-overlap window).

**x-min / x-max** spin boxes in nm with a *Full* reset. Transient; not persisted.

### Masked bands become NaN, not dropped

Matplotlib breaks a line at NaN, so a masked region reads as a gap. Dropping the
points instead would connect the band on either side and draw a straight segment
across the excluded window that looks like measured data.

This matters more in CR views than in reflectance. `hull_cr` sets excluded bands
to exactly 1.0 by construction, so plotting them unmasked draws a flat plateau at
the continuum that is not a measurement at all. Masking is therefore not
cosmetic in CR mode, which is why the default is on.

### Axis labels follow the mode

`Reflectance (I/F), polygon mean` · `Hull-CR (ratio), polygon mean` ·
`Linear-CR (ratio), polygon mean`. The "polygon mean" qualifier is always
present, because the CR-of-mean distinction is invisible otherwise.

### Graceful failure on non-59-band layers

If the clicked layer's band count is not 59, the CR modes are **disabled** with a
tooltip naming the band count and explaining that CR is defined on the 59-band
model grid. Reflectance view continues to work exactly as today. Silently
computing a hull over a mismatched wavelength grid would produce a plausible,
wrong spectrum.

## Out of scope

No settings persistence, no per-trace view mode, no spectrum export, no
band-depth readout. Each is a state-management problem larger than the feature it
would add. The x-range resets when the dock is rebuilt, which is acceptable for a
transient inspection tool.

## Risks

- **Divergence between `crism_cr.py` and the pipeline** is the principal risk and
  is mitigated only by the parity test. If `data/continuum_removal.py` changes,
  that test fails and must be reconciled deliberately.
- **QGIS numpy version** is whatever QGIS ships. The port uses only
  `asarray`, `isfinite`, `max`, `clip`, `nan_to_num`, `interp`, `polyfit`-free
  least squares via `lstsq`, and boolean indexing — all long-stable API.
- **Linear-CR is unfamiliar** to read for a spectroscopist used to hull removal.
  It is included because it is half of what the dual-CR model consumes and the
  channel in which alteration's 1–2 µm arch survives; the axis label names it
  explicitly.
