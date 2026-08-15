# -*- coding: utf-8 -*-
"""
CRISM Spectrum Viewer — main plugin class.

Click a polygon feature that has band_00..band_58 attributes to plot its
CRISM reflectance spectrum in a dock widget.  Wavelengths are loaded from a
sidecar *_wavelengths.json file located next to the source GeoPackage.
"""

import json
import os
from glob import glob

import numpy as np

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsWkbTypes,
)
from qgis.gui import QgsMapToolEmitPoint

# Matplotlib — bundled with QGIS on all platforms
import matplotlib
matplotlib.use("Qt5Agg")  # must be set before importing pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Continuum removal — a standalone numpy-only port of data/continuum_removal.py.
# It CANNOT be imported from the repo: this runs in QGIS's interpreter, not the
# `crism` conda env. tests/test_plugin_cr_parity.py asserts the port matches the
# pipeline value-for-value, which is the only guard against the viewer drawing a
# plausible spectrum that is not the one the model consumed.
try:
    from . import crism_cr
except (ImportError, ValueError):  # module loaded outside its package
    import crism_cr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAND_COLUMNS = [f"band_{i:02d}" for i in range(59)]  # MRDR default; layers may have more (VRDR=87)
MAX_TRACES = 5

# View modes (shared by every trace; per-trace modes are out of scope)
MODE_REFLECTANCE = "Reflectance"
MODE_HULL_CR = "Hull-CR"
MODE_LINEAR_CR = "Linear-CR"
VIEW_MODES = [MODE_REFLECTANCE, MODE_HULL_CR, MODE_LINEAR_CR]
CR_MODES = (MODE_HULL_CR, MODE_LINEAR_CR)

# band_NN attributes are polygon MEAN reflectances, so a CR of them is
# CR-of-the-mean, not the mean of per-pixel CRs. Conventional, but it has to be
# said on the axis or it reads as a per-pixel spectrum.
Y_LABELS = {
    MODE_REFLECTANCE: "Reflectance (I/F), polygon mean",
    MODE_HULL_CR: "Hull-CR (ratio), polygon mean",
    MODE_LINEAR_CR: "Linear-CR (ratio), polygon mean",
}

X_FULL_MIN, X_FULL_MAX = 400, 2500  # nm; the full 59-band model window, rounded


def _feature_band_columns(field_names):
    """All band_NN columns present on the layer, sorted by index (any count).
    Replaces the hardcoded 59-band assumption so VRDR (87) and MRDR (59) both work."""
    import re
    cols = [n for n in field_names if re.fullmatch(r"band_\d+", n)]
    return sorted(cols, key=lambda s: int(s.split("_", 1)[1]))
TRACE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
PLUGIN_NAME = "CRISM Spectrum Viewer"
MENU_LABEL = "View CRISM Spectrum"


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------
class SpectrumDock(QDockWidget):
    """Dockable widget that renders overlaid CRISM spectra."""

    def __init__(self, parent=None):
        super().__init__(PLUGIN_NAME, parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._traces = []  # list of (label, wavelengths, reflectances)

        # --- figure ---
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(220)

        # --- controls ---
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear)
        self._legend_chk = QCheckBox("Show legend")
        self._legend_chk.setChecked(True)
        self._legend_chk.setToolTip("Toggle the legend; when shown, drag it anywhere on the plot.")
        self._legend_chk.toggled.connect(self._redraw)

        self._mode_cmb = QComboBox()
        self._mode_cmb.addItems(VIEW_MODES)
        self._mode_cmb.setToolTip(
            "Reflectance, upper-hull continuum removal, or linear (least-squares "
            "line) continuum removal. Applies to every trace."
        )
        self._mode_cmb.currentIndexChanged.connect(self._redraw)

        # Default ON: in CR mode the excluded bands are 1.0 BY CONSTRUCTION, so
        # showing them draws a flat plateau at the continuum that is not a
        # measurement at all.
        self._badbands_chk = QCheckBox("Mask bad bands")
        self._badbands_chk.setChecked(True)
        self._badbands_chk.setToolTip(
            "Hide band 0 (410.1 nm blue-edge artefact) and bands 16-19 "
            "(1000-1065 nm detector overlap). 59-band layers only."
        )
        self._badbands_chk.toggled.connect(self._redraw)

        # x-range: transient, not persisted; resets when the dock is rebuilt.
        self._xmin_spn = QSpinBox()
        self._xmax_spn = QSpinBox()
        for spn, val in ((self._xmin_spn, X_FULL_MIN), (self._xmax_spn, X_FULL_MAX)):
            spn.setRange(0, 10000)
            spn.setSingleStep(50)
            spn.setSuffix(" nm")
            spn.setValue(val)
            spn.setToolTip("x-axis limit in nm; press Full to autoscale again.")
            spn.valueChanged.connect(self._on_xrange_changed)
        self._x_full = True   # True -> let matplotlib autoscale x
        self._full_btn = QPushButton("Full")
        self._full_btn.setToolTip("Reset the x-axis to the full wavelength range.")
        self._full_btn.clicked.connect(self._reset_xrange)

        self._status = QLabel("")
        self._status.setWordWrap(True)

        # --- layout ---
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._canvas, stretch=1)
        controls = QHBoxLayout()
        controls.addWidget(self._clear_btn)
        controls.addWidget(self._legend_chk)
        controls.addWidget(QLabel("View:"))
        controls.addWidget(self._mode_cmb)
        controls.addWidget(self._badbands_chk)
        controls.addStretch(1)
        layout.addLayout(controls)
        xrow = QHBoxLayout()
        xrow.addWidget(QLabel("x:"))
        xrow.addWidget(self._xmin_spn)
        xrow.addWidget(QLabel("to"))
        xrow.addWidget(self._xmax_spn)
        xrow.addWidget(self._full_btn)
        xrow.addStretch(1)
        layout.addLayout(xrow)
        layout.addWidget(self._status)
        self.setWidget(container)

        self._update_cr_availability()
        self._redraw()

    # ------------------------------------------------------------------
    def add_trace(self, label, wavelengths, reflectances, n_bands=None):
        """Append a new spectrum trace; drop oldest if > MAX_TRACES.

        n_bands is the source layer's band_NN column count (before any
        wavelength/reflectance truncation); it decides CR availability.
        """
        if n_bands is None:
            n_bands = len(reflectances)
        self._traces.append((label, wavelengths, reflectances, n_bands))
        if len(self._traces) > MAX_TRACES:
            self._traces = self._traces[-MAX_TRACES:]
        self._update_cr_availability()
        self._redraw()
        self._status.setText(f"Last: {label}")

    def clear(self):
        self._traces.clear()
        self._update_cr_availability()
        self._status.setText("Plot cleared.")
        self._redraw()

    # ------------------------------------------------------------------
    @staticmethod
    def _trace_on_grid(trace):
        """True if a trace really sits on the 59-band model grid.

        `n_bands` is the LAYER's band_NN column count, which is not always
        len(rf): the map tool truncates a trace to min(len(wavelengths),
        len(reflectances)), so an 87-band VRDR layer paired with a stray 59-entry
        sidecar would otherwise arrive here looking like model input. A truncated
        trace is off-grid whatever its length.
        """
        _, _, rf, n_bands = trace
        return n_bands == crism_cr.N_BANDS and len(rf) == crism_cr.N_BANDS

    def _cr_block_reason(self):
        """None if CR is available, else a phrase naming why it is not."""
        reasons = []
        for trace in self._traces:
            if self._trace_on_grid(trace):
                continue
            _, _, rf, n_bands = trace
            if n_bands != crism_cr.N_BANDS:
                reasons.append(f"{n_bands} bands")
            else:
                reasons.append(f"{len(rf)} of {n_bands} bands with wavelengths")
        if not reasons:
            return None
        return ", ".join(sorted(set(reasons)))

    def _update_cr_availability(self):
        """Enable the CR modes only on the 59-band model grid.

        CR is defined on that grid: `hull_cr` interpolates a hull over its fixed
        wavelengths and excludes fixed indices. Computing it on a VRDR-style
        87-band layer would silently produce a plausible, wrong spectrum, so the
        modes are disabled instead, with a tooltip naming the real band count.
        """
        reason = self._cr_block_reason()
        enabled = reason is None
        model = self._mode_cmb.model()
        for i, name in enumerate(VIEW_MODES):
            if name not in CR_MODES:
                continue
            item = model.item(i) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(enabled)
        if enabled:
            self._mode_cmb.setToolTip(
                "Reflectance, upper-hull continuum removal, or linear "
                "(least-squares line) continuum removal. Applies to every trace."
            )
        else:
            self._mode_cmb.setToolTip(
                f"Continuum removal is disabled: this layer has {reason}, not "
                f"the {crism_cr.N_BANDS}-band model grid "
                f"({crism_cr.WAVELENGTHS_59[0]:.0f}-"
                f"{crism_cr.WAVELENGTHS_59[-1]:.0f} nm) CR is defined on."
            )
            if self._mode_cmb.currentText() in CR_MODES:
                # Fall back to reflectance rather than draw a wrong-grid hull.
                self._mode_cmb.setCurrentIndex(VIEW_MODES.index(MODE_REFLECTANCE))
                self._status.setText(
                    f"Continuum removal unavailable ({reason}); "
                    "showing reflectance."
                )

    def _current_mode(self):
        mode = self._mode_cmb.currentText()
        if mode in CR_MODES and self._cr_block_reason() is not None:
            return MODE_REFLECTANCE
        return mode if mode in VIEW_MODES else MODE_REFLECTANCE

    def _on_xrange_changed(self, _value):
        self._x_full = False
        self._redraw()

    def _reset_xrange(self):
        for spn, val in ((self._xmin_spn, X_FULL_MIN), (self._xmax_spn, X_FULL_MAX)):
            spn.blockSignals(True)
            spn.setValue(val)
            spn.blockSignals(False)
        self._x_full = True
        self._redraw()

    def _transform(self, trace, mode):
        """Reflectances -> plotted y, with masked bands as NaN (never dropped).

        Matplotlib breaks a line at NaN, so a masked window reads as a gap.
        Dropping the points would join the bands either side with a straight
        segment that looks like measured data.
        """
        y = np.asarray(trace[2], dtype=np.float64)
        on_grid = self._trace_on_grid(trace)
        if mode == MODE_HULL_CR and on_grid:
            y = np.asarray(crism_cr.hull_cr(y), dtype=np.float64)
        elif mode == MODE_LINEAR_CR and on_grid:
            y = np.asarray(crism_cr.linear_cr(y), dtype=np.float64)
        if self._badbands_chk.isChecked() and on_grid:
            # The bad-band indices are grid-specific, so they are only applied
            # to the 59-band grid they were derived on.
            y = y.copy()
            y[crism_cr.bad_band_mask()] = np.nan
        return y

    # ------------------------------------------------------------------
    def _redraw(self):
        mode = self._current_mode()
        self._ax.cla()
        self._ax.set_xlabel("Wavelength (nm)", fontsize=8)
        self._ax.set_ylabel(Y_LABELS[mode], fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, alpha=0.3)

        for i, trace in enumerate(self._traces):
            label, wl = trace[0], trace[1]
            color = TRACE_COLORS[i % len(TRACE_COLORS)]
            self._ax.plot(wl, self._transform(trace, mode), color=color,
                          linewidth=1.2, label=label)

        if not self._x_full:
            lo, hi = self._xmin_spn.value(), self._xmax_spn.value()
            if hi > lo:
                self._ax.set_xlim(lo, hi)

        if self._traces and self._legend_chk.isChecked():
            leg = self._ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
            if leg is not None:
                try:
                    leg.set_draggable(True)   # click-drag the legend anywhere on the plot
                except Exception:
                    pass

        self._canvas.draw()


# ---------------------------------------------------------------------------
# Map tool
# ---------------------------------------------------------------------------
class SpectrumMapTool(QgsMapToolEmitPoint):
    """Left-click map tool that identifies a polygon and emits its spectrum."""

    def __init__(self, canvas, dock, iface):
        super().__init__(canvas)
        self._dock = dock
        self._iface = iface

    # ------------------------------------------------------------------
    def canvasReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton:
            return

        layer = self._iface.activeLayer()
        if layer is None or layer.type() != layer.VectorLayer:
            self._iface.statusBarIface().showMessage(
                "CRISM Spectrum Viewer: activate a vector layer first.", 4000
            )
            return

        # Check the layer has the expected band columns
        field_names = {f.name() for f in layer.fields()}
        band_cols = _feature_band_columns(field_names)
        if not band_cols:
            self._iface.statusBarIface().showMessage(
                "CRISM Spectrum Viewer: active layer has no band_NN columns.", 4000
            )
            return

        # Build a small bbox query around the click — setFilterRect prunes by
        # geometry BOUNDING BOX (fast but coarse), so we then test actual
        # point-in-polygon containment among the candidates.
        point = self.toLayerCoordinates(layer, event.mapPoint())
        tol = self._tolerance_in_layer_units(layer)
        rect = QgsRectangle(
            point.x() - tol, point.y() - tol,
            point.x() + tol, point.y() + tol,
        )
        click_geom = QgsGeometry.fromPointXY(QgsPointXY(point.x(), point.y()))

        request = QgsFeatureRequest().setFilterRect(rect)
        candidates = list(layer.getFeatures(request))

        feature = None
        # Prefer the smallest-area polygon that actually CONTAINS the click —
        # if polygons nest, this picks the most specific one.
        contained = []
        for feat in candidates:
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if geom.contains(click_geom):
                contained.append((geom.area(), feat))
        if contained:
            contained.sort(key=lambda t: t[0])
            feature = contained[0][1]
        else:
            # Fallback: nearest polygon by centroid distance, but only if there
            # were any candidates (small rounding errors can put the click
            # slightly outside a polygon edge).
            if candidates:
                best_dist = float("inf")
                for feat in candidates:
                    geom = feat.geometry()
                    if geom is None or geom.isEmpty():
                        continue
                    d = geom.distance(click_geom)
                    if d < best_dist:
                        best_dist = d
                        feature = feat

        if feature is None:
            self._iface.statusBarIface().showMessage(
                "CRISM Spectrum Viewer: no feature found at click location.", 3000
            )
            return

        # Extract reflectance values (however many band_NN columns the layer has)
        reflectances = [float(feature[b]) for b in band_cols]

        # Load wavelengths from sidecar (or fall back to indices)
        wavelengths = _load_wavelengths(layer, len(band_cols))

        # Build a human-readable label
        label = _make_label(feature, field_names)

        # Guard against any wavelength/reflectance length mismatch (mixed instruments)
        n = min(len(wavelengths), len(reflectances))
        self._dock.add_trace(label, wavelengths[:n], reflectances[:n],
                             n_bands=len(band_cols))

        # Ensure dock is visible
        if self._dock.isHidden():
            self._dock.show()

    # ------------------------------------------------------------------
    def _tolerance_in_layer_units(self, layer):
        """Return a search tolerance in LAYER CRS units, roughly 5 px on canvas.

        We map two screen pixels (one px apart) into layer coordinates and
        take their distance. Avoids fragile internal QGIS conversion helpers.
        """
        canvas = self.canvas()
        try:
            p_a = self.toLayerCoordinates(layer, canvas.mapSettings().mapToPixel().toMapCoordinates(0, 0))
            p_b = self.toLayerCoordinates(layer, canvas.mapSettings().mapToPixel().toMapCoordinates(1, 0))
            one_pixel = ((p_b.x() - p_a.x()) ** 2 + (p_b.y() - p_a.y()) ** 2) ** 0.5
            if one_pixel > 0:
                return one_pixel * 5
        except Exception:
            pass
        # Fallback: assume layer ≈ map CRS
        return canvas.mapUnitsPerPixel() * 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_wavelengths(layer, n_default=59):
    """Load wavelengths from sidecar JSON next to the layer source file.

    Accepts files named <stem>_wavelengths.json; falls back to any
    *_wavelengths.json in the same directory; finally falls back to
    band indices 0..n_default-1.
    """
    source = layer.source()
    # source may be "/path/to/file.gpkg|layername=foo"
    filepath = source.split("|")[0].strip()
    directory = os.path.dirname(filepath)
    stem = os.path.splitext(os.path.basename(filepath))[0]

    # 1. Exact stem match
    exact = os.path.join(directory, f"{stem}_wavelengths.json")
    if os.path.isfile(exact):
        return _parse_wavelength_json(exact)

    # 2. Any *_wavelengths.json in the same directory
    candidates = glob(os.path.join(directory, "*_wavelengths.json"))
    if candidates:
        return _parse_wavelength_json(candidates[0])

    # 3. Fall back: band indices
    return list(range(n_default))


def _parse_wavelength_json(path):
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        return data["wavelengths_nm"]
    except Exception:
        return list(range(59))


def _make_label(feature, field_names):
    """Build a short label string from optional well-known attributes."""
    parts = []
    for col in ("tile_id", "category", "count_px"):
        if col in field_names:
            val = feature[col]
            if val is not None:
                parts.append(str(val))
    return " | ".join(parts) if parts else f"fid={feature.id()}"


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------
class CRISMSpectrumViewer:
    """Main plugin class, instantiated by QGIS via classFactory."""

    def __init__(self, iface):
        self._iface = iface
        self._action = None
        self._dock = None
        self._tool = None
        self._prev_tool = None

    # ------------------------------------------------------------------
    def initGui(self):  # noqa: N802
        """Called by QGIS to set up UI elements."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()

        self._action = QAction(icon, MENU_LABEL, self._iface.mainWindow())
        self._action.setCheckable(True)
        self._action.setToolTip(
            "Activate the CRISM spectrum picker: click a polygon to plot its spectrum."
        )
        self._action.triggered.connect(self._toggle_tool)

        # Toolbar
        self._iface.addToolBarIcon(self._action)

        # Menu
        self._iface.addPluginToMenu(PLUGIN_NAME, self._action)

        # Dock widget (hidden until first use)
        self._dock = SpectrumDock(self._iface.mainWindow())
        self._iface.addDockWidget(Qt.RightDockWidgetArea, self._dock)
        self._dock.hide()

    def unload(self):
        """Called by QGIS on plugin unload — clean up everything."""
        if self._action is not None:
            self._iface.removeToolBarIcon(self._action)
            self._iface.removePluginMenu(PLUGIN_NAME, self._action)
            self._action.deleteLater()
            self._action = None

        if self._dock is not None:
            self._iface.removeDockWidget(self._dock)
            self._dock.deleteLater()
            self._dock = None

        # Restore previous map tool
        if self._prev_tool is not None:
            self._iface.mapCanvas().setMapTool(self._prev_tool)
            self._prev_tool = None

        self._tool = None

    # ------------------------------------------------------------------
    def _toggle_tool(self, checked):
        canvas = self._iface.mapCanvas()
        if checked:
            # Activate our map tool
            self._prev_tool = canvas.mapTool()
            if self._tool is None:
                self._tool = SpectrumMapTool(canvas, self._dock, self._iface)
            canvas.setMapTool(self._tool)
            # Show dock
            if self._dock.isHidden():
                self._dock.show()
        else:
            # Deactivate — restore previous tool
            if self._prev_tool is not None:
                canvas.setMapTool(self._prev_tool)
                self._prev_tool = None
            else:
                canvas.unsetMapTool(self._tool)

    def _on_tool_deactivated(self):
        """Keep action state in sync if user changes tool externally."""
        if self._action is not None:
            self._action.setChecked(False)
