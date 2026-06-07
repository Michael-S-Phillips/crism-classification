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

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QDockWidget,
    QLabel,
    QPushButton,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAND_COLUMNS = [f"band_{i:02d}" for i in range(59)]  # MRDR default; layers may have more (VRDR=87)
MAX_TRACES = 5


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
        self._status = QLabel("")
        self._status.setWordWrap(True)

        # --- layout ---
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._clear_btn)
        layout.addWidget(self._status)
        self.setWidget(container)

        self._redraw()

    # ------------------------------------------------------------------
    def add_trace(self, label, wavelengths, reflectances):
        """Append a new spectrum trace; drop oldest if > MAX_TRACES."""
        self._traces.append((label, wavelengths, reflectances))
        if len(self._traces) > MAX_TRACES:
            self._traces = self._traces[-MAX_TRACES:]
        self._redraw()
        self._status.setText(f"Last: {label}")

    def clear(self):
        self._traces.clear()
        self._status.setText("Plot cleared.")
        self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self):
        self._ax.cla()
        self._ax.set_xlabel("Wavelength (nm)", fontsize=8)
        self._ax.set_ylabel("Reflectance (I/F)", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, alpha=0.3)

        for i, (label, wl, rf) in enumerate(self._traces):
            color = TRACE_COLORS[i % len(TRACE_COLORS)]
            self._ax.plot(wl, rf, color=color, linewidth=1.2, label=label)

        if self._traces:
            self._ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

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
        self._dock.add_trace(label, wavelengths[:n], reflectances[:n])

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
