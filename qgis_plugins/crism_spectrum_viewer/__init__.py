# -*- coding: utf-8 -*-
"""CRISM Spectrum Viewer QGIS Plugin — entry point."""


def classFactory(iface):  # noqa: N802
    """Required by QGIS plugin loader."""
    from .plugin import CRISMSpectrumViewer
    return CRISMSpectrumViewer(iface)
