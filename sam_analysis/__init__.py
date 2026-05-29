"""SAM (Spectral Angle Mapper) analysis module for CRISM Argyre tiles.

Provides:
- endmembers.load_endmember_library — loader for per-class mean spectra.
- sam.spectral_angle / sam_raster — pure-numpy SAM core.
- embedding_sam — embedding-space SAM-analog using the champion encoder.
- diagnostic — classifier-plag-in-SAM diagnostic figures + hard-negative export.
- run_argyre_sam — top-level driver script.
- find_argyre_mtrdr — locate MTRDR scenes spatially overlapping Argyre tiles.
"""

__all__ = [
    "endmembers",
    "sam",
    "embedding_sam",
    "diagnostic",
]
