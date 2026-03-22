"""App-wide constants."""
import os

# Spectral constants (shared with pipeline)
CRISM_NODATA = 65535.0
BAD_BAND_RANGES = [(1040, 1070)]   # nm — single bad band ~1056 nm (S/L detector boundary)
WAV_MIN, WAV_MAX = 500, 2600       # display range (nm)

# CRISM mrral false-color band indices (1-based)
FC_BANDS = (60, 34, 21)  # R=2529 nm, G=1506 nm, B=1079 nm

# Tile-image downscale for display — longest edge becomes this many pixels
TILE_DISPLAY_MAX_PX = 1024

# Verification column names added to GeoPackage
COL_VERDICT    = 'verdict'          # 'correct' | 'incorrect' | None
COL_CONFIDENCE = 'verify_conf'      # 'low' | 'moderate' | 'high' | None
COL_NOTE       = 'verify_note'      # free-text string
COL_TIMESTAMP  = 'verified_at'      # ISO-8601 string
