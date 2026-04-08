"""Spectral extraction utilities."""
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry.base import BaseGeometry

from app.config import BAD_BAND_RANGES, CRISM_NODATA, WAV_MAX, WAV_MIN


def _read_wavelengths(src: rasterio.DatasetReader) -> np.ndarray:
    """Parse wavelengths (nm) from per-band ENVI-style tags."""
    try:
        return np.array(
            [float(src.tags(i)['wavelength']) for i in range(1, src.count + 1)],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError):
        return np.linspace(410, 3920, src.count).astype(np.float32)


def _interp_bad_bands(spec: np.ndarray, wav: np.ndarray) -> np.ndarray:
    spec = spec.copy()
    good = np.ones(len(wav), dtype=bool)
    for lo, hi in BAD_BAND_RANGES:
        good &= ~((wav >= lo) & (wav <= hi))
    bad_idx = np.where(~good)[0]
    if len(bad_idx) == 0:
        return spec
    good_idx = np.where(good)[0]
    spec[bad_idx] = np.interp(wav[bad_idx], wav[good_idx], spec[good_idx])
    return spec


def _wav_mask(wav: np.ndarray) -> np.ndarray:
    return (wav >= WAV_MIN) & (wav <= WAV_MAX)


def _extract_masked(img_path: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract pixel spectra for True pixels in mask. Returns (N, B) and wavelengths."""
    with rasterio.open(img_path) as src:
        wav = _read_wavelengths(src)
        wmask = _wav_mask(wav)
        wav_sel = wav[wmask]

        n_px = int(mask.sum())
        n_bands = int(wmask.sum())
        spectra = np.full((n_px, n_bands), np.nan, dtype=np.float32)

        for bi, band_global in enumerate(np.where(wmask)[0]):
            raw = src.read(int(band_global) + 1).astype(np.float32)
            raw[raw == CRISM_NODATA] = np.nan
            raw[np.abs(raw) > 1] = np.nan
            spectra[:, bi] = raw[mask]

    return spectra, wav_sel


def _summarise(spectra: np.ndarray, wav: np.ndarray) -> dict:
    mean = np.nanmean(spectra, axis=0)
    std  = np.nanstd(spectra, axis=0)
    mean = _interp_bad_bands(mean, wav)
    std  = _interp_bad_bands(std, wav)
    return {
        'wavelengths': wav.tolist(),
        'mean':        mean.tolist(),
        'std':         std.tolist(),
        'n_pixels':    int((~np.isnan(spectra[:, 0])).sum()),
    }


def extract_polygon_spectrum(img_path: str, geometry: BaseGeometry) -> dict:
    """Return mean±std spectrum for all pixels within geometry."""
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        transform = src.transform

    mask = rasterio.features.geometry_mask(
        [geometry], out_shape=(h, w), transform=transform, invert=True
    )
    if not mask.any():
        return {'wavelengths': [], 'mean': [], 'std': [], 'n_pixels': 0}

    spectra, wav = _extract_masked(img_path, mask)
    return _summarise(spectra, wav)


def extract_roi_spectrum(
    img_path: str,
    row: int,
    col: int,
    radius: int = 5,
) -> dict:
    """Return mean±std spectrum for a square ROI of ±radius pixels."""
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width

    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)

    mask = np.zeros((h, w), dtype=bool)
    mask[r0:r1, c0:c1] = True

    spectra, wav = _extract_masked(img_path, mask)
    return _summarise(spectra, wav)
