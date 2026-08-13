"""Tests for ``scripts/extract_mrrsu_features.py``: extracting the 60 mrrsu
summary parameters at each labeled pixel.

The whole point of this module is ROW ALIGNMENT with the input parquet: Tasks
4 and 5 join labels to parameters positionally, so a reorder, dropped row, or
row/col transposition attaches every label to the wrong pixel's parameters and
produces a plausible-looking but meaningless baseline, with no error anywhere.
The fingerprint tests below (`row*1000 + col` baked into the fake cube) are
built to catch exactly that class of bug -- a shape or row-count check cannot.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.extract_mrrsu_features import extract_features, _smooth_nanmean


class FakeCube:
    """Cube whose value encodes its own (row, col) so misalignment is provable."""
    def __init__(self, h=40, w=50):
        self.h, self.w = h, w

    def read(self, tile_id):
        cube = np.zeros((self.h, self.w, 60), dtype=np.float32)
        rr, cc = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing='ij')
        cube[..., 0] = rr * 1000 + cc          # unique, recoverable fingerprint
        return cube, [f'B{i}' for i in range(60)]


def _df():
    rows = [dict(tile_id='t0001', pixel_row=r, pixel_col=c, split='train')
            for r, c in [(3, 7), (11, 2), (0, 0), (39, 49), (20, 20)]]
    return pd.DataFrame(rows)


def test_rows_align_with_the_input_parquet():
    df = _df()
    out = extract_features(df, {'t0001': 'fake.img'}, reader=FakeCube().read)
    assert len(out) == len(df)
    # B0 encodes row*1000+col; every row must recover ITS OWN coordinates.
    for i, r in df.iterrows():
        assert out['B0'].iloc[i] == r['pixel_row'] * 1000 + r['pixel_col'], (
            f"row {i} got another pixel's parameters")


def test_out_of_bounds_pixel_is_nan_not_wrapped():
    """Negative or oversized indices must not silently wrap to a valid pixel."""
    df = pd.DataFrame([dict(tile_id='t0001', pixel_row=999, pixel_col=0,
                            split='train')])
    out = extract_features(df, {'t0001': 'fake.img'}, reader=FakeCube().read)
    assert np.isnan(out['B0'].iloc[0])


def test_row_order_survives_multiple_tiles_processed_out_of_order():
    """Rows for different tiles are interleaved in the input; grouping by tile
    for I/O efficiency must not change which output row a given input row's
    parameters land in. A groupby-then-concat implementation that resets row
    order (instead of scattering back into the original positions) would pass
    the single-tile test above but fail here."""
    class TwoTileFakeCube:
        def read(self, tile_id):
            h, w = 40, 50
            cube = np.zeros((h, w, 60), dtype=np.float32)
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            # offset second tile's fingerprint so tile identity is also provable
            offset = 0 if tile_id == 't0001' else 100000
            cube[..., 0] = rr * 1000 + cc + offset
            return cube, [f'B{i}' for i in range(60)]

    rows = [
        dict(tile_id='t0001', pixel_row=3, pixel_col=7, split='train'),
        dict(tile_id='t0002', pixel_row=5, pixel_col=9, split='train'),
        dict(tile_id='t0001', pixel_row=11, pixel_col=2, split='train'),
        dict(tile_id='t0002', pixel_row=20, pixel_col=1, split='train'),
        dict(tile_id='t0001', pixel_row=0, pixel_col=0, split='train'),
    ]
    df = pd.DataFrame(rows)
    out = extract_features(df, {'t0001': 'a.img', 't0002': 'b.img'},
                           reader=TwoTileFakeCube().read)
    assert len(out) == len(df)
    for i, r in df.iterrows():
        offset = 0 if r['tile_id'] == 't0001' else 100000
        expected = r['pixel_row'] * 1000 + r['pixel_col'] + offset
        assert out['B0'].iloc[i] == expected, (
            f"row {i} (tile {r['tile_id']}) got another row's parameters")


def test_feature_columns_are_named_by_real_parameter_not_positional():
    """Downstream tasks index features by name (OLINDEX3, BD1300, ...) and never
    decode a header themselves -- columns must carry the real names the reader
    returns, not p0..p59."""
    df = _df()
    out = extract_features(df, {'t0001': 'fake.img'}, reader=FakeCube().read)
    assert list(out.columns) == [f'B{i}' for i in range(60)]
    assert 'p0' not in out.columns


def test_divergent_band_order_across_tiles_raises():
    """A tile whose header lists bands in a different order would otherwise
    silently shift every parameter for that tile alone. This never happens
    among the 1,764 real mrrsu headers on this machine (verified: exactly one
    band order), but the check is cheap and must fire if a divergent product
    ever appears."""
    class MismatchedCube:
        def read(self, tile_id):
            cube = np.zeros((10, 10, 60), dtype=np.float32)
            if tile_id == 't0001':
                names = [f'B{i}' for i in range(60)]
            else:
                names = [f'B{i}' for i in range(59, -1, -1)]  # reversed order
            return cube, names

    df = pd.DataFrame([
        dict(tile_id='t0001', pixel_row=0, pixel_col=0, split='train'),
        dict(tile_id='t0002', pixel_row=0, pixel_col=0, split='train'),
    ])
    with pytest.raises(ValueError, match='band order differs'):
        extract_features(df, {'t0001': 'a.img', 't0002': 'b.img'},
                         reader=MismatchedCube().read)


def test_smooth_mean_ignores_nan_neighbours_instead_of_propagating():
    """A single nodata pixel must not blank its whole 7x7 neighbourhood: the
    smoothed mean has to be computed over the VALID neighbours only, not a
    plain mean that treats NaN as poisoning the whole window (a plain mean
    over a window containing any NaN is itself NaN, or corrupts the sum if
    NaNs were zero-filled without correcting the denominator)."""
    cube = np.full((9, 9, 1), 2.0, dtype=np.float32)
    cube[4, 4, 0] = np.nan   # single nodata pixel at the center
    out = _smooth_nanmean(cube, size=7)
    # Every value away from the border is a window of 2.0s with one NaN
    # excluded -- the nan-aware mean must still be exactly 2.0, not NaN and
    # not skewed by treating the missing value as 0.
    assert np.isclose(out[4, 4, 0], 2.0)
    assert not np.isnan(out[4, 4, 0])


def test_smooth_true_produces_different_values_than_smooth_false():
    """--smooth must actually change the extracted values (it exists because
    RPEAK1 is a regional discriminant per data/mrrsu_aux.py) -- if smoothing
    silently no-ops, --smooth is dead weight."""
    class GradientCube:
        def read(self, tile_id):
            h, w = 20, 20
            cube = np.zeros((h, w, 60), dtype=np.float32)
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            cube[..., 0] = (rr + cc).astype(np.float32)  # spatial gradient
            return cube, [f'B{i}' for i in range(60)]

    df = pd.DataFrame([dict(tile_id='t0001', pixel_row=10, pixel_col=10,
                            split='train')])
    raw = extract_features(df, {'t0001': 'x.img'}, smooth=False,
                           reader=GradientCube().read)
    smoothed = extract_features(df, {'t0001': 'x.img'}, smooth=True,
                                reader=GradientCube().read)
    assert raw['B0'].iloc[0] == 20.0          # 10 + 10, unsmoothed
    assert smoothed['B0'].iloc[0] == 20.0      # mean of a symmetric window is unchanged
    # Use an off-center pixel where the gradient makes smoothing visible.
    df2 = pd.DataFrame([dict(tile_id='t0001', pixel_row=1, pixel_col=1,
                             split='train')])
    raw2 = extract_features(df2, {'t0001': 'x.img'}, smooth=False,
                            reader=GradientCube().read)
    smoothed2 = extract_features(df2, {'t0001': 'x.img'}, smooth=True,
                                 reader=GradientCube().read)
    assert raw2['B0'].iloc[0] != smoothed2['B0'].iloc[0]
