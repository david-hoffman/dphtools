"""Approved coordinate standard errors imply inverse-variance drift weights."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils.beads import calc_drift


def _calculate_unchanged(tracks, **options):
    originals = [track.copy(deep=True) for track in tracks]
    try:
        result = calc_drift(tracks, weighted="coords", **options)
    finally:
        for track, original in zip(tracks, originals):
            assert_frame_equal(track, original)
    assert isinstance(result, pd.DataFrame)
    return result


@pytest.mark.parametrize("diagnostics", [False, True])
def test_coordinate_standard_errors_give_inverse_variance_weights(diagnostics):
    tracks = [
        pd.DataFrame(
            {
                "slice": [2, 6],
                "x0": [0.0, 2.0],
                "y0": [0.0, -4.0],
                "sigma_x": [1.0, 1.0],
                "sigma_y": [2.0, 2.0],
            }
        ),
        pd.DataFrame(
            {
                "slice": [2, 6],
                "x0": [0.0, 10.0],
                "y0": [0.0, 8.0],
                "sigma_x": [2.0, 2.0],
                "sigma_y": [1.0, 1.0],
            }
        ),
    ]
    try:
        result = _calculate_unchanged(tracks, diagnostics=diagnostics)
        # x: (-1 - 5/4)/(1 + 1/4) = -1.8; y uses the reversed weights.
        assert_array_equal(result.index, [2, 6])
        assert_allclose(result[["x0", "y0"]], [[-1.8, -2.8], [1.8, 2.8]], rtol=0, atol=1e-12)
        if diagnostics:
            assert plt.get_fignums()
            for number in plt.get_fignums():
                plt.figure(number).canvas.draw()
    finally:
        plt.close("all")


def test_observation_dependent_errors_missing_frames_and_two_arithmetic_centerings():
    tracks = [
        pd.DataFrame(
            {
                "slice": [2, 4, 8],
                "x0": [10.0, 12.0, 17.0],
                "y0": [5.0, 11.0, 8.0],
                "sigma_x": [1.0, 2.0, 1.0],
                "sigma_y": [2.0, 1.0, 3.0],
            }
        ),
        pd.DataFrame(
            {
                "slice": [2, 8],
                "x0": [-5.0, 5.0],
                "y0": [20.0, 24.0],
                "sigma_x": [2.0, 1.0],
                "sigma_y": [1.0, 2.0],
            }
        ),
        pd.DataFrame(
            {
                "slice": [4, 8],
                "x0": [100.0, 106.0],
                "y0": [-10.0, -4.0],
                "sigma_x": [1.0, 3.0],
                "sigma_y": [2.0, 1.0],
            }
        ),
    ]
    # Center each track using its own arithmetic mean. Before final centering,
    # x=(-17/5, -13/5, 84/19), y=(-11/5, 9/5, 18/7).
    # Their arithmetic means are -10/19 and 76/105, respectively.
    expected = np.array([[-273 / 95, -307 / 105], [-197 / 95, 113 / 105], [94 / 19, 194 / 105]])
    result = _calculate_unchanged(tracks)
    assert_array_equal(result.index, [2, 4, 8])
    assert_allclose(result[["x0", "y0"]], expected, rtol=0, atol=1e-12)
    assert_allclose(result[["x0", "y0"]].mean(), [0, 0], rtol=0, atol=1e-12)

    # Changing coordinate units must also scale their standard errors.
    # Arbitrary static positions still disappear at each track's first mean.
    converted = [track.copy(deep=True) for track in tracks]
    for index, track in enumerate(converted):
        track["x0"] = 3 * track["x0"] + 20 * index
        track["y0"] = 0.5 * track["y0"] - 7 * index
        track["sigma_x"] *= 3
        track["sigma_y"] *= 0.5
    transformed = _calculate_unchanged(converted)
    assert_array_equal(transformed.index, [2, 4, 8])
    assert_allclose(transformed[["x0", "y0"]], expected * [3, 0.5], rtol=0, atol=1e-12)
