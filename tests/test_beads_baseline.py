"""SETUP-001 L2/L3: coordinate centering and known shared drift."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils import beads


def test_remove_coord_mean_preserves_coordinate_differences_and_metadata():
    frame = pd.DataFrame(
        {"x0": [2.0, 4.0, 6.0], "y0": [8.0, 4.0, 0.0], "amp": [1, 2, 3]}, index=[2, 5, 9]
    )
    result = beads.remove_coord_mean(frame)
    assert_allclose(result[["x0", "y0"]], [[-2, 4], [0, 0], [2, -4]])
    assert_array_equal(result["amp"], [1, 2, 3])
    assert_array_equal(result.index, [2, 5, 9])


def test_remove_coord_mean_custom_coordinates_and_single_row():
    frame = pd.DataFrame({"u": [7.0], "v": [-3.0], "label": ["fiducial"]})
    result = beads.remove_coord_mean(frame, coords=["u", "v"])
    assert_allclose(result[["u", "v"]], [[0, 0]])
    assert result["label"].tolist() == ["fiducial"]


@pytest.mark.parametrize("diagnostics", [False, True])
def test_calc_drift_identical_centered_tracks(diagnostics):
    tracks = [
        pd.DataFrame(
            {"slice": [0, 1, 2], "x0": [-1.0, 0.0, 1.0], "y0": [2.0, 0.0, -2.0], "amp": amplitude}
        )
        for amplitude in (np.ones(3), np.full(3, 3.0))
    ]
    result = beads.calc_drift(tracks, diagnostics=diagnostics)
    assert_allclose(result[["x0", "y0"]], [[-1, 2], [0, 0], [1, -2]])
    assert_array_equal(result.index, [0, 1, 2])
    if diagnostics:
        import matplotlib.pyplot as plt

        plt.close("all")


def test_calc_drift_amplitude_weighted_mean():
    # Both tracks are centered so their absolute-position datum is immaterial.
    tracks = [
        pd.DataFrame(
            {"slice": [0, 1, 2], "x0": [-1.0, 0.0, 1.0], "y0": [2.0, 0.0, -2.0], "amp": [1.0] * 3}
        ),
        pd.DataFrame(
            {"slice": [0, 1, 2], "x0": [-3.0, 0.0, 3.0], "y0": [4.0, 0.0, -4.0], "amp": [3.0] * 3}
        ),
    ]
    result = beads.calc_drift(tracks, weighted="amp")
    assert_allclose(result[["x0", "y0"]], [[-2.5, 3.5], [0, 0], [2.5, -3.5]])


def test_calc_drift_custom_coordinate_and_frame_names():
    track = pd.DataFrame(
        {"frame": [2, 4, 6], "u": [-2.0, 0.0, 2.0], "v": [1.0, 0.0, -1.0], "amp": [1.0] * 3}
    )
    result = beads.calc_drift([track], coords=["u", "v"], frame_name="frame")
    assert_allclose(result[["u", "v"]], [[-2, 1], [0, 0], [2, -1]])
    assert_array_equal(result.index, [2, 4, 6])


@pytest.mark.parametrize(
    "coords, frame_name",
    [(["x0", "y0"], "frame"), (["u", "v"], "slice"), (["u", "v"], "frame")],
)
def test_calc_drift_named_columns_preserve_weighting_and_remove_track_offsets(coords, frame_name):
    tracks = [
        pd.DataFrame(
            {
                frame_name: [2, 4, 6],
                coords[0]: [9.0, 10.0, 11.0],
                coords[1]: [8.0, 6.0, 4.0],
                "amp": [1.0] * 3,
            }
        ),
        pd.DataFrame(
            {
                frame_name: [2, 4, 6],
                coords[0]: [-23.0, -20.0, -17.0],
                coords[1]: [9.0, 5.0, 1.0],
                "amp": [3.0] * 3,
            }
        ),
    ]
    # Centering removes the static bead positions. Weights 1:3 give the same
    # established amplitude-weighted drift, independently of column spelling.
    result = beads.calc_drift(tracks, coords=coords, frame_name=frame_name, weighted="amp")
    assert_allclose(result[coords], [[-2.5, 3.5], [0, 0], [2.5, -3.5]])
    assert_array_equal(result.index, [2, 4, 6])


@pytest.mark.parametrize("coords, frame_name", [(["x0", "y0"], "slice"), (["u", "v"], "frame")])
def test_calc_drift_empty_string_selects_unweighted_mean(coords, frame_name):
    tracks = [
        pd.DataFrame(
            {
                frame_name: [2, 4, 6],
                coords[0]: [-1.0, 0.0, 1.0],
                coords[1]: [2.0, 0.0, -2.0],
                "amp": [1.0] * 3,
            }
        ),
        pd.DataFrame(
            {
                frame_name: [2, 4, 6],
                coords[0]: [-3.0, 0.0, 3.0],
                coords[1]: [4.0, 0.0, -4.0],
                "amp": [3.0] * 3,
            }
        ),
    ]
    # G2: weighted="" gives both tracks equal weight, despite their amplitudes.
    # At frame 2, x=(-1-3)/2=-2 and y=(2+4)/2=3; both tracks are centered.
    result = beads.calc_drift(tracks, coords=coords, frame_name=frame_name, weighted="")
    assert_allclose(result[coords], [[-2, 3], [0, 0], [2, -3]])
    assert_array_equal(result.index, [2, 4, 6])
