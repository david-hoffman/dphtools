"""SETUP-001 L2/L5: elementary geometry and morphological background invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils import rolling_ball


@pytest.mark.parametrize(
    "vector, expected", [([3.0, 4.0], 25), ([-2.0, 1.0, 2.0], 9), ([0.0, 0.0], 0)]
)
def test_squared_euclidean_norm(vector, expected):
    assert_allclose(rolling_ball.sq_norm(np.array(vector)), expected)


@pytest.mark.parametrize("offset", [(0.0, 0.0), (10.0, -7.0)])
def test_circumcircle_right_triangle(offset):
    points = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]) + offset
    center, radius = rolling_ball.circumcircle(points, np.array([0, 1, 2]))
    assert_allclose(center, np.array([2.0, 1.5]) + offset, atol=1e-12)
    assert_allclose(radius, 2.5, atol=1e-12)


@pytest.mark.parametrize("top", [False, True])
@pytest.mark.parametrize("shape, spacing", [((9,), None), ((7, 9), 1), ((5, 7, 9), (1, 2, 1))])
def test_rolling_ball_constant_background(shape, spacing, top):
    data = np.full(shape, 3.0)
    residual, background = rolling_ball.rolling_ball_filter(data, 2, spacing=spacing, top=top)
    assert residual.shape == background.shape == data.shape
    assert_allclose(background, data, atol=1e-12)
    assert_allclose(residual, 0, atol=1e-12)


@pytest.mark.parametrize("top", [False, True])
def test_rolling_ball_background_and_residual_reconstruct_input(top):
    data = np.full((9, 9), 3.0)
    data[4, 4] = -4 if top else 10
    residual, background = rolling_ball.rolling_ball_filter(data, 2, top=top, mode="reflect")
    assert_allclose(residual + background, data, atol=1e-12)
    assert residual[4, 4] != 0
    # Opening lies below the image; closing lies above it.
    if top:
        assert np.all(background >= data - 1e-12)
    else:
        assert np.all(background <= data + 1e-12)


def test_rolling_ball_vertical_offset_only_changes_background():
    data = np.zeros((9, 9))
    data[4, 4] = 8
    residual, background = rolling_ball.rolling_ball_filter(data, 2)
    shifted_residual, shifted_background = rolling_ball.rolling_ball_filter(data + 7, 2)
    assert_allclose(shifted_residual, residual, atol=1e-12)
    assert_allclose(shifted_background, background + 7, atol=1e-12)
