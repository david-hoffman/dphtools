"""Owner-approved active, right-handed xyz angle contract; radians throughout."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools import utils

# Float64 matrices have unit-scale entries. This allows accumulated trig and
# matrix arithmetic roundoff, while resolving errors far below a microradian.
ATOL = 2e-12


def rotation_xyz(x, y, z):
    """Construct elementary active column-vector rotations, without extraction."""
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def assert_principal_rotation(angles, matrix):
    """Check the tuple API, principal ranges and the represented rotation."""
    assert isinstance(angles, tuple)
    assert len(angles) == 3
    values = np.asarray(angles)
    assert values.shape == (3,)
    assert np.isfinite(values).all()
    assert np.all(np.abs(values) <= np.array([np.pi, np.pi / 2, np.pi]) + ATOL)
    assert_allclose(rotation_xyz(*angles), matrix, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "expected",
    [
        (0, 0, 0),
        (np.pi / 2, 0, 0),
        (-np.pi / 2, 0, 0),
        (0, np.pi / 3, 0),
        (0, -np.pi / 3, 0),
        (0, 0, np.pi / 2),
        (0, 0, -np.pi / 2),
        (0.37, -0.61, 1.12),
        (-2.4, 0.7, 2.2),
        (2.5, -1.0, -2.7),
    ],
    ids=["identity", "x+", "x-", "y+", "y-", "z+", "z-", "mixed", "q2", "q3"],
)
def test_calc_angles_recovers_ordinary_principal_angles(expected):
    matrix = rotation_xyz(*expected)
    original = matrix.copy()
    angles = utils.calc_angles(matrix)
    assert_array_equal(matrix, original)
    assert_principal_rotation(angles, original)
    assert_allclose(angles, expected, rtol=0, atol=ATOL)


def test_positive_z_quarter_turn_maps_x_to_y_with_original_keyword_api():
    matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    original = matrix.copy()
    angles = utils.calc_angles(mat_b=matrix)
    assert_array_equal(matrix, original)
    assert_principal_rotation(angles, original)
    assert_allclose(angles, (0, 0, np.pi / 2), rtol=0, atol=ATOL)
    assert_allclose(rotation_xyz(*angles) @ [1, 0, 0], [0, 1, 0], rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "seed, expected",
    [
        ((0.2, 2.1, -0.4), (0.2 - np.pi, np.pi - 2.1, np.pi - 0.4)),
        ((-0.3, -2.0, 0.5), (np.pi - 0.3, 2.0 - np.pi, 0.5 - np.pi)),
    ],
    ids=["positive-y-second-chart", "negative-y-second-chart"],
)
def test_calc_angles_selects_principal_chart(seed, expected):
    # Shifting x and z by pi and reflecting y gives the same rotation.
    matrix = rotation_xyz(*seed)
    original = matrix.copy()
    angles = utils.calc_angles(matrix)
    assert_array_equal(matrix, original)
    assert_principal_rotation(angles, original)
    assert_allclose(angles, expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "seed",
    [(np.pi, 0, 0), (-np.pi, 0, 0), (0, 0, np.pi), (0, 0, -np.pi), (np.pi, 0.4, np.pi)],
    ids=["x+pi", "x-pi", "z+pi", "z-pi", "both-pi"],
)
def test_calc_angles_allows_equivalent_principal_endpoints(seed):
    matrix = rotation_xyz(*seed)
    original = matrix.copy()
    angles = utils.calc_angles(matrix)
    assert_array_equal(matrix, original)
    assert_principal_rotation(angles, original)


@pytest.mark.parametrize(
    "sign, x, z, expected_x",
    [
        (1, 0, 0, 0),
        (1, 0.7, -0.4, 1.1),
        (1, -2.2, 2.1, 2 * np.pi - 4.3),
        (1, 2.4, -2.0, 4.4 - 2 * np.pi),
        (-1, 0, 0, 0),
        (-1, 0.7, -0.4, 0.3),
        (-1, -2.2, -2.1, 2 * np.pi - 4.3),
        (-1, 2.4, 2.0, 4.4 - 2 * np.pi),
    ],
    ids=["+zero", "+mixed", "+wrap-up", "+wrap-down", "-zero", "-mixed", "-wrap-up", "-wrap-down"],
)
def test_calc_angles_exact_gimbal_lock_warns_and_sets_z_zero(sign, x, z, expected_x):
    # cos(y)=0 and sin(y)=sign exactly: no near-lock threshold is exercised.
    ry = np.array([[0.0, 0.0, sign], [0.0, 1.0, 0.0], [-sign, 0.0, 0.0]])
    matrix = rotation_xyz(0, 0, z) @ ry @ rotation_xyz(x, 0, 0)
    original = matrix.copy()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            angles = utils.calc_angles(matrix)
    finally:
        # Retain category/message/location even if the call or assertions fail.
        for warning in caught:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    assert_array_equal(matrix, original)
    assert_principal_rotation(angles, original)
    # At +pi/2 only x-z survives; at -pi/2 only x+z survives.
    assert_allclose(angles, (expected_x, sign * np.pi / 2, 0), rtol=0, atol=ATOL)
    assert caught, "An exact gimbal lock must emit a warning."
