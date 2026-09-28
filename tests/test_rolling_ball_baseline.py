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


def test_circumcircle_selects_indexed_scalene_triangle():
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]])
    points = np.concatenate((triangle, triangle + [10, -7]))
    center, radius = rolling_ball.circumcircle(points, np.array([3, 4, 5]))
    # (2,1) is sqrt(5) from each original vertex. Select the translated triangle.
    assert_allclose(center, [12, -6], atol=1e-12)
    assert_allclose(radius, np.sqrt(5), atol=1e-12)


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_circumcircle_equilateral_geometry_is_independent_of_vertex_order(order):
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3)]])
    center, radius = rolling_ball.circumcircle(points, np.array(order))
    assert_allclose(center, [1, 1 / np.sqrt(3)], atol=1e-12)
    assert_allclose(radius, 2 / np.sqrt(3), atol=1e-12)


@pytest.mark.parametrize("mode", ["reflect", "wrap", "nearest"])
def test_rolling_ball_top_and_bottom_are_dual_under_sign_inversion(mode):
    data = np.array([0.0, 1.0, 4.0, -2.0, 0.5, 3.0, 1.0])
    bottom_residual, bottom_background = rolling_ball.rolling_ball_filter(data, 2, mode=mode)
    top_residual, top_background = rolling_ball.rolling_ball_filter(-data, 2, top=True, mode=mode)
    assert_allclose(top_residual, -bottom_residual, atol=1e-12)
    assert_allclose(top_background, -bottom_background, atol=1e-12)


@pytest.mark.parametrize("top", [False, True])
def test_rolling_ball_anisotropic_spacing_commutes_with_axis_permutation(top):
    data = np.array([[0.0, 1.0, 4.0, 1.0], [2.0, -1.0, 3.0, 0.0], [1.0, 5.0, 0.0, 2.0]])
    residual, background = rolling_ball.rolling_ball_filter(data, 3, spacing=(1, 2), top=top)
    transposed_residual, transposed_background = rolling_ball.rolling_ball_filter(
        data.T, 3, spacing=(2, 1), top=top
    )
    assert_allclose(transposed_background, background.T, atol=1e-12)
    assert_allclose(transposed_residual, residual.T, atol=1e-12)


def test_rolling_ball_scalar_spacing_matches_isotropic_vector():
    data = np.arange(35.0).reshape(5, 7) % 8
    scalar = rolling_ball.rolling_ball_filter(data, 3, spacing=2)
    vector = rolling_ball.rolling_ball_filter(data, 3, spacing=(2, 2))
    assert_allclose(scalar, vector, atol=1e-12)


def accurate_filter_coordinates(data, **kwargs):
    """Allow either the documented Nx2 array or the observed coordinate-array pair."""
    result = rolling_ball.rolling_ball_filter_accurate(
        data, 2.0, bounds_error=False, fill_value="extrapolate", **kwargs
    )
    return np.column_stack(result) if isinstance(result, tuple) else np.asarray(result)


@pytest.mark.parametrize("top", [False, True])
def test_accurate_rolling_geometry_preserves_side_and_coordinate_translation(top):
    data = np.column_stack((np.arange(7.0), [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]))
    result = accurate_filter_coordinates(data, top=top)
    assert result.shape == data.shape
    assert_allclose(result[:, 0], data[:, 0], atol=1e-12)
    assert np.isfinite(result).all()
    if top:
        assert np.all(result[:, 1] >= data[:, 1] - 1e-12)
    else:
        assert np.all(result[:, 1] <= data[:, 1] + 1e-12)
    # Rigid translation cannot change ball geometry. Extrapolation is explicitly
    # requested so the test does not choose an unspecified endpoint policy.
    translation = np.array([10.0, -3.0])
    shifted = accurate_filter_coordinates(data + translation, top=top)
    assert_allclose(shifted, result + translation, atol=1e-12)


def test_accurate_rolling_top_bottom_duality_under_vertical_reflection():
    data = np.column_stack((np.arange(7.0), [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]))
    reflection = np.array([1.0, -1.0])
    bottom = accurate_filter_coordinates(data, top=False)
    reflected_top = accurate_filter_coordinates(data * reflection, top=True)
    assert_allclose(reflected_top, bottom * reflection, atol=1e-12)


@pytest.mark.parametrize("top", [False, True])
def test_rolling_ball_background_is_idempotent_under_the_same_morphological_filter(top):
    data = np.array([0.0, 1.0, 4.0, -2.0, 0.5, 3.0, 1.0, 0.0, 2.0])
    _, background = rolling_ball.rolling_ball_filter(data, 3, top=top, mode="wrap")
    residual, repeated_background = rolling_ball.rolling_ball_filter(
        background, 3, top=top, mode="wrap"
    )
    # Both morphological opening and closing are idempotent.
    assert_allclose(repeated_background, background, atol=1e-12)
    assert_allclose(residual, 0, atol=1e-12)


@pytest.mark.parametrize("top", [False, True])
def test_rolling_ball_preserves_a_linear_ramp_away_from_edges(top):
    data = 3 + 0.1 * np.arange(31.0)
    residual, background = rolling_ball.rolling_ball_filter(data, 3, top=top)
    # Erosion and dilation add opposite constants to an affine ramp. On the
    # interior, their composition restores it for any fixed structuring element.
    assert_allclose(background[8:-8], data[8:-8], atol=1e-12)
    assert_allclose(residual[8:-8], 0, atol=1e-12)


def test_rolling_ball_module_entry_point_completes_under_agg(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    # Execute the real __main__ path in a fresh Python process. Diagnostics are
    # installed before product imports; no production arguments are changed.
    bootstrap = r"""
import warnings
def format_warning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"
warnings.formatwarning = format_warning
import sys
sys.excepthook = lambda category, message, tb: print(f"{category.__name__}: {message}", file=sys.stderr)
warning_count = 0
def show_warning(message, category, filename, lineno, file=None, line=None):
    global warning_count
    warning_count += 1
    (file or sys.stderr).write(format_warning(message, category, filename, lineno))
warnings.showwarning = show_warning
import runpy
try:
    runpy.run_module("dphtools.utils.rolling_ball", run_name="__main__", alter_sys=True)
finally:
    print(f"ROLLING_BALL_SMOKE_WARNINGS={warning_count}", flush=True)
"""
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(completed.stdout, end="")
    assert completed.returncode == 0, completed.stdout
    # No generated values, random trajectory, or elapsed time are asserted.
