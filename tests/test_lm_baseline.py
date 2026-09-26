"""SETUP-001 L2/L3/L5: public least-squares contracts, including documented errors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils.lm import curve_fit, lm


def line(x, slope, intercept):
    return slope * x + intercept


def line_jacobian(x, slope, intercept):
    return np.column_stack((x, np.ones_like(x)))


@pytest.mark.parametrize("method", ["lm", "trf", "dogbox", "ls", "mle", "pyls"])
def test_curve_fit_documented_methods_recover_exact_line(method):
    x = np.arange(5.0)
    parameters, covariance = curve_fit(
        line, x, 2 * x + 3, p0=[1.0, 1.0], jac=line_jacobian, method=method, maxfev=100
    )
    assert_allclose(parameters, [2, 3], rtol=1e-6, atol=1e-8)
    assert covariance.shape == (2, 2)


def test_curve_fit_infers_parameter_count_and_starting_point():
    x = np.arange(5.0)
    parameters, _ = curve_fit(line, x, 2 * x + 3, maxfev=100)
    assert_allclose(parameters, [2, 3], atol=1e-8)


@pytest.mark.parametrize("method", ["lm", "ls"])
@pytest.mark.parametrize("matrix_sigma", [False, True])
def test_curve_fit_absolute_covariance_for_known_linear_design(method, matrix_sigma):
    x = np.array([-1.0, 0.0, 1.0])
    sigma = 4 * np.eye(3) if matrix_sigma else np.full(3, 2.0)
    parameters, covariance = curve_fit(
        line,
        x,
        2 * x + 3,
        p0=[1.0, 1.0],
        sigma=sigma,
        absolute_sigma=True,
        jac=line_jacobian,
        method=method,
        maxfev=100,
    )
    # inv(J.T @ J) * sigma**2; centered x makes the cross term zero.
    assert_allclose(parameters, [2, 3], atol=1e-7)
    assert_allclose(covariance, [[2, 0], [0, 4 / 3]], atol=1e-7)


@pytest.mark.parametrize("method", ["lm", "ls"])
def test_curve_fit_relative_covariance_scales_with_residual_variance(method):
    x = np.array([-1.0, 0.0, 1.0])
    # Residual [1, -2, 1] is orthogonal to both design columns.
    y = 2 * x + 3 + np.array([1.0, -2.0, 1.0])
    parameters, covariance = curve_fit(
        line,
        x,
        y,
        p0=[1.0, 1.0],
        jac=line_jacobian,
        method=method,
        absolute_sigma=False,
        maxfev=100,
    )
    assert_allclose(parameters, [2, 3], atol=1e-7)
    # Residual sum of squares 6, degrees of freedom 3-2 = 1.
    assert_allclose(covariance, [[3, 0], [0, 2]], atol=1e-7)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", ["x", "y"])
def test_curve_fit_rejects_nonfinite_input_by_default(bad, location):
    x = np.arange(5.0)
    y = 2 * x + 3
    (x if location == "x" else y)[2] = bad
    with pytest.raises(ValueError):
        curve_fit(line, x, y, p0=[1.0, 1.0], maxfev=100)


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_curve_fit_respects_parameter_bounds(method):
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    parameters, _ = curve_fit(
        line,
        x,
        2 * x + 3,
        p0=[0.5, 2.0],
        bounds=([0, 0], [1, 5]),
        jac=line_jacobian,
        method=method,
        maxfev=100,
    )
    assert_allclose(parameters, [1, 3], atol=1e-6)


def test_curve_fit_lm_rejects_underdetermined_problem():
    with pytest.raises(TypeError):
        curve_fit(line, np.array([1.0]), np.array([2.0]), p0=[1, 1], method="lm", maxfev=20)


def test_lm_documented_finite_difference_jacobian():
    result, _ = lm(lambda p: np.array([p[0] - 3, 2 * (p[0] - 3)]), [1.0], maxfev=50)
    assert_allclose(result, [3], atol=1e-7)


def test_lm_analytic_jacobian_recovers_known_minimum():
    result, _ = lm(
        lambda p: np.array([p[0] - 3, 2 * (p[0] - 3)]),
        [1.0],
        Dfun=lambda p: np.array([[1.0], [2.0]]),
        maxfev=50,
    )
    assert_allclose(result, [3], atol=1e-7)


def test_lm_analytic_jacobian_with_derivatives_across_rows():
    result, _ = lm(
        lambda p, target: np.array([p[0] - target, 2 * (p[0] - target)]),
        [1.0],
        args=(3.0,),
        Dfun=lambda p, target: np.array([[1.0], [2.0]]),
        col_deriv=False,
        maxfev=50,
    )
    assert_allclose(result, [3], atol=1e-7)
