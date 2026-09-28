"""SETUP-001 M1-M5: supported fits and explicit custom-solver limitations.

The maintenance packet supersedes the copied SciPy option promises. Expected
parameters come from exact linear designs or the packet's Poisson score equation.
Custom covariance is tested only for unweighted linear least squares.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils.lm import curve_fit, lm


def line(x, slope, intercept):
    return slope * x + intercept


def line_jacobian(x, slope, intercept):
    return np.column_stack((x, np.ones_like(x)))


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox", "ls", "mle"])
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


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox"])
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


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox"])
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


@pytest.mark.parametrize("method", [None, "trf", "dogbox"])
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


def test_lm_rejects_missing_analytic_jacobian():
    with pytest.raises(NotImplementedError):
        lm(lambda p: np.array([p[0] - 3, 2 * (p[0] - 3)]), [1.0], maxfev=50)


def test_lm_analytic_jacobian_recovers_known_minimum():
    result, _ = lm(
        lambda p: np.array([p[0] - 3, 2 * (p[0] - 3)]),
        [1.0],
        Dfun=lambda p: np.array([[1.0], [2.0]]),
        maxfev=50,
    )
    assert_allclose(result, [3], atol=1e-7)


@pytest.mark.parametrize("full_output, tuple_length", [(False, 2), (True, 5)])
def test_lm_legacy_flag_and_result_tuple_shape(full_output, tuple_length):
    output = lm(
        lambda p: np.array([p[0] - 3, 2 * (p[0] - 3)]),
        [1.0],
        Dfun=lambda p: np.array([[1.0], [2.0]]),
        col_deriv=True,
        full_output=full_output,
        maxfev=50,
    )
    assert len(output) == tuple_length
    assert_allclose(output[0], [3], atol=1e-7)


def test_lm_rejects_false_col_deriv_flag():
    with pytest.raises(NotImplementedError):
        lm(
            lambda p: np.array([p[0] - 3, 2 * (p[0] - 3)]),
            [1.0],
            Dfun=lambda p: np.array([[1.0], [2.0]]),
            col_deriv=False,
            maxfev=50,
        )


def test_curve_fit_rejects_unknown_pyls_method():
    x = np.arange(5.0)
    with pytest.raises(TypeError):
        curve_fit(line, x, 2 * x + 3, p0=[1, 1], jac=line_jacobian, method="pyls")


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize("sigma", [2.0, np.full(5, 2.0), 4 * np.eye(5)])
def test_custom_curve_fit_rejects_weighting(method, sigma):
    x = np.arange(5.0)
    with pytest.raises(NotImplementedError):
        curve_fit(line, x, 2 * x + 3, p0=[1, 1], jac=line_jacobian, method=method, sigma=sigma)


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize("bounds", [(0, np.inf), (-np.inf, 10), ([0, 0], [10, 10])])
def test_custom_curve_fit_rejects_finite_bounds(method, bounds):
    x = np.arange(5.0)
    with pytest.raises(NotImplementedError):
        curve_fit(line, x, 2 * x + 3, p0=[1, 1], jac=line_jacobian, method=method, bounds=bounds)


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize("jac", [None, "2-point", "3-point"])
def test_custom_curve_fit_rejects_missing_analytic_jacobian(method, jac):
    x = np.arange(5.0)
    with pytest.raises(NotImplementedError):
        curve_fit(line, x, 2 * x + 3, p0=[1, 1], method=method, jac=jac)


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("residual_scale", [1, 2])
def test_custom_ls_covariance_is_unscaled_without_rescaling_option(absolute_sigma, residual_scale):
    x = np.array([-1.0, 0.0, 1.0])
    y = 2 * x + 3 + residual_scale * np.array([1.0, -2.0, 1.0])
    parameters, covariance = curve_fit(
        line,
        x,
        y,
        p0=[1.0, 1.0],
        jac=line_jacobian,
        method="ls",
        absolute_sigma=absolute_sigma,
        maxfev=100,
    )
    assert_allclose(parameters, [2, 3], atol=1e-7)
    # M4: absolute_sigma supplies no custom rescaling capability. J.T @ J
    # is diag(2, 3), regardless of the residual variance or this inherited flag.
    assert_allclose(covariance, [[0.5, 0], [0, 1 / 3]], atol=1e-7)


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox"])
def test_scipy_methods_retain_numerical_derivatives(method):
    x = np.arange(5.0)
    parameters, covariance = curve_fit(line, x, 2 * x + 3, p0=[1, 1], method=method)
    assert_allclose(parameters, [2, 3], atol=1e-7)
    assert covariance.shape == (2, 2)


@pytest.mark.parametrize("method", ["trf", "dogbox"])
@pytest.mark.parametrize("jac", ["2-point", "3-point"])
def test_scipy_methods_retain_numerical_derivative_selectors(method, jac):
    x = np.arange(5.0)
    parameters, covariance = curve_fit(line, x, 2 * x + 3, p0=[1, 1], method=method, jac=jac)
    assert_allclose(parameters, [2, 3], atol=1e-7)
    assert covariance.shape == (2, 2)


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox"])
@pytest.mark.parametrize("matrix_sigma", [False, True])
def test_scipy_weighting_changes_the_fitted_optimum(method, matrix_sigma):
    x = np.array([-1.0, 0.0, 1.0])
    sigma = np.diag([1.0, 4.0, 1.0]) if matrix_sigma else np.array([1.0, 2.0, 1.0])
    parameters, covariance = curve_fit(
        line,
        x,
        np.array([1.0, 0.0, 5.0]),
        p0=[1, 1],
        sigma=sigma,
        absolute_sigma=True,
        method=method,
        jac=line_jacobian,
    )
    # J.T @ W @ J = diag(2, 9/4), J.T @ W @ y = [4, 6].
    assert_allclose(parameters, [2, 8 / 3], atol=1e-7)
    assert_allclose(covariance, [[0.5, 0], [0, 4 / 9]], atol=1e-7)


def exposure_model(exposure, log_rate):
    """An unconstrained parameter with strictly positive model predictions."""
    return np.exp(log_rate) * exposure


def exposure_jacobian(exposure, log_rate):
    return (np.exp(log_rate) * exposure)[:, None]


@pytest.mark.parametrize("counts", [[1, 4, 3, 8], [0, 4, 0, 12], [0, 0, 0, 16]])
@pytest.mark.parametrize("starting_rate", [0.5, 5.0])
def test_poisson_mle_matches_exposure_oracle_including_zero_counts(counts, starting_rate):
    exposure = np.array([1.0, 2.0, 4.0, 8.0])
    counts = np.array(counts, dtype=float)
    parameters, covariance = curve_fit(
        exposure_model,
        exposure,
        counts,
        p0=[np.log(starting_rate)],
        jac=exposure_jacobian,
        method="mle",
        maxfev=1000,
    )
    # D'(a)/2 = sum(exposure) - sum(counts)/a. Each case has total 16,
    # hence a_MLE = 16/15, including the exposures of the zero-count bins.
    assert_allclose(np.exp(parameters), [16 / 15], rtol=1e-5, atol=1e-8)
    # Only the established tuple/matrix shape is claimed for Poisson covariance.
    assert covariance.shape == (1, 1)


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox", "ls"])
@pytest.mark.parametrize("counts, expected", [([1, 4, 3, 8], 1), ([0, 4, 0, 12], 104 / 85)])
def test_least_squares_remains_distinct_from_poisson_mle(method, counts, expected):
    exposure = np.array([1.0, 2.0, 4.0, 8.0])
    parameters, covariance = curve_fit(
        exposure_model,
        exposure,
        np.array(counts, dtype=float),
        p0=[np.log(0.5)],
        jac=exposure_jacobian,
        method=method,
        maxfev=1000,
    )
    # a_LS = dot(exposure, counts)/dot(exposure, exposure), not 16/15.
    assert_allclose(np.exp(parameters), [expected], rtol=1e-5, atol=1e-8)
    assert covariance.shape == (1, 1)


@pytest.mark.parametrize("start", [0.1, 2.0], ids=["overshooting-start", "nearby-start"])
@pytest.mark.parametrize(
    "stopping, expected_status",
    [
        ({"maxfev": 1, "ftol": 0, "xtol": 0}, 5),
        ({"maxfev": 100, "ftol": 1, "xtol": 0}, 1),
        ({"maxfev": 100, "ftol": 0, "xtol": 1e6}, 2),
        ({"maxfev": 100, "gtol": 1e6}, 4),
    ],
    ids=["iteration-limit", "objective-stop", "step-stop", "gradient-stop"],
)
def test_low_level_ls_diagnostics_describe_accepted_point(start, stopping, expected_status):
    calls = []

    def residual(p):
        calls.append(p.copy())
        return np.array([p[0] ** 2 - 1, 2 * (p[0] ** 2 - 1)])

    result, covariance, info, message, status = lm(
        residual,
        [start],
        Dfun=lambda p: np.array([[2 * p[0]], [4 * p[0]]]),
        full_output=True,
        **stopping,
    )
    # Independently evaluate the polynomial and its derivative at returned p.
    # No diagnostic evaluation calls the instrumented callback.
    expected_residual = np.array([result[0] ** 2 - 1, 2 * (result[0] ** 2 - 1)])
    assert_allclose(info["fvec"], expected_residual, atol=1e-12)
    assert_allclose(info["fjac"], [[2 * result[0]], [4 * result[0]]], atol=1e-12)
    assert info["nfev"] == len(calls)
    assert 2.5 * (result[0] ** 2 - 1) ** 2 <= 2.5 * (start**2 - 1) ** 2 + 1e-12
    assert covariance is None
    assert isinstance(message, str) and message
    assert status == expected_status


@pytest.mark.parametrize("starting_rate", [0.5, 20.0])
@pytest.mark.parametrize(
    "stopping, expected_status",
    [
        ({"maxfev": 1, "ftol": 0, "xtol": 0}, 5),
        ({"maxfev": 100, "ftol": 1, "xtol": 0}, 1),
        ({"maxfev": 100, "ftol": 0, "xtol": 1e6}, 2),
        ({"maxfev": 100, "gtol": 1e6}, 4),
    ],
    ids=["iteration-limit", "objective-stop", "step-stop", "gradient-stop"],
)
def test_low_level_poisson_diagnostics_describe_accepted_point(
    starting_rate, stopping, expected_status
):
    exposures = np.array([1.0, 2.0, 4.0, 8.0])
    counts = np.array([0.0, 4.0, 0.0, 12.0])
    calls = []

    def predictions(p):
        calls.append(p.copy())
        return np.exp(p[0]) * exposures, counts

    result, covariance, info, message, status = lm(
        predictions,
        [np.log(starting_rate)],
        Dfun=lambda p: (np.exp(p[0]) * exposures)[:, None],
        method="mle",
        full_output=True,
        **stopping,
    )
    accepted_rate = np.exp(result[0])
    assert_allclose(info["fvec"], accepted_rate * exposures, atol=1e-12)
    assert_allclose(info["fjac"], (accepted_rate * exposures)[:, None], atol=1e-12)
    assert info["nfev"] == len(calls)
    # All terms independent of a cancel in the objective difference:
    # D(a)/2 - D(a0)/2 = 15*(a-a0) - 16*log(a/a0).
    objective_change = 15 * (accepted_rate - starting_rate) - 16 * np.log(
        accepted_rate / starting_rate
    )
    assert objective_change <= 1e-10
    assert covariance is None
    assert isinstance(message, str) and message
    assert status == expected_status


def test_low_level_singular_problem_reports_real_function_call_count():
    calls = []

    def residual(p):
        calls.append(p.copy())
        return np.array([2.0, -1.0])

    result, covariance, info, message, status = lm(
        residual, [3.0], Dfun=lambda p: np.zeros((2, 1)), full_output=True, maxfev=3
    )
    assert_allclose(result, [3])
    assert_allclose(info["fvec"], [2, -1])
    assert_allclose(info["fjac"], np.zeros((2, 1)))
    assert info["nfev"] == len(calls)
    assert covariance is None
    assert isinstance(message, str) and message
    # An accepted zero-change step may meet objective tolerance; a proposed
    # zero step may meet step tolerance, or singular solves may exhaust the limit.
    # Default gtol disables gradient convergence.
    assert status in (1, 2, 5)


@pytest.mark.parametrize("method", [None, "lm", "trf", "dogbox", "ls", "mle"])
def test_curve_fit_full_output_keeps_fit_and_diagnostics_consistent(method):
    x = np.arange(5.0)
    result, covariance, info, message, status = curve_fit(
        line,
        x,
        2 * x + 3,
        p0=[1, 1],
        jac=line_jacobian,
        method=method,
        full_output=True,
        maxfev=100,
    )
    assert_allclose(result, [2, 3], atol=1e-7)
    assert covariance.shape == (2, 2)
    assert isinstance(message, str) and message
    assert status in (1, 2, 3, 4)
    if method in ("trf", "dogbox"):
        assert info is None
    elif method in ("ls", "mle"):
        expected = 2 * x + 3 if method == "mle" else np.zeros_like(x)
        assert_allclose(info["fvec"], expected, atol=1e-7)
        assert_allclose(info["fjac"], np.column_stack((x, np.ones_like(x))), atol=1e-12)
        assert info["nfev"] >= 1


def test_low_level_solver_rejects_unimplemented_extra_argument_forwarding():
    with pytest.raises(NotImplementedError):
        lm(
            lambda p, target: np.array([p[0] - target]),
            [1.0],
            args=(3.0,),
            Dfun=lambda p, target: np.ones((1, 1)),
        )


def test_poisson_invalid_trial_cannot_replace_valid_accepted_point():
    exposures = np.array([1.0, 2.0, 4.0, 8.0])
    counts = np.array([0.0, 4.0, 0.0, 12.0])
    calls = []

    def predictions(p):
        calls.append(p.copy())
        return p[0] * exposures, counts

    result, _, info, _, status = lm(
        predictions,
        [20.0],
        Dfun=lambda p: exposures[:, None],
        method="mle",
        maxfev=1,
        ftol=0,
        xtol=0,
        full_output=True,
    )
    # This unconstrained linear model permits invalid proposals. Returned state
    # must remain valid and correspond to p, regardless of damping choices.
    assert result[0] > 0
    assert_allclose(info["fvec"], result[0] * exposures, atol=1e-12)
    assert_allclose(info["fjac"], exposures[:, None], atol=1e-12)
    assert info["nfev"] == len(calls)
    assert 15 * (result[0] - 20) - 16 * np.log(result[0] / 20) <= 1e-10
    assert status == 5


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_custom_curve_fit_rejects_explicit_unit_weights(method):
    x = np.arange(5.0)
    with pytest.raises(NotImplementedError):
        curve_fit(
            line, x, 2 * x + 3, p0=[1, 1], jac=line_jacobian, method=method, sigma=np.ones(5)
        )


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_custom_curve_fit_rejects_false_legacy_derivative_flag(method):
    x = np.arange(5.0)
    with pytest.raises(NotImplementedError):
        curve_fit(line, x, 2 * x + 3, p0=[1, 1], jac=line_jacobian, method=method, col_deriv=False)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_custom_curve_fit_accepts_array_valued_unbounded_limits(method):
    x = np.arange(5.0)
    result, covariance = curve_fit(
        line,
        x,
        2 * x + 3,
        p0=[1, 1],
        jac=line_jacobian,
        method=method,
        bounds=(np.full(2, -np.inf), np.full(2, np.inf)),
    )
    assert_allclose(result, [2, 3], atol=1e-7)
    assert covariance.shape == (2, 2)


@pytest.mark.parametrize("method", [None, "trf", "dogbox"])
def test_delegated_bounded_full_output_retains_documented_placeholders(method):
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result, covariance, info, message, status = curve_fit(
        line,
        x,
        2 * x + 3,
        p0=[0.5, 2],
        bounds=([0, 0], [1, 5]),
        jac=line_jacobian,
        method=method,
        full_output=True,
    )
    # Centered x leaves the intercept 3; the constrained optimal slope is 1.
    assert_allclose(result, [1, 3], atol=1e-6)
    assert covariance.shape == (2, 2)
    assert info is None
    assert message == "No error"
    assert status == 1
