"""SETUP-001 L2/L3/L5: supplemental exponential and count-model baselines."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils import fitfuncs


@pytest.mark.parametrize("amplitude, rate, offset", [(3, 0.5, 2), (-3, 0.5, -2), (2, 0, 4)])
def test_exponent_documented_formula(amplitude, rate, offset):
    x = np.array([0.0, 1.0, 2.0])
    expected = amplitude * np.exp(-rate * x) + offset
    assert_allclose(fitfuncs.exponent(x, amplitude, rate, offset), expected)


@pytest.mark.parametrize("bias", [(), (4.0,)])
def test_multi_exp_documented_sum(bias):
    x = np.array([0.0, 0.5, 2.0])
    expected = 3 * np.exp(-2 * x) - np.exp(-0.5 * x) + sum(bias)
    assert_allclose(fitfuncs.multi_exp(x, 3, 2, -1, 0.5, *bias), expected)


@pytest.mark.parametrize("bias", [(), (4.0,)])
def test_multi_exp_jacobian_analytic_derivatives(bias):
    x = np.array([0.0, 0.5, 2.0])
    columns = [np.exp(-2 * x), -3 * x * np.exp(-2 * x), np.exp(-0.5 * x), x * np.exp(-0.5 * x)]
    if bias:
        columns.append(np.ones_like(x))
    assert_allclose(fitfuncs.multi_exp_jac(x, 3, 2, -1, 0.5, *bias), np.column_stack(columns))


@pytest.mark.parametrize("sign", [-1, 1])
def test_exponent_fit_independently_generated_decay(sign):
    x = np.linspace(0, 5, 41)
    data = sign * (6 * np.exp(-0.75 * x) + 2)
    parameters, covariance = fitfuncs.exponent_fit(data, x)
    assert_allclose(parameters, [sign * 6, 0.75, sign * 2], rtol=1e-6, atol=1e-8)
    assert covariance.shape == (3, 3)


@pytest.mark.parametrize("explicit_x", [False, True])
def test_exponent_fit_without_offset(explicit_x):
    x = np.arange(20.0)
    data = 5 * np.exp(-0.2 * x)
    parameters, covariance = fitfuncs.exponent_fit(data, x if explicit_x else None, offset=False)
    assert_allclose(parameters, [5, 0.2], rtol=1e-6, atol=1e-8)
    assert covariance.shape == (2, 2)


@pytest.mark.parametrize("offset", [False, True])
def test_multi_exp_fit_single_component(offset):
    x = np.linspace(0, 5, 41)
    expected = [6, 0.75, 2] if offset else [6, 0.75]
    data = 6 * np.exp(-0.75 * x) + (2 if offset else 0)
    parameters, covariance = fitfuncs.multi_exp_fit(
        data, x, components=1, offset=offset, maxfev=500
    )
    assert_allclose(parameters, expected, rtol=1e-5, atol=1e-7)
    assert covariance.shape == (len(expected), len(expected))


def test_multi_exp_fit_resolves_two_distinct_components():
    x = np.linspace(0, 8, 81)
    data = 4 * np.exp(-0.35 * x) + 2 * np.exp(-1.5 * x) + 0.75
    parameters, covariance = fitfuncs.multi_exp_fit(
        data, x, components=2, offset=True, maxfev=1000
    )
    components = sorted(zip(parameters[:-1:2], parameters[1:-1:2]), key=lambda pair: pair[1])
    assert_allclose(components, [[4, 0.35], [2, 1.5]], rtol=1e-5, atol=1e-7)
    assert_allclose(parameters[-1], 0.75, rtol=1e-5, atol=1e-7)
    assert covariance.shape == (5, 5)


@pytest.mark.parametrize("shape, mean", [(1.0, 2.0), (2.0, 6.0), (0.5, 3.0)])
def test_negative_binomial_uses_requested_mean(shape, mean):
    distribution = fitfuncs.NegBinom(shape, mean)
    assert_allclose(distribution.mean(), mean)


def test_multi_exp_jacobian_at_zero_amplitude_and_zero_rate():
    x = np.array([0.0, 0.5, 2.0])
    # First component has no amplitude: its rate derivative is zero.
    # Second has rate zero: d/dA = 1 and d/dk = -A*x.
    expected = np.column_stack((np.exp(-2 * x), np.zeros(3), np.ones(3), -3 * x, np.ones(3)))
    assert_allclose(fitfuncs.multi_exp_jac(x, 0, 2, 3, 0, 4), expected)


@pytest.mark.parametrize("diagnostics", [False, True])
def test_loglog_power_fit_predicts_an_independent_curve_and_intercept(diagnostics):
    import matplotlib.pyplot as plt

    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    # log(y) = log(6) - 1.5*log(x), so the log-log regression is exact.
    try:
        parameters = fitfuncs.estimate_power_law(x, 6 * x**-1.5, diagnostics=diagnostics)
        unseen = np.array([1.5, 3.0, 7.0])
        assert_allclose(fitfuncs.power_law(unseen, *parameters), 6 * unseen**-1.5, rtol=1e-12)
        # 6*x**(-3/2) = 3/4 gives x=4; no parameter-order convention is invented.
        assert_allclose(fitfuncs.power_intercept(parameters, value=0.75), 4, rtol=1e-12)
        if diagnostics:
            assert plt.get_fignums()
            plt.gcf().canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize("xmin", [1.0, 2.5])
def test_power_percentile_and_its_documented_inverse_agree(xmin):
    x = np.array([1.0, 2.0, 4.0, 8.0])
    parameters = fitfuncs.estimate_power_law(x, 3 * x**-2)
    for percentile in (0.2, 0.5, 0.8):
        value = fitfuncs.power_percentile(percentile, parameters, xmin=xmin)
        assert np.isfinite(value) and value >= xmin
        assert_allclose(
            fitfuncs.power_percentile_inv(value, parameters, xmin=xmin), percentile, atol=1e-12
        )
    # This tests inverse consistency only, not an unspecified distribution estimator.


def test_power_law_jacobian_is_the_derivative_of_the_public_model():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    parameters = np.asarray(fitfuncs.estimate_power_law(x, 6 * x**-1.5))
    jacobian = fitfuncs.power_law_jac(x, *parameters)
    # Independent central differences verify the derivative relation, without
    # guessing the public packet's unspecified power-law parameter conventions.
    step = 1e-5
    columns = []
    for index in range(len(parameters)):
        direction = np.zeros_like(parameters)
        direction[index] = step
        columns.append(
            (
                fitfuncs.power_law(x, *(parameters + direction))
                - fitfuncs.power_law(x, *(parameters - direction))
            )
            / (2 * step)
        )
    assert_allclose(jacobian, np.column_stack(columns), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("offset", [False, True])
def test_multi_exp_fit_default_sample_axis_reconstructs_exact_decay(offset):
    x = np.arange(40.0)
    data = 5 * np.exp(-0.2 * x) + (1.25 if offset else 0)
    parameters, covariance = fitfuncs.multi_exp_fit(data, components=1, offset=offset, maxfev=500)
    paired_parameters = parameters[:-1] if offset else parameters
    predicted = np.zeros_like(x)
    for amplitude, rate in zip(paired_parameters[::2], paired_parameters[1::2]):
        predicted += amplitude * np.exp(-rate * x)
    if offset:
        predicted += parameters[-1]
    assert_allclose(predicted, data, rtol=1e-6, atol=1e-8)
    assert covariance.shape == (len(parameters), len(parameters))


def test_multi_exp_fit_two_components_without_bias_on_shifted_sample_axis():
    x = np.linspace(0.5, 10.5, 81)
    data = 4 * np.exp(-0.3 * x) + 2 * np.exp(-1.4 * x)
    parameters, covariance = fitfuncs.multi_exp_fit(
        data, x, components=2, offset=False, maxfev=1000
    )
    predicted = sum(a * np.exp(-k * x) for a, k in zip(parameters[::2], parameters[1::2]))
    assert_allclose(predicted, data, rtol=1e-5, atol=1e-7)
    components = sorted(zip(parameters[::2], parameters[1::2]), key=lambda pair: pair[1])
    assert_allclose(components, [[4, 0.3], [2, 1.4]], rtol=1e-5, atol=1e-7)
    assert covariance.shape == (4, 4)


def test_multi_exp_jacobian_matches_analytic_growing_component():
    x = np.array([-0.5, 0.0, 1.0])
    expected = np.column_stack((np.exp(0.2 * x), -3 * x * np.exp(0.2 * x), np.ones(3)))
    assert_allclose(fitfuncs.multi_exp(x, 3, -0.2, 4), 3 * np.exp(0.2 * x) + 4)
    assert_allclose(fitfuncs.multi_exp_jac(x, 3, -0.2, 4), expected)


@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("component_option", [{}, {"components": None}])
def test_automatic_exponential_component_selection_remains_unsupported(offset, component_option):
    x = np.arange(40.0)
    data = 5 * np.exp(-0.2 * x) + (1.25 if offset else 0)
    with pytest.raises(NotImplementedError):
        fitfuncs.multi_exp_fit(data, offset=offset, **component_option)


@pytest.mark.parametrize("alpha, xmin, xmax", [(2.5, 3, 20), (1.5, 7, 9)])
def test_discrete_power_law_generator_stays_within_requested_support(alpha, xmin, xmax):
    import random

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(7821)
        np.random.seed(7821)
        samples = np.array(
            [fitfuncs.powerlaw_prng(alpha, xmin=xmin, xmax=xmax) for _ in range(64)]
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    # Integer support and bounds do not select a fitted estimator or tail formula.
    assert samples.shape == (64,)
    assert np.isfinite(samples).all()
    integer_values = np.floor(samples)
    assert_allclose(samples, integer_values, rtol=0, atol=0)
    assert np.all((samples >= xmin) & (samples <= xmax))


def test_power_law_explicit_cutoff_retains_samples_and_updates_clipped_state():
    data = np.array([0.75, 4.5, 1.25, 2.5, 3.25, 6.5, 4.5, 1.5, 9.0, 12.0, 18.0, 25.0])
    model = fitfuncs.PowerLaw(data.copy())
    for cutoff in (2.0, 5.0):
        assert not np.any(data == cutoff)
        model.fit(xmin=cutoff)
        # Compare multisets: clipping preserves duplicates but need not preserve
        # order. Neither cutoff inclusion nor a fitted estimator is selected.
        assert_allclose(np.sort(model.clipped_data), np.sort(data[data > cutoff]), rtol=0, atol=0)
