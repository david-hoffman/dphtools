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
