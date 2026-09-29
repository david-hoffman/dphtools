"""Paired power-law regression components plus an optional additive constant."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import power_law, power_law_jac


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ((3.0, 2.0, 5.0), [8.0, 5.75]),
        ((-2.5,), [-2.5, -2.5]),
        ((3.0, 2.0, -2.0, 1.0, 5.0), [6.0, 4.75]),
        ((3.0, 2.0, -2.0, 1.0), [1.0, -0.25]),
        ((3.0, 2.0, 0.0), [3.0, 0.75]),
    ],
    ids=["approved-offset", "constant-only", "two-pairs-offset", "pairs-only", "zero-offset"],
)
def test_power_law_offset_and_pairs_have_independent_values(parameters, expected):
    x = np.array([1.0, 2.0])
    original = x.copy()
    try:
        actual = power_law(x, *parameters)
    finally:
        assert_array_equal(x, original)
    assert actual.shape == x.shape
    assert_allclose(actual, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("constant", [(), (5.0,)])
def test_power_law_jacobian_has_analytic_component_and_offset_derivatives(constant):
    x = np.array([0.5, 1.0, 2.0, 4.0])
    original = x.copy()
    # d(A*x^-alpha)/dA=x^-alpha; d/dalpha=-A*log(x)*x^-alpha.
    rows = [
        [
            value**-2,
            -3 * math.log(value) * value**-2,
            value**-0.5,
            2 * math.log(value) * value**-0.5,
        ]
        + ([1.0] if constant else [])
        for value in x
    ]
    try:
        actual = power_law_jac(x, 3.0, 2.0, -2.0, 0.5, *constant)
    finally:
        assert_array_equal(x, original)
    assert actual.shape == (x.size, 4 + len(constant))
    assert_allclose(actual, rows, rtol=1e-12, atol=1e-13)


def test_lone_constant_jacobian_is_a_column_of_ones():
    x = np.array([0.5, 1.0, 3.0])
    original = x.copy()
    try:
        actual = power_law_jac(x, -2.5)
    finally:
        assert_array_equal(x, original)
    assert actual.shape == (x.size, 1)
    assert_allclose(actual, np.ones((x.size, 1)), rtol=0, atol=0)
