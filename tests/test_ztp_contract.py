"""Approved zero-truncated Poisson estimator, validation, and boundary contract."""

import inspect
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import fit_ztp

# Independent 60-digit Decimal bisection of lambda/(1-exp(-lambda)) = mean.
# This is the conditional model, not the ordinary Poisson sample-mean estimate.
RATE_CASES = [
    ([1] * 9 + [2], 0.1937475579949905),
    ([1, 1, 1, 2], 0.46421275437881665),
    ([1, 1, 2, 4], 1.5936242600400401),
    ([2, 3, 4], 2.821439372122079),
    ([2, 5, 8], 4.965114231744276),
    ([17, 20, 23], 19.999999958776926),
]


def _fit_unchanged(data):
    original = data.copy()
    try:
        return fit_ztp(data)
    finally:
        assert_array_equal(data, original)


def _assert_rate(rate, expected, observations):
    assert np.asarray(rate).shape == ()
    assert np.isrealobj(rate)
    assert np.isfinite(rate) and rate > 0
    # About six relative decimal places allows numerical optimization and
    # float32 sample-mean rounding, while separating the conditional estimator.
    assert_allclose(rate, expected, rtol=2e-6, atol=2e-8)
    mean = math.fsum(float(x) for x in observations) / len(observations)
    implied_mean = float(rate) / -math.expm1(-float(rate))
    assert_allclose(implied_mean, mean, rtol=2e-6, atol=2e-8)


def test_fit_ztp_preserves_public_signature():
    signature = inspect.signature(fit_ztp)
    assert tuple(signature.parameters) == ("data",)
    assert signature.parameters["data"].default is inspect.Parameter.empty
    assert signature.parameters["data"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize(
    "values, expected",
    RATE_CASES,
    ids=["mean-1.1", "mean-1.25", "approved-mean-2", "mean-3", "mean-5", "mean-20"],
)
def test_conditional_maximum_likelihood_rate(values, expected, dtype):
    data = np.array(values, dtype=dtype)
    rate = _fit_unchanged(data)
    _assert_rate(rate, expected, data)


@pytest.mark.parametrize(
    "values",
    [[4, 2, 1, 1], [1, 1, 2, 4] * 7, [1, 3], [2], [2, 2, 2]],
    ids=["permuted", "replicated", "same-mean-distinct", "singleton", "same-mean-identical"],
)
def test_rate_depends_only_on_mean_including_permutation_and_replication(values):
    original = np.array([1, 1, 2, 4])
    alternate = np.array(values)
    first = _fit_unchanged(original)
    second = _fit_unchanged(alternate)
    _assert_rate(first, 1.5936242600400401, original)
    _assert_rate(second, 1.5936242600400401, alternate)
    assert_allclose(second, first, rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("count", [1, 8])
def test_all_ones_rejects_unattained_positive_rate_supremum(dtype, count):
    with pytest.raises(ValueError):
        _fit_unchanged(np.ones(count, dtype=dtype))


@pytest.mark.parametrize(
    "data",
    [
        np.array([], dtype=float),
        np.array([], dtype=np.int64),
        np.array(2.0),
        np.array([[1, 2], [3, 4]]),
        np.array([1.0, np.nan, 4.0]),
        np.array([1.0, np.inf, 4.0]),
        np.array([1.0, -np.inf, 4.0]),
        np.array([0, 2, 4]),
        np.array([0.0, 2.0, 4.0]),
        np.array([-1, 2, 4]),
        np.array([-1.0, 2.0, 4.0]),
        np.array([1.0, 1.5, 4.0]),
        np.array([False, True, True]),
        np.array([True, True]),
        np.array([1 + 0j, 2 + 0j, 4 + 0j]),
        np.array([1 + 1j, 2 + 0j, 4 + 0j]),
        np.array(["one", "two", "four"]),
        np.array([1, "bad", 4], dtype=object),
    ],
    ids=[
        "empty-float",
        "empty-int",
        "scalar-array",
        "matrix",
        "nan",
        "positive-inf",
        "negative-inf",
        "zero-int",
        "zero-float",
        "negative-int",
        "negative-float",
        "fractional",
        "bool",
        "all-true",
        "complex-real-values",
        "complex",
        "strings",
        "object",
    ],
)
def test_invalid_observations_raise_value_error_and_preserve_input(data):
    with pytest.raises(ValueError):
        _fit_unchanged(data)


def test_fit_ztp_requires_numpy_array_input():
    data = [1, 1, 2, 4]
    original = data.copy()
    try:
        with pytest.raises(ValueError):
            fit_ztp(data)
    finally:
        assert data == original
