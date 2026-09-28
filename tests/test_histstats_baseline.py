"""SETUP-001 L2/L3: histogram moments with independently expanded samples."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils import histstats


@pytest.mark.parametrize("multiple", [1, 7])
def test_histogram_mean_and_population_variance(multiple):
    # Expanded sample is [0, 2, 2, 4]: mean 2, variance (4+0+0+4)/4 = 2.
    weights = multiple * np.array([1, 2, 1])
    bins = np.array([0.0, 2.0, 4.0])
    assert_allclose(histstats.hist_mean(weights, bins), 2)
    assert_allclose(histstats.hist_var(weights, bins), 2)


def test_histogram_zero_weight_bin_does_not_contribute():
    weights = np.array([1, 0, 3])
    bins = np.array([0.0, 2.0, 4.0])
    assert_allclose(histstats.hist_mean(weights, bins), 3)
    assert_allclose(histstats.hist_var(weights, bins), 3)


@pytest.mark.parametrize("order, expected", [(2, 1), (3, 0), (4, 2)])
def test_standardized_histogram_moments(order, expected):
    # The module explicitly cites standardized moments: mu_k / sigma**k.
    assert_allclose(
        histstats.hist_moment(np.array([1, 2, 1]), np.array([0.0, 2.0, 4.0]), k=order),
        expected,
        atol=1e-14,
    )


def test_default_histogram_moment_is_skewness():
    # [0, 0, 0, 4] has mean 1, variance 3, central third moment 6.
    assert_allclose(histstats.hist_moment(np.array([3, 1]), np.array([0.0, 4.0])), 2 / np.sqrt(3))


@pytest.mark.parametrize("multiple", [0.25, 1.0, 7.0])
@pytest.mark.parametrize(
    "statistic, kwargs, expected",
    [
        (histstats.hist_mean, {}, 3),
        (histstats.hist_var, {}, 3),
        (histstats.hist_moment, {"k": 3}, -2 / np.sqrt(3)),
        (histstats.hist_moment, {"k": 4}, 7 / 3),
    ],
    ids=["mean", "variance", "skewness", "kurtosis"],
)
def test_asymmetric_population_statistics_use_each_weight_once(
    multiple, statistic, kwargs, expected
):
    weights = multiple * np.array([1.0, 0.0, 3.0])
    bins = np.array([0.0, 100.0, 4.0])
    # [0, 4, 4, 4]: mean 3, variance 3, central moments mu3=-6, mu4=21.
    assert_allclose(statistic(weights, bins, **kwargs), expected)


def test_histogram_translation_and_scale_obey_population_definitions():
    weights = np.array([3.0, 1.0])
    bins = np.array([10.0, 18.0])
    # This is [0,0,0,4] stretched by 2 and translated by 10.
    assert_allclose(histstats.hist_mean(weights, bins), 12)
    assert_allclose(histstats.hist_var(weights, bins), 12)
    assert_allclose(histstats.hist_moment(weights, bins), 2 / np.sqrt(3))
