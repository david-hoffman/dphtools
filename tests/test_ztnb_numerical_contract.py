"""Supplemental post-implementation acceptance evidence for ZTNB numerics.

Oracles use the approved conditional PMF, Decimal gamma ratios, and both
limiting distributions. They neither inspect nor prescribe product algorithms.
The 12-term Stirling remainder at x >= 64 is below 2e-42; correlated ratios
also preserve vanishing differences from the logarithmic boundary.
"""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import fit_ztnb

D = Decimal
STIRLING = (
    (1, 12),
    (-1, 360),
    (1, 1260),
    (-1, 1680),
    (1, 1188),
    (-691, 360360),
    (1, 156),
    (-3617, 122400),
    (43867, 244188),
    (-174611, 125400),
    (77683, 5796),
    (-236364091, 1506960),
)


def _stirling(x):
    value = (x - D("0.5")) * x.ln() - x
    for r, (numerator, denominator) in enumerate(STIRLING, 1):
        value += D(numerator) / (denominator * x ** (2 * r - 1))
    return value


def _log_gamma_ratio(x, y):
    """Log Gamma(x)/Gamma(y); common shift correlates truncation errors."""
    shift = max(0, 64 - int(min(x, y)))
    value = _stirling(x + shift) - _stirling(y + shift)
    ratio = D(1)
    for j in range(shift):
        ratio *= (x + j) / (y + j)
    return value - ratio.ln()


def _digamma(x):
    shift = max(0, 64 - int(x))
    y = x + shift
    value = y.ln() - 1 / (2 * y)
    for r, (numerator, denominator) in enumerate(STIRLING, 1):
        value -= D((2 * r - 1) * numerator) / (denominator * y ** (2 * r))
    for j in range(shift):
        value -= 1 / (x + j)
    return value


def _positive_mass(exponent):
    # Error below exp(-2000) < 3e-869, below our 700-digit arithmetic.
    return D(1) if exponent > 2000 else 1 - (-exponent).exp()


def _statistics(shape, mean, counts):
    """Grouped (positive integer, multiplicity) counts; scores in log(a), log(t)."""
    a, m = D(shape), D(mean)
    t = (1 + m / a).ln()
    z = _positive_mass(a * t)
    log_q = _positive_mass(t).ln()
    n = sum(weight for _, weight in counts)
    total = sum(D(k) * weight for k, weight in counts)
    common = a.ln() - a * t - z.ln()
    gamma_a = _log_gamma_ratio(1 + a, D(1))
    psi_a = _digamma(1 + a)
    likelihood = D(0)
    score_a = D(0)
    for k, weight in counts:
        k = D(k)
        if k == 1:
            gamma_increment = psi_increment = D(0)
        else:
            gamma_increment = _log_gamma_ratio(k + a, k) - gamma_a
            psi_increment = _digamma(k + a) - psi_a
        likelihood += weight * (common - k.ln() + gamma_increment + k * log_q)
        score_a += weight * (1 / a + psi_increment)
    score_a -= n * t / z
    score_t = total / (t.exp() - 1) - n * a / z
    return likelihood, (a * score_a / n, t * score_t / n), m / z


def _boundary_likelihoods(counts):
    n = sum(weight for _, weight in counts)
    total = sum(D(k) * weight for k, weight in counts)
    average = total / n
    rate = average
    if average < 2000:
        for _ in range(30):
            z = _positive_mass(rate)
            delta = (rate / z - average) / ((z - rate * (-rate).exp()) / z**2)
            rate -= delta
            if abs(delta) < D("1e-660"):
                break
    poisson = total * rate.ln() - n * rate - n * _positive_mass(rate).ln()
    poisson -= sum(weight * _log_gamma_ratio(D(k) + 1, D(1)) for k, weight in counts)
    t = average.ln() + (1 + average.ln()).ln()
    for _ in range(30):
        e = t.exp()
        delta = ((e - 1).ln() - t.ln() - average.ln()) / (e / (e - 1) - 1 / t)
        t -= delta
        if abs(delta) < D("1e-660"):
            break
    logarithmic = total * _positive_mass(t).ln() - n * t.ln()
    logarithmic -= sum(weight * D(k).ln() for k, weight in counts)
    return poisson, logarithmic, rate, t


_NUMERICAL_FAILURE = object()
APPROVED = [1, 1, 1, 2, 2, 3, 4, 6]
LOW_VARIANCE = [1] * 9 + [2] * 3 + [3, 4]
ORDINARY = [
    (APPROVED, (2.1234896339948812, 1.8331910760864318), -13.400242369262732),
    (LOW_VARIANCE, (1.217647708464976, 0.6183625983366804), -14.419074638823873),
]
# Independently bracketed roots of the conditional shape score at 85 digits.
# At these roots, P(0) < exp(-8000); untruncated means equal sample means to
# much better precision than any tolerance used below.
LARGE_INTERIOR = [
    ([9850.0, 10000.0, 10150.0], 19998.083293398527, -18.680509234002715),
    ([990000.0, 1000000.0, 1010000.0], 15227.832895756724, -31.27963084206631),
]


def _call_preserving_state(data, policy, record_property, initial=None, allow_failure=True):
    original = data.copy()
    initial_original = None if initial is None else initial.copy()
    previous_policy = np.geterr().copy()
    try:
        with np.errstate(all=policy):
            requested_policy = np.geterr().copy()
            try:
                result = fit_ztnb(data) if initial is None else fit_ztnb(data, x0=initial)
            except RuntimeError as error:
                record_property(
                    "numerical_outcome",
                    {"status": "RuntimeError", "message": str(error), "numpy_policy": policy},
                )
                assert str(error).strip(), "Numerical failure needs an informative diagnosis"
                if not allow_failure:
                    raise
                return _NUMERICAL_FAILURE
            finally:
                assert np.geterr() == requested_policy
            record_property("numerical_outcome", {"status": "returned", "numpy_policy": policy})
            return result
    finally:
        assert np.geterr() == previous_policy
        assert_array_equal(data, original)
        if initial is not None:
            assert_array_equal(initial, initial_original)


def _parameters(result):
    values = np.asarray(result)
    assert values.shape == (2,)
    assert np.isrealobj(values)
    assert np.isfinite(values).all() and np.all(values > 0)
    return tuple(map(float, values))


def _assert_ordinary_fit(result, observations, reference, expected_likelihood, record_property):
    a, m = _parameters(result)
    t = math.log1p(m / a)
    z = -math.expm1(-a * t)
    log_q = math.log(-math.expm1(-t))
    n = len(observations)
    likelihood = math.fsum(
        math.fsum(math.log(a + j) - math.log(j + 1) for j in range(k))
        - a * t
        + k * log_q
        - math.log(z)
        for k in observations
    )
    score_a = math.fsum(1 / (a + j) for k in observations for j in range(k)) - n * t / z
    score_t = sum(observations) / math.expm1(t) - n * a / z
    scores = [a * score_a / n, t * score_t / n]
    record_property(
        "fit_evidence", {"parameters": [a, m], "likelihood": likelihood, "scores": scores}
    )
    assert_allclose([a, m], reference, rtol=2e-4, atol=2e-6)
    assert_allclose(likelihood, expected_likelihood, rtol=0, atol=2e-7)
    assert_allclose(scores, [0, 0], rtol=0, atol=2e-5)
    assert_allclose(m / z, sum(observations) / n, rtol=2e-5, atol=2e-7)


def _assert_stress_fit(result, counts, record_property, reference=None):
    shape, mean = _parameters(result)
    # 700 digits resolve sums/differences involving every binary64 exponent
    # in these fixtures, including loss of a subnormal shape at the boundary.
    with localcontext() as context:
        context.prec = 700
        likelihood, scores, conditional_mean = _statistics(D(shape), D(mean), counts)
        poisson, logarithmic, _, _ = _boundary_likelihoods(counts)
        n = sum(weight for _, weight in counts)
        average = sum(D(k) * weight for k, weight in counts) / n
        gaps = [likelihood - poisson, likelihood - logarithmic]
        record_property(
            "fit_evidence",
            {
                "parameters": [shape, mean],
                "likelihood": str(likelihood),
                "scores": [str(score) for score in scores],
                "conditional_mean_relative_error": str(conditional_mean / average - 1),
                "likelihood_gains_over_boundaries": [str(gap) for gap in gaps],
            },
        )
        assert abs(conditional_mean / average - 1) < D("2e-6")
        assert all(abs(score) < D("1e-4") for score in scores)
        # A finite interior maximum must beat both optimized boundary models.
        # A near-boundary cap with small dimensionless scores is insufficient.
        assert all(gap > 0 for gap in gaps)
        if reference is not None:
            expected_shape, expected_likelihood = reference
            assert_allclose(shape, expected_shape, rtol=2e-3, atol=0)
            assert abs(likelihood - D(str(expected_likelihood))) / n < D("1e-6")


@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize(
    "observations,reference,likelihood", ORDINARY, ids=["approved", "low-variance"]
)
def test_ordinary_default_controls_succeed_under_both_numpy_policies(
    policy, observations, reference, likelihood, record_property
):
    data = np.array(observations)
    result = _call_preserving_state(data, policy, record_property, allow_failure=False)
    _assert_ordinary_fit(result, observations, reference, likelihood, record_property)


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_subnormal_positive_start_returns_mle_or_informative_failure(policy, record_property):
    data = np.array(APPROVED, dtype=float)
    initial = np.array([1e-310, 1.0])
    result = _call_preserving_state(data, policy, record_property, initial=initial)
    if result is not _NUMERICAL_FAILURE:
        _assert_ordinary_fit(result, APPROVED, ORDINARY[0][1], ORDINARY[0][2], record_property)


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_repeated_largest_finite_integral_counts_require_boundary_diagnosis(
    policy, record_property
):
    # For every integer k>1, finite NB = a nondegenerate gamma mixture of
    # Poissons. Conditioning on K>0 reweights its mixing density by
    # 1-exp(-lambda), still continuously positive for every lambda>0.
    # Its P(K=k | K>0) is strictly below max_lambda P_ZTP(K=k).
    # The unique maximizing rate solves lambda/(1-exp(-lambda))=k;
    # the conditional mean rises strictly from 1 to infinity. At fixed mean
    # lambda, shape -> infinity concentrates the gamma mixing distribution.
    # Thus for repeated k the NB likelihood has only a Poisson-boundary
    # supremum. This argument is specific to identical counts; sample
    # underdispersion alone does not justify rejecting an interior fit.
    largest = np.finfo(np.float64).max
    assert math.isfinite(largest) and largest.is_integer()
    data = np.full(3, largest)
    result = _call_preserving_state(data, policy, record_property)
    assert result is _NUMERICAL_FAILURE, "Repeated identical counts cannot have a finite NB MLE"


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_many_ones_and_extreme_stored_integer_require_valid_fit_or_diagnosis(
    policy, record_property
):
    data = np.concatenate((np.ones(10000), [3e305]))
    assert math.isfinite(float(data[-1])) and float(data[-1]).is_integer()
    result = _call_preserving_state(data, policy, record_property)
    if result is not _NUMERICAL_FAILURE:
        _assert_stress_fit(result, [(1, 10000), (int(data[-1]), 1)], record_property)


@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize(
    "observations,shape,likelihood", LARGE_INTERIOR, ids=["mean-10000", "mean-1000000"]
)
def test_large_symmetric_counts_require_conditional_mle_or_diagnosis(
    policy, observations, shape, likelihood, record_property
):
    data = np.array(observations)
    result = _call_preserving_state(data, policy, record_property)
    if result is not _NUMERICAL_FAILURE:
        _assert_stress_fit(
            result,
            [(int(k), 1) for k in observations],
            record_property,
            reference=(shape, likelihood),
        )
