"""Proposed histogram decomposition of the approved conditional NB estimator.

Supplemental post-implementation gap evidence. The private interface is a
proposal; its absence is structural red, not a public scientific regression.
No optimizer, random sequence, or internal traversal is prescribed here.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils import fitfuncs

# Grouped versions of the two approved samples, with the independently
# validated prior finite-product likelihood references. Counts are dimensionless.
INTERIOR = [
    (
        [1, 2, 3, 4, 6],
        [3, 2, 1, 1, 1],
        (2.1234896339948812, 1.8331910760864318),
        -13.400242369262732,
    ),
    ([1, 2, 3, 4], [9, 3, 1, 1], (1.217647708464976, 0.6183625983366804), -14.419074638823873),
]
_NUMERICAL_FAILURE = object()


def _component_call(counts, frequencies, record_property, *, maxiter=None, allow_failure=False):
    before_counts, before_frequencies = counts.copy(), frequencies.copy()
    try:
        component = getattr(fitfuncs, "_fit_ztnb_profile", None)
        if not callable(component):
            message = (
                "STRUCTURAL_RED: proposed _fit_ztnb_profile component is absent or not callable"
            )
            record_property("structural_red", message)
            pytest.fail(message, pytrace=False)
        options = {} if maxiter is None else {"maxiter": maxiter}
        try:
            # Omitting maxiter exercises the proposed ordinary default budget.
            # This seed is solely a computational initializer, not a model input.
            result = component(counts, frequencies, seed=0.5, **options)
        except RuntimeError as error:
            record_property(
                "component_outcome",
                {"status": "RuntimeError", "message": str(error), "maxiter": maxiter},
            )
            assert str(error).strip(), "Numerical failure requires an informative diagnosis"
            if not allow_failure:
                raise
            return _NUMERICAL_FAILURE
        record_property("component_outcome", {"status": "returned", "maxiter": maxiter})
        return result
    finally:
        assert_array_equal(counts, before_counts)
        assert_array_equal(frequencies, before_frequencies)


def _public_call(data):
    before = data.copy()
    try:
        return fitfuncs.fit_ztnb(data)
    finally:
        assert_array_equal(data, before)


def _assert_estimate(
    result, counts, frequencies, reference, reference_likelihood, record_property, label
):
    parameters = np.asarray(result)
    assert parameters.shape == (2,)
    assert np.isrealobj(parameters)
    assert np.isfinite(parameters).all() and np.all(parameters > 0)
    a, m = map(float, parameters)
    t = math.log1p(m / a)
    z = -math.expm1(-a * t)
    log_q = math.log(-math.expm1(-t))
    # Python integer totals and rational weights describe the exact histogram.
    # Normalizing the likelihood preserves its maximizer under replication and
    # avoids forming an enormous expanded array or an enormous total objective.
    n = sum(map(int, frequencies))
    weights = [Fraction(int(f), n) for f in frequencies]
    average = Fraction(sum(int(k) * int(f) for k, f in zip(counts, frequencies)), n)
    log_probabilities = [
        math.fsum(math.log(a + j) - math.log(j + 1) for j in range(int(k)))
        - a * t
        + int(k) * log_q
        - math.log(z)
        for k in counts
    ]
    likelihood_per_observation = math.fsum(
        float(weight) * value for weight, value in zip(weights, log_probabilities)
    )
    score_a = (
        math.fsum(
            float(weight) * math.fsum(1 / (a + j) for j in range(int(k)))
            for k, weight in zip(counts, weights)
        )
        - t / z
    )
    score_t = float(average) / math.expm1(t) - a / z
    scores = [a * score_a, t * score_t]
    record_property(
        label,
        {
            "parameters": parameters.tolist(),
            "observations": n,
            "exact_conditional_sample_mean": str(average),
            "likelihood_per_observation": likelihood_per_observation,
            "scores_per_observation": scores,
        },
    )
    # Same accuracy policy as the accepted ordinary public cases. Expressing
    # likelihood tolerance per base observation retains it under replication.
    base_n, base_likelihood = reference_likelihood
    assert_allclose(parameters, reference, rtol=2e-4, atol=2e-6)
    assert_allclose(
        likelihood_per_observation, base_likelihood / base_n, rtol=0, atol=2e-7 / base_n
    )
    assert_allclose(scores, [0, 0], rtol=0, atol=2e-5)
    assert_allclose(m / z, float(average), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize(
    "bins,multiplicities,reference,likelihood", INTERIOR, ids=["approved", "low-variance"]
)
def test_default_histogram_component_matches_independent_mle_and_expanded_public_fit(
    bins, multiplicities, reference, likelihood, record_property
):
    counts, frequencies = np.array(bins), np.array(multiplicities)
    baseline = (sum(multiplicities), likelihood)
    # Only these small controls are materialized (8 and 14 observations).
    public_result = _public_call(np.repeat(counts, frequencies))
    _assert_estimate(
        public_result, counts, frequencies, reference, baseline, record_property, "public_fit"
    )
    result = _component_call(counts, frequencies, record_property)
    _assert_estimate(
        result, counts, frequencies, reference, baseline, record_property, "component_fit"
    )
    # Both estimates also independently satisfy the stricter reference bounds.
    assert_allclose(result, public_result, rtol=4e-4, atol=4e-6)


@pytest.mark.parametrize("transformation", ["replication", "bin-permutation"])
def test_histogram_replication_and_bin_permutation_preserve_estimator(
    transformation, record_property
):
    bins, multiplicities, reference, likelihood = INTERIOR[0]
    counts, frequencies = np.array(bins), np.array(multiplicities)
    if transformation == "replication":
        frequencies = 3 * frequencies
    else:
        order = [3, 0, 4, 1, 2]
        counts, frequencies = counts[order], frequencies[order]
    result = _component_call(counts, frequencies, record_property)
    _assert_estimate(
        result, counts, frequencies, reference, (8, likelihood), record_property, "component_fit"
    )


@pytest.mark.parametrize("ones", [15, 2**54 - 1], ids=["modest", "not-materializable"])
def test_histograms_on_one_and_two_require_a_boundary_diagnosis(ones, record_property):
    # Let R = P_NB(2)/P_NB(1) = (a+1)*q/2 and match Poisson rate lambda=2R.
    # For k>=3, the ratio of NB P(k)/P(1) to the Poisson ratio is
    # product((a+j)/(a+1), j=1..k-1) > 1. Ratios for k=1,2 agree.
    # Normalization therefore makes NB P(1) and P(2) strictly smaller.
    # Every finite NB pair is dominated for every positive pair of frequencies.
    # The best truncated Poisson is approached as a -> infinity at fixed mean,
    # so there is only a boundary supremum, not a finite positive NB MLE.
    counts = np.array([1, 2], dtype=np.int64)
    frequencies = np.array([ones, 1], dtype=np.int64)
    exact_mean = Fraction(ones + 2, ones + 1)
    assert exact_mean > 1  # Includes exactly 1 + 2**-54; this is not all-ones data.
    record_property("exact_conditional_sample_mean", str(exact_mean))
    result = _component_call(counts, frequencies, record_property, allow_failure=True)
    assert result is _NUMERICAL_FAILURE, "No finite NB MLE exists for this histogram"


def test_modest_boundary_histogram_has_an_expanded_public_control(record_property):
    with pytest.raises(RuntimeError) as caught:
        _public_call(np.repeat(np.array([1, 2]), [15, 1]))
    assert str(caught.value).strip()
    record_property("public_boundary_diagnosis", str(caught.value))


def test_huge_frequency_replication_returns_accurate_mle_or_numerical_diagnosis(record_property):
    bins, multiplicities, reference, likelihood = INTERIOR[1]
    counts = np.array(bins, dtype=np.int64)
    frequencies = np.array([f * 2**50 for f in multiplicities], dtype=np.int64)
    # Positive integer replication multiplies the complete log likelihood by
    # 2**50 and leaves its exact finite maximizer unchanged. Never expand it.
    result = _component_call(counts, frequencies, record_property, allow_failure=True)
    if result is not _NUMERICAL_FAILURE:
        _assert_estimate(
            result,
            counts,
            frequencies,
            reference,
            (14, likelihood),
            record_property,
            "component_fit",
        )


def test_real_one_iteration_budget_returns_accurate_mle_or_numerical_diagnosis(record_property):
    bins, multiplicities, reference, likelihood = INTERIOR[0]
    counts, frequencies = np.array(bins), np.array(multiplicities)
    # A real constrained call with the algorithm active; either valid outcome
    # is allowed. This observation cannot prove iteration-budget enforcement.
    result = _component_call(counts, frequencies, record_property, maxiter=1, allow_failure=True)
    if result is not _NUMERICAL_FAILURE:
        _assert_estimate(
            result,
            counts,
            frequencies,
            reference,
            (8, likelihood),
            record_property,
            "component_fit",
        )
