"""Conditional negative-binomial MLEs from independent integer-count likelihoods."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import NegBinom, fit_ztnb

APPROVED = [1, 1, 1, 2, 2, 3, 4, 6]
LOW_VARIANCE = [1] * 9 + [2] * 3 + [3, 4]
# Independent 60-digit Decimal profiling: solve the conditional-mean equation
# for t=-log(p), then bisect the shape score. No product output enters these.
REFERENCES = {
    "approved": ((2.1234896339948812, 1.8331910760864318), -13.400242369262732),
    "low-variance": ((1.217647708464976, 0.6183625983366804), -14.419074638823873),
}


def _conditional_likelihood_and_scores(shape, mean, observations):
    """Finite products replace gamma/digamma for positive integer observations."""
    t = math.log1p(mean / shape)
    normalizer = -math.expm1(-shape * t)
    log_q = math.log(-math.expm1(-t))
    values = [int(value) for value in observations]
    log_likelihood = math.fsum(
        math.fsum(math.log(shape + j) - math.log(j + 1) for j in range(k))
        - shape * t
        + k * log_q
        - math.log(normalizer)
        for k in values
    )
    score_a = math.fsum(1 / (shape + j) for k in values for j in range(k))
    score_a -= len(values) * t / normalizer
    score_t = sum(values) / math.expm1(t) - len(values) * shape / normalizer
    # Dimensionless derivatives per observation in (log(shape), log(t)).
    scores = np.array([shape * score_a, t * score_t]) / len(values)
    return log_likelihood, scores


def _fit_unchanged(data, initial=None):
    original = data.copy()
    initial_original = None if initial is None else np.array(initial, copy=True)
    try:
        return fit_ztnb(data) if initial is None else fit_ztnb(data, x0=initial)
    finally:
        assert_array_equal(data, original)
        if initial is not None:
            assert_array_equal(initial, initial_original)


def _assert_interior_estimate(result, observations, reference, record_property):
    parameters = np.asarray(result)
    assert parameters.shape == (2,)
    assert np.isrealobj(parameters)
    assert np.isfinite(parameters).all() and np.all(parameters > 0)
    shape, mean = map(float, parameters)
    likelihood, scores = _conditional_likelihood_and_scores(shape, mean, observations)
    expected_parameters, expected_likelihood = reference
    record_property(
        "conditional_mle",
        {
            "parameters": parameters.tolist(),
            "log_likelihood": likelihood,
            "dimensionless_scores_per_observation": scores.tolist(),
        },
    )
    # These optima are separated from both limiting models by >0.13 log units.
    # Parameter tolerance permits numerical minimization of the shallow shape
    # direction; likelihood and score checks independently require stationarity.
    assert_allclose(parameters, expected_parameters, rtol=2e-4, atol=2e-6)
    assert_allclose(likelihood, expected_likelihood, rtol=0, atol=2e-7)
    assert_allclose(scores, [0, 0], rtol=0, atol=2e-5)
    conditional_mean = mean / -math.expm1(-shape * math.log1p(mean / shape))
    assert_allclose(conditional_mean, np.mean(observations), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize(
    "case, dtype",
    [
        ("approved", np.int64),
        ("low-variance", np.int64),
        ("low-variance", np.float32),
        ("low-variance", np.uint64),
    ],
    ids=["approved", "low-variance", "whole-valued-float", "unsigned"],
)
def test_ordinary_interior_mle_succeeds_from_default_start(case, dtype, record_property):
    data = np.array(APPROVED if case == "approved" else LOW_VARIANCE, dtype=dtype)
    if case == "low-variance":
        # Zero truncation invalidates an untruncated variance > mean heuristic.
        assert np.var(data.astype(float)) < np.mean(data)
    result = _fit_unchanged(data)
    _assert_interior_estimate(result, data, REFERENCES[case], record_property)


@pytest.mark.parametrize(
    "case, dtype, initial",
    [
        ("approved", np.float64, [0.25, 2.0]),
        ("approved", np.int32, (8.0, 4.0)),
        ("low-variance", np.float32, [0.25, 2.0]),
        ("low-variance", np.uint64, (8.0, 4.0)),
    ],
    ids=[
        "approved-float-alternate-start",
        "approved-large-start",
        "low-variance-float-alternate-start",
        "low-variance-unsigned",
    ],
)
def test_alternate_start_returns_same_mle_or_informative_numerical_failure(
    case, dtype, initial, record_property
):
    data = np.array(APPROVED if case == "approved" else LOW_VARIANCE, dtype=dtype)
    if isinstance(initial, list):
        initial = np.array(initial)
    try:
        result = _fit_unchanged(data, initial)
    except RuntimeError as error:
        # Only the default start has the approved ordinary-case success
        # guarantee. Retain allowed numerical failures as diagnoses, not fits.
        frames = []
        traceback = error.__traceback__
        while traceback is not None:
            code = traceback.tb_frame.f_code
            frames.append(
                {
                    "filename": code.co_filename,
                    "line": traceback.tb_lineno,
                    "function": code.co_name,
                }
            )
            traceback = traceback.tb_next
        record_property(
            "alternate_start_numerical_failure",
            {
                "category": type(error).__name__,
                "message": str(error),
                "frames": frames,
            },
        )
        assert str(error).strip()
        return
    _assert_interior_estimate(result, data, REFERENCES[case], record_property)


@pytest.mark.parametrize("replications", [1, 3])
def test_permutation_and_replication_preserve_the_conditional_estimator(
    replications, record_property
):
    data = np.tile(np.array(APPROVED), replications)[::-1]
    result = _fit_unchanged(data)
    parameters, likelihood = REFERENCES["approved"]
    _assert_interior_estimate(
        result, data, (parameters, replications * likelihood), record_property
    )


@pytest.mark.parametrize("shape, mean", [(2.0, 6.0), (0.5, 1.5)])
def test_negative_binomial_public_distribution_uses_shape_and_untruncated_mean(shape, mean):
    values = np.array([0, 1, 2, 5, 9])
    p = shape / (shape + mean)
    # (a)_k/k! * p**a * (1-p)**k, independently expanded into scalar products.
    probabilities = [
        math.prod((shape + j) / (j + 1) for j in range(int(k))) * p**shape * (1 - p) ** int(k)
        for k in values
    ]
    distribution = NegBinom(shape, mean)
    assert_allclose(distribution.pmf(values), probabilities, rtol=1e-12, atol=1e-14)
    assert_allclose(distribution.mean(), mean, rtol=1e-12, atol=1e-14)
    assert_allclose(distribution.var(), mean + mean * mean / shape, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "data",
    [
        np.array([]),
        np.array(2),
        np.array([[1, 2], [3, 4]]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
        np.array([0, 2, 3]),
        np.array([-1, 2, 3]),
        np.array([1.0, 1.5, 3.0]),
        np.array([True, False]),
        np.array([True, True]),
        np.array([1 + 0j, 2 + 0j]),
        np.array([1 + 1j, 2 + 0j]),
        np.array(["1", "2"]),
        np.array([1, "bad"], dtype=object),
        [1, 2, 3],
    ],
    ids=[
        "empty",
        "scalar",
        "matrix",
        "nan",
        "positive-inf",
        "negative-inf",
        "zero",
        "negative",
        "fractional",
        "bool",
        "all-true",
        "complex-real-values",
        "complex",
        "strings",
        "object",
        "not-numpy",
    ],
)
def test_invalid_observations_raise_value_error_without_mutation(data):
    with pytest.raises(ValueError):
        _fit_unchanged(data)


@pytest.mark.parametrize(
    "initial",
    [
        0.5,
        [0.5],
        [0.5, 0.5, 0.5],
        [[0.5, 0.5]],
        [0.0, 0.5],
        [0.5, 0.0],
        [-1.0, 0.5],
        [0.5, -1.0],
        [np.nan, 0.5],
        [0.5, np.inf],
        [1.0, "bad"],
    ],
    ids=[
        "scalar",
        "short",
        "long",
        "matrix",
        "zero-shape",
        "zero-mean",
        "negative-shape",
        "negative-mean",
        "nan",
        "inf",
        "nonnumeric",
    ],
)
def test_invalid_initial_pair_raises_value_error_without_mutation(initial):
    with pytest.raises(ValueError):
        _fit_unchanged(np.array(APPROVED), initial)


@pytest.mark.parametrize("data", [np.ones(1, dtype=np.int64), np.ones(5, dtype=float)])
def test_all_ones_rejects_unattained_positive_parameter_supremum(data):
    with pytest.raises(ValueError):
        _fit_unchanged(data)


def test_identical_twos_have_only_a_poisson_boundary_optimum(record_property):
    # A finite NB is a nondegenerate Poisson-Gamma mixture. After conditioning
    # on positive counts it remains a mixture of zero-truncated Poissons.
    # Each conditional P(2) <= max_lambda P_ZTP(2), strictly after mixing over
    # a continuous positive rate distribution. The unique maximum satisfies
    # lambda/(1-exp(-lambda))=2. NB approaches it as shape -> infinity with
    # mean -> lambda. Thus this sample has no finite positive NB maximizer.
    rate = 1.5936242600400401
    assert_allclose(rate / -math.expm1(-rate), 2, rtol=0, atol=1e-14)
    with pytest.raises(RuntimeError) as caught:
        _fit_unchanged(np.array([2, 2, 2, 2]))
    assert str(caught.value).strip()
    record_property(
        "boundary_diagnostic",
        {
            "category": type(caught.value).__name__,
            "message": str(caught.value),
        },
    )
