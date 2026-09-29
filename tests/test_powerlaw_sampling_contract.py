"""Real generators and conditional bootstrap with independent probability oracles."""

import copy
import math
import random

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import zeta

from dphtools.utils.fitfuncs import PowerLaw

CASES = ["continuous-unbounded", "continuous-bounded", "discrete-unbounded", "discrete-bounded"]


@pytest.fixture(autouse=True)
def isolate_random_streams():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    random.seed(8317)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _snapshot(model):
    # Availability depends on completed public operations. Preserve values
    # without choosing a representation for an unbounded xmax. ks_data and
    # ks_tests are calculate_p outputs, checked separately below.
    fields = ("data", "C", "alpha", "xmin", "xmax", "alpha_error", "ks_statistics", "clipped_data")
    return {name: copy.deepcopy(getattr(model, name)) for name in fields if hasattr(model, name)}


def _assert_state(model, before):
    for name, value in before.items():
        actual = getattr(model, name)
        assert_array_equal(actual, value, err_msg=f"Public fitted field {name} changed")
        if isinstance(value, np.ndarray):
            assert actual.dtype == value.dtype


def _diagnostic(error):
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append(
            {"filename": code.co_filename, "line": traceback.tb_lineno, "function": code.co_name}
        )
        traceback = traceback.tb_next
    return {"category": type(error).__name__, "message": str(error), "frames": frames}


def _model(case, large=False, bootstrap=False):
    discrete = case.startswith("discrete")
    bounded = case.endswith("-bounded")
    if case == "continuous-unbounded":
        if bootstrap:
            # Independent inverse-CDF observations; no product fit/generator supplies data.
            uniform = np.random.default_rng(421).random(80)
            data = 1 / (1 - uniform)
        else:
            data = np.full(4096 if large else 80, math.e)
        beta = len(data) / math.fsum(math.log(float(x)) for x in data)
        parameters, upper = (beta, 1 + beta), math.inf
    elif case == "continuous-bounded":
        data = np.repeat([1.025, 1.08, 1.15, 1.25, 1.38, 1.55, 1.78, 2.12, 2.65, 3.5, 30.0], 10)
        if large:
            data = np.tile(data, 40)
        parameters, upper = (1.5246865836407402, 2.1746821314411457), 3.5
    elif case == "discrete-unbounded":
        data = np.repeat([1, 2, 4, 20], [30, 12, 10, 10])
        if large:
            data = np.tile(data, 64)
        parameters, upper = (0.5183425136587252, 1.7698533438102164), math.inf
    else:
        data = np.repeat([1, 2, 3, 4, 5, 6, 20], [80, 20, 9, 5, 3, 2, 10])
        if large:
            data = np.tile(data, 40)
        parameters, upper = (0.8, 2.0), 2
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1, opt_max=bounded)
    finally:
        assert_array_equal(data, original)
    assert_allclose(result, parameters, rtol=2e-6, atol=2e-8)
    assert_array_equal(np.sort(model.clipped_data), np.sort(data[data <= upper]))
    return model, data, upper, parameters[1], discrete


def _cdf(x, alpha, upper, discrete):
    if x < 1:
        return 0.0
    if discrete:
        if math.isinf(upper):
            return float(1 - zeta(alpha, math.floor(x) + 1) / zeta(alpha, 1))
        support = range(1, int(upper) + 1)
        return math.fsum(k**-alpha for k in support if k <= x) / math.fsum(
            k**-alpha for k in support
        )
    numerator = -math.expm1(-(alpha - 1) * math.log(x))
    return (
        numerator if math.isinf(upper) else numerator / -math.expm1(-(alpha - 1) * math.log(upper))
    )


def _draw(model, count, upper, discrete):
    samples = np.asarray(model.gen_power_law())
    assert samples.size == count
    assert np.isrealobj(samples) and np.isfinite(samples).all()
    samples = samples.reshape(-1)
    assert np.all((samples >= 1) & (samples <= upper))
    if discrete:
        assert_array_equal(samples, np.floor(samples))
    return samples


@pytest.mark.parametrize("case", CASES)
def test_generator_has_independent_model_probabilities_and_one_draw_per_retained_sample(
    case, record_property
):
    model, data, upper, alpha, discrete = _model(case, large=True)
    original, before = data.copy(), _snapshot(model)
    np.random.seed(27091)
    try:
        samples = _draw(model, len(model.clipped_data), upper, discrete)
    finally:
        assert_array_equal(data, original)
        _assert_state(model, before)
    thresholds = [1] if case == "discrete-bounded" else ([1, 3, 20] if discrete else [2, 4, 10])
    if case == "continuous-bounded":
        thresholds = [1.4, 2.0, 3.0]
    observed, expected = [], []
    for threshold in thresholds:
        observed.append(float(np.mean(samples > threshold)))
        expected.append(1 - _cdf(threshold, alpha, upper, discrete))
    record_property(
        "tail_probabilities",
        {
            "thresholds": thresholds,
            "actual": observed,
            "expected": expected,
            "count": samples.size,
        },
    )
    # At N>=3968, Hoeffding's two-sided bound at 0.055 is <7.6e-11
    # per fixed threshold. These probabilities come from independent models.
    assert samples.size >= 3968
    assert_allclose(observed, expected, rtol=0, atol=0.055)
    indicators = samples > thresholds[0]
    pairs = indicators[: 2 * (samples.size // 2)].reshape(-1, 2)
    joint = float(np.mean(np.all(pairs, axis=1)))
    # Disjoint pairs are independent under the required IID generator.
    # For >=1984 pairs, tolerance .085 gives a bound below 8e-13.
    assert_allclose(joint, expected[0] ** 2, rtol=0, atol=0.085)
    if math.isinf(upper):
        assert np.max(samples) > np.max(data)  # Observed maxima do not cap support.


@pytest.mark.parametrize("case", CASES)
def test_resetting_only_numpy_seed_reproduces_draws_and_preserves_fitted_state(case):
    model, data, upper, _, discrete = _model(case)
    original, before = data.copy(), _snapshot(model)
    try:
        np.random.seed(11837)
        first = _draw(model, len(model.clipped_data), upper, discrete)
        # Python's random stream is deliberately not reset here.
        np.random.seed(11837)
        second = _draw(model, len(model.clipped_data), upper, discrete)
        assert_array_equal(second, first)
    finally:
        assert_array_equal(data, original)
        _assert_state(model, before)


def _ks(samples, alpha, upper, discrete):
    values, counts = np.unique(samples, return_counts=True)
    total, largest = 0, 0.0
    for value, count in zip(values, counts):
        left = _cdf(value - 1 if discrete else value, alpha, upper, discrete)
        right = _cdf(value, alpha, upper, discrete)
        largest = max(
            largest, abs(total / len(samples) - left), abs((total + count) / len(samples) - right)
        )
        total += count
    return float(largest)


def _bootstrap_diagnostics(model, probability, iterations, observed, record_property, label):
    # Copy the first call's diagnostics before a repeat call can replace them.
    ks_data = np.asarray(model.ks_data)
    scores = np.array(model.ks_tests, copy=True)
    assert ks_data.ndim == 0 and np.isrealobj(ks_data) and np.isfinite(ks_data)
    assert 0 <= ks_data <= 1
    assert scores.size == iterations and np.isrealobj(scores) and np.isfinite(scores).all()
    scores = scores.reshape(-1)
    assert np.all((scores >= 0) & (scores <= 1))
    # No tolerance enters this comparison: use exactly the exposed numbers.
    comparisons = scores >= ks_data
    expected = int(np.count_nonzero(comparisons)) / iterations
    record_property(
        label,
        {
            "ks_data": float(ks_data),
            "independent_observed_ks": observed,
            "ks_tests": scores.tolist(),
            "comparisons_ge": comparisons.tolist(),
            "iterations": iterations,
            "expected_probability": expected,
            "actual_probability": float(probability),
        },
    )
    # Score tolerance allows the finite likelihood solver's parameter error;
    # it does not change >= comparisons or define a bootstrap tie policy.
    assert_allclose(ks_data, observed, rtol=0, atol=2e-6)
    assert np.ndim(probability) == 0 and np.isrealobj(probability) and np.isfinite(probability)
    assert 0 <= probability <= 1
    assert_allclose(probability, expected, rtol=0, atol=4 * np.finfo(float).eps)
    return float(ks_data), scores


@pytest.mark.parametrize("case", CASES)
def test_conditional_bootstrap_diagnostics_fraction_and_same_method_seed_reproducibility(
    case, record_property
):
    # Bounded controls retain 4,000 observations. Independent enumeration
    # revalidates their selected windows after replication. Across both
    # 31-replicate calls, conservative natural-boundary probabilities are
    # below 1e-35 (continuous) and 1.1e-60 (binary), independent of RNG stream.
    model, data, upper, alpha, discrete = _model(
        case, large=case.endswith("-bounded"), bootstrap=True
    )
    original, before = data.copy(), _snapshot(model)
    # These parameters and retained samples come from the independent fixture
    # oracle, not from product bootstrap output or its random stream.
    observed = _ks(data[data <= upper], alpha, upper, discrete)
    iterations, seed = 31, 50119
    record_property("bootstrap_request", {"seed": seed, "num": iterations, "observed": observed})
    try:
        np.random.seed(seed)
        first = model.calculate_p(num=iterations)
        first_observed, first_scores = _bootstrap_diagnostics(
            model, first, iterations, observed, record_property, "first_bootstrap"
        )
        _assert_state(model, before)
        # Only the same public method is replayed. No generator/fit stream
        # relationship is prescribed, and Python's stream is not reset.
        np.random.seed(seed)
        second = model.calculate_p(num=iterations)
        second_observed, second_scores = _bootstrap_diagnostics(
            model, second, iterations, observed, record_property, "second_bootstrap"
        )
        assert_array_equal(second_scores, first_scores)
        assert second_observed == first_observed
        assert second == first
    finally:
        assert_array_equal(data, original)
        _assert_state(model, before)


def test_singleton_continuous_bootstrap_refits_exponent_in_every_replicate(record_property):
    lower = 2.0
    data = np.array([lower * math.e])
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=lower)
    finally:
        assert_array_equal(data, original)
    # beta=1/log(x/L)=1, alpha=2, C=beta*L**beta=2.
    assert_allclose(result, (2.0, 2.0), rtol=2e-6, atol=2e-8)
    assert_array_equal(model.clipped_data, data)
    before = _snapshot(model)
    expected_score = -math.expm1(-1)
    iterations, seed = 31, 24017
    record_property("singleton_request", {"lower": lower, "num": iterations, "seed": seed})
    try:
        np.random.seed(seed)
        probability = model.calculate_p(num=iterations)
        _, scores = _bootstrap_diagnostics(
            model, probability, iterations, expected_score, record_property, "singleton_bootstrap"
        )
        # Every singleton Y>L has beta_hat*log(Y/L)=1. At its only
        # empirical jump, KS=max(1-exp(-1), exp(-1))=1-exp(-1).
        # An unchanged exponent instead gives a nonconstant random score.
        assert_allclose(scores, expected_score, rtol=0, atol=2e-6)
        # The returned probability was checked against actual recorded >=
        # comparisons above; analytic equality does not imply an exact p=1.
    finally:
        assert_array_equal(data, original)
        _assert_state(model, before)


@pytest.mark.parametrize("invalid", [0, -2, 1.5, np.nan, np.inf])
def test_invalid_bootstrap_iteration_count_reports_error_and_preserves_state(
    invalid, record_property
):
    model, data, _, _, _ = _model("discrete-unbounded")
    original, before = data.copy(), _snapshot(model)
    try:
        # Require an operable positive-count control so a general bootstrap
        # failure cannot masquerade as invalid-iteration validation.
        np.random.seed(9187)
        control = model.calculate_p(num=1)
        assert np.ndim(control) == 0 and np.isfinite(control) and control in (0, 1)
        _assert_state(model, before)
        # The contract requires an error but does not prescribe its class.
        with pytest.raises(Exception) as caught:
            model.calculate_p(num=invalid)
        assert str(caught.value).strip()
        record_property(
            "iteration_rejection",
            _diagnostic(caught.value),
        )
    finally:
        assert_array_equal(data, original)
        _assert_state(model, before)


def test_natural_boundary_bootstrap_replicate_reports_error_instead_of_discarding(record_property):
    # On {1,2}, alpha_hat=log(K/(50-K))/log(2). Thus a finite
    # alpha>1 exists exactly for integer K in [34,49]. Under the fitted
    # model K~Binomial(50,17/25); compute its failure probability exactly.
    denominator = 25**50
    failure_numerator = sum(math.comb(50, k) * 17**k * 8 ** (50 - k) for k in range(34)) + 17**50
    assert 5 * failure_numerator > 2 * denominator
    failure_probability = failure_numerator / denominator
    record_property(
        "boundary_probability",
        {
            "failure_numerator": failure_numerator,
            "failure_denominator": denominator,
            "failure_probability": failure_probability,
            "no_failure_in_64": (1 - failure_probability) ** 64,
            "conservative_no_failure_bound": 0.6**64,
        },
    )
    data = np.repeat([1, 2], [34, 16])
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1, opt_max=True)
    finally:
        assert_array_equal(data, original)
    assert_allclose(result, (34 / 50, math.log(34 / 16) / math.log(2)), rtol=2e-6, atol=2e-8)
    assert_array_equal(np.sort(model.clipped_data), data)
    before = _snapshot(model)
    try:
        # This is a fixed bootstrap seed, not a seed selected by replaying
        # gen_power_law. Under the required IID model, P(no failure in 64)
        # is less than .6**64 < 6.34e-15. Successful controls above exercise
        # bootstrap on finite, non-boundary data through the real API.
        np.random.seed(1830)
        record_property("boundary_bootstrap_request", {"seed": 1830, "num": 64})
        with pytest.raises(Exception) as caught:
            model.calculate_p(num=64)
        assert str(caught.value).strip()
        record_property(
            "replicate_rejection",
            _diagnostic(caught.value),
        )
    finally:
        assert_array_equal(data, original)
        _assert_state(model, before)
