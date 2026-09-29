"""Approved PowerLaw cutoff contract, tested only through its public interface.

Continuous expectations use the packet's closed-form likelihood and both sides
of empirical jumps. Discrete constants come from independently differentiated
Euler--Maclaurin sums, not a fitted product result or continuous approximation.
See reports/A-powerlaw-cutoff/handoff.md for derivations and precision checks.
"""

import inspect
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw


def _continuous_reference(data, cutoff):
    tail = np.sort(np.asarray(data, dtype=float)[data >= cutoff])
    shape = len(tail) / math.fsum(math.log(float(x) / cutoff) for x in tail)
    values, counts = np.unique(tail, return_counts=True)
    after = np.cumsum(counts) / len(tail)
    before = (np.cumsum(counts) - counts) / len(tail)
    model = -np.expm1(-shape * np.log(values / cutoff))
    distance = max(np.max(np.abs(after - model)), np.max(np.abs(before - model)))
    return (shape * cutoff**shape, 1 + shape), float(distance)


def _fit_unchanged(data, **options):
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(**options)
    finally:
        assert_array_equal(data, original)
    return model, result


def _assert_fit(model, result, original, cutoff, parameters, scores, *, discrete=False):
    # Discrete optimization receives a looser parameter allowance than the
    # closed-form continuous fit. Neither allowance can hide a model change.
    rtol, atol = (1e-6, 2e-8) if discrete else (2e-10, 2e-12)
    score_atol = 2e-7 if discrete else 2e-12
    if original.dtype == np.dtype(np.float32):
        # The public contract does not require promoting float32 arithmetic.
        rtol = atol = score_atol = 8 * np.finfo(np.float32).eps
    assert len(result) == 2
    assert_allclose(result, parameters, rtol=rtol, atol=atol)
    assert_allclose((model.C, model.alpha), parameters, rtol=rtol, atol=atol)
    assert_allclose(result, (model.C, model.alpha), rtol=0, atol=0)
    assert model.xmin == cutoff
    assert np.isfinite([model.C, model.alpha]).all()
    assert model.C > 0 and model.alpha > 1
    assert_array_equal(np.sort(model.clipped_data), np.sort(original[original >= cutoff]))
    distances = np.asarray(model.ks_statistics).reshape(-1)
    assert distances.shape == (len(scores),)
    assert np.all((0 <= distances) & (distances <= 1))
    assert_allclose(distances, scores, rtol=0, atol=score_atol)


def _coarse_sample(dtype=float):
    return np.repeat(np.array([1, 2, 4, 8], dtype=dtype), [70, 20, 20, 20])


def test_public_fit_signature_is_preserved():
    # Signature metadata only: no source, bytecode, or private interface reads.
    signature = inspect.signature(PowerLaw.fit)
    assert tuple(signature.parameters) == ("self", "xmin", "xmin_max", "opt_max")
    assert signature.parameters["xmin"].default is None
    assert signature.parameters["xmin_max"].default == 200
    assert signature.parameters["opt_max"].default is False
    assert all(
        parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(
    "values, cutoff",
    [
        ([2.0, 3.0, 8.0], 2.0),
        ([0.0, 2.0, 2.0, 3.0, 8.0], 2.0),
        ([0.0, 1.0, 3.0, 4.0, 12.0], 2.0),
        ([0.0, 0.75, 1.5, 1.5, 2.25, 6.0], 1.5),
        ([1.0, 2.0, 2.0, 2.0, 2.0], 1.0),
    ],
    ids=[
        "approved-three",
        "zeros-and-cutoff-ties",
        "unobserved-cutoff",
        "fractional",
        "left-jump",
    ],
)
def test_continuous_explicit_likelihood_and_two_sided_ks(values, cutoff):
    data = np.array(values)
    parameters, score = _continuous_reference(data, cutoff)
    model, result = _fit_unchanged(data, xmin=cutoff)
    _assert_fit(model, result, data, cutoff, parameters, [score])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cap, selected", [(1, 1), (1.5, 1), (2, 2), (200, 2)])
def test_approved_automatic_selection_candidate_order_and_inclusive_cap(dtype, cap, selected):
    data = np.concatenate([np.zeros(7, dtype=dtype), _coarse_sample(dtype)])
    expected_scores = [7 / 13] if cap < 2 else [7 / 13, 1 / 3]
    parameters, _ = _continuous_reference(data, selected)
    model, result = _fit_unchanged(data, xmin_max=cap)
    _assert_fit(model, result, data, selected, parameters, expected_scores)
    if selected == 2:
        assert len(model.clipped_data) == 60
        tolerance = 8 * np.finfo(dtype).eps if dtype == np.float32 else 2e-10
        assert_allclose(model.alpha, 1 + 1 / math.log(2), rtol=tolerance)


@pytest.mark.parametrize("factor, selected", [(100.0, 200.0), (101.0, 101.0)])
def test_default_cap_is_200_inclusive_and_does_not_cap_observations(factor, selected):
    data = factor * _coarse_sample()
    parameters, _ = _continuous_reference(data, selected)
    scores = [7 / 13, 1 / 3] if factor == 100 else [7 / 13]
    model, result = _fit_unchanged(data)
    _assert_fit(model, result, data, selected, parameters, scores)
    assert np.max(model.clipped_data) == 8 * factor


@pytest.mark.parametrize("count_at_two", [24, 25])
def test_automatic_floor_is_applied_to_each_inclusive_tail(count_at_two):
    data = np.repeat([1.0, 2.0, 4.0], [10, count_at_two, 25])
    cutoffs = [1.0] if count_at_two == 24 else [1.0, 2.0]
    references = [_continuous_reference(data, cutoff) for cutoff in cutoffs]
    scores = [reference[1] for reference in references]
    selected_index = min(range(len(cutoffs)), key=lambda i: scores[i])
    model, result = _fit_unchanged(data)
    _assert_fit(
        model, result, data, cutoffs[selected_index], references[selected_index][0], scores
    )


def test_degenerate_observed_cutoff_is_ineligible_even_with_fifty_observations():
    data = np.repeat([1.0, 2.0], [50, 50])
    parameters, score = _continuous_reference(data, 1.0)
    model, result = _fit_unchanged(data)
    _assert_fit(model, result, data, 1.0, parameters, [score])


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_exactly_fifty_nondegenerate_observations_are_eligible(dtype):
    data = np.repeat(np.array([1, 2], dtype=dtype), [25, 25])
    if np.issubdtype(dtype, np.integer):
        parameters = (0.7117406031863725, 2.353828175624703)
        score = 0.21174060318637246
    else:
        parameters, score = _continuous_reference(data, 1.0)
    model, result = _fit_unchanged(data)
    _assert_fit(model, result, data, 1, parameters, [score], discrete=dtype == np.int64)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize(
    "values, options",
    [
        ([1] * 24 + [2] * 25, {}),
        ([0] * 20 + [1] * 24 + [2] * 25, {}),
        ([1] * 50, {}),
        ([0] * 60, {}),
        ([201] * 25 + [402] * 25, {}),
        ([1] * 25 + [2] * 25, {"xmin_max": 0.5}),
    ],
    ids=[
        "49-positive",
        "zeros-do-not-count",
        "all-at-cutoff",
        "all-zero",
        "above-default",
        "below-min",
    ],
)
def test_no_eligible_automatic_candidate_raises_value_error(dtype, values, options):
    data = np.array(values, dtype=dtype)
    with pytest.raises(ValueError):
        _fit_unchanged(data, **options)


@pytest.mark.parametrize(
    "delta, selected", [(0.0, 1.0), (5e-11, 1.0), (7.5e-11, 1.0), (5e-10, 2.0)]
)
def test_natural_ks_ties_use_absolute_tolerance_and_smallest_cutoff(delta, selected):
    # At cutoff 1, the left side of the jump at 2 has D = 0.5 + delta.
    # At cutoff 2, its 45/90 atom fixes D = 0.5. Both tails are nondegenerate.
    # 7.5e-11 is inside the absolute 1e-10 allowance, but outside an incorrect
    # relative allowance of 0.5 * 1e-10; the smallest cutoff must still win.
    log_survival = -math.log(0.4 - delta)
    upper = 2 ** (20 / (9 * log_survival) - 1)
    data = np.repeat([1.0, 2.0, upper], [10, 45, 45])
    parameters, _ = _continuous_reference(data, selected)
    model, result = _fit_unchanged(data)
    _assert_fit(model, result, data, selected, parameters, [0.5 + delta, 0.5])


# (values, cutoff, exact-discrete C, exact-discrete alpha, ordinary discrete KS).
# Values were derived with 48-digit Decimal arithmetic at (N, terms)=(64,10)
# and (128,12). All recorded parameter/score differences are below 1e-24.
DISCRETE_CASES = [
    ([2, 3, 8], 2, 2.047789747575585, 2.2034990389100895, 0.18306891643543627),
    ([2, 3, 8], 1, 0.4179227483216648, 1.5630115616600486, 0.4179227483216648),
    ([0, 2, 2, 3, 8, 14], 2, 1.5813764478059491, 2.0136048277374196, 0.19790886802993023),
    ([2, 8, 8, 8, 8], 2, 1.0159066173182737, 1.7373656193121413, 0.4884550907926876),
]


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
@pytest.mark.parametrize(
    "values, cutoff, c, alpha, score",
    DISCRETE_CASES,
    ids=["approved-three", "unobserved-cutoff", "zeros-ties-gaps", "gap-maximum"],
)
def test_exact_discrete_likelihood_infinite_support_and_gap_ks(
    dtype, values, cutoff, c, alpha, score
):
    data = np.array(values, dtype=dtype)
    model, result = _fit_unchanged(data, xmin=cutoff)
    _assert_fit(model, result, data, cutoff, (c, alpha), [score], discrete=True)


@pytest.mark.parametrize("cap", [1, 2, 200])
def test_discrete_automatic_selection_uses_exact_likelihood_for_every_candidate(cap):
    data = np.concatenate([np.zeros(7, dtype=np.int64), _coarse_sample(np.int64)])
    scores = [0.11249714503303818]
    if cap >= 2:
        scores.append(0.25238603450294046)
    model, result = _fit_unchanged(data, xmin_max=cap)
    _assert_fit(
        model, result, data, 1, (0.5818132280251532, 1.9273004803635327), scores, discrete=True
    )


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_permuting_observations_preserves_selection_and_fit(dtype):
    data = _coarse_sample(dtype)
    shuffled = np.random.default_rng(731).permutation(data)
    first, first_result = _fit_unchanged(data)
    second, second_result = _fit_unchanged(shuffled)
    assert second.xmin == first.xmin
    assert_allclose(second_result, first_result, rtol=1e-7, atol=1e-9)
    assert_allclose(second.ks_statistics, first.ks_statistics, rtol=0, atol=2e-8)
    assert_array_equal(np.sort(second.clipped_data), np.sort(first.clipped_data))


@pytest.mark.parametrize("factor", [0.25, 3.5])
def test_continuous_scaling_preserves_alpha_and_scores_and_scales_cutoff(factor):
    data = _coarse_sample()
    base, _ = _fit_unchanged(data, xmin_max=2)
    scaled_data = data * factor
    scaled, result = _fit_unchanged(scaled_data, xmin_max=2 * factor)
    parameters, _ = _continuous_reference(scaled_data, 2 * factor)
    _assert_fit(scaled, result, scaled_data, 2 * factor, parameters, [7 / 13, 1 / 3])
    assert scaled.xmin == factor * base.xmin
    assert_allclose(scaled.alpha, base.alpha, rtol=2e-10, atol=2e-12)
    assert_allclose(scaled.ks_statistics, base.ks_statistics, rtol=0, atol=2e-12)


@pytest.mark.parametrize(
    "data",
    [
        np.array([], dtype=float),
        np.array([], dtype=np.int64),
        np.array(2.0),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([1.0, np.nan, 4.0]),
        np.array([1.0, np.inf, 4.0]),
        np.array([1.0, -np.inf, 4.0]),
        np.array([-1.0, 2.0, 4.0]),
        np.array([-1, 2, 4]),
        np.array([1 + 0j, 2 + 0j, 4 + 0j]),
        np.array([1 + 1j, 2 + 0j, 4 + 0j]),
        np.array([False, True, True]),
        np.array(["one", "two", "four"]),
        np.array([1, "bad", 4], dtype=object),
    ],
    ids=[
        "empty-float",
        "empty-int",
        "scalar-array",
        "matrix",
        "nan",
        "inf",
        "negative-inf",
        "negative-float",
        "negative-int",
        "complex-real-values",
        "complex",
        "bool",
        "strings",
        "object",
    ],
)
@pytest.mark.parametrize("options", [{}, {"xmin": 1}], ids=["automatic", "explicit"])
def test_invalid_data_raise_value_error_without_prescribing_validation_stage(data, options):
    with pytest.raises(ValueError):
        _fit_unchanged(data, **options)


def test_numpy_array_input_is_required():
    data = [1.0, 2.0, 4.0]
    original = data.copy()
    try:
        with pytest.raises(ValueError):
            PowerLaw(data).fit(xmin=1)
    finally:
        assert data == original


@pytest.mark.parametrize("cap", [0, -1, np.nan, np.inf, -np.inf, [2, 3], np.array([2, 3])])
def test_automatic_cap_must_be_a_positive_finite_scalar(cap):
    with pytest.raises(ValueError):
        _fit_unchanged(_coarse_sample(), xmin_max=cap)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("cutoff", [0, -1, np.nan, np.inf, -np.inf, 9, 8])
def test_invalid_explicit_cutoff_empty_tail_and_degenerate_tail_raise_value_error(dtype, cutoff):
    data = np.array([0, 2, 3, 8, 8], dtype=dtype)
    with pytest.raises(ValueError):
        _fit_unchanged(data, xmin=cutoff)


def test_discrete_cutoff_must_be_integral():
    with pytest.raises(ValueError):
        _fit_unchanged(np.array([2, 3, 8]), xmin=1.5)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize(
    "invalid_options",
    [{"xmin": 0}, {"xmin": 9}, {"xmin": 8}, {"xmin_max": np.nan}, {}],
    ids=[
        "invalid-cutoff",
        "empty-tail",
        "degenerate-tail",
        "invalid-cap",
        "no-eligible-candidate",
    ],
)
def test_failed_refit_preserves_all_prior_public_fitted_state(dtype, invalid_options):
    data = np.array([0, 1, 1, 2, 4, 8], dtype=dtype)
    model, result = _fit_unchanged(data, xmin=1)
    assert np.isfinite(result).all()
    before = (model.xmin, model.C, model.alpha)
    before_tail = np.array(model.clipped_data, copy=True)
    before_scores = np.array(model.ks_statistics, copy=True)
    original = data.copy()
    try:
        with pytest.raises(ValueError):
            model.fit(**invalid_options)
    finally:
        assert_array_equal(data, original)
        assert_array_equal((model.xmin, model.C, model.alpha), before)
        assert_array_equal(model.clipped_data, before_tail)
        assert_array_equal(model.ks_statistics, before_scores)


def test_successful_refits_restore_original_samples_and_replace_single_score():
    data = np.array([0.0, 2.0, 2.0, 3.0, 4.0, 8.0])
    original = data.copy()
    model = PowerLaw(data)
    try:
        for cutoff in (2.0, 3.0, 2.0):
            parameters, score = _continuous_reference(original, cutoff)
            result = model.fit(cutoff, 200, False)
            _assert_fit(model, result, original, cutoff, parameters, [score])
    finally:
        assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("count", [1, 3], ids=["singleton", "constant-tail"])
def test_constant_tail_strictly_above_unobserved_cutoff_has_finite_mle(dtype, count):
    data = np.full(count, 2, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        # Independent infinite-support score equation: zeta'/zeta + log(2)=0.
        # The KS maximum is at unobserved integer 1, where F_model(1)=C.
        parameters = (0.5634337517198817, 1.8791006722784631)
        score = 0.5634337517198817
    else:
        # n cancels from n/(n*log(2)); the left jump is 1-exp(-1).
        parameters = (1 / math.log(2), 1 + 1 / math.log(2))
        score = -math.expm1(-1)
    model, result = _fit_unchanged(data, xmin=1)
    _assert_fit(model, result, data, 1, parameters, [score], discrete=dtype == np.int64)
