"""Approved upper-window selection through public parameters and retained data."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw


def _fit(data, **options):
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(opt_max=True, **options)
    finally:
        assert_array_equal(data, original)
    return model, result


def _assert_selection(model, result, data, lower, upper, parameters):
    assert len(result) == 2
    assert np.isfinite(result).all()
    assert result[0] > 0 and result[1] > 1
    # Independent scalar likelihood roots permit numerical optimization error.
    # These margins are far smaller than competing-window parameter changes.
    assert_allclose(result, parameters, rtol=2e-6, atol=2e-8)
    assert_allclose([model.C, model.alpha], result, rtol=0, atol=0)
    assert model.xmin == lower
    retained = data[(data >= lower) & (data <= upper)]
    assert_array_equal(np.sort(model.clipped_data), np.sort(retained))
    assert len(retained) >= 50


def _bounded_data(discrete):
    if discrete:
        return np.repeat([1, 2, 3, 4, 5, 6, 20], [80, 20, 9, 5, 3, 2, 10])
    return np.repeat([1.025, 1.08, 1.15, 1.25, 1.38, 1.55, 1.78, 2.12, 2.65, 3.5, 30.0], 10)


@pytest.mark.parametrize("discrete", [False, True], ids=["continuous", "discrete"])
def test_observed_upper_endpoint_wins_with_fixed_unclipped_lower_bound(discrete):
    data = np.concatenate([np.zeros(7, dtype=int if discrete else float), _bounded_data(discrete)])
    model, result = _fit(data, xmin=1)
    if discrete:
        # On {1,2}, (80,20) exactly matches alpha=2 and C=4/5. KS=0.
        upper, parameters = 2, (0.8, 2.0)
    else:
        # Scalar likelihood: E_alpha[log(X)] = empirical mean(log(X)).
        # Enumerated eligible windows have their unique smallest KS at U=3.5.
        upper, parameters = 3.5, (1.5246865836407402, 2.1746821314411457)
        assert not np.any(data == 1)  # Explicit xmin need not be observed.
    _assert_selection(model, result, data, 1, upper, parameters)
    assert np.max(model.clipped_data) == upper  # The observed endpoint is inclusive.
    assert len(model.clipped_data) == 100


@pytest.mark.parametrize("discrete", [False, True], ids=["whole-floats", "integers"])
def test_unbounded_candidate_and_dtype_specific_likelihood_with_integer_gaps(discrete):
    data = np.repeat(np.array([1, 2, 4, 20], dtype=int if discrete else float), [30, 12, 10, 10])
    model, result = _fit(data, xmin=1)
    if discrete:
        # Ordinary discrete KS includes F(k-1) at jumps/gaps: unbounded
        # D=0.0928981221 beats [1,4] D=0.1014329694 and [1,20] D=0.1553882943.
        parameters = (0.5183425136587252, 1.7698533438102164)
    else:
        beta = len(data) / math.fsum(math.log(float(x)) for x in data)
        parameters = (beta, 1 + beta)
        # Bounded [1,20] ties at D=30/62, so larger upper support must win.
    _assert_selection(model, result, data, 1, math.inf, parameters)


@pytest.mark.parametrize("cap, lower", [(1.0, 1.0), (2.0, 2.0)])
def test_automatic_lower_candidates_obey_inclusive_cap_during_upper_search(cap, lower):
    data = np.repeat([1.0, 2.0, 4.0, 8.0], [70, 20, 20, 20])
    tail = data[data >= lower]
    beta = len(tail) / math.fsum(math.log(float(x) / lower) for x in tail)
    model, result = _fit(data, xmin_max=cap)
    _assert_selection(model, result, data, lower, math.inf, (beta * lower**beta, 1 + beta))
    assert np.max(model.clipped_data) == 8  # The cap bounds lower candidates only.


def test_explicit_fractional_lower_bound_is_fixed_during_upper_search():
    data = np.repeat([1.0, 2.0, 4.0, 8.0], [70, 20, 20, 20])
    lower = 1.5
    # Finite windows have no interior alpha>1 for this retained sample.
    beta = 1 / math.log(4 / lower)
    model, result = _fit(data, xmin=lower)
    _assert_selection(model, result, data, lower, math.inf, (beta * lower**beta, 1 + beta))


def test_exact_ks_ties_choose_most_observations_then_larger_upper_support():
    data = np.repeat([1.0, 2.0, 4.0], [50, 25, 25])
    # D=1/2 for [1,4], [1,infinity), and [2,infinity). The first two
    # retain 100 observations, the third 50. Larger upper breaks the first tie.
    # [2,4] has mean(log(X/2))=log(2)/2: alpha=1 boundary, not eligible.
    beta = 4 / (3 * math.log(2))
    model, result = _fit(data)
    _assert_selection(model, result, data, 1, math.inf, (beta, 1 + beta))


@pytest.mark.parametrize("delta, lower", [(7.5e-11, 1.0), (5e-10, 2.0)])
def test_near_ties_use_absolute_ks_tolerance_after_excluding_boundary_windows(delta, lower):
    upper = 2 ** (20 / (9 * -math.log(0.4 - delta)) - 1)
    data = np.repeat([1.0, 2.0, upper], [10, 45, 45])
    # Every eligible finite window has its unconstrained optimum at alpha<=1.
    # Unbounded D(1)=.5+delta, D(2)=.5. The 7.5e-11 case distinguishes
    # the absolute 1e-10 allowance from a relative allowance of .5e-10.
    tail = data[data >= lower]
    beta = len(tail) / math.fsum(math.log(float(x) / lower) for x in tail)
    model, result = _fit(data)
    _assert_selection(model, result, data, lower, math.inf, (beta * lower**beta, 1 + beta))


def test_fifty_observations_and_unbounded_interior_survive_bounded_alpha_one_boundary():
    data = np.repeat([1.0, 2.0], [25, 25])
    # Bounded mean(log X)=log(2)/2 gives only alpha=1. The unbounded
    # likelihood has the finite interior alpha=1+2/log(2), and must be tried.
    beta = 2 / math.log(2)
    model, result = _fit(data, xmin=1)
    _assert_selection(model, result, data, 1, math.inf, (beta, 1 + beta))


@pytest.mark.parametrize(
    "data",
    [np.repeat([1.0, 2.0], [24, 25]), np.ones(50)],
    ids=["only-49-even-with-explicit-xmin", "all-at-lower-bound"],
)
def test_no_eligible_window_raises_value_error(data):
    with pytest.raises(ValueError):
        _fit(data, xmin=1)


@pytest.mark.parametrize("discrete", [False, True], ids=["continuous", "discrete"])
@pytest.mark.parametrize(
    "options", [{"xmin": 20}, {"xmin_max": np.nan}], ids=["fewer-than-50", "invalid-lower-cap"]
)
def test_failed_upper_refit_preserves_prior_public_parameters_support_and_scores(
    discrete, options
):
    data = _bounded_data(discrete)
    original = data.copy()
    model, _ = _fit(data, xmin=1)
    before = (model.xmin, model.C, model.alpha)
    before_tail = np.array(model.clipped_data, copy=True)
    before_scores = np.array(model.ks_statistics, copy=True)
    try:
        with pytest.raises(ValueError):
            model.fit(opt_max=True, **options)
    finally:
        assert_array_equal(data, original)
        assert_array_equal((model.xmin, model.C, model.alpha), before)
        assert_array_equal(model.clipped_data, before_tail)
        assert_array_equal(model.ks_statistics, before_scores)
