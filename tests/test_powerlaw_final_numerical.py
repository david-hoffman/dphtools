"""Supplemental public-input boundaries; no implementation or fitted-state injection."""

from copy import deepcopy
from decimal import Decimal, MAX_EMAX, MIN_EMIN, localcontext
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw


def _decimal(value):
    """Retain finite scalar range, including a huge decimal normalizer."""
    assert not isinstance(value, (str, bytes, complex, np.complexfloating, bool, np.bool_))
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, np.integer)):
        result = Decimal(int(value))
    elif callable(getattr(value, "as_integer_ratio", None)):
        numerator, denominator = value.as_integer_ratio()
        result = Decimal(numerator) / Decimal(denominator)
    else:
        result = Decimal(str(value))
    assert result.is_finite()
    return result


def _context(context):
    context.prec = 90
    context.Emax, context.Emin = MAX_EMAX, MIN_EMIN


def _parameters(model, result, alpha, log_c, *, tolerance=2e-6):
    assert len(result) == 2
    with localcontext() as context:
        _context(context)
        c, a = map(_decimal, result)
        assert c > 0 and a > 1
        assert _decimal(model.C) == c and _decimal(model.alpha) == a
        assert_allclose(float(a), float(alpha), rtol=tolerance, atol=0)
        assert_allclose(float(c.ln()), float(log_c), rtol=0, atol=tolerance)


def _observations(model, data, lower):
    actual, expected = np.asarray(model.clipped_data), data[data >= lower]
    assert actual.shape == expected.shape
    # Mixed NumPy integer/float equality can hide rounding above 2**53.
    with localcontext() as context:
        _context(context)
        assert _decimal(model.xmin) == _decimal(lower)
        assert sorted(map(_decimal, actual)) == sorted(map(_decimal, expected))


def _snapshot(model):
    fields = ("data", "C", "alpha", "xmin", "xmax", "alpha_error", "ks_statistics", "clipped_data")
    return {name: deepcopy(getattr(model, name)) for name in fields if hasattr(model, name)}


def _preserved(model, before):
    for name, expected in before.items():
        actual = getattr(model, name)
        assert_array_equal(actual, expected, err_msg=f"Public field {name} changed")
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype
            # Check integer values without NumPy's mixed-dtype promotion.
            if np.issubdtype(expected.dtype, np.integer):
                assert list(map(int, actual.flat)) == list(map(int, expected.flat))


def _error(record_property, label, error):
    assert str(error).strip()
    record_property(label, {"category": type(error).__name__, "message": str(error)})


def _continuous_reference(data, lower):
    with localcontext() as context:
        _context(context)
        log_lower = _decimal(lower).ln()
        retained = data[data >= lower]
        logs = [_decimal(x).ln() - log_lower for x in retained]
        beta = Decimal(len(logs)) / sum(logs)
        return 1 + beta, beta.ln() + beta * log_lower


def test_ordinary_discrete_control_succeeds_and_preserves_state_on_error(record_property):
    # On {1,2}, empirical p(1)=.8 gives 2**(-alpha)=1/4, alpha=2,
    # C=.8 and KS=0. The unbounded alternative has positive mass above 2.
    data = np.repeat(np.array([1, 2], dtype=np.int64), [40, 10])
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1, opt_max=True)
        _parameters(model, result, 2, math.log(0.8))
        _observations(model, original, 1)
        before = _snapshot(model)
        record_property("mandatory_discrete_control_success", sorted(before))
        with pytest.raises(ValueError) as caught:
            model.fit(xmin=2, opt_max=True)
        _error(record_property, "invalid_discrete_refit", caught.value)
        _preserved(model, before)
        record_property("discrete_preservation_completed", True)
    finally:
        assert_array_equal(data, original)


@pytest.mark.parametrize("lower", [2**53, 2**53 + 1], ids=["float64-spacing", "inexact-cutoff"])
def test_adjacent_exact_integers_are_eligible_and_numerical_refit_is_atomic(
    lower, record_property
):
    data = np.array([lower, lower + 1], dtype=np.int64)
    original = data.copy()
    model = PowerLaw(data)
    before = None
    try:
        # At L=floor(t/2), sum/integral bounds give the continuous limit
        # alpha=1+1/log(2), log(C)=log(beta)+beta*log(L), with O(1/t) error.
        control_lower = lower // 2
        try:
            result = model.fit(xmin=control_lower)
        except RuntimeError as error:
            _error(record_property, "initial_fit_numerical_failure_no_snapshot", error)
        else:
            alpha, log_c = _continuous_reference(data, control_lower)
            _parameters(model, result, alpha, log_c, tolerance=2e-5)
            _observations(model, original, control_lower)
            before = _snapshot(model)
            record_property("initial_fit_success", sorted(before))
        try:
            result = model.fit(xmin=lower)
        except RuntimeError as error:
            _error(record_property, "adjacent_fit_numerical_failure", error)
            if before is not None:
                _preserved(model, before)
                record_property("numerical_refit_preservation_completed", True)
        else:
            # With j=k-t, normalized weights tend to exp(-a*j), a=alpha/t.
            # E(j)=1/2 gives exp(-a)=1/3 and a=log(3). The omitted finite-t
            # corrections are <1e-12; verify_oracles.py checks signs/bounds.
            with localcontext() as context:
                _context(context)
                c, alpha = map(_decimal, result)
                assert c > 0 and alpha > 1
                assert _decimal(model.C) == c and _decimal(model.alpha) == alpha
                a = alpha / Decimal(lower)
                assert_allclose(float(a), math.log(3), rtol=0, atol=2e-6)
                log_scaled_c = c.ln() - alpha * Decimal(lower).ln()
                expected = (1 - (-a).exp()).ln()
                assert abs(log_scaled_c - expected) < Decimal("2e-6")
            _observations(model, original, lower)
            # F(t)=2/3 and F(t+1)=8/9 in the geometric limit; the maximum
            # empirical/model difference is F(t)-1/2 = 1/6.
            assert_allclose(model.ks_statistics, [1 / 6], rtol=0, atol=2e-6)
            record_property("adjacent_fit_success_oracles_completed", True)
    finally:
        assert_array_equal(data, original)
        assert data.dtype == original.dtype


@pytest.mark.parametrize(
    "lower, upper",
    [(1e-300, 1e300), (np.nextafter(0.0, 1.0), np.finfo(float).max)],
    ids=["600-decades", "float64-endpoints"],
)
def test_wide_finite_continuous_samples_use_log_ratios_or_report_numerical_failure(
    lower, upper, record_property
):
    data = np.array([lower, 1.0, upper])
    original = data.copy()
    model = PowerLaw(data)
    before = None
    try:
        try:
            result = model.fit(xmin=1.0)
        except RuntimeError as error:
            _error(record_property, "initial_fit_numerical_failure_no_snapshot", error)
        else:
            _parameters(model, result, *_continuous_reference(data, 1.0))
            _observations(model, original, 1.0)
            before = _snapshot(model)
            record_property("initial_fit_success", sorted(before))
        try:
            result = model.fit(xmin=lower)
        except RuntimeError as error:
            _error(record_property, "wide_fit_numerical_failure", error)
            if before is not None:
                _preserved(model, before)
                record_property("numerical_refit_preservation_completed", True)
        else:
            alpha, log_c = _continuous_reference(original, lower)
            _parameters(model, result, alpha, log_c)
            _observations(model, original, lower)
            with localcontext() as context:
                _context(context)
                beta = alpha - 1
                cdf = [1 - (-beta * (_decimal(x).ln() - _decimal(lower).ln())).exp() for x in data]
                score = max(
                    abs(Decimal(j + side) / 3 - probability)
                    for j, probability in enumerate(cdf)
                    for side in (0, 1)
                )
            assert_allclose(model.ks_statistics, [float(score)], rtol=0, atol=2e-6)
            record_property("wide_fit_success_oracles_completed", True)
    finally:
        assert_array_equal(data, original)


@pytest.mark.parametrize("ones", [25, 40], ids=["alpha-one-bounded-boundary", "finite-interior"])
def test_finite_upper_search_controls_succeed_and_invalid_refits_preserve_state(
    ones, record_property
):
    data = np.repeat([1.0, 2.0], [ones, 50 - ones])
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1.0, opt_max=True)
        _parameters(model, result, *_continuous_reference(data, 1.0))
        _observations(model, original, 1.0)
        # Both eligible windows for the 40/10 case have D=.8, so infinity
        # wins. For 25/25 the bounded optimum has alpha=1 and is ineligible.
        # The selected C/alpha distinguish infinity without a new sentinel.
        before = _snapshot(model)
        record_property(
            "mandatory_finite_control_success", {"ones": ones, "fields": sorted(before)}
        )
        with pytest.raises(ValueError) as caught:
            model.fit(xmin=2.0, opt_max=True)
        _error(record_property, "invalid_refit", caught.value)
        _preserved(model, before)
        record_property("invalid_refit_preservation_completed", True)
    finally:
        assert_array_equal(data, original)


def test_nearly_coincident_root_checks_parameters_and_error_atomicity(
    record_property,
):
    """Check finite parameters/error atomicity; upper-support selection is unobserved."""
    record_property("upper_support_selection_observed", False)
    data = np.repeat([1.0, 2.0], [49, 1])
    original = data.copy()
    try:
        model = PowerLaw(data)
        # The explicit unbounded formula is a required successful control.
        parameters = _continuous_reference(data, 1.0)
        _parameters(model, model.fit(xmin=1.0), *parameters)
        _observations(model, original, 1.0)
        before = _snapshot(model)
        record_property("steep_unbounded_control_success", sorted(before))
        try:
            result = model.fit(xmin=1.0, opt_max=True)
        except RuntimeError as error:
            # Bounded beta is 50/log(2) minus about 6.96e-19. A finite
            # solution exists; reporting numerical inability is permitted,
            # not required. No solver path or exact error text is selected.
            _error(record_property, "bounded_root_numerical_failure", error)
            _preserved(model, before)
            record_property("bounded_root_preservation_completed", True)
        else:
            # Bounded and unbounded parameters agree within these tolerances.
            # Check finite parameters and retained data only; upper-support
            # selection is unobserved here. The 40/10 control distinguishes it.
            _parameters(model, result, *parameters)
            _observations(model, original, 1.0)
            record_property("bounded_root_finite_parameters_and_observations_checked", True)
    finally:
        assert_array_equal(data, original)


def _log_samples(samples, count, log_lower):
    values = np.asarray(samples)
    assert values.size == count
    # Object arrays and extended-precision outputs remain admissible.
    with localcontext() as context:
        _context(context)
        logarithms = []
        for value in values.flat:
            number = _decimal(value)
            assert number > 0
            logarithm = float(number.ln())
            assert math.isfinite(logarithm) and logarithm >= log_lower - 1e-10
            logarithms.append(logarithm)
    return np.array(logarithms)


@pytest.mark.parametrize("heavy", [False, True], ids=["ordinary-success", "unbounded-heavy-tail"])
def test_fitted_generation_has_natural_range_errors_or_finite_distribution_and_preserves_state(
    heavy, record_property
):
    # Fixed in advance, independent of any product RNG output. A successful
    # ordinary control is compulsory; only the heavy-tail call may fail.
    seed = 29629
    count = 262144 if heavy else 4096
    lower, value = (np.nextafter(0.0, 1.0), 1e300) if heavy else (1.0, math.e)
    data = np.full(count, value)
    original = data.copy()
    random_state = np.random.get_state()
    try:
        model = PowerLaw(data)
        try:
            result = model.fit(xmin=lower)
        except RuntimeError as error:
            if not heavy:
                raise
            _error(record_property, "heavy_fit_failure_generation_not_reached", error)
            return
        # Constant observations above an unobserved cutoff are nondegenerate.
        # n cancels; computing this reference need not traverse 262144 values.
        alpha, log_c = _continuous_reference(np.array([value]), lower)
        _parameters(model, result, alpha, log_c)
        before = _snapshot(model)
        record_property("generation_initial_fit_success", sorted(before))
        outcomes = []
        first = None
        for attempt in range(2):
            np.random.seed(seed)
            try:
                samples = model.gen_power_law()
            except Exception as error:
                if not heavy:
                    raise
                # No generation exception class is selected by the packet.
                _error(record_property, f"generation_error_{attempt}", error)
                outcomes.append("error")
            else:
                with localcontext() as context:
                    _context(context)
                    log_lower = float(_decimal(lower).ln())
                logs = _log_samples(samples, count, log_lower)
                beta = float(alpha - 1)
                if heavy:
                    # P(log X > 12000)=exp(-beta*(12000-log L)) > .00013.
                    # P(no exceedance) < exp(-262144*.00013) < 1.6e-15.
                    # This threshold exceeds standard floating ranges on the
                    # supplied hosts; it does not cap allowable output types.
                    probability = math.exp(-beta * (12000 - log_lower))
                    assert probability > 0.00013
                    assert np.any(logs > 12000)
                    threshold = log_lower + 1 / beta
                    expected = math.exp(-1)
                    tolerance = 0.02
                else:
                    threshold, expected, tolerance = math.log(4), 0.25, 0.055
                actual = float(np.mean(logs > threshold))
                assert abs(actual - expected) < tolerance
                outcomes.append("finite-success")
                record_property(
                    f"generation_success_{attempt}", {"tail": actual, "expected": expected}
                )
                if first is None:
                    first = deepcopy(samples)
                else:
                    assert_array_equal(samples, first)
            finally:
                _preserved(model, before)
        assert outcomes[0] == outcomes[1]
        record_property("generation_outcomes_and_preservation", outcomes)
    finally:
        np.random.set_state(random_state)
        assert_array_equal(data, original)
