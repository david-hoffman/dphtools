"""Source-blind component math and public discrete-generation range checks."""

from copy import deepcopy
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw, _powerlaw_discrete_partition


def _error(record_property, name, error):
    assert str(error).strip()
    record_property(name, {"category": type(error).__name__, "message": str(error)})


def _partition_pair(result):
    assert len(result) == 2
    s, m = map(float, result)
    assert math.isfinite(s) and math.isfinite(m) and s >= 1 and m >= 0
    return s, m


@pytest.mark.parametrize("alpha,lower,upper", [(1.0, 2, 7), (2.5, 3, 11), (3.0, 7, 7)])
def test_partition_finite_and_singleton_controls_succeed(alpha, lower, upper):
    with localcontext() as context:
        context.prec = 60
        logs = [(Decimal(k) / lower).ln() for k in range(lower, upper + 1)]
        weights = [(-Decimal.from_float(alpha) * log).exp() for log in logs]
        expected = (sum(weights), sum(w * log for w, log in zip(weights, logs)))
    actual = _partition_pair(_powerlaw_discrete_partition(alpha, lower, upper))
    assert_allclose(actual, list(map(float, expected)), rtol=2e-12, atol=2e-14)


def test_partition_infinite_control_succeeds_with_independently_bounded_tail():
    # For alpha=3,L=2, both summands decrease after N. Integrals from
    # N+1 and N bound the remaining sum; neither product component is an oracle.
    n = 8192
    weights = [(k / 2) ** -3 for k in range(2, n + 1)]
    s = math.fsum(weights)
    m = math.fsum(w * math.log(k / 2) for k, w in enumerate(weights, start=2))
    lower_s, upper_s = (4 / r**2 for r in (n + 1, n))
    lower_m, upper_m = ((4 * math.log(r / 2) + 2) / r**2 for r in (n + 1, n))
    actual_s, actual_m = _partition_pair(_powerlaw_discrete_partition(3.0, 2, math.inf))
    assert s + lower_s - 2e-12 <= actual_s <= s + upper_s + 2e-12
    assert m + lower_m - 2e-12 <= actual_m <= m + upper_m + 2e-12


def test_extreme_finite_partition_exponent_is_not_forced_to_fail(record_property):
    # For a>=64, S-1 and M are bounded by their a=64 values. Bound the
    # decreasing tail with its k=2 term plus integral from 2 to infinity.
    s_tail = 2.0**-64 * (1 + 2 / 63)
    m_tail = 2.0**-64 * (math.log(2) + 2 * (math.log(2) / 63 + 1 / 63**2))
    assert s_tail < 6e-20 and m_tail < 4e-20
    record_property("independent_tail_bounds", {"S_minus_one": s_tail, "M": m_tail})
    try:
        result = _powerlaw_discrete_partition(1e30, 1, math.inf)
    except RuntimeError as error:
        _error(record_property, "extreme_partition_numerical_failure", error)
    else:
        s, m = _partition_pair(result)
        # A zero moment is legitimate numerical underflow, not a failure.
        assert abs(s - 1) <= 2e-14 and m <= 2e-14
        record_property("extreme_partition_success_checked", [s, m])


@pytest.mark.parametrize(
    "discrete,bounded", [(True, True), (True, False), (False, True), (False, False)]
)
def test_cdf_ordinary_controls_succeed(discrete, bounded):
    data = np.array([1, 2, 3], dtype=int if discrete else float)
    original = data.copy()
    model = PowerLaw(data)
    values = np.array([0, 1, 2, 3, 4, 8], dtype=float)
    before_values = values.copy()
    if discrete:
        lower, upper = (2, 4) if bounded else (1, math.inf)
        expected = (
            [0, 0, 36 / 61, 52 / 61, 1, 1]
            if bounded
            else [math.fsum(k**-2 for k in range(1, int(x) + 1)) * 6 / math.pi**2 for x in values]
        )
    else:
        lower, upper = 2, (8 if bounded else math.inf)
        expected = [0, 0, 0, 4 / 9, 2 / 3, 1] if bounded else [0, 0, 0, 1 / 3, 1 / 2, 3 / 4]
    try:
        actual = np.asarray(model._cdf(values, lower, upper, 2.0), dtype=float)
        assert actual.shape == values.shape and np.isfinite(actual).all()
        assert np.all((actual >= 0) & (actual <= 1)) and np.all(np.diff(actual) >= 0)
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)
    finally:
        assert_array_equal(data, original)
        assert_array_equal(values, before_values)


def test_large_integer_lower_cdf_preserves_small_positive_probabilities(record_property):
    lower, alpha = 10**15, 1.1
    data = np.array([1, 2, 3], dtype=np.int64)
    original = data.copy()
    values = np.array([lower - 1, lower, lower + 1, 2 * lower], dtype=np.int64)
    original_values = values.copy()
    with localcontext() as context:
        context.prec = 60
        a, l = Decimal.from_float(alpha), Decimal(lower)
        integral = l / (a - 1)
        # Decreasing summands give I<=S<=I+1. The first two numerators
        # are exact finite sums, so these bounds retain relative accuracy.
        numerators = [Decimal(1), 1 + (-a * ((l + 1) / l).ln()).exp()]
        intervals = [(p / (integral + 1), p / integral) for p in numerators]
        far = 1 - ((1 - a) * Decimal(2).ln()).exp()
    record_property("small_cdf_intervals", [[str(x) for x in pair] for pair in intervals])
    model = PowerLaw(data)
    try:
        try:
            result = model._cdf(values, lower, math.inf, alpha)
        except RuntimeError as error:
            _error(record_property, "large_lower_cdf_numerical_failure", error)
        else:
            result = np.asarray(result, dtype=float)
            assert result.shape == values.shape and np.isfinite(result).all()
            assert np.all((result >= 0) & (result <= 1)) and np.all(np.diff(result) >= 0)
            assert result[0] == 0
            for actual, (lo, hi) in zip(result[1:3], intervals):
                # Relative slack, no absolute floor that could accept zero
                # or an 11%-wrong rounded subtraction near 1e-16.
                assert float(lo) * (1 - 2e-8) <= actual <= float(hi) * (1 + 2e-8)
            # The integral CDF at 2L differs by <3/I, below 4e-16 here.
            assert_allclose(result[3], float(far), rtol=0, atol=2e-12)
            record_property("large_lower_cdf_success_checked", result.tolist())
    finally:
        assert_array_equal(data, original)
        assert_array_equal(values, original_values)


# Independent infinite-series references for the PUBLIC fitted generator.
_LOG_K = np.log(np.arange(1, 8193, dtype=float))


def _zeta_bounds(alpha):
    weights = np.exp(-alpha * _LOG_K)
    s, m = math.fsum(weights), math.fsum(weights * _LOG_K)
    beta = alpha - 1
    pairs = []
    for r in (8193, 8192):
        power = math.exp(-beta * math.log(r))
        pairs.append((s + power / beta, m + power * (math.log(r) / beta + 1 / beta**2)))
    # Sum/integral enclosures plus generous rounding padding on positive sums.
    return tuple((pairs[0][i] * (1 - 2e-12), pairs[1][i] * (1 + 2e-12)) for i in (0, 1))


def _mean_bounds(alpha):
    (sl, sh), (ml, mh) = _zeta_bounds(alpha)
    return ml / sh, mh / sl


def _fit_bracket(value):
    target = math.log(value)
    lo, hi = 1.001, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ml, mh = _mean_bounds(mid)
        if ml > target:
            lo = mid
        elif mh < target:
            hi = mid
        else:
            break
    assert _mean_bounds(lo)[0] > target > _mean_bounds(hi)[1]
    assert hi - lo < 2e-5
    return lo, hi


def _tail_bounds(alpha, threshold):
    (sl, sh), _ = _zeta_bounds(alpha)
    if threshold <= 32:
        prefix = math.fsum(k**-alpha for k in range(1, threshold + 1))
        return 1 - prefix / sl, 1 - prefix / sh
    # Entire integer tail starting at r: integral <= sum <= integral+f(r).
    r, beta = threshold + 1, alpha - 1
    integral = math.exp(-beta * math.log(r)) / beta
    return integral / sh, (integral + math.exp(-alpha * math.log(r))) / sl


def _integer(value):
    assert not isinstance(value, (str, bytes, bool, np.bool_, complex, np.complexfloating))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if callable(getattr(value, "as_integer_ratio", None)):
        numerator, denominator = value.as_integer_ratio()
        assert denominator == 1
        return numerator
    number = Decimal(str(value))
    assert number.is_finite() and number == number.to_integral_value()
    return int(number)


def _snapshot(model):
    fields = ("data", "C", "alpha", "xmin", "xmax", "alpha_error", "ks_statistics", "clipped_data")
    return {key: deepcopy(getattr(model, key)) for key in fields if hasattr(model, key)}


def _preserved(model, before):
    for key, expected in before.items():
        actual = getattr(model, key)
        assert_array_equal(actual, expected, err_msg=f"Public field {key} changed")
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    "value", [2, 2**52], ids=["ordinary-discrete-control", "discrete-range-boundary"]
)
def test_public_discrete_generator_range_distribution_independence_and_preservation(
    value, record_property
):
    count, seed = 4096, 29731
    heavy = value != 2
    lo, hi = _fit_bracket(value)
    thresholds = [1, 2**16 - 1, 2**63 - 1] if heavy else [1, 4, 16]
    intervals = [(_tail_bounds(hi, k)[0], _tail_bounds(lo, k)[1]) for k in thresholds]
    # This is a range witness, not a selected output cap. Wider finite integer
    # storage may succeed. A narrow representation can report numerical failure.
    if heavy:
        assert intervals[-1][0] > 0.25
    record_property(
        "independent_generator_reference",
        {
            "alpha_bracket": [lo, hi],
            "thresholds": thresholds,
            "tail_intervals": intervals,
            "seed": seed,
            "count": count,
        },
    )
    data = np.full(count, value, dtype=np.int64)
    original, random_state = data.copy(), np.random.get_state()
    try:
        model = PowerLaw(data)
        try:
            c, alpha = model.fit(xmin=1)
        except RuntimeError as error:
            if not heavy:
                raise
            _error(record_property, "boundary_fit_failure_generation_not_reached", error)
            return
        assert math.isfinite(c) and math.isfinite(alpha) and c > 0 and alpha > 1
        assert lo - 2e-6 <= alpha <= hi + 2e-6
        (sl, sh), _ = _zeta_bounds(float(alpha))
        assert (1 / sh) * (1 - 2e-6) <= c <= (1 / sl) * (1 + 2e-6)
        assert_allclose([model.C, model.alpha], [c, alpha], rtol=0, atol=0)
        assert model.xmin == 1
        assert_array_equal(model.clipped_data, original)
        before = _snapshot(model)
        record_property(
            "real_fit_success", {"C": float(c), "alpha": float(alpha), "fields": sorted(before)}
        )
        outcomes, first = [], None
        for attempt in range(2):
            np.random.seed(seed)
            try:
                samples = model.gen_power_law()
            except Exception as error:
                if not heavy:
                    raise
                _error(record_property, f"range_error_{attempt}", error)
                outcomes.append("error")
            else:
                array = np.asarray(samples)
                assert array.size == count
                integers = [_integer(x) for x in array.flat]
                assert all(x >= 1 for x in integers)
                probabilities = []
                for k, (pl, ph) in zip(thresholds, intervals):
                    observed = sum(x > k for x in integers) / count
                    assert pl - 0.055 <= observed <= ph + 0.055
                    probabilities.append(observed)
                # Disjoint pair events must have probability p**2. Using
                # .08 stochastic slack after numerical error, the two-sided
                # Hoeffding bound for 2048 pairs is below 8.3e-12.
                index = 1 if heavy else 0
                indicators = np.array([x > thresholds[index] for x in integers])
                joint = float(np.mean(np.all(indicators.reshape(-1, 2), axis=1)))
                pl, ph = intervals[index]
                assert pl**2 - 0.085 <= joint <= ph**2 + 0.085
                if first is not None:
                    assert integers == first
                first = integers
                outcomes.append("finite-success")
                record_property(
                    f"finite_generator_success_{attempt}",
                    {"tails": probabilities, "pair_probability": joint},
                )
            finally:
                _preserved(model, before)
        assert outcomes[0] == outcomes[1]
        record_property("generator_outcomes_and_preservation", outcomes)
    finally:
        np.random.set_state(random_state)
        assert_array_equal(data, original)
