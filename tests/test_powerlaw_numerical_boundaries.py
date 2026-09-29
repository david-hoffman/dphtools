"""Conditional finite-output and atomic-state checks at a justified C boundary.

The mathematical normalizer exceeds standard floating ranges. Arbitrary-precision
success is permitted and verified in logarithms; a numerical failure must be a
RuntimeError. No implementation path or floating return dtype is required.
"""

from copy import deepcopy
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw

LOWER, UPPER = 1e200, 1.001e200


def _reference_parameters():
    # These are the actual stored float64 inputs, not exact decimal literals.
    with localcontext() as context:
        context.prec = 60
        lower, upper = Decimal.from_float(LOWER), Decimal.from_float(UPPER)
        shape = 2 / (upper / lower).ln()
        log_c = shape.ln() + shape * lower.ln()
        control_shape = Decimal(110) / (25 * (lower.ln() + upper.ln()))
        assert shape > 1000 and log_c > 460000
        return float(1 + shape), float(log_c), float(control_shape)


def _positive_finite_log(value):
    """Check real finiteness without narrowing a huge C to a machine float.

    Decimal, Python integers, rational/as_integer_ratio scalars, NumPy floating
    scalars, and decimal-string arbitrary-precision reals can retain their range.
    Only the logarithm, which is moderate for this fixture, becomes a float.
    """
    if isinstance(value, Decimal):
        number = value
    elif callable(getattr(value, "as_integer_ratio", None)):
        numerator, denominator = value.as_integer_ratio()
        assert numerator > 0 and denominator > 0
        logarithm = math.log(numerator) - math.log(denominator)
        assert math.isfinite(logarithm)
        return logarithm
    else:
        number = Decimal(str(value))
    assert number.is_finite() and number > 0
    with localcontext() as context:
        context.prec = 60
        logarithm = float(number.ln())
    assert math.isfinite(logarithm)
    return logarithm


def _data(include_lower_candidate):
    tail = np.repeat(np.array([LOWER, UPPER]), 25)
    if include_lower_candidate:
        return np.concatenate([np.ones(60), tail])
    return tail


def _assert_parameters(model, result, expected_log_c, expected_alpha, log_c_atol):
    assert len(result) == 2
    returned = [_positive_finite_log(value) for value in result]
    fitted = [_positive_finite_log(model.C), _positive_finite_log(model.alpha)]
    assert returned[1] > 0  # alpha > 1
    assert_allclose(returned[0], expected_log_c, rtol=0, atol=log_c_atol)
    assert_allclose(returned[1], math.log(expected_alpha), rtol=0, atol=2e-10)
    assert_allclose(fitted, returned, rtol=0, atol=0)


def _assert_observations_and_scores(model, data, cutoff, expected_scores):
    assert model.xmin == cutoff
    assert_array_equal(np.sort(model.clipped_data), np.sort(data[data >= cutoff]))
    # Valid scores are bounded by one, so float conversion cannot overflow a
    # legitimate arbitrary-precision score as it could the large normalizer.
    scores = np.asarray(model.ks_statistics, dtype=float).reshape(-1)
    assert scores.shape == (len(expected_scores),)
    assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    assert_allclose(scores, expected_scores, rtol=0, atol=2e-12)


def _assert_large_success(model, result, data, automatic):
    alpha, log_c, _ = _reference_parameters()
    # log(C) sensitivity is about log(t)*d(alpha); 2e-4 allows alpha relative
    # error ~2e-10 here, without ever constructing or coercing C as float64.
    _assert_parameters(model, result, log_c, alpha, log_c_atol=2e-4)
    scores = [6 / 11, 0.5] if automatic else [0.5]
    _assert_observations_and_scores(model, data, LOWER, scores)


def _assert_control_success(model, result, data):
    _, _, shape = _reference_parameters()
    _assert_parameters(model, result, math.log(shape), 1 + shape, log_c_atol=2e-10)
    _assert_observations_and_scores(model, data, 1, [6 / 11])


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


def _snapshot(model):
    return {
        "xmin": deepcopy(model.xmin),
        "C": deepcopy(model.C),
        "alpha": deepcopy(model.alpha),
        "clipped_data": np.array(model.clipped_data, copy=True),
        "ks_statistics": np.array(model.ks_statistics, copy=True),
    }


def _assert_unchanged_state(model, before):
    for name in ("xmin", "C", "alpha"):
        assert getattr(model, name) == before[name]
    assert_array_equal(model.clipped_data, before["clipped_data"])
    assert_array_equal(model.ks_statistics, before["ks_statistics"])


@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "all-candidates"])
def test_eligible_large_normalizer_reports_failure_or_correct_finite_fit(
    automatic, record_property
):
    data = _data(include_lower_candidate=automatic)
    original = data.copy()
    assert np.isfinite(data).all() and np.all(data > 0)
    options = {"xmin_max": LOWER} if automatic else {"xmin": LOWER}
    try:
        model = PowerLaw(data)
        try:
            result = model.fit(**options)
        except RuntimeError as error:
            record_property("eligible_numerical_failure", _diagnostic(error))
        else:
            _assert_large_success(model, result, original, automatic)
            record_property("outcome", "independently verified finite large-normalizer success")
    finally:
        assert_array_equal(data, original)


def test_representable_lower_cutoff_control_retains_the_large_observations():
    data = _data(include_lower_candidate=True)
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1)
        _assert_control_success(model, result, original)
    finally:
        assert_array_equal(data, original)


@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "all-candidates"])
def test_eligible_numerical_refit_failure_preserves_prior_public_state(automatic, record_property):
    data = _data(include_lower_candidate=True)
    original = data.copy()
    try:
        model = PowerLaw(data)
        initial = model.fit(xmin=1)
        _assert_control_success(model, initial, original)
        before = _snapshot(model)
        options = {"xmin_max": LOWER} if automatic else {"xmin": LOWER}
        try:
            result = model.fit(**options)
        except RuntimeError as error:
            record_property("eligible_numerical_refit_failure", _diagnostic(error))
            _assert_unchanged_state(model, before)
        else:
            _assert_large_success(model, result, original, automatic)
            record_property("outcome", "independently verified finite large-normalizer refit")
    finally:
        assert_array_equal(data, original)
