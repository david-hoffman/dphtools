"""Public ZTP numerical outcomes under warning and raising NumPy policies."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import fit_ztp


def _assert_positive_scalar(rate):
    assert np.asarray(rate).shape == ()
    assert np.isrealobj(rate)
    assert np.isfinite(rate) and rate > 0


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_ordinary_fit_succeeds_under_numpy_error_policy(policy):
    data = np.array([1, 1, 2, 4], dtype=np.float64)
    original = data.copy()
    previous_policy = np.geterr()
    try:
        with np.errstate(over=policy, invalid=policy):
            rate = fit_ztp(data)
            _assert_positive_scalar(rate)
            # Independent 90-digit Decimal bisection, bracket [1, 2].
            # Match the approved tests' optimization tolerance (about 6 digits).
            assert_allclose(rate, 1.5936242600400400923, rtol=2e-6, atol=2e-8)
            implied_mean = float(rate) / -math.expm1(-float(rate))
            assert_allclose(implied_mean, 2.0, rtol=2e-6, atol=2e-8)
    finally:
        assert np.geterr() == previous_policy
        assert_array_equal(data, original)


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_extreme_finite_sample_fits_or_reports_numerical_failure(policy, record_property):
    data = np.array([1e308, 1e308], dtype=np.float64)
    original = data.copy()
    previous_policy = np.geterr()
    # This stored binary64 value is finite, positive and exactly integer-valued.
    # Equal observations have mean m without needing an overflowing reduction.
    mean = float(data[0])
    try:
        with np.errstate(over=policy, invalid=policy):
            try:
                rate = fit_ztp(data)
            except RuntimeError as exc:
                # Only this eligible extreme fit may report numerical inability.
                # Keep its diagnostic without prescribing internal wording.
                record_property("fit_outcome", "RuntimeError")
                record_property("failure_message", str(exc))
                assert str(exc).strip()
            else:
                record_property("fit_outcome", "finite_rate")
                _assert_positive_scalar(rate)
                # For h(r) = m*(1-exp(-r))-r, h(m)<0 and h(m-1)>0
                # because log(m)<m-1. In real arithmetic, m-1 < rate < m.
                # Hence using m as oracle costs <1/m (<2e-308) relative error.
                # Divide before comparing to avoid overflow in the test itself;
                # 2e-6 retains the approved suite's solver accuracy allowance.
                assert_allclose(float(rate) / mean, 1.0, rtol=2e-6, atol=0)
    finally:
        assert np.geterr() == previous_policy
        assert_array_equal(data, original)
