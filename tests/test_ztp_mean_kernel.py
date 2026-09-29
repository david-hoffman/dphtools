"""Proposed real conditional-likelihood component, without public API changes."""

import math

import numpy as np
import pytest

from dphtools.utils import fitfuncs


def _required_mean_kernel():
    kernel = getattr(fitfuncs, "_fit_ztp_mean", None)
    assert callable(kernel), (
        "Structural refactoring red: proposed dphtools.utils.fitfuncs._fit_ztp_mean "
        "is not available as a callable. This does not establish a public numerical bug."
    )
    return kernel


def _assert_root(rate, expected):
    scalar = np.asarray(rate)
    assert scalar.shape == (), f"expected a scalar rate, got shape {scalar.shape}"
    assert np.isrealobj(scalar), "the fitted rate must be real"
    value = scalar.item()
    assert math.isfinite(value) and value > 0, f"invalid fitted rate: {value}"
    # Independent 110-digit Decimal bisection; use the approved public suite's
    # six-relative-digit solver allowance, with no absolute small-rate floor.
    ratio = float(value) / expected
    assert math.isclose(
        ratio, 1.0, rel_tol=2e-6, abs_tol=0
    ), f"rate {value} differs from independent root {expected}; ratio {ratio}"


@pytest.mark.parametrize(
    "mean, expected",
    [(2.0, 1.5936242600400400923), (20.0, 19.9999999587769258519)],
    ids=["mean-two", "mean-twenty"],
)
def test_ordinary_mean_kernel_fits_the_conditional_likelihood(mean, expected):
    kernel = _required_mean_kernel()
    rate = kernel(mean)
    _assert_root(rate, expected)
    # A separate stable evaluation checks the conditional-mean equation.
    implied_mean = float(rate) / -math.expm1(-float(rate))
    assert math.isclose(implied_mean, mean, rel_tol=2e-6, abs_tol=0)


def test_near_one_mean_kernel_fits_or_reports_numerical_inability(record_property):
    kernel = _required_mean_kernel()
    mean = float(1 + 1e-13)
    # Exact binary64 excess is 225 / 2**51, not exactly 1e-13.
    # Bisection gives 1.998401444325215212623658984126...e-13.
    try:
        rate = kernel(mean)
    except RuntimeError as exc:
        record_property("near_one_outcome", "RuntimeError")
        record_property("near_one_diagnostic", str(exc))
        assert str(exc).strip(), "numerical inability requires an informative diagnostic"
    else:
        record_property("near_one_outcome", "finite_rate")
        _assert_root(rate, 1.9984014443252152126e-13)
