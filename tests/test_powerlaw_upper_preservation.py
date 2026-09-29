"""Public upper-window tie priorities and complete failed-refit preservation."""

import copy
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw


def _assert_fit(model, result, parameters, lower, retained):
    # Independent likelihood roots; allow numerical optimizer error while
    # separating competing windows by much larger parameter differences.
    assert_allclose(result, parameters, rtol=2e-6, atol=2e-8)
    assert_allclose([model.C, model.alpha], result, rtol=0, atol=0)
    assert model.xmin == lower
    assert_array_equal(np.sort(model.clipped_data), np.sort(retained))


def _snapshot(model):
    fields = ("data", "C", "alpha", "xmin", "xmax", "alpha_error", "ks_statistics", "clipped_data")
    # Availability depends on completed public operations. Copy existing
    # values without choosing any representation for an unbounded xmax.
    return {name: copy.deepcopy(getattr(model, name)) for name in fields if hasattr(model, name)}


def _assert_preserved(model, before):
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


def test_equal_count_scaled_windows_choose_smaller_lower_bound(record_property):
    # Fifty bounded-alpha=2 quantiles including endpoints. Their fitted
    # alpha need not equal the generating alpha. Multiplication by 128 is
    # exact in binary, preserving both copies' dimensionless likelihood.
    base = 1 / (1 - 0.75 * np.arange(50) / 49)
    data = np.concatenate([base, 128 * base])
    original = data.copy()
    # Independent enumeration of all 5,150 candidate pairs found 732
    # interior fits. Only [1,4] and [128,512] attain KS=.02, each with
    # n=50. The next valid score is .08988047073853078; the best unbounded
    # score is .09177739977279314. The approved 1e-10 tie rule therefore
    # reaches smaller-lower priority, with retained counts exactly equal.
    record_property(
        "independent_selection",
        {
            "tied_windows": [[1, 4, 50], [128, 512, 50]],
            "ks": 0.02,
            "next_ks": 0.08988047073853078,
            "best_unbounded_ks": 0.09177739977279314,
        },
    )
    try:
        model = PowerLaw(data)
        result = model.fit(xmin_max=512, opt_max=True)
        _assert_fit(model, result, (1.3177664016166413, 1.9782315607400307), 1, base)
        if hasattr(model, "xmax"):
            assert model.xmax == 4
    finally:
        assert_array_equal(data, original)


def test_equal_count_fixed_lower_tie_chooses_unbounded_upper_through_fit_state():
    data = np.repeat([1.0, 2.0, 4.0], [50, 25, 25])
    original = data.copy()
    # With explicit L=1, the candidates are U=1,2,4,infinity. U=1 is
    # degenerate; U=2 has n=75 and KS=2/3. Both U=4 and infinity retain
    # 100 observations and have KS=1/2. Their likelihood optima differ:
    # bounded alpha=2.1251540661545896; unbounded alpha=1+4/(3*log(2)).
    # Public C/alpha distinguish the winner without an infinity sentinel.
    beta = 4 / (3 * math.log(2))
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1, opt_max=True)
        _assert_fit(model, result, (beta, 1 + beta), 1, data)
    finally:
        assert_array_equal(data, original)


@pytest.mark.parametrize(
    "bounded", [True, False], ids=["bounded-continuous", "unbounded-discrete"]
)
def test_failed_upper_refit_preserves_all_available_public_fit_fields(bounded, record_property):
    if bounded:
        data = np.repeat([1.025, 1.08, 1.15, 1.25, 1.38, 1.55, 1.78, 2.12, 2.65, 3.5, 30.0], 10)
        parameters = (1.5246865836407402, 2.1746821314411457)
        retained = data[data <= 3.5]
    else:
        data = np.repeat([1, 2, 4, 20], [30, 12, 10, 10])
        parameters = (0.5183425136587252, 1.7698533438102164)
        retained = data
    original = data.copy()
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1, opt_max=bounded)
    finally:
        assert_array_equal(data, original)
    # A real, independently checked successful fit is required before the
    # rejection can count as a preservation check. No fitted state is injected.
    _assert_fit(model, result, parameters, 1, retained)
    before = _snapshot(model)
    record_property(
        "positive_control_and_snapshot",
        {"fields": sorted(before), "fit": [float(value) for value in result]},
    )
    try:
        # Only ten samples remain above explicit L=20 in either fixture.
        # Automatic upper selection requires at least fifty even for a
        # fixed lower bound, so every possible window is ineligible.
        with pytest.raises(ValueError) as caught:
            model.fit(xmin=20, opt_max=True)
        record_property("failed_refit", _diagnostic(caught.value))
    finally:
        assert_array_equal(data, original)
        _assert_preserved(model, before)
    record_property("preservation_assertions_completed", True)


@pytest.mark.parametrize(
    "values,counts",
    [([1, 2, 1000], [40, 10, 1]), ([1, 2, 3, 100, 1000], [10000, 1, 1, 1, 1])],
    ids=["one-remote-outlier", "strong-skew"],
)
def test_discrete_outliers_are_excluded_by_the_independent_binary_winner(
    values, counts, record_property
):
    data = np.repeat(np.array(values, dtype=int), counts)
    original = data.copy()
    retained = data[data <= 2]
    expected_c = counts[0] / (counts[0] + counts[1])
    expected_alpha = math.log(counts[0] / counts[1]) / math.log(2)
    # On {1,2}, P(1)=1/(1+2**(-alpha)); its unique likelihood optimum
    # exactly matches the empirical masses. Independent enumeration includes
    # every observed upper endpoint and infinity. All other interior fits
    # have KS>1e-6, well outside the approved 1e-10 selection tie band.
    try:
        model = PowerLaw(data)
        result = model.fit(xmin=1, xmin_max=2, opt_max=True)
        _assert_fit(model, result, (expected_c, expected_alpha), 1, retained)
        # |dP(1)/dalpha|=log(2)*P(1)*P(2): at most .111 here.
        # An absolute alpha error 2e-10 gives KS error below 2.23e-11.
        # These margins are many thousands of double-precision ulps, yet
        # below the selection tie band; no exact floating zero is required.
        assert_allclose(model.alpha, expected_alpha, rtol=0, atol=2e-10)
        assert_allclose(model.C, expected_c, rtol=0, atol=2e-11)
        fitted_p_one = 1 / (1 + 2.0 ** (-model.alpha))
        assert_allclose(fitted_p_one, expected_c, rtol=0, atol=3e-11)
        if hasattr(model, "xmax"):
            assert model.xmax == 2
        before = _snapshot(model)
        record_property(
            "successful_binary_selection",
            {
                "retained_count": len(retained),
                "expected_C": expected_c,
                "expected_alpha": expected_alpha,
                "actual_C": float(model.C),
                "actual_alpha": float(model.alpha),
                "ks_from_normalized_binary_fit": abs(float(fitted_p_one) - expected_c),
                "snapshot_fields": sorted(before),
            },
        )
        try:
            # Empty explicit support is an approved ValueError, after the
            # required successful selection. It cannot make a failed fit pass.
            with pytest.raises(ValueError) as caught:
                model.fit(xmin=1001, opt_max=True)
            record_property(
                "empty_refit_error",
                {"category": type(caught.value).__name__, "message": str(caught.value)},
            )
        finally:
            _assert_preserved(model, before)
        record_property("outlier_preservation_completed", True)
    finally:
        assert_array_equal(data, original)
