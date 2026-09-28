"""Supplemental R1-R3/U1 public runtime, alignment, matching, and Git contracts."""

import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils import registration
from dphtools.utils.lm import lm


@pytest.mark.parametrize(
    "model",
    [
        registration.TranslationCPD,
        registration.RigidCPD,
        registration.SimilarityCPD,
        registration.AffineCPD,
    ],
)
def test_constructed_registration_representation_preserves_inputs_and_subsequent_fit(model):
    moving = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-1.0, 3.0]])
    fixed = moving + [0.2, -0.3]
    original_fixed, original_moving = fixed.copy(), moving.copy()
    reg = model(fixed, moving)
    representation, string = repr(reg), str(reg)
    assert isinstance(representation, str) and representation.strip()
    assert isinstance(string, str) and string.strip()
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)
    reg(maxiters=100, dist_tol=1e-8)
    assert_allclose(reg.transform(original_moving), original_fixed, atol=1e-5)
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)


@pytest.mark.parametrize("operation", ["estimate", "fit-normalized", "fit-unnormalized"])
def test_base_registration_rejects_calls_requiring_unimplemented_overrides(
    operation, record_property
):
    moving = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-1.0, 3.0]])
    reg = registration.BaseCPD(moving + [0.2, -0.3], moving.copy())
    with pytest.raises(NotImplementedError) as caught:
        if operation == "estimate":
            reg.estimate()
        else:
            reg(normalization=operation == "fit-normalized", init_var=0.5, maxiters=3)
    record_property("unsupported_base_error", f"{type(caught.value).__name__}: {caught.value}")


@pytest.mark.parametrize("normalization", [False, True])
@pytest.mark.parametrize(
    "dimension, iterations, initial_variance", [(2, 1, 0.5), (2, 4, 2.0), (3, 1, 2.0), (3, 4, 0.5)]
)
def test_translation_explicit_variance_bounded_fit_returns_consistent_finite_state(
    normalization, dimension, iterations, initial_variance, record_property
):
    moving = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 2.0, -2.0], [3.0, 4.0, 3.0], [-1.0, 3.0, 1.5]]
    )[:, :dimension].copy()
    displacement = np.array([0.2, -0.3, 0.1])[:dimension]
    fixed = moving + displacement
    original_fixed, original_moving = fixed.copy(), moving.copy()
    reg = registration.TranslationCPD(fixed, moving)
    returned_points = reg(
        normalization=normalization,
        init_var=initial_variance,
        maxiters=iterations,
        tol=0,
        dist_tol=0,
    )
    matrix, shift = np.asarray(reg.B), np.asarray(reg.translation).reshape(-1)
    record_property("returned_translation", shift.tolist())
    assert matrix.shape == (dimension, dimension)
    assert shift.shape == (dimension,)
    assert np.isfinite(matrix).all() and np.isfinite(shift).all()
    assert_allclose(matrix, np.eye(dimension), rtol=0, atol=1e-12)
    transformed = reg.transform(original_moving)
    assert np.shape(returned_points) == original_moving.shape
    assert np.isfinite(returned_points).all()
    assert_allclose(returned_points, transformed, rtol=1e-12, atol=1e-12)
    assert_allclose(transformed, original_moving + shift, rtol=1e-12, atol=1e-12)
    # Independently known translations fix the residual geometry even before
    # convergence: every corresponding point has error shift-displacement.
    # No amount of progress in a fixed number of iterations is prescribed.
    assert_allclose(
        transformed - original_fixed,
        np.broadcast_to(shift - displacement, transformed.shape),
        atol=1e-12,
    )
    unseen = np.array([[2.5, 1.5, -0.5], [-0.5, 2.0, 4.0]])[:, :dimension]
    assert_allclose(reg.transform(unseen), unseen + shift, rtol=1e-12, atol=1e-12)
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)


@pytest.mark.parametrize("iterations", [2, 5])
def test_bounded_two_parameter_lm_returns_one_accepted_parameter_point(
    iterations, record_property
):
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    start = np.array([0.1, -3.0])
    original_start = start.copy()
    calls = []

    def residual(p):
        calls.append(p.copy())
        return (p[0] ** 2 - 2) * x + np.exp(p[1]) - 3

    def jacobian(p):
        return np.column_stack((2 * p[0] * x, np.full(x.size, np.exp(p[1]))))

    outcome = run_lm_allowing_informative_numerical_failure(
        record_property,
        residual,
        start,
        Dfun=jacobian,
        maxfev=iterations,
        ftol=0,
        xtol=0,
        gtol=0,
        full_output=True,
    )
    assert_array_equal(start, original_start)
    if outcome is None:
        return
    result, covariance, info, message, status = outcome
    expected_residual = (result[0] ** 2 - 2) * x + np.exp(result[1]) - 3
    expected_jacobian = np.column_stack((2 * result[0] * x, np.full(x.size, np.exp(result[1]))))
    initial_residual = (0.1**2 - 2) * x + np.exp(-3) - 3
    assert np.isfinite(result).all()
    assert np.isfinite(info["fvec"]).all() and np.isfinite(info["fjac"]).all()
    assert_allclose(info["fvec"], expected_residual, rtol=1e-12, atol=1e-12)
    assert_allclose(info["fjac"], expected_jacobian, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(expected_residual) <= np.linalg.norm(initial_residual) + 1e-12
    assert info["nfev"] == len(calls)
    assert covariance is None
    assert isinstance(message, str) and message
    assert status in (1, 2, 4, 5)
    assert_array_equal(start, original_start)


def test_extreme_finite_ls_model_cannot_accept_nonfinite_or_worse_trial(record_property):
    exposure = np.array([1.0, 2.0, 4.0])
    calls = []
    trial_values = []

    def residual(p):
        calls.append(p.copy())
        values = (np.exp(p[0]) - 2) * exposure
        trial_values.append(values.copy())
        return values

    outcome = run_lm_allowing_informative_numerical_failure(
        record_property,
        residual,
        [-350.0],
        Dfun=lambda p: (np.exp(p[0]) * exposure)[:, None],
        method="ls",
        maxfev=3,
        ftol=0,
        xtol=0,
        gtol=0,
        full_output=True,
    )
    if outcome is None:
        return
    result, covariance, info, message, status = outcome
    record_property(
        "nonfinite_model_evaluations", sum(not np.isfinite(value).all() for value in trial_values)
    )
    record_property(
        "termination",
        {
            "status": status,
            "message": message,
            "parameters": result.tolist(),
            "nfev": info["nfev"],
            "actual_calls": len(calls),
        },
    )
    assert np.isfinite(result).all()
    assert np.isfinite(info["fvec"]).all() and np.isfinite(info["fjac"]).all()
    rate = np.exp(result[0])
    assert np.isfinite(rate) and rate > 0
    assert_allclose(info["fvec"], (rate - 2) * exposure, rtol=1e-12, atol=0)
    assert_allclose(info["fjac"], (rate * exposure)[:, None], rtol=1e-12, atol=0)
    # RSS is 21*(rate-2)**2. Compare its square root to avoid overflow in
    # the independent oracle. The input objective and derivative are finite.
    assert abs(rate - 2) <= abs(np.exp(-350.0) - 2) + 1e-12
    assert info["nfev"] == len(calls)
    assert covariance is None
    assert isinstance(message, str) and message
    assert status in (1, 2, 4, 5)


@pytest.mark.parametrize("starting_log_rate", [-400.0, 400.0])
def test_extreme_finite_poisson_model_keeps_consistent_accepted_state(
    starting_log_rate, record_property
):
    exposure = np.array([1.0, 2.0, 4.0, 8.0])
    counts = np.array([0.0, 4.0, 0.0, 12.0])
    original_counts = counts.copy()
    calls = []
    trial_values = []

    def predictions(p):
        calls.append(p.copy())
        values = np.exp(p[0]) * exposure
        trial_values.append(values.copy())
        return values, counts

    outcome = run_lm_allowing_informative_numerical_failure(
        record_property,
        predictions,
        [starting_log_rate],
        Dfun=lambda p: (np.exp(p[0]) * exposure)[:, None],
        method="mle",
        maxfev=3,
        ftol=0,
        xtol=0,
        gtol=0,
        full_output=True,
    )
    assert_array_equal(counts, original_counts)
    if outcome is None:
        return
    result, covariance, info, message, status = outcome
    record_property(
        "nonfinite_model_evaluations", sum(not np.isfinite(value).all() for value in trial_values)
    )
    record_property(
        "termination",
        {
            "status": status,
            "message": message,
            "parameters": result.tolist(),
            "nfev": info["nfev"],
            "actual_calls": len(calls),
        },
    )
    assert np.isfinite(result).all()
    assert np.isfinite(info["fvec"]).all() and np.isfinite(info["fjac"]).all()
    rate, initial_rate = np.exp(result[0]), np.exp(starting_log_rate)
    assert np.isfinite(rate) and rate > 0
    assert_allclose(info["fvec"], rate * exposure, rtol=1e-12, atol=0)
    assert_allclose(info["fjac"], (rate * exposure)[:, None], rtol=1e-12, atol=0)
    # From the supplied Poisson objective, all count-only terms cancel.
    objective_change = 15 * (rate - initial_rate) - 16 * (result[0] - starting_log_rate)
    assert objective_change <= 1e-12 * max(1, 15 * initial_rate)
    assert info["nfev"] == len(calls)
    assert covariance is None
    assert isinstance(message, str) and message
    assert status in (1, 2, 4, 5)
    assert_array_equal(counts, original_counts)


def run_lm_allowing_informative_numerical_failure(record_property, *args, **kwargs):
    """R1 permits diagnosed arithmetic limits, not arbitrary callback/API errors."""
    try:
        result = lm(*args, **kwargs)
        assert result is not None, "The solver returned no result or numerical-failure diagnostic."
        return result
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        assert str(error).strip(), "Numerical failure must carry a diagnostic."
        record_property("numerical_failure", f"{type(error).__name__}: {error}")
    except (ValueError, RuntimeError) as error:
        # Some numerical libraries use ValueError/RuntimeError for nonfinite
        # arrays or singular solves. Unrelated errors must still fail the test.
        numerical_diagnostic = re.search(
            r"non[- ]?finite|overflow|underflow|singular|not positive definite"
            r"|\binfs?\b|\bnans?\b|cannot.*represent",
            str(error),
            re.IGNORECASE,
        )
        if numerical_diagnostic is None:
            raise
        record_property("numerical_failure", f"{type(error).__name__}: {error}")
    return None


@pytest.mark.parametrize("iterations", [1, 3])
def test_finite_analytic_ls_stress_preserves_accepted_state_and_objective(
    iterations, record_property
):
    a = 1 / 101
    start = np.array([0.0])
    calls = []

    def residual(p):
        calls.append(p.copy())
        return np.array([1e150 * (1 - (p[0] / a) ** 2), 1e-100 * (p[0] - 1)])

    def jacobian(p):
        return np.array([[-2e150 * p[0] / a**2], [1e-100]])

    outcome = run_lm_allowing_informative_numerical_failure(
        record_property,
        residual,
        start,
        Dfun=jacobian,
        method="ls",
        full_output=True,
        maxfev=iterations,
        ftol=0,
        xtol=0,
        gtol=0,
    )
    record_property("actual_calls", len(calls))
    assert_array_equal(start, [0])
    if outcome is None:
        return
    result, covariance, info, message, status = outcome
    record_property(
        "termination",
        {
            "status": status,
            "message": message,
            "parameters": result.tolist(),
            "nfev": info["nfev"],
        },
    )
    assert np.isfinite(result).all()
    expected = np.array([1e150 * (1 - (result[0] / a) ** 2), 1e-100 * (result[0] - 1)])
    expected_jacobian = np.array([[-2e150 * result[0] / a**2], [1e-100]])
    assert np.isfinite(info["fvec"]).all() and np.isfinite(info["fjac"]).all()
    assert_allclose(info["fvec"], expected, rtol=1e-12, atol=0)
    assert_allclose(info["fjac"], expected_jacobian, rtol=1e-12, atol=0)
    # Initial squared residual is about 1e300, still representable. hypot
    # compares the equivalent norm without overflowing on a bad returned step.
    assert math.hypot(*expected) <= math.hypot(1e150, -1e-100) * (1 + 1e-12)
    assert info["nfev"] == len(calls)
    assert covariance is None
    assert isinstance(message, str) and message
    assert status in (1, 2, 4, 5)


@pytest.mark.parametrize("normalization", [False, True])
def test_tiny_positive_variance_large_translation_keeps_finite_consistent_state(normalization):
    moving = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-1.0, 3.0]])
    fixed = moving + [1000, -900]
    original_fixed, original_moving = fixed.copy(), moving.copy()
    reg = registration.TranslationCPD(fixed, moving)
    returned = reg(init_var=1e-12, normalization=normalization, maxiters=1)
    shift = np.asarray(reg.translation).reshape(-1)
    assert np.isfinite(shift).all() and np.isfinite(returned).all()
    assert_allclose(reg.B, np.eye(2), rtol=0, atol=1e-12)
    assert_allclose(returned, reg.transform(original_moving), rtol=1e-12, atol=1e-12)
    assert_allclose(returned, original_moving + shift, rtol=1e-12, atol=1e-12)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0], [5.0, -7.0]])
    transformed = reg.transform(unseen)
    assert_allclose(transformed, unseen + shift, rtol=1e-12, atol=1e-12)
    assert_allclose(np.diff(transformed, axis=0), np.diff(unseen, axis=0), atol=1e-10)
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)


@pytest.mark.parametrize("normalization", [False, True])
@pytest.mark.parametrize("fixed_count, moving_count", [(4, 5), (5, 4)])
def test_unequal_cloud_translation_returns_finite_parameters_and_preserves_geometry(
    normalization, fixed_count, moving_count
):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-1.0, 3.0]])
    fixed = points[:fixed_count].copy() + [0.2, -0.3]
    moving = points[:moving_count].copy()
    original_fixed, original_moving = fixed.copy(), moving.copy()
    reg = registration.TranslationCPD(fixed, moving)
    returned = reg(init_var=2.0, normalization=normalization, maxiters=2)
    shift = np.asarray(reg.translation).reshape(-1)
    assert shift.shape == (2,) and np.isfinite(shift).all()
    assert np.shape(returned) == original_moving.shape and np.isfinite(returned).all()
    assert_allclose(reg.B, np.eye(2), rtol=0, atol=1e-12)
    assert_allclose(returned, original_moving + shift, rtol=1e-12, atol=1e-12)
    assert_allclose(reg.transform(original_moving), returned, rtol=1e-12, atol=1e-12)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0], [5.0, -7.0]])
    transformed = reg.transform(unseen)
    assert_allclose(transformed, unseen + shift, rtol=1e-12, atol=1e-12)
    assert_allclose(np.diff(transformed, axis=0), np.diff(unseen, axis=0), atol=1e-12)
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)


def test_alignment_unequal_matched_sets_returns_finite_translation_and_unseen_geometry():
    fixed = pd.DataFrame([[0, 0], [2, 0], [10, 10]], columns=["x0", "y0"])
    moving = pd.DataFrame([[1, 0], [3.5, 0], [10.1, 10]], columns=fixed.columns)
    original_fixed, original_moving = fixed.copy(deep=True), moving.copy(deep=True)
    reg = registration.align(fixed, moving, only2d=True, model="translation", iters=3)
    shift = np.asarray(reg.translation).reshape(-1)
    assert shift.shape == (2,) and np.isfinite(shift).all()
    assert_allclose(reg.B, np.eye(2), rtol=0, atol=1e-12)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0], [5.0, -7.0]])
    transformed = reg.transform(unseen)
    assert np.isfinite(transformed).all()
    assert_allclose(transformed, unseen + shift, rtol=1e-12, atol=1e-12)
    assert_allclose(np.diff(transformed, axis=0), np.diff(unseen, axis=0), atol=1e-12)
    assert_frame_equal(fixed, original_fixed)
    assert_frame_equal(moving, original_moving)


@pytest.mark.parametrize("permuted", [False, True])
def test_noisy_translation_alignment_matches_independent_mean_shift(permuted):
    fixed = pd.DataFrame([[0, 0], [10, 0], [0, 10], [10, 10]], columns=["x0", "y0"])
    moving = pd.DataFrame(
        [[0.2, 0.1], [10.1, 0.2], [0.3, 10.1], [10.2, 10.3]], columns=fixed.columns
    )
    if permuted:
        fixed = fixed.iloc[[2, 0, 3, 1]].copy()
        moving = moving.iloc[[1, 3, 0, 2]].copy()
    original_fixed, original_moving = fixed.copy(deep=True), moving.copy(deep=True)
    reg = registration.align(fixed, moving, only2d=True, model="translation", iters=10)
    # Corresponding clusters are about 10 units apart. Their fixed-moving
    # offsets average to (-.8/4, -.7/4); no row order supplies correspondence.
    expected_shift = np.array([-0.2, -0.175])
    assert_allclose(np.asarray(reg.translation).reshape(-1), expected_shift, atol=1e-12)
    assert_allclose(reg.B, np.eye(2), rtol=0, atol=1e-12)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0]])
    assert_allclose(reg.transform(unseen), unseen + expected_shift, atol=1e-12)
    assert_frame_equal(fixed, original_fixed)
    assert_frame_equal(moving, original_moving)


def test_alignment_one_pass_cannot_report_both_success_and_failure(
    caplog, capsys, record_property
):
    fixed = pd.DataFrame([[0, 0], [10, 0], [0, 10], [10, 10]], columns=["x0", "y0"])
    moving = pd.DataFrame(
        [[0.2, 0.1], [10.1, 0.2], [0.3, 10.1], [10.2, 10.3]], columns=fixed.columns
    )
    with caplog.at_level(logging.DEBUG):
        registration.align(
            fixed, moving, only2d=True, model="translation", iters=1, atol=1e-12, rtol=0
        )
    captured = capsys.readouterr()
    messages = [record.getMessage() for record in caplog.records]
    messages.extend((captured.out + "\n" + captured.err).splitlines())
    record_property("alignment_diagnostics", messages)
    # only2d invokes one coordinate pass. No exact text, logger, convergence
    # speed, or requirement to emit a particular completion message is chosen.
    failure = re.compile(
        r"\bfail\w*\b|\bunsuccess\w*\b|\b(?:not|without)\s+converg\w*\b"
        r"|\b(?:exhaust\w*|exceed\w*)\b|iteration.*limit.*reach",
        re.I,
    )
    success = re.compile(r"\bsucceed\w*\b|\bsuccess\w*\b|\bconverged\b", re.I)
    failed = [message for message in messages if failure.search(message)]
    succeeded = [
        message for message in messages if not failure.search(message) and success.search(message)
    ]
    assert not (failed and succeeded), f"Conflicting completion diagnostics: {messages}"


@pytest.mark.parametrize("bad_coordinate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", ["fixed", "moving"])
def test_alignment_nonfinite_coordinates_raise_ordinary_validation_error(bad_coordinate, location):
    fixed = pd.DataFrame(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], columns=["x0", "y0"]
    )
    moving = fixed + [0.2, -0.3]
    (fixed if location == "fixed" else moving).iloc[1, 0] = bad_coordinate
    with pytest.raises((TypeError, ValueError)):
        registration.align(fixed, moving, only2d=True, model="translation", iters=2)


@pytest.mark.parametrize("method", ["tree", "brute"])
@pytest.mark.parametrize(
    "radius, fixed_members, moving_members",
    [(0.35, {0, 1}, {1, 3}), (1.1, {0, 1, 2, 3}, {0, 1, 2, 3})],
)
def test_tree_and_brute_matching_recover_known_radius_memberships(
    method, radius, fixed_members, moving_members
):
    fixed = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 9.0], [10.0, 7.0]])
    moving = np.array([[0.3, 8.6], [0.1, 0.2], [10.9, 7.4], [4.2, -0.1], [30.0, -20.0]])
    original_fixed, original_moving = fixed.copy(), moving.copy()
    actual_fixed, actual_moving = registration.closest_point_matches(
        fixed, moving, method=method, r=radius
    )
    # Independent nearest distances are sqrt(.05), sqrt(.05), .5, sqrt(.97).
    # Membership only: do not prescribe ordering, multiplicities, or pairing.
    assert set(actual_fixed) == fixed_members
    assert set(actual_moving) == moving_members
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)


@pytest.mark.parametrize("percentile", [10, 50, 90])
def test_distance_percentile_memberships_are_invariant_to_common_translation_and_scale(
    percentile, record_property
):
    fixed = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 9.0], [10.0, 7.0]])
    # Dyadic coordinates and power-of-two scales make the transformed
    # coordinate differences exact; distinct nearest distances avoid a tie
    # changing membership merely because of decimal roundoff.
    moving = np.array([[0.375, 8.5], [0.125, 0.25], [11.0, 7.25], [4.375, -0.125], [30.0, -20.0]])
    original_fixed, original_moving = fixed.copy(), moving.copy()
    first = registration.closest_point_matches(
        fixed, moving, method="brute", percentile=percentile
    )
    expected = tuple(set(indices) for indices in first)
    record_property(
        "original_memberships", [sorted(int(index) for index in group) for group in expected]
    )
    for scale, translation in [(1.0, [100.0, -30.0]), (0.125, [0.0, 0.0]), (8.0, [100.0, -30.0])]:
        transformed_fixed = scale * fixed + translation
        transformed_moving = scale * moving + translation
        before_fixed, before_moving = transformed_fixed.copy(), transformed_moving.copy()
        actual = registration.closest_point_matches(
            transformed_fixed, transformed_moving, method="brute", percentile=percentile
        )
        record_property(
            f"transformed_memberships_scale_{scale}",
            [sorted(int(index) for index in group) for group in actual],
        )
        assert tuple(set(indices) for indices in actual) == expected
        assert_array_equal(transformed_fixed, before_fixed)
        assert_array_equal(transformed_moving, before_moving)
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)


def test_matching_rejects_unknown_method():
    # R3 selects rejection, without selecting its exception family.
    with pytest.raises(Exception):
        registration.closest_point_matches(
            np.array([[0.0, 0.0], [1.0, 2.0]]),
            np.array([[0.1, 0.0], [1.1, 2.0]]),
            method="not-a-method",
        )


def test_model_selection_rejects_a_class_outside_registration_hierarchy():
    class NotARegistration:
        pass

    with pytest.raises(Exception):
        registration.choose_model(NotARegistration)


@pytest.mark.parametrize("git_absent", [False, True], ids=["nonrepository", "git-absent"])
def test_git_unavailability_is_observable_in_an_isolated_real_child(
    tmp_path, git_absent, record_property
):
    # The child has its own working/requested directories and Git discovery
    # ceiling. Only its PATH changes; no repository, global environment, or
    # subprocess implementation is replaced. Run pytest with a role-owned
    # --basetemp during A evidence collection.
    working, requested, empty_path = [tmp_path / name for name in ("working", "requested", "bin")]
    for directory in (working, requested, empty_path):
        directory.mkdir()
    environment = os.environ.copy()
    for variable in list(environment):
        if variable.startswith("GIT_"):
            del environment[variable]
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CEILING_DIRECTORIES=str(tmp_path.resolve()),
        MPLBACKEND="Agg",
    )
    if git_absent:
        environment["PATH"] = str(empty_path.resolve())
    # Install source-free diagnostics before any dependency or library import.
    script = r"""
import json
import logging
import sys
import warnings

sys.dont_write_bytecode = True
sys.excepthook = lambda kind, error, tb: sys.stderr.write(f"{kind.__name__}: {error}\n")
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{filename}:{lineno}: {category.__name__}: {message}\n"
)
logging.Formatter.formatException = lambda self, info: f"{info[0].__name__}: {info[1]}"
receipt = {"phase": "import", "warnings": [], "logs": []}

def show_warning(message, category, filename, lineno, file=None, line=None):
    receipt["warnings"].append({"category": category.__name__, "message": str(message),
                                "filename": filename, "lineno": lineno})
    (file or sys.stderr).write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = show_warning

class ReceiptLog(logging.Handler):
    def emit(self, record):
        receipt["logs"].append({"level": record.levelno, "message": record.getMessage()})

logging.getLogger().addHandler(ReceiptLog())
logging.getLogger().setLevel(logging.DEBUG)
sys.path.insert(0, sys.argv[1])
status = 2
try:
    from dphtools.utils import get_git
    receipt["phase"] = "get_git"
    result = get_git(sys.argv[2])
    if isinstance(result, bytes):
        result = result.decode(errors="replace")
    receipt["result"] = result if result is None or isinstance(result, (str, bool)) else repr(result)
    receipt["phase"] = "returned"
    status = 0
except Exception as error:
    # Preserve every error as evidence. The parent accepts only the ordinary
    # external-command exceptions selected below, never arbitrary exceptions.
    receipt["exception"] = {"class": type(error).__name__, "message": str(error)}
    sys.stderr.write(f"{type(error).__name__}: {error}\n")
receipt["status"] = status
print("GIT_BOUNDARY_RECEIPT " + json.dumps(receipt), flush=True)
raise SystemExit(status)
"""
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(Path(__file__).resolve().parents[1]),
            str(requested.resolve()),
        ],
        cwd=working,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    record_property(
        "git_child_process",
        {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr},
    )
    receipts = [
        line.removeprefix("GIT_BOUNDARY_RECEIPT ")
        for line in process.stdout.splitlines()
        if line.startswith("GIT_BOUNDARY_RECEIPT ")
    ]
    assert len(receipts) == 1, "The isolated Git call did not return an evidence receipt."
    receipt = json.loads(receipts[0])
    if "exception" in receipt:
        assert receipt["phase"] == "get_git", receipt
        assert receipt["exception"]["class"] in (
            "FileNotFoundError",
            "CalledProcessError",
        ), receipt
        assert receipt["exception"]["message"].strip(), receipt
        assert process.returncode == receipt["status"] == 2
    else:
        assert process.returncode == receipt["status"] == 0
        assert receipt["phase"] == "returned"
        result = receipt["result"]
        unavailable_value = result is None or result is False or result == "" or result == b""
        # Accept an informative unavailable result/diagnostic, without freezing
        # an undocumented fallback string or querying the surrounding checkout.
        diagnostics = "\n".join(
            [str(result), process.stderr, *[entry["message"] for entry in receipt["logs"]]]
        )
        unavailable_diagnostic = re.search(
            r"unknown|unavailable|not (?:a )?git repository|not found|not installed"
            r"|no such file|\bfatal\b|\berror\b|\bfail\w*\b|\babsent\b",
            diagnostics,
            re.I,
        )
        assert unavailable_value or unavailable_diagnostic, receipt


def test_paired_positive_one_dimensional_similarity_reports_its_linear_scale(record_property):
    # R1's supported special case fixes the multiplier without selecting a
    # multidimensional, reflected, affine, or angular definition of scale.
    multiplier, displacement = 2.0, 3.5
    moving = np.array([[-7.0], [-2.0], [1.0], [6.0], [14.0]])
    fixed = multiplier * moving + displacement
    held_out = np.array([[-10.0], [-0.75], [3.25], [20.0]])
    original_fixed, original_moving = fixed.copy(), moving.copy()
    original_held_out = held_out.copy()
    reg = registration.SimilarityCPD(fixed, moving)
    # The paired estimate has known correspondences and needs no iterative
    # convergence-speed assumption.
    reg.estimate()
    transformed = reg.transform(moving)
    transformed_held_out = reg.transform(held_out)
    reported_scale = np.asarray(reg.scale)
    record_property(
        "positive_similarity_1d",
        {
            "reported_scale": reported_scale.tolist(),
            "transformed": np.asarray(transformed).tolist(),
            "transformed_held_out": np.asarray(transformed_held_out).tolist(),
        },
    )
    assert_array_equal(fixed, original_fixed)
    assert_array_equal(moving, original_moving)
    assert_array_equal(held_out, original_held_out)
    assert np.shape(transformed) == original_fixed.shape
    assert np.shape(transformed_held_out) == original_held_out.shape
    assert_allclose(transformed, original_fixed, rtol=1e-12, atol=1e-12)
    assert_allclose(
        transformed_held_out,
        multiplier * original_held_out + displacement,
        rtol=1e-12,
        atol=1e-12,
    )
    # Accept a scalar or any one-element array layout for this single scale.
    assert reported_scale.size == 1
    assert np.isfinite(reported_scale).all() and np.all(reported_scale > 0)
    assert_allclose(reported_scale, multiplier, rtol=1e-12, atol=1e-12)


def test_lpsvd_odd_wide_window_matches_analytic_terms_and_held_out_signal(record_property):
    from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

    sample_count, model_order, lfactor = 83, 4, 0.8
    # The packet's rank condition permits bias removal: L=floor(83*.8)=66,
    # leaving 17 prediction equations. Both counts strictly exceed M=4.
    sample = np.arange(sample_count, dtype=float)
    signal = 2.4 * np.exp(-0.025 * sample) * np.cos(math.tau * 0.109375 * sample + 0.45)
    signal += 0.9 * np.exp(-0.045 * sample) * np.cos(math.tau * 0.234375 * sample + 3.6)
    original_signal = signal.copy()
    expected_terms = np.array(
        [
            [0.45, -0.234375, -0.045, -3.6],
            [1.2, -0.109375, -0.025, -0.45],
            [1.2, 0.109375, -0.025, 0.45],
            [0.45, 0.234375, -0.045, 3.6],
        ]
    )
    fitted = LPSVD(signal, M=model_order, lfactor=lfactor, removebias=True)
    assert_array_equal(signal, original_signal)
    fields = ["amps", "freqs", "damps", "phase"]
    actual = fitted.sort_values("freqs")[fields].to_numpy()
    record_property("wide_window_analytic_terms", actual.tolist())
    assert actual.shape == expected_terms.shape
    # Separated, moderately damped, noiseless float64 components use the
    # existing real-signal tests' allowances for SVD/root-solving roundoff:
    # 1e-8 relative, 1e-9 absolute, and 1e-8 radians for wrapped phase.
    assert_allclose(actual[:, :3], expected_terms[:, :3], rtol=1e-8, atol=1e-9)
    phase_error = (actual[:, 3] - expected_terms[:, 3] + math.pi) % math.tau - math.pi
    assert_allclose(phase_error, 0, rtol=0, atol=1e-8)

    stop = sample_count + 19
    expected = np.array(
        [
            2.4 * math.exp(-0.025 * n) * math.cos(math.tau * 0.109375 * n + 0.45)
            + 0.9 * math.exp(-0.045 * n) * math.cos(math.tau * 0.234375 * n + 3.6)
            for n in range(stop)
        ]
    )
    # Use only physical coefficient fields; uncertainty is outside this case.
    coefficients = fitted[fields].copy()
    original_coefficients = coefficients.copy(deep=True)
    template = np.zeros(stop, dtype=float)
    original_template = template.copy()
    reconstructed = reconstruct_signal(
        coefficients, template, ampcutoff=0, freqcutoff=0, dampcutoff=0
    )
    assert reconstructed.shape == expected.shape
    record_property(
        "wide_window_max_absolute_signal_errors",
        {
            "training": float(
                np.max(np.abs(reconstructed[:sample_count] - expected[:sample_count]))
            ),
            "held_out": float(
                np.max(np.abs(reconstructed[sample_count:] - expected[sample_count:]))
            ),
        },
    )
    assert_allclose(reconstructed[:sample_count], expected[:sample_count], rtol=1e-8, atol=1e-9)
    assert_allclose(reconstructed[sample_count:], expected[sample_count:], rtol=1e-8, atol=1e-9)
    assert_array_equal(signal, original_signal)
    assert_array_equal(template, original_template)
    assert_frame_equal(coefficients, original_coefficients)
