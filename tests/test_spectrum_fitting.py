"""Independent public-entry-point tests for SPECTRUM-FITTING-001 contract R3.

Expected spectra use the contract's unit-height definitions, not product helpers.
See docs/tasks/SPECTRUM-FITTING-001-tests.md for mapping and tolerances.
Run with the source-free launcher in the A packet during blind A/B work.
"""

import json
import math
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile

FAMILIES = ("gauss", "lorentz", "voigt")
ROW_SIZE = {"gauss": 3, "lorentz": 3, "voigt": 4}
PARAM_RTOL = 3e-5
PARAM_ATOL = 2e-5
MODEL_ATOL = 2e-7


@pytest.fixture(scope="session")
def fit():
    # Importing the approved public module is allowed; never inspect its source.
    from dphtools.utils.fitfuncs import spectrum_fit

    return spectrum_fit


def _model(x, rows, family, background=()):
    """Contract equations: point samples, individual component peak heights."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for row in rows:
        amplitude, center, width = row[:3]
        d = x - center
        if family == "gauss":
            profile = np.exp(-0.5 * (d / width) ** 2)
        elif family == "lorentz":
            profile = width**2 / (d**2 + width**2)
        else:
            gamma = row[3]
            profile = voigt_profile(d, width, gamma) / voigt_profile(0, width, gamma)
        y += amplitude * profile
    if len(background):
        y += background[0]
    if len(background) == 2:
        y += background[1] * (x - x[0])
    return y


def _rows(family):
    rows = np.array([[4.2, 14.3, 1.1], [2.7, 18.2, 1.5]])
    if family == "voigt":
        rows[:, 2] = [0.9, 1.2]
        rows = np.column_stack((rows, [0.5, 0.7]))
    return rows


def _starts(rows):
    starts = rows.copy()
    starts[:, 0] *= 0.88
    starts[:, 1] += np.array([0.12, -0.16])[: len(rows)]
    starts[:, 2:] *= 1.15
    return starts


def _noise(n):
    i = np.arange(n, dtype=float)
    return 0.018 * (np.sin(1.63 * i) + 0.6 * np.cos(0.71 * i))


def _jacobian(x, parameters, family, npeak):
    """Independent five-point differences of the stated physical model."""
    size = ROW_SIZE[family] * npeak

    def model_at(p):
        return _model(x, p[:size].reshape(npeak, -1), family, p[size:])

    columns = []
    for j, value in enumerate(parameters):
        h = np.finfo(float).eps ** 0.2 * max(1.0, abs(value))
        step = np.zeros_like(parameters)
        step[j] = h
        columns.append(
            (
                model_at(parameters - 2 * step)
                - 8 * model_at(parameters - step)
                + 8 * model_at(parameters + step)
                - model_at(parameters + 2 * step)
            )
            / (12 * h)
        )
    return np.column_stack(columns)


def _check_result(result, x, data, family, npeak, nbackground):
    peaks = np.asarray(result.peak_parameters)
    background = np.asarray(result.background_parameters)
    fitted = np.asarray(result.fitted)
    residuals = np.asarray(result.residuals)
    covariance = np.asarray(result.covariance)
    nparameter = npeak * ROW_SIZE[family] + nbackground
    assert peaks.shape == (npeak, ROW_SIZE[family])
    assert background.shape == (nbackground,)
    assert fitted.shape == residuals.shape == np.asarray(data).shape
    assert covariance.shape == (nparameter, nparameter)
    for array in (peaks, background, fitted, residuals, covariance):
        assert np.isrealobj(array)
        assert np.all(np.isfinite(array))
    assert np.all(peaks[:, 0] > 0)
    assert np.all(peaks[:, 2:] > 0)
    assert np.all(np.diff(peaks[:, 1]) >= 0)
    assert_allclose(fitted, _model(x, peaks, family, background), rtol=2e-10, atol=2e-10)
    assert_allclose(residuals, np.asarray(data) - fitted, rtol=2e-12, atol=2e-12)
    assert_allclose(covariance, covariance.T, rtol=2e-10, atol=2e-12)


def _check_covariance_and_objective(result, x, data, family):
    """RSS/(n-p) (J.T J)^-1 in sorted physical coordinates; unweighted optimum."""
    peaks = np.asarray(result.peak_parameters)
    p = np.concatenate((peaks.ravel(), result.background_parameters))
    j = _jacobian(x, p, family, len(peaks))
    residual = np.asarray(data) - _model(x, peaks, family, result.background_parameters)
    norms = np.linalg.norm(j, axis=0)
    # The test fixtures must be locally identifiable in dimensionless coordinates.
    assert np.linalg.cond(j / norms) < 1e4
    rss = residual @ residual
    assert rss > 1e-8  # Nonzero noise makes covariance scale discriminating.
    expected = np.linalg.inv(j.T @ j) * rss / (len(x) - len(p))
    scales = np.sqrt(np.diag(expected))
    assert np.all(scales > 0)
    # Normalization measures every entry in its natural uncertainty units.
    assert_allclose(
        np.asarray(result.covariance) / np.outer(scales, scales),
        expected / np.outer(scales, scales),
        rtol=0.01,
        atol=2e-4,
    )
    assert np.min(np.linalg.eigvalsh(np.asarray(result.covariance))) >= -1e-12
    # J.T residual == 0 at an unconstrained, unweighted least-squares optimum.
    assert np.max(np.abs(j.T @ residual) / (norms * np.linalg.norm(residual))) < 3e-4


def _preserving_call(fit, data, xdata=None, **kwargs):
    inputs = [value for value in (data, xdata, kwargs.get("guesses")) if value is not None]
    snapshots = [np.array(value, copy=True) for value in inputs]
    try:
        result = fit(data, xdata, **kwargs)
    finally:
        for value, snapshot in zip(inputs, snapshots):
            assert_array_equal(value, snapshot)
    return result


@pytest.mark.parametrize("name", ["gauss", "Gaussian", "GAUSS", "gAuSsIaN"])
@pytest.mark.parametrize("as_list", [False, True])
def test_s01_full_gaussian_guesses_and_schema(fit, name, as_list):
    x = np.linspace(-6, 8, 281)
    truth = np.array([[4.7, 0.65, 0.83]])
    guesses = np.array([[3.9, 0.82, 1.0]])
    y = _model(x, truth, "gauss")
    if as_list:
        x, y, guesses = x.tolist(), y.tolist(), guesses.tolist()
    result = _preserving_call(fit, y, x, peak_type=name, guesses=guesses, background="none")
    _check_result(result, x, y, "gauss", 1, 0)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("name", ["lorentz", "Lorentzian", "LORENTZ", "lOrEnTzIaN"])
def test_s02_full_lorentzian_guesses(fit, name):
    x = np.linspace(-8, 9, 341)
    truth = np.array([[3.4, -0.9, 1.27]])
    y = _model(x, truth, "lorentz")
    result = _preserving_call(
        fit, y, x, peak_type=name, guesses=[[2.9, -0.7, 1.5]], background="none"
    )
    _check_result(result, x, y, "lorentz", 1, 0)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("name", ["voigt", "VoIgT"])
def test_s03_full_voigt_guesses_independent_widths(fit, name):
    x = np.linspace(-9, 11, 401)
    truth = np.array([[5.3, 0.35, 0.72, 0.46]])
    y = _model(x, truth, "voigt")
    result = _preserving_call(
        fit, y, x, peak_type=name, guesses=[[4.6, 0.5, 0.85, 0.38]], background="none"
    )
    _check_result(result, x, y, "voigt", 1, 0)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_s04_center_guesses_joint_components(fit, family):
    x = np.linspace(7, 31, 481)
    truth = _rows(family)
    y = _model(x, truth, family, [0.7])
    centers = np.array([18.35, 14.15])
    result = _preserving_call(fit, y, x, peak_type=family, guesses=centers)
    _check_result(result, x, y, family, 2, 1)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.background_parameters, [0.7], rtol=PARAM_RTOL, atol=PARAM_ATOL)


def test_s04_center_guesses_identifiable_overlap_with_one_maximum(fit):
    x = np.linspace(-7, 8, 601)
    truth = np.array([[3.3, -0.6, 0.8], [2.1, 0.8, 1.0]])
    y = _model(x, truth, "gauss", [0.4])
    assert len(find_peaks(y)[0]) == 1  # Fixture property, not a discovery promise.
    result = _preserving_call(fit, y, x, guesses=[-0.65, 0.9])
    _check_result(result, x, y, "gauss", 2, 1)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("truncated_side", ["left", "right"])
def test_s04_s08_center_guess_broad_component_truncated_half_height(fit, family, truncated_side):
    # Post-implementation correction: a missing half-height crossing does not
    # remove the curvature that identifies this component's physical width.
    row = [3.4, 7.3, 2.5] if family != "voigt" else [3.4, 7.3, 2.5, 1.25]
    truth = np.array([row])
    offsets = (-0.4, 3.0) if truncated_side == "left" else (-3.0, 0.4)
    x = row[1] + row[2] * np.linspace(*offsets, 241)
    y = _model(x, truth, family)
    truncated = y[0] if truncated_side == "left" else y[-1]
    observed_tail = y[-1] if truncated_side == "left" else y[0]
    assert truncated > row[0] / 2 > observed_tail
    result = _preserving_call(
        fit, y, x, peak_type=family, guesses=np.array([row[1]]), background="none"
    )
    _check_result(result, x, y, family, 1, 0)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
    assert_allclose(result.residuals, 0, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize(
    "controls,count",
    [
        ({}, 3),
        ({"prominence": 0.5}, 2),
        ({"distance": 60}, 2),
        ({"prominence": 0.1, "distance": 1.5}, 3),
    ],
)
@pytest.mark.parametrize("coordinate_scale", [None, 0.01])
def test_s05_automatic_discovery_controls_and_index_default(
    fit, controls, count, coordinate_scale
):
    indices = np.arange(201, dtype=float)
    truth = np.array([[3.5, 45, 5], [0.25, 90, 4], [2.6, 145, 6]])
    y = _model(indices, truth, "gauss", [0.4])
    x = indices if coordinate_scale is None else coordinate_scale * indices
    kwargs = dict(controls)
    if coordinate_scale is not None:
        kwargs["xdata"] = x
    result = _preserving_call(fit, y, **kwargs)
    _check_result(result, x, y, "gauss", count, 1)
    expected = truth.copy() if count == 3 else truth[[0, 2]].copy()
    expected[:, 1:] *= 1.0 if coordinate_scale is None else coordinate_scale
    if count == 3:
        assert_allclose(result.peak_parameters, expected, rtol=PARAM_RTOL, atol=PARAM_ATOL)
        assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
    else:
        # The omitted weak component changes the optimum; only candidate count,
        # location, reconstruction, and unweighted stationarity are promised.
        assert_allclose(result.peak_parameters[:, 1], expected[:, 1], rtol=0, atol=0.003)
        _check_covariance_and_objective(result, x, y, "gauss")


@pytest.mark.parametrize("family", FAMILIES)
def test_s06_constant_background_and_covariance(fit, family):
    x = np.linspace(7, 31, 81)
    truth = _rows(family)[:1]
    y = _model(x, truth, family, [1.25]) + _noise(len(x))
    result = _preserving_call(fit, y, x, peak_type=family, guesses=_starts(truth))
    _check_result(result, x, y, family, 1, 1)
    assert_allclose(result.peak_parameters, truth, rtol=0, atol=0.04)
    assert_allclose(result.background_parameters, [1.25], rtol=0, atol=0.01)
    _check_covariance_and_objective(result, x, y, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_s07_linear_background_first_coordinate_origin(fit, family):
    x = np.linspace(7, 31, 241)
    truth = _rows(family)
    coefficients = np.array([1.1, -0.025])
    y = _model(x, truth, family, coefficients)
    result = _preserving_call(
        fit, y, x, peak_type=family, guesses=_starts(truth), background="linear"
    )
    _check_result(result, x, y, family, 2, 2)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.background_parameters, coefficients, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_s08_no_background_multicomponent_sum(fit, family):
    x = np.linspace(7, 31, 241)
    truth = _rows(family)
    y = _model(x, truth, family)
    result = _preserving_call(
        fit, y, x, peak_type=family, guesses=_starts(truth), background="none"
    )
    _check_result(result, x, y, family, 2, 0)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_s09_nonuniform_units_full_sorted_covariance_and_storage(fit, family):
    x = 7 + 24 * np.linspace(0, 1, 181) ** 1.8
    truth = _rows(family)
    y = _model(x, truth, family, [1.1, -0.025]) + _noise(len(x))
    guesses = _starts(truth)[::-1].copy()
    result = _preserving_call(fit, y, x, peak_type=family, guesses=guesses, background="linear")
    _check_result(result, x, y, family, 2, 2)
    assert_allclose(result.peak_parameters, truth, rtol=0, atol=0.05)
    assert_allclose(result.background_parameters, [1.1, -0.025], rtol=0, atol=0.01)
    _check_covariance_and_objective(result, x, y, family)
    snapshots = [a.copy() for a in (x, y, guesses)]
    for name in ("peak_parameters", "background_parameters", "covariance", "fitted", "residuals"):
        array = np.asarray(getattr(result, name))
        if array.size and array.flags.writeable:
            array.flat[0] += 1
        for original, snapshot in zip((x, y, guesses), snapshots):
            assert_array_equal(original, snapshot)


@pytest.fixture
def valid_data():
    x = np.linspace(-4, 4, 41)
    return x, _model(x, [[3, 0.2, 0.9]], "gauss")


@pytest.mark.parametrize("shape", [(), (41, 1), (1, 41, 1)])
def test_s10_non_1d_data(fit, valid_data, shape):
    x, y = valid_data
    if shape == ():
        # Scalar rejection also follows from too few observations; only the
        # matrix/3-D cases independently discriminate dimension validation.
        data, x = y[0], x[:1]
    else:
        data = y.reshape(shape)
    with pytest.raises(ValueError):
        fit(data, x, guesses=[[3, 0.2, 0.9]], background="none")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_s11_nonfinite_data(fit, valid_data, value):
    x, y = valid_data
    y[10] = value
    with pytest.raises(ValueError):
        fit(y, x, guesses=[0.2])


@pytest.mark.parametrize("imaginary", [0, 0.1])
def test_s12_complex_data(fit, valid_data, imaginary):
    x, y = valid_data
    y = y.astype(complex)
    y[10] += imaginary * 1j
    with pytest.raises(ValueError):
        fit(y, x, guesses=[0.2])


@pytest.mark.parametrize("shape", ["scalar", "matrix", "short", "long"])
def test_s13_coordinate_shape_length(fit, valid_data, shape):
    x, y = valid_data
    coordinates = {
        "scalar": x[0],
        "matrix": x.reshape(41, 1),
        "short": x[:-1],
        "long": np.append(x, x[-1] + (x[-1] - x[-2])),
    }[shape]
    with pytest.raises(ValueError):
        fit(y, coordinates, guesses=[[3, 0.2, 0.9]], background="none")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1j, 0j])
def test_s14_coordinate_real_finite_domain(fit, valid_data, value):
    x, y = valid_data
    if isinstance(value, complex):
        x = x.astype(complex)
        x[10] += value
    else:
        x[10] = value
    with pytest.raises(ValueError):
        fit(y, x, guesses=[0.2])


@pytest.mark.parametrize("change", ["duplicate", "descending", "one_reversal"])
def test_s15_strict_coordinate_order(fit, valid_data, change):
    x, y = valid_data
    if change == "duplicate":
        x[10] = x[9]
    elif change == "descending":
        x = x[::-1]
    else:
        x[[10, 11]] = x[[11, 10]]
    with pytest.raises(ValueError):
        fit(y, x, guesses=[0.2])


@pytest.mark.parametrize(
    "guesses",
    [[], np.empty((0, 3)), np.empty((1, 0)), 0.2, np.ones((1, 1, 3)), [[1, 0, 1], [2, 0]]],
)
def test_s16_empty_or_malformed_guesses(fit, valid_data, guesses):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=guesses)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("delta", [-1, 1])
def test_s16_wrong_full_guess_row_width(fit, valid_data, family, delta):
    guesses = np.ones((1, ROW_SIZE[family] + delta))
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], peak_type=family, guesses=guesses)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1j, 0j])
@pytest.mark.parametrize("centers_only", [False, True])
def test_s17_guess_real_finite_domain(fit, valid_data, value, centers_only):
    guesses = [value] if centers_only else [[3, value, 0.9]]
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=guesses)


@pytest.mark.parametrize(
    "family,field",
    [
        ("gauss", 0),
        ("gauss", 2),
        ("lorentz", 0),
        ("lorentz", 2),
        ("voigt", 0),
        ("voigt", 2),
        ("voigt", 3),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1j, 0j])
def test_s17_nonfinite_amplitude_or_width(fit, valid_data, family, field, value):
    guesses = np.ones(
        (1, ROW_SIZE[family]), dtype=complex if isinstance(value, complex) else float
    )
    guesses[0, field] = value
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], peak_type=family, guesses=guesses)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("value", [0, -1])
def test_s18_nonpositive_initial_amplitude(fit, valid_data, family, value):
    guesses = np.ones((1, ROW_SIZE[family]))
    guesses[0, 0] = value
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], peak_type=family, guesses=guesses)


@pytest.mark.parametrize(
    "family,field", [("gauss", 2), ("lorentz", 2), ("voigt", 2), ("voigt", 3)]
)
@pytest.mark.parametrize("value", [0, -0.5])
def test_s19_nonpositive_initial_width(fit, valid_data, family, field, value):
    guesses = np.ones((1, ROW_SIZE[family]))
    guesses[0, field] = value
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], peak_type=family, guesses=guesses)


@pytest.mark.parametrize("family", ["gauss", "lorentz"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("initial_outside", [False, True])
def test_s20_free_initial_and_fitted_edge_tail_centers(fit, family, side, initial_outside):
    x = np.linspace(0, 6, 241)
    center = -0.4 if side == "left" else 6.4
    start = (
        (-0.65 if initial_outside else 0.15)
        if side == "left"
        else (6.65 if initial_outside else 5.85)
    )
    truth = np.array([[3.5, center, 1.1]])
    y = _model(x, truth, family)
    result = _preserving_call(
        fit, y, x, peak_type=family, guesses=[[3.1, start, 0.95]], background="none"
    )
    _check_result(result, x, y, family, 1, 0)
    assert_allclose(result.peak_parameters, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
    assert result.peak_parameters[0, 1] < x[0] or result.peak_parameters[0, 1] > x[-1]


@pytest.mark.parametrize("value", ["voight", "mixed", "", None, 3, ["gauss"]])
def test_s21_invalid_profile(fit, valid_data, value):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=[0.2], peak_type=value)


@pytest.mark.parametrize("value", ["quadratic", "", None, 3, ["constant"]])
def test_s22_invalid_background(fit, valid_data, value):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=[0.2], background=value)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf, 1j, "high", [0, 10]])
def test_s23_invalid_prominence(fit, valid_data, value):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], prominence=value)


@pytest.mark.parametrize("value", [0, 0.5, np.nan, np.inf, -np.inf, 1j, "far", [2, 3]])
def test_s24_invalid_distance(fit, valid_data, value):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], distance=value)


@pytest.mark.parametrize("guesses", [[0.2], [[3, 0.2, 0.9]]])
@pytest.mark.parametrize(
    "control", [{"prominence": 0}, {"distance": 1}, {"prominence": 0.1, "distance": 2}]
)
def test_s25_explicit_guesses_reject_discovery_controls(fit, valid_data, guesses, control):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=guesses, **control)


@pytest.mark.parametrize("data", [np.zeros(41), np.arange(41), -np.arange(41)])
def test_s26_no_local_maxima(fit, data):
    with pytest.raises(ValueError):
        fit(data)


def test_s26_no_peaks_meet_prominence(fit, valid_data):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], prominence=10)


def test_s27_empty_data(fit):
    with pytest.raises(ValueError):
        fit([])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("background,nbackground", [("none", 0), ("constant", 1), ("linear", 2)])
@pytest.mark.parametrize("sample_delta", [-1, 0])
@pytest.mark.parametrize("npeak", [1, 2])
def test_s27_positive_residual_degrees_of_freedom(
    fit, family, background, nbackground, sample_delta, npeak
):
    n = npeak * ROW_SIZE[family] + nbackground + sample_delta
    x = np.linspace(-2, 2, n)
    guesses = [[3, 0.2, 0.9]] if family != "voigt" else [[3, 0.2, 0.9, 0.4]]
    if npeak == 2:
        guesses.append([2, -0.7, 0.6] if family != "voigt" else [2, -0.7, 0.6, 0.3])
    y = _model(x, guesses, family)
    with pytest.raises(ValueError):
        fit(y, x, peak_type=family, guesses=guesses, background=background)


@pytest.mark.parametrize("value", [0, -1, 1.5, np.nan, np.inf, "10", None, [10]])
def test_s28_invalid_evaluation_limit(fit, valid_data, value):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=[0.2], max_nfev=value)


def test_s29_exhausted_default_optimizer_preserves_inputs(fit):
    x = np.linspace(-6, 6, 121)
    y = _model(x, [[4, 0.4, 0.9]], "gauss")
    guesses = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError):
        _preserving_call(fit, y, x, guesses=guesses, background="none", max_nfev=1)


def test_s29_failed_custom_status_preserves_inputs(fit, valid_data):
    def nonconvergent(residual, initial, *, bounds, max_nfev):
        return SimpleNamespace(
            x=np.asarray(initial).copy(), success=False, message="evaluation limit reached"
        )

    with pytest.raises(RuntimeError):
        _preserving_call(
            fit,
            valid_data[1],
            valid_data[0],
            guesses=np.array([[3, 0.2, 0.9]]),
            optimizer=nonconvergent,
        )


def test_s29_finite_subnormal_voigt_numerical_boundary_preserves_inputs(fit):
    # V(s*d; s*sigma, s*gamma) = V(d; sigma, gamma)/s. Its unit-height
    # ratio stays finite even when the separate normalized densities overflow.
    width = np.finfo(float).tiny / 1024
    x = width * np.linspace(-4, 4, 81)
    scaled_x = x / width
    scaled_truth = np.array([[3.0, 0.0, 1.0, 1.0]])
    y = _model(scaled_x, scaled_truth, "voigt")
    guesses = np.array([[3.0, 0.0, width, width]])
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(guesses))
    assert np.all(guesses[:, [0, 2, 3]] > 0)
    try:
        result = _preserving_call(fit, y, x, peak_type="voigt", guesses=guesses, background="none")
    except RuntimeError as failure:
        # Numerical inability is allowed; input validation and leaked solver
        # exceptions are not. A robust finite solution is also allowed below.
        print("Finite subnormal Voigt numerical failure:", failure)
    else:
        peaks = np.asarray(result.peak_parameters)
        assert peaks.shape == (1, 4)
        assert np.isrealobj(peaks) and np.all(np.isfinite(peaks))
        assert np.all(peaks[:, [0, 2, 3]] > 0)
        scaled_peaks = peaks.copy()
        scaled_peaks[:, 1:] /= width
        assert_allclose(scaled_peaks, scaled_truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
        assert np.asarray(result.background_parameters).shape == (0,)
        covariance = np.asarray(result.covariance)
        assert covariance.shape == (4, 4) and np.isrealobj(covariance)
        for values in (result.fitted, result.residuals):
            values = np.asarray(values)
            assert values.shape == y.shape
            assert np.isrealobj(values) and np.all(np.isfinite(values))
        assert_allclose(
            result.fitted, _model(scaled_x, scaled_peaks, "voigt"), rtol=2e-10, atol=2e-10
        )
        assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
        assert_allclose(result.residuals, y - result.fitted, rtol=2e-12, atol=2e-12)
        assert_allclose(result.residuals, 0, rtol=0, atol=MODEL_ATOL)
        print("Finite subnormal Voigt returned a verified finite exact-model fit")


def _stable_narrow_component(x, row, family):
    """R3 model; physical Jacobian columns scaled by (height, width, width)."""
    amplitude, center, width = map(float, row)
    model, sensitivity = [], []
    for coordinate in x:
        d = float(coordinate) - center
        if family == "gauss":
            # Even the largest finite height times exp(-64**2/2) rounds to
            # zero. Bound d/width before division; never square the width.
            z = d / width if abs(d) <= 64 * width else math.copysign(math.inf, d)
            value = math.exp(math.log(amplitude) - 0.5 * z * z)
            derivatives = (value, value * z, value * z * z) if value else (0.0,) * 3
        else:
            scale = max(abs(d), width)
            t, u = d / scale, width / scale
            denominator = t * t + u * u
            # Log evaluation retains height*profile when profile alone would
            # underflow. The scaled denominator is at least one.
            value = math.exp(
                math.log(amplitude)
                + 2 * (math.log(width) - math.log(scale))
                - math.log(denominator)
            )
            derivatives = (
                value,
                value * 2 * t * u / denominator,
                value * 2 * t * t / denominator,
            )
        model.append(value)
        sensitivity.append(derivatives)
    return np.asarray(model), np.asarray(sensitivity)


@pytest.mark.parametrize("family", ["gauss", "lorentz"])
def test_s29_smallest_positive_width_point_samples_preserve_inputs(fit, family):
    # Post-implementation regression: this is a valid point-sampled component,
    # much narrower than the sampling distance, not an identifiable-width case.
    x = np.linspace(-1, 1, 41)
    y = np.zeros(41)
    y[20] = 2
    width = np.nextafter(0.0, 1.0)
    guesses = [[2.0, 0.0, width]]
    assert width == np.finfo(float).smallest_subnormal and width > 0
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(guesses))
    assert len(y) > 3
    initial_model, _ = _stable_narrow_component(x, guesses[0], family)
    assert_array_equal(initial_model, y)
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        try:
            result = _preserving_call(
                fit, y, x, peak_type=family, guesses=guesses, background="none"
            )
        except RuntimeError as failure:
            print(f"Smallest-positive-width {family} numerical failure: {failure}")
            return
        except Exception as failure:
            print(f"Smallest-positive-width {family} leaked {type(failure).__name__}: {failure}")
            raise
        finally:
            for diagnostic in diagnostics:
                print(
                    warnings.formatwarning(
                        diagnostic.message,
                        diagnostic.category,
                        diagnostic.filename,
                        diagnostic.lineno,
                        line="",
                    ),
                    file=sys.stderr,
                    end="",
                )
    peaks = np.asarray(result.peak_parameters)
    assert peaks.shape == (1, 3)
    assert np.isrealobj(peaks) and np.all(np.isfinite(peaks))
    assert np.all(peaks[:, [0, 2]] > 0)
    background = np.asarray(result.background_parameters)
    assert background.shape == (0,) and np.isrealobj(background)
    for values in (result.fitted, result.residuals):
        values = np.asarray(values)
        assert values.shape == y.shape
        assert np.isrealobj(values) and np.all(np.isfinite(values))
    model, jacobian = _stable_narrow_component(x, peaks[0], family)
    assert_allclose(result.fitted, model, rtol=2e-10, atol=2e-10)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
    assert_allclose(result.residuals, y - result.fitted, rtol=2e-12, atol=2e-12)
    assert_allclose(result.residuals, 0, rtol=0, atol=MODEL_ATOL)
    covariance = np.asarray(result.covariance)
    assert covariance.shape == (3, 3) and np.isrealobj(covariance)
    assert_allclose(covariance, covariance.T, rtol=2e-10, atol=2e-12, equal_nan=True)
    diagonal = np.diag(covariance)
    assert np.all(diagonal[np.isfinite(diagonal)] >= 0)
    # Positive column rescaling preserves physical rank and estimability.
    # Check the returned solution, without demanding any particular width.
    scales = np.max(np.abs(jacobian), axis=0)
    normalized = jacobian / np.where(scales > 0, scales, 1)
    rank = np.linalg.matrix_rank(normalized)
    if rank < 3:
        unidentifiable = [
            j for j in range(3) if np.linalg.matrix_rank(np.delete(normalized, j, axis=1)) == rank
        ]
        assert np.all(~np.isfinite(diagonal[unidentifiable]))
        assert diagnostics
    if np.any(~np.isfinite(covariance)):
        assert diagnostics
    print(f"Smallest-positive-width {family} returned a verified sampled-model fit")


def test_s30_default_is_real_lm_with_positive_domain_and_free_centers():
    # A fresh process observes the public SciPy dependency before product import,
    # including implementations using a direct imported solver binding.
    script = r"""
import warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{category.__name__}: {message} ({filename}:{lineno})\n"
)
import sys
import traceback
sys.excepthook = lambda kind, value, tb: traceback.print_exception(kind, value, None, limit=0)
import json
import numpy as np
import scipy.optimize as optimize
real_solver = optimize.least_squares
calls = []
def observed_solver(*args, **kwargs):
    calls.append(kwargs.get("method", "trf"))
    return real_solver(*args, **kwargs)
optimize.least_squares = observed_solver
from dphtools.utils.fitfuncs import spectrum_fit
x = np.linspace(0, 6, 241)
y = 3.5 * np.exp(-0.5 * ((x + 0.4) / 1.1)**2)
a = spectrum_fit(y, x, guesses=[[3.1, 0.15, 0.95]], background="none")
b = spectrum_fit(y, x, guesses=[[3.1, 0.15, 0.95]], background="none", optimizer="lm")
print("SPECTRUM_LM_EVIDENCE=" + json.dumps({
    "methods": calls, "default": np.asarray(a.peak_parameters).tolist(),
    "explicit": np.asarray(b.peak_parameters).tolist(),
    "fitted": np.asarray(a.fitted).tolist(),
}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=os.environ.copy()
    )
    # Retain all warnings, messages, and the child exit status, without source.
    print("LM probe exit status:", child.returncode)
    if child.stderr:
        print(child.stderr, end="")
    records = []
    for line in child.stdout.splitlines():
        if line.startswith("SPECTRUM_LM_EVIDENCE="):
            records.append(json.loads(line.split("=", 1)[1]))
        else:
            print(line)
    assert child.returncode == 0
    assert len(records) == 1
    evidence = records[0]
    assert len(evidence["methods"]) >= 2
    assert set(evidence["methods"]) == {"lm"}
    truth = [[3.5, -0.4, 1.1]]
    assert_allclose(evidence["default"], truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(evidence["explicit"], truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert_allclose(evidence["default"], evidence["explicit"], rtol=2e-8, atol=2e-8)
    x = np.linspace(0, 6, 241)
    assert_allclose(evidence["fitted"], _model(x, truth, "gauss"), rtol=0, atol=MODEL_ATOL)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "background,coefficients", [("none", ()), ("constant", (1.1,)), ("linear", (1.1, -0.025))]
)
def test_s31_custom_optimizer_physical_protocol_and_covariance(
    fit, family, background, coefficients
):
    x = 7 + 24 * np.linspace(0, 1, 161) ** 1.8
    truth = _rows(family)
    y = _model(x, truth, family, coefficients) + _noise(len(x))
    guesses = _starts(truth)[::-1].copy()
    size = guesses.size
    n = size + len(coefficients)
    calls = []

    def adapter(residual, initial, *, bounds, max_nfev):
        initial = np.asarray(initial)
        assert initial.shape == (n,)
        assert np.isrealobj(initial) and np.all(np.isfinite(initial))
        assert_allclose(initial[:size], guesses.ravel(), rtol=0, atol=0)
        lower, upper = map(np.asarray, bounds)
        expected_lower = np.full(n, -np.inf)
        for i in range(len(guesses)):
            base = i * ROW_SIZE[family]
            expected_lower[base] = 0
            expected_lower[base + 2 : base + ROW_SIZE[family]] = 0
        assert_array_equal(lower, expected_lower)
        assert_array_equal(upper, np.full(n, np.inf))
        assert max_nfev == 2345
        # Physical parameter ordering stays fixed even though centers are unsorted.
        for shift in (0.0, 0.17):
            trial = initial.copy()
            trial[0] += shift
            trial[1] -= shift
            trial[2] += shift
            if family == "voigt":
                trial[3] += 0.5 * shift
            if len(coefficients):
                trial[size] -= shift
            if len(coefficients) == 2:
                trial[-1] += 0.1 * shift
            values = residual(trial)
            assert np.asarray(values).shape == y.shape
            assert_allclose(
                values,
                y - _model(x, trial[:size].reshape(guesses.shape), family, trial[size:]),
                rtol=2e-10,
                atol=2e-10,
            )
        solution = least_squares(residual, initial, bounds=bounds, max_nfev=max_nfev, method="trf")
        calls.append(solution.success)
        # Deliberately supply no Jacobian or covariance: the fitter owns them.
        return SimpleNamespace(
            x=solution.x, success=bool(solution.success), message=solution.message
        )

    result = _preserving_call(
        fit,
        y,
        x,
        peak_type=family,
        guesses=guesses,
        background=background,
        optimizer=adapter,
        max_nfev=2345,
    )
    assert calls and all(calls)
    _check_result(result, x, y, family, 2, len(coefficients))
    assert_allclose(result.peak_parameters, truth, rtol=0, atol=0.05)
    if len(coefficients):
        assert_allclose(result.background_parameters, coefficients, rtol=0, atol=0.01)
    _check_covariance_and_objective(result, x, y, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_s31_coincident_components_warn_without_finite_individual_height_uncertainties(
    fit, family
):
    # Post-implementation correction: (a1+t, a2-t) leaves the spectrum
    # unchanged for coincident equal-width components. Both height columns
    # of the physical Jacobian are identical, even at an exact minimizer.
    row = [2.0, 0.25, 0.9] if family != "voigt" else [2.0, 0.25, 0.9, 0.4]
    truth = np.array([row, row])
    x = np.linspace(-4, 4, 161)
    y = _model(x, truth, family)
    guesses = truth.copy()
    calls = []

    def exact_minimizer(residual, initial, *, bounds, max_nfev):
        solution = truth.ravel().copy()
        assert_allclose(residual(solution), 0, rtol=0, atol=2e-12)
        calls.append(True)
        return SimpleNamespace(x=solution, success=True)

    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        try:
            result = _preserving_call(
                fit,
                y,
                x,
                peak_type=family,
                guesses=guesses,
                background="none",
                optimizer=exact_minimizer,
            )
        finally:
            # Retain warning diagnostics during blind runs without source lines.
            for diagnostic in diagnostics:
                print(
                    warnings.formatwarning(
                        diagnostic.message,
                        diagnostic.category,
                        diagnostic.filename,
                        diagnostic.lineno,
                        line="",
                    ),
                    file=sys.stderr,
                    end="",
                )
    assert calls and diagnostics
    peaks = np.asarray(result.peak_parameters)
    assert peaks.shape == truth.shape
    assert np.isrealobj(peaks) and np.all(np.isfinite(peaks))
    assert_allclose(peaks, truth, rtol=PARAM_RTOL, atol=PARAM_ATOL)
    assert np.asarray(result.background_parameters).shape == (0,)
    for values in (result.fitted, result.residuals):
        values = np.asarray(values)
        assert values.shape == y.shape
        assert np.isrealobj(values) and np.all(np.isfinite(values))
    assert_allclose(result.fitted, _model(x, peaks, family), rtol=2e-10, atol=2e-10)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
    assert_allclose(result.residuals, y - result.fitted, rtol=2e-12, atol=2e-12)
    covariance = np.asarray(result.covariance)
    assert covariance.shape == (truth.size, truth.size) and np.isrealobj(covariance)
    # No specific nonfinite encoding, warning wording, or covariance algorithm.
    assert np.all(~np.isfinite(np.diag(covariance)[[0, ROW_SIZE[family]]]))


@pytest.mark.parametrize("optimizer", ["trf", "dogbox", "LM", None, 42, ["lm"]])
def test_s32_invalid_optimizer_selection(fit, valid_data, optimizer):
    with pytest.raises(ValueError):
        fit(valid_data[1], valid_data[0], guesses=[0.2], optimizer=optimizer)


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing_x",
        "missing_success",
        "matrix",
        "scalar",
        "short",
        "long",
        "nan",
        "inf",
        "complex",
        "success_string",
        "success_integer",
        "background_nan",
    ],
)
def test_s33_malformed_custom_optimizer_output(fit, valid_data, defect):
    def malformed(residual, initial, *, bounds, max_nfev):
        p = np.asarray(initial).copy()
        if defect == "none":
            return None
        if defect == "missing_x":
            return SimpleNamespace(success=True)
        if defect == "missing_success":
            return SimpleNamespace(x=p)
        if defect == "matrix":
            p = p.reshape(1, -1)
        elif defect == "scalar":
            p = 1.0
        elif defect == "short":
            p = p[:-1]
        elif defect == "long":
            p = np.append(p, 0.0)
        elif defect == "nan":
            p[0] = np.nan
        elif defect == "inf":
            p[1] = np.inf
        elif defect == "complex":
            p = p.astype(complex)
            p[0] += 1j
        elif defect == "background_nan":
            p[-1] = np.nan
        success = (
            "True" if defect == "success_string" else 1 if defect == "success_integer" else True
        )
        return SimpleNamespace(x=p, success=success)

    with pytest.raises(RuntimeError):
        _preserving_call(
            fit,
            valid_data[1],
            valid_data[0],
            guesses=np.array([[3, 0.2, 0.9]]),
            optimizer=malformed,
        )


@pytest.mark.parametrize(
    "family,field",
    [
        ("gauss", 0),
        ("gauss", 2),
        ("lorentz", 0),
        ("lorentz", 2),
        ("voigt", 0),
        ("voigt", 2),
        ("voigt", 3),
    ],
)
@pytest.mark.parametrize("value", [0, -0.1])
def test_s33_invalid_final_positive_domain_and_input_storage(
    fit, valid_data, family, field, value
):
    guesses = np.array([[3, 0.2, 0.9]]) if family != "voigt" else np.array([[3, 0.2, 0.9, 0.4]])

    def invalid(residual, initial, *, bounds, max_nfev):
        p = np.asarray(initial).copy()
        p[field] = value
        # Writable working storage must not alias caller guesses. R3 does not
        # require the supplied initial array itself to be writable.
        working = np.asarray(initial)
        if working.flags.writeable:
            working[: len(guesses.ravel())] = 9.0
        return SimpleNamespace(x=p, success=True)

    with pytest.raises(RuntimeError):
        _preserving_call(
            fit, valid_data[1], valid_data[0], peak_type=family, guesses=guesses, optimizer=invalid
        )


def test_s09_s31_gaussian_physical_covariance_transforms_with_x_units(fit):
    # Supplementary post-implementation regression. Derivatives come directly
    # from m=A*exp(-z**2/2), z=(x-c)/sigma, in physical (A,c,sigma) units.
    x = np.linspace(-4, 4, 201)
    truth = np.array([[3.0, 0.0, 1.0]])
    profile = np.exp(-0.5 * x * x)
    model = 3 * profile
    jacobian = np.column_stack((profile, model * x, model * x * x))
    norms = np.linalg.norm(jacobian, axis=0)
    assert np.linalg.cond(jacobian / norms) < 3
    q, _ = np.linalg.qr(jacobian, mode="reduced")
    i = np.arange(len(x))
    raw_noise = 0.01 * (np.sin(1.63 * i) + 0.6 * np.cos(0.71 * i))
    noise = raw_noise - q @ (q.T @ raw_noise)
    y = model + noise
    residuals = y - model  # Use the represented data, including addition roundoff.
    rss = residuals @ residuals
    assert 0.001 < rss < 0.1
    assert np.max(np.abs(jacobian.T @ residuals) / (norms * np.linalg.norm(residuals))) < 1e-12

    # For half the residual sum of squares, H=J.T@J-sum(r_i*m_i'').
    # Bound the analytic curvature correction to prove a strict local minimum;
    # no iterative optimizer output is used to establish the fixture's solution.
    gram = jacobian.T @ jacobian
    correction = np.zeros((3, 3))
    correction[0, 1] = residuals @ (profile * x)
    correction[0, 2] = residuals @ (profile * x * x)
    correction[1, 1] = residuals @ (model * (x * x - 1))
    correction[1, 2] = residuals @ (model * (x**3 - 2 * x))
    correction[2, 2] = residuals @ (model * (x**4 - 3 * x * x))
    correction += np.triu(correction, 1).T
    assert np.linalg.norm(correction, 2) < 0.01 * np.min(np.linalg.eigvalsh(gram))
    expected = np.linalg.inv(gram) * rss / (len(x) - truth.size)
    uncertainties = np.sqrt(np.diag(expected))
    uncertainty_units = np.outer(uncertainties, uncertainties)
    converted_covariances = []

    for unit_scale in (1.0, 1e-6):
        units = np.array([1.0, unit_scale, unit_scale])
        physical_truth = truth * units
        physical_x = x * unit_scale
        calls = []

        def known_local_minimizer(residual, initial, *, bounds, max_nfev):
            assert_allclose(initial, physical_truth.ravel(), rtol=0, atol=0)
            assert_array_equal(bounds[0], [0, -np.inf, 0])
            assert_array_equal(bounds[1], [np.inf, np.inf, np.inf])
            assert max_nfev == 10
            solution = physical_truth.ravel().copy()
            assert_allclose(residual(solution), residuals, rtol=2e-12, atol=2e-12)
            trial = (truth + [[0.1, 0.05, 0.03]]) * units
            assert_allclose(
                residual(trial.ravel()),
                y - _model(physical_x, trial, "gauss"),
                rtol=2e-12,
                atol=2e-12,
            )
            calls.append(True)
            # Return only a legitimate physical minimizer and success status.
            # The fitter independently calculates its Jacobian and covariance.
            return SimpleNamespace(x=solution, success=True)

        result = _preserving_call(
            fit,
            y,
            physical_x,
            guesses=physical_truth.copy(),
            background="none",
            optimizer=known_local_minimizer,
            max_nfev=10,
        )
        assert calls == [True]
        _check_result(result, physical_x, y, "gauss", 1, 0)
        assert_allclose(result.peak_parameters / units, truth, rtol=2e-12, atol=2e-12)
        assert_allclose(result.fitted, model, rtol=2e-12, atol=2e-12)
        assert_allclose(result.residuals, residuals, rtol=2e-12, atol=2e-12)
        # C'=D C D: convert every physical covariance entry back to base units.
        converted = np.asarray(result.covariance) / np.outer(units, units)
        converted_covariances.append(converted)
        assert_allclose(
            converted / uncertainty_units,
            expected / uncertainty_units,
            rtol=0.01,
            atol=2e-4,
            err_msg=f"Analytic physical covariance at x-unit scale {unit_scale:g}",
        )

    assert_allclose(
        converted_covariances[1] / uncertainty_units,
        converted_covariances[0] / uncertainty_units,
        rtol=0.01,
        atol=2e-4,
        err_msg="Physical covariance must transform as D C D when x units change",
    )


def _stationary_gaussian_failure_fixture():
    """Analytic physical derivatives, projected noise, and positive curvature."""
    x = np.linspace(-4, 4, 201)
    truth = np.array([[3.0, 0.0, 1.0]])
    profile = np.exp(-0.5 * x * x)
    model = 3 * profile
    jacobian = np.column_stack((profile, model * x, model * x * x))
    norms = np.linalg.norm(jacobian, axis=0)
    assert np.linalg.cond(jacobian / norms) < 3
    q, _ = np.linalg.qr(jacobian, mode="reduced")
    i = np.arange(len(x))
    raw = 0.01 * (np.sin(1.63 * i) + 0.6 * np.cos(0.71 * i))
    y = model + raw - q @ (q.T @ raw)
    residuals = y - model
    assert np.max(np.abs(jacobian.T @ residuals) / (norms * np.linalg.norm(residuals))) < 1e-12
    gram = jacobian.T @ jacobian
    correction = np.zeros((3, 3))
    correction[0, 1] = residuals @ (profile * x)
    correction[0, 2] = residuals @ (profile * x * x)
    correction[1, 1] = residuals @ (model * (x * x - 1))
    correction[1, 2] = residuals @ (model * (x**3 - 2 * x))
    correction[2, 2] = residuals @ (model * (x**4 - 3 * x * x))
    correction += np.triu(correction, 1).T
    assert np.linalg.norm(correction, 2) < 0.01 * np.min(np.linalg.eigvalsh(gram))
    covariance = np.linalg.inv(gram) * (residuals @ residuals) / (len(x) - 3)
    return x, truth, y, model, residuals, jacobian, covariance


def _numerical_failure_outcome(fit, data, x, **kwargs):
    """Accept only RuntimeError failure; retain every warning and input check."""
    result = None
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        try:
            result = _preserving_call(fit, data, x, **kwargs)
        except RuntimeError as failure:
            print("Public numerical failure:", failure)
        else:
            assert result is not None
        finally:
            for diagnostic in diagnostics:
                print(
                    warnings.formatwarning(
                        diagnostic.message,
                        diagnostic.category,
                        diagnostic.filename,
                        diagnostic.lineno,
                        line="",
                    ),
                    file=sys.stderr,
                    end="",
                )
    return result, diagnostics


def _check_scaled_gaussian_failure_success(result, x, y, truth, data_unit):
    """Verify finite public fit arrays without squaring a large data unit."""
    units = np.array([data_unit, 1.0, 1.0])
    peaks = np.asarray(result.peak_parameters)
    assert peaks.shape == (1, 3)
    assert np.isrealobj(peaks) and np.all(np.isfinite(peaks))
    assert np.all(peaks[:, [0, 2]] > 0)
    assert_allclose(peaks / units, truth / units, rtol=2e-12, atol=2e-12)
    background = np.asarray(result.background_parameters)
    assert background.shape == (0,) and np.isrealobj(background)
    fitted, residuals = map(np.asarray, (result.fitted, result.residuals))
    for values in (fitted, residuals):
        assert values.shape == y.shape
        assert np.isrealobj(values) and np.all(np.isfinite(values))
    scaled_peaks = peaks / units
    assert_allclose(fitted / data_unit, _model(x, scaled_peaks, "gauss"), rtol=2e-10, atol=2e-12)
    assert_allclose(residuals / data_unit, (y - fitted) / data_unit, rtol=2e-12, atol=2e-12)
    return residuals / data_unit


def _check_large_unit_covariance(result, expected, units, diagnostics):
    """Check physical entries and variance overflow in the returned dtype."""
    covariance = np.asarray(result.covariance)
    assert covariance.shape == (3, 3) and np.isrealobj(covariance)
    assert_allclose(covariance, covariance.T, rtol=2e-10, atol=2e-12, equal_nan=True)
    diagonal = np.diag(covariance)
    assert np.all(diagonal[np.isfinite(diagonal)] >= 0)
    if np.any(~np.isfinite(covariance)):
        assert diagnostics
    if not np.any(expected):
        # An exact zero-residual identifiable minimizer has zero covariance.
        assert_array_equal(covariance, np.zeros((3, 3)))
        return
    # Keep the limit in the returned dtype: a wider maximum can exceed float64.
    log_limit = np.log(np.finfo(covariance.dtype).max)
    for j in range(3):
        if math.log(expected[j, j]) + 2 * math.log(units[j]) > log_limit:
            assert not np.isfinite(diagonal[j])
            assert diagnostics
    # Sequential divisions avoid an overflowing outer product of data units.
    converted = covariance / units[:, None] / units[None, :]
    uncertainty_units = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    finite = np.isfinite(covariance)
    assert_allclose(
        converted[finite] / uncertainty_units[finite],
        expected[finite] / uncertainty_units[finite],
        rtol=0.01,
        atol=2e-4,
    )


@pytest.mark.parametrize("height", [1e200, np.finfo(float).max])
def test_s29_s31_extreme_finite_height_exact_custom_minimum(fit, height):
    x = np.arange(-4, 5, dtype=float)
    profile = np.exp(-0.5 * x * x)
    y = height * profile
    truth = np.array([[height, 0.0, 1.0]])
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(truth))
    relative_jacobian = np.column_stack((profile, profile * x, profile * x * x))
    assert np.linalg.cond(relative_jacobian) < 3
    calls = []

    def exact_minimizer(residual, initial, *, bounds, max_nfev):
        assert_array_equal(initial, truth.ravel())
        assert_array_equal(bounds[0], [0, -np.inf, 0])
        assert_array_equal(bounds[1], [np.inf, np.inf, np.inf])
        assert max_nfev == 10
        assert_allclose(residual(initial) / height, 0, rtol=0, atol=2e-12)
        calls.append(True)
        return SimpleNamespace(x=initial.copy(), success=True)

    result, diagnostics = _numerical_failure_outcome(
        fit,
        y,
        x,
        guesses=truth.copy(),
        background="none",
        optimizer=exact_minimizer,
        max_nfev=10,
    )
    if result is None:
        return
    assert calls == [True]
    scaled_residuals = _check_scaled_gaussian_failure_success(result, x, y, truth, height)
    assert_allclose(result.fitted / height, profile, rtol=2e-10, atol=2e-12)
    assert_allclose(scaled_residuals, 0, rtol=0, atol=2e-12)
    expected = (
        np.linalg.inv(relative_jacobian.T @ relative_jacobian)
        * (scaled_residuals @ scaled_residuals)
        / (len(x) - 3)
    )
    _check_large_unit_covariance(result, expected, np.array([height, 1, 1]), diagnostics)
    print("Extreme finite height returned a verified exact-model fit:", height)


def test_s29_s31_large_noise_units_unrepresentable_height_variance(fit):
    x, base_truth, base_y, model, residuals, jacobian, expected = (
        _stationary_gaussian_failure_fixture()
    )
    data_unit = 1e160
    units = np.array([data_unit, 1, 1])
    truth = base_truth * units
    y = base_y * data_unit
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(truth))
    assert math.log(expected[0, 0]) + 2 * math.log(data_unit) > math.log(np.finfo(float).max)
    # Scaling finite point data is a unit change. In scaled physical coordinates
    # the stationary solution and positive objective Hessian are unchanged.
    represented_noise = y / data_unit - model
    norms = np.linalg.norm(jacobian, axis=0)
    assert (
        np.max(
            np.abs(jacobian.T @ represented_noise) / (norms * np.linalg.norm(represented_noise))
        )
        < 1e-12
    )
    calls = []

    def stationary_minimizer(residual, initial, *, bounds, max_nfev):
        assert_array_equal(initial, truth.ravel())
        assert_array_equal(bounds[0], [0, -np.inf, 0])
        assert_array_equal(bounds[1], [np.inf, np.inf, np.inf])
        assert max_nfev == 10
        assert_allclose(residual(initial) / data_unit, residuals, rtol=2e-12, atol=2e-12)
        calls.append(True)
        return SimpleNamespace(x=initial.copy(), success=True)

    result, diagnostics = _numerical_failure_outcome(
        fit,
        y,
        x,
        guesses=truth.copy(),
        background="none",
        optimizer=stationary_minimizer,
        max_nfev=10,
    )
    if result is None:
        return
    assert calls == [True]
    actual_noise = _check_scaled_gaussian_failure_success(result, x, y, truth, data_unit)
    assert_allclose(result.fitted / data_unit, model, rtol=2e-12, atol=2e-12)
    assert_allclose(actual_noise, residuals, rtol=2e-12, atol=2e-12)
    _check_large_unit_covariance(result, expected, units, diagnostics)
    print("Large noise units returned a verified fit with physical covariance")


@pytest.mark.parametrize("boundary", ["optimizer_lstsq", "covariance_svd"])
def test_s29_s31_public_numerical_backend_failure(fit, monkeypatch, boundary):
    x, truth, y, model, residuals, jacobian, expected = _stationary_gaussian_failure_fixture()
    calls, injections = [], []

    def minimizing_adapter(residual, initial, *, bounds, max_nfev):
        assert_array_equal(initial, truth.ravel())
        assert_array_equal(bounds[0], [0, -np.inf, 0])
        assert_array_equal(bounds[1], [np.inf, np.inf, np.inf])
        assert max_nfev == 10
        observed = residual(initial)
        assert_allclose(observed, residuals, rtol=2e-12, atol=2e-12)
        calls.append(True)
        if boundary == "optimizer_lstsq":
            # A real Gauss-Newton stationarity check at the proved strict local
            # minimum. The unpatched least-squares step is zero to roundoff.
            step = np.linalg.lstsq(jacobian, observed, rcond=None)[0]
            assert_allclose(step, 0, rtol=0, atol=2e-12)
        return SimpleNamespace(x=initial.copy(), success=True)

    def unavailable_decomposition(matrix, *args, **kwargs):
        matrix = np.asarray(matrix)
        assert matrix.ndim >= 2 and matrix.size
        assert np.all(np.isfinite(matrix))
        if boundary == "optimizer_lstsq":
            assert matrix.ndim == 2 and np.isrealobj(matrix)
            assert_allclose(matrix, jacobian, rtol=0, atol=0)
            assert_allclose(args[0], residuals, rtol=2e-12, atol=2e-12)
        injections.append(matrix.shape)
        print(
            "Injecting public numerical failure:", boundary, "finite operand shape:", matrix.shape
        )
        raise np.linalg.LinAlgError("Numerical decomposition did not converge (injected)")

    # Patch only documented NumPy boundaries. A covariance implementation may
    # use another valid algorithm/binding or recover; verified success is allowed.
    with monkeypatch.context() as patch:
        patch.setattr(
            np.linalg,
            "lstsq" if boundary == "optimizer_lstsq" else "svd",
            unavailable_decomposition,
        )
        result, diagnostics = _numerical_failure_outcome(
            fit,
            y,
            x,
            guesses=truth.copy(),
            background="none",
            optimizer=minimizing_adapter,
            max_nfev=10,
        )
    assert calls
    print("Public numerical boundary:", boundary, "injected operands:", injections)
    if result is None:
        assert injections  # An unrelated rejection cannot satisfy this case.
        return
    _check_result(result, x, y, "gauss", 1, 0)
    assert_allclose(result.peak_parameters, truth, rtol=2e-12, atol=2e-12)
    assert_allclose(result.fitted, model, rtol=2e-12, atol=2e-12)
    assert_allclose(result.residuals, residuals, rtol=2e-12, atol=2e-12)
    uncertainty_units = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    assert_allclose(
        result.covariance / uncertainty_units,
        expected / uncertainty_units,
        rtol=0.01,
        atol=2e-4,
    )
    print("Public numerical boundary returned an independently verified successful fit")


@pytest.mark.parametrize("family", ["gauss", "lorentz"])
@pytest.mark.parametrize("optimizer_mode", ["exact", "lm"])
def test_s29_large_origin_narrow_width_point_boundary(fit, family, optimizer_mode):
    # Valid point observations; coordinate spacing exceeds this positive width.
    # A numerical failure or a verified success is permitted, not exact recovery.
    x = 1e12 + np.arange(9, dtype=float)
    guesses = np.array([[2.0, x[4], 1e-6]])
    y, _ = _stable_narrow_component(x, guesses[0], family)
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(guesses))
    assert np.all(guesses[:, [0, 2]] > 0) and len(y) > guesses.size
    assert guesses[0, 2] < np.spacing(x[4]) < np.min(np.diff(x))
    roundoff_floor = 8 * np.finfo(float).smallest_subnormal
    calls = []

    def exact_minimizer(residual, initial, *, bounds, max_nfev):
        assert_allclose(initial, guesses.ravel(), rtol=0, atol=0)
        assert_array_equal(bounds[0], [0, -np.inf, 0])
        assert_array_equal(bounds[1], [np.inf, np.inf, np.inf])
        assert max_nfev == 10
        values = np.asarray(residual(guesses.ravel().copy()))
        assert values.shape == y.shape and np.isrealobj(values)
        assert np.all(np.isfinite(values))
        # The generating row has zero mathematical RSS, hence is a minimizer.
        # Relative-to-data roundoff resolves the narrow Lorentzian tails.
        assert np.all(np.abs(values) <= 2e-12 * np.abs(y) + roundoff_floor)
        calls.append(True)
        return SimpleNamespace(x=guesses.ravel().copy(), success=True)

    options = {"optimizer": exact_minimizer, "max_nfev": 10} if optimizer_mode == "exact" else {}
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        try:
            result = _preserving_call(
                fit, y, x, peak_type=family, guesses=guesses, background="none", **options
            )
        except RuntimeError as failure:
            print(f"Large-origin {family}/{optimizer_mode} numerical failure: {failure}")
            return
        except Exception as failure:
            print(
                f"Large-origin {family}/{optimizer_mode} leaked {type(failure).__name__}: {failure}"
            )
            raise
        finally:
            for diagnostic in diagnostics:
                print(
                    warnings.formatwarning(
                        diagnostic.message,
                        diagnostic.category,
                        diagnostic.filename,
                        diagnostic.lineno,
                        line="",
                    ),
                    file=sys.stderr,
                    end="",
                )
    if optimizer_mode == "exact":
        assert calls == [True]
    peaks = np.asarray(result.peak_parameters)
    assert peaks.shape == (1, 3)
    assert np.isrealobj(peaks) and np.all(np.isfinite(peaks))
    assert np.all(peaks[:, [0, 2]] > 0)
    background = np.asarray(result.background_parameters)
    assert background.shape == (0,) and np.isrealobj(background)
    for values in (result.fitted, result.residuals):
        values = np.asarray(values)
        assert values.shape == y.shape
        assert np.isrealobj(values) and np.all(np.isfinite(values))
    model, sensitivity = _stable_narrow_component(x, peaks[0], family)
    assert_allclose(result.fitted, model, rtol=2e-10, atol=roundoff_floor)
    assert_allclose(result.fitted, y, rtol=0, atol=MODEL_ATOL)
    assert_allclose(result.residuals, y - result.fitted, rtol=2e-12, atol=2e-12)
    assert_allclose(result.residuals, 0, rtol=0, atol=MODEL_ATOL)
    covariance = np.asarray(result.covariance)
    assert covariance.shape == (3, 3) and np.isrealobj(covariance)
    assert_allclose(covariance, covariance.T, rtol=2e-10, atol=2e-12, equal_nan=True)
    diagonal = np.diag(covariance)
    assert np.all(diagonal[np.isfinite(diagonal)] >= 0)
    # Assess the returned row analytically; do not prescribe derivative steps.
    scales = np.max(np.abs(sensitivity), axis=0)
    normalized = sensitivity / np.where(scales > 0, scales, 1)
    rank = np.linalg.matrix_rank(normalized)
    if rank < 3:
        unidentifiable = [
            j for j in range(3) if np.linalg.matrix_rank(np.delete(normalized, j, axis=1)) == rank
        ]
        assert np.all(~np.isfinite(diagonal[unidentifiable]))
        assert diagnostics
    if np.any(~np.isfinite(covariance)):
        assert diagnostics
    print(f"Large-origin {family}/{optimizer_mode} returned a verified sampled-model fit")


def _gaussian_background_physical_jacobian(x, parameters):
    """R4 analytic model derivatives in (height V, center nm, sigma nm, b0 V)."""
    amplitude, center, sigma, _ = parameters
    z = (x - center) / sigma
    profile = np.exp(-0.5 * z * z)
    return np.column_stack(
        (
            profile,
            amplitude * profile * z / sigma,
            amplitude * profile * z * z / sigma,
            np.ones_like(x),
        )
    )


def _resolved_gaussian_voltage_fixture():
    """Illustrative analog voltage; independent additive read noise, not counts."""
    x = np.linspace(500, 510, 201)  # Wavelength in nm; 0.05 nm sample spacing.
    physical = np.array([1.2, 505.1, 0.7, 0.4])  # V, nm, nm, V.
    noiseless = _model(x, physical[:3].reshape(1, 3), "gauss", physical[3:])
    noise = np.random.default_rng(20261005).normal(0, 0.01, len(x))  # 0.01 V rms.
    guesses = np.array([[1.0, 505.0, 0.85]])
    return x, noiseless + noise, guesses, physical, noiseless, noise


@pytest.mark.parametrize("optimizer_mode", ["default_lm", "real_custom_trf"])
def test_s06_s09_s31_resolved_voltage_gaussian_full_physical_covariance(fit, optimizer_mode):
    # R4 physical-scope correction. Fixture and bounds were frozen before calls.
    # The two successful-fit cases replace only the artificial 1e12-offset cases.
    x, data, guesses, truth, noiseless, noise = _resolved_gaussian_voltage_fixture()
    snapshots = [value.copy() for value in (x, data, guesses)]
    calls = []

    def real_physical_optimizer(residual, initial, *, bounds, max_nfev):
        # Existing S31/S33 cases cover protocol defects. This is a real fit in
        # physical coordinates, with no Jacobian/covariance returned to fitter.
        solution = least_squares(
            residual,
            initial,
            jac=lambda parameters: -_gaussian_background_physical_jacobian(x, parameters),
            bounds=bounds,
            method="trf",
            x_scale=[1.2, 0.7, 0.7, 0.4],
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )
        calls.append(bool(solution.success))
        return SimpleNamespace(
            x=solution.x, success=bool(solution.success), message=str(solution.message)
        )

    options = {} if optimizer_mode == "default_lm" else {"optimizer": real_physical_optimizer}
    # Ordinary representative data must fit successfully; RuntimeError fails.
    result = _preserving_call(
        fit, data, x, guesses=guesses, background="constant", max_nfev=2000, **options
    )
    if optimizer_mode == "real_custom_trf":
        assert calls == [True]
    _check_result(result, x, data, "gauss", 1, 1)

    peaks = np.asarray(result.peak_parameters)
    background = np.asarray(result.background_parameters)
    fitted = np.asarray(result.fitted)
    residuals = np.asarray(result.residuals)
    covariance = np.asarray(result.covariance)
    physical = np.concatenate((peaks.ravel(), background))
    # Absolute recovery bounds: 0.020 V, 0.012 nm, 0.012 nm, 0.006 V.
    # These generous noise-derived bounds are not confidence-interval claims.
    recovery_bounds = np.array([0.020, 0.012, 0.012, 0.006])
    assert np.all(np.abs(physical - truth) <= recovery_bounds)
    assert np.sqrt(np.mean((fitted - noiseless) ** 2)) < 0.006  # V rms.
    rms = np.sqrt(np.mean(residuals**2))
    assert 0.006 < rms < 0.014  # V; resolved noise must remain in residuals.

    jacobian = _gaussian_background_physical_jacobian(x, physical)
    norms = np.linalg.norm(jacobian, axis=0)
    condition = np.linalg.cond(jacobian / norms)
    assert condition < 4  # Bounds the roundoff sensitivity of the inverse oracle.
    rss = residuals @ residuals  # V**2; actual residuals, not nominal noise variance.
    # Generating truth is feasible: a meaningful least-squares fit improves it.
    assert rss <= (noise @ noise) * (1 + 1e-6)
    assert np.max(np.abs(jacobian.T @ residuals) / (norms * np.linalg.norm(residuals))) < 3e-4
    expected = np.linalg.inv(jacobian.T @ jacobian) * rss / (len(x) - 4)
    scales = np.sqrt(np.diag(expected))
    assert np.all(np.isfinite(expected)) and np.all(scales > 0)
    units = np.outer(scales, scales)
    normalized = covariance / units
    expected_normalized = expected / units
    # Every variance/correlation is compared in its own sigma_i*sigma_j units.
    # A <=1e-4 columnwise derivative error and condition <4 imply <0.00141
    # normalized inverse error; 0.002 also allows arithmetic/solver roundoff.
    assert_allclose(normalized, normalized.T, rtol=0, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(normalized)) >= -1e-10
    assert_allclose(normalized, expected_normalized, rtol=0, atol=0.002)
    print(
        f"{optimizer_mode}: residual rms={rms:.9g} V; RSS={rss:.9g} V**2; "
        f"condition={condition:.9g}; "
        f"max normalized covariance error={np.max(np.abs(normalized - expected_normalized)):.9g}"
    )

    # Result storage as well as the fit/optimizer must preserve caller inputs.
    for values in (peaks, background, fitted, residuals, covariance):
        if values.flags.writeable:
            values.flat[0] += 1
        for value, snapshot in zip((x, data, guesses), snapshots):
            assert_array_equal(value, snapshot)
