"""Supplemental public boundaries from PUBLIC-BOUNDARIES-CONTRACT.md.

These are maintenance tests, not original test-first evidence. Existing tests
already cover numerical derivatives, solver state, and registration matching.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from numpy.testing import assert_allclose, assert_array_equal

from dphtools import display, utils
from dphtools.utils import beads, fitfuncs, registration
from dphtools.utils.lm import curve_fit


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_colorline_renders_caller_colormap_at_requested_normalized_values():
    fig, ax = plt.subplots()
    display.colorline(
        [0, 2, 5],
        [1, -1, 3],
        z=[10, 20],
        cmap=ListedColormap(["red", "blue"]),
        norm=Normalize(10, 20),
        ax=ax,
    )
    fig.canvas.draw()
    collection = ax.collections[0]
    assert_allclose(collection.get_segments(), [[[0, 1], [2, -1]], [[2, -1], [5, 3]]])
    assert_allclose(collection.get_colors(), [[1, 0, 0, 1], [0, 0, 1, 1]])


@pytest.mark.parametrize("figsize", [2, 8])
def test_wrap_name_preserves_nonwhitespace_content(figsize):
    name = "/calibration/Repeated Measurements/sample_A/long_directory_name_0123456789"
    wrapped = display.wrap_name(name, figsize)
    assert isinstance(wrapped, str)
    # Wrapping can insert line breaks or replace interword whitespace. It must
    # retain all letters, punctuation, and their order; no width is prescribed.
    assert "".join(wrapped.split()) == "".join(name.split())


@pytest.mark.parametrize("gamma", [0.5, 2.0])
def test_power_normalization_scalar_vector_and_inverse_agree(gamma):
    norm = display.SymPowerNorm(gamma, vmin=1, vmax=5)
    values = np.array([1.0, 2.0, 4.0, 5.0])
    expected = ((values - 1) / 4) ** gamma
    assert_allclose(norm(values), expected, atol=1e-12)
    for value, scaled in zip(values, expected):
        assert_allclose(norm(float(value)), scaled, atol=1e-12)
        assert_allclose(norm.inverse(float(scaled)), value, atol=1e-12)


@pytest.mark.parametrize("bounds", [{}, {"vmin": 2}, {"vmax": 10}])
def test_power_normalization_call_supplies_missing_bounds(bounds):
    norm = display.SymPowerNorm(2, **bounds)
    assert_allclose(norm([2.0, 4.0, 10.0]), [0, 1 / 16, 1], atol=1e-12)
    assert (norm.vmin, norm.vmax) == (2, 10)
    assert_allclose(norm.inverse([0, 1 / 16, 1]), [2, 4, 10], atol=1e-12)


@pytest.mark.parametrize("explicit_axis", [False, True])
def test_recolor_changes_only_selected_axis_and_preserves_both_lines(explicit_axis):
    fig, (selected, other) = plt.subplots(1, 2)
    (selected_line,) = selected.plot([0, 2], [3, -1], color="red")
    (other_line,) = other.plot([1, 4], [-2, 7], color="green")
    # The explicit target deliberately differs from the current axis.
    plt.sca(other if explicit_axis else selected)
    options = {"ax": selected} if explicit_axis else {}
    display.recolor(ListedColormap(["cyan"]), **options)
    assert_allclose(to_rgba(selected_line.get_color())[:3], [0, 1, 1])
    assert_allclose(to_rgba(other_line.get_color()), to_rgba("green"))
    for line, x, y in [(selected_line, [0, 2], [3, -1]), (other_line, [1, 4], [-2, 7])]:
        assert_array_equal(line.get_xdata(), x)
        assert_array_equal(line.get_ydata(), y)
    fig.canvas.draw()


def test_histogram_and_cumulative_create_axes_when_none_supplied():
    plt.close("all")
    display.hist_and_cumulative(np.array([1.0, 2.0, 2.0, 3.0]))
    assert plt.get_fignums()
    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert any(axis.patches for axis in fig.axes)
    assert any(axis.lines for axis in fig.axes)
    fig.canvas.draw()


@pytest.mark.parametrize("value", [0, 17])
def test_constant_image_adjustment_keeps_pixels_and_usable_display_bounds(value):
    image = np.full((16, 16), value, dtype=np.uint8)
    # Exercise the established automatic display entry point. The extracted
    # auto_adjust return schema needs clarification; no tuple/mapping choice
    # is needed to assert this basic non-destructive display bound.
    display.display_grid({"constant": image}, auto=True)
    fig = plt.gcf()
    artists = [artist for axis in fig.axes for artist in axis.images]
    assert len(artists) == 1
    artist = artists[0]
    fig.canvas.draw()
    low, high = artist.get_clim()
    assert np.isfinite([low, high]).all()
    assert low <= value <= high
    assert_array_equal(image, np.full((16, 16), value, dtype=np.uint8))
    assert_array_equal(artist.get_array(), image)


def test_git_description_matches_real_disposable_tagged_repository(tmp_path, monkeypatch):
    # Keep every mutation in the temporary repository; do not use checkout Git.
    for variable in list(os.environ):
        if variable.startswith("GIT_"):
            monkeypatch.delenv(variable)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Boundary Test",
                "-c",
                "user.email=boundary@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-C",
                str(repository),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init")
    tracked = repository / "sample.txt"
    tracked.write_text("first\n")
    git("add", "sample.txt")
    git("commit", "-m", "First sample")
    git("tag", "-a", "public-boundary-v1", "-m", "Boundary fixture")
    for content in (None, "second\n"):
        if content is not None:
            tracked.write_text(content)
            git("commit", "-am", "Second sample")
        expected = git("describe")
        if content is None:
            assert expected == "public-boundary-v1"
        actual = utils.get_git(str(repository))
        # The public packet does not choose bytes versus text or final newline.
        actual = actual.decode() if isinstance(actual, bytes) else actual
        # Git documents both exact-tag and --long descriptions. The packet
        # does not select one: https://git-scm.com/docs/git-describe
        assert actual.strip() in {expected, git("describe", "--long")}


@pytest.mark.parametrize("number", [1200.0, 0.0012, -1200.0, -0.0012])
def test_latex_scientific_notation_preserves_value_without_dollar_delimiters(number):
    rendered = utils.latex_format_e(number, pre=2)
    assert "$" not in rendered
    # Require explicit multiplication and a braced exponent or one unsigned digit.
    # Values have exact short mantissas, avoiding a rounding policy.
    compact = re.sub(r"\s+", "", rendered)
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\\times|\\cdot)10\^" r"(?:\{([+-]?\d+)\}|(\d))",
        compact,
    )
    assert match is not None, rendered
    exponent = int(match.group(2) or match.group(3))
    assert_allclose(float(match.group(1)) * 10.0**exponent, number, rtol=1e-12)


def test_timer_reports_elapsed_quantity_and_preserves_body_return(capsys, caplog):
    marker = object()
    sleep_seconds = 0.025

    def timed_body():
        with utils.EasyTimer("boundary duration"):
            time.sleep(sleep_seconds)
            return marker

    start = time.perf_counter()
    with caplog.at_level(logging.DEBUG):
        result = timed_body()
    elapsed = time.perf_counter() - start
    assert result is marker
    captured = capsys.readouterr()
    emitted = captured.out + captured.err + "\n".join(r.getMessage() for r in caplog.records)
    assert "boundary duration" in emitted
    quantities = re.findall(
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        r"(nanoseconds?|microseconds?|milliseconds?|seconds?|ns|[uµμ]s|ms|s)\b",
        emitted,
    )
    consistent = []
    for magnitude, unit in quantities:
        scale = (
            1e-9
            if unit.startswith("n")
            else (
                1e-6
                if unit.startswith(("u", "µ", "μ", "micro"))
                else 1e-3 if unit.startswith(("ms", "milli")) else 1
            )
        )
        seconds = float(magnitude) * scale
        # Tolerance follows the printed resolution; no precision or unit-switch
        # threshold is required. The outer clock bounds the measured context.
        resolution = 10.0 ** Decimal(magnitude).as_tuple().exponent * scale
        consistent.append(sleep_seconds - resolution <= seconds <= elapsed + resolution)
    assert any(consistent), emitted


def test_timer_does_not_hide_or_replace_context_body_exception():
    error = RuntimeError("intentional context body failure")
    with pytest.raises(RuntimeError) as caught:
        with utils.EasyTimer("exception boundary"):
            raise error
    assert caught.value is error


@pytest.mark.parametrize("shape", [(5, 5), (4, 6)])
def test_same_shape_centered_identity_convolution_preserves_all_pixels(shape):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape) ** 2 - 7
    kernel = np.zeros(shape)
    kernel[tuple(length // 2 for length in shape)] = 1
    assert_allclose(utils.fftconvolve_fast(data, kernel), data, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("data", [np.array([]), np.full(8, np.nan)], ids=["empty", "all-nan"])
@pytest.mark.parametrize("fit", ["exponent", "multi-exp"])
def test_no_finite_exponential_observations_cannot_return_a_successful_fit(data, fit):
    # The packet specifies no successful fit but no exception family. An
    # ordinary exception meets only that narrow promise, not error-message QA.
    with pytest.raises(Exception):
        if fit == "exponent":
            fitfuncs.exponent_fit(data)
        else:
            fitfuncs.multi_exp_fit(data, components=1)


@pytest.mark.parametrize("method, expected_rate", [("ls", 104 / 85), ("mle", 16 / 15)])
def test_custom_fit_finite_data_with_finite_check_disabled(method, expected_rate):
    exposure = np.array([1.0, 2.0, 4.0, 8.0])
    parameters, covariance = curve_fit(
        lambda x, log_rate: np.exp(log_rate) * x,
        exposure,
        [0.0, 4.0, 0.0, 12.0],
        p0=[0.0],
        jac=lambda x, log_rate: (np.exp(log_rate) * x)[:, None],
        method=method,
        check_finite=False,
        maxfev=1000,
    )
    # LS: sum(x*y)/sum(x*x)=104/85; Poisson: sum(y)/sum(x)=16/15.
    assert_allclose(np.exp(parameters), [expected_rate], rtol=1e-5)
    assert covariance.shape == (1, 1)


@pytest.mark.parametrize("method, expected_rate", [("ls", 104 / 85), ("mle", 16 / 15)])
@pytest.mark.parametrize("check_finite", [False, True])
def test_custom_fit_passes_predictor_object_to_model_and_derivative(
    method, expected_rate, check_finite
):
    class Predictor:
        exposure = np.array([1.0, 2.0, 4.0, 8.0])

    predictor = Predictor()

    def model(x, log_rate):
        assert x is predictor
        return np.exp(log_rate) * x.exposure

    def jacobian(x, log_rate):
        assert x is predictor
        return (np.exp(log_rate) * x.exposure)[:, None]

    parameters, covariance = curve_fit(
        model,
        predictor,
        [0.0, 4.0, 0.0, 12.0],
        p0=[0.0],
        jac=jacobian,
        method=method,
        check_finite=check_finite,
        maxfev=1000,
    )
    assert_allclose(np.exp(parameters), [expected_rate], rtol=1e-5)
    assert covariance.shape == (1, 1)


def test_curve_fit_rejects_unknown_method_beyond_named_unsupported_pyls():
    with pytest.raises(TypeError):
        curve_fit(
            lambda x, rate: rate * x,
            np.array([1.0, 2.0, 4.0]),
            [2.0, 4.0, 8.0],
            p0=[1.0],
            jac=lambda x, rate: x[:, None],
            method="not-a-solver",
        )


def test_custom_poisson_convergence_failure_is_not_reported_as_success():
    exposure = np.array([1.0, 2.0, 4.0, 8.0])
    counts = np.array([0.0, 5.0, 8.0, 16.0])

    def model(x, rate):
        return rate * x

    def jacobian(x, rate):
        return x[:, None]

    # At rate=2, residual [2,-1,0,0] is orthogonal to exposure. The
    # documented SciPy initializer can finish here, even with this low budget.
    parameters, _ = curve_fit(
        model, exposure, counts, p0=[2.0], jac=jacobian, method="lm", maxfev=1
    )
    assert_allclose(parameters, [2], atol=1e-12)
    # The Poisson optimum is 29/15, so a real custom iteration is still needed.
    with pytest.raises(RuntimeError):
        curve_fit(model, exposure, counts, p0=[2.0], jac=jacobian, method="mle", maxfev=1)


def test_single_track_drift_uses_named_frame_index_and_removes_static_offset():
    frames = pd.Index([2, 5, 9], name="slice")
    track = pd.DataFrame(
        {"x0": [12.0, 14.0, 10.0], "y0": [-4.0, 2.0, 5.0], "amp": [1.0, 1.0, 1.0]},
        index=frames,
    )
    result = beads.calc_drift([track], frames_index=frames)
    assert_array_equal(result.index, frames)
    assert_allclose(result[["x0", "y0"]], [[0, -5], [2, 1], [-2, 4]], atol=1e-12)


def test_fitted_registration_has_usable_text_representations():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-1.0, 3.0]])
    reg = registration.TranslationCPD(points, points + [0.2, -0.3])
    reg(maxiters=100, dist_tol=1e-8)
    assert_allclose(reg.transform(points + [0.2, -0.3]), points, atol=1e-5)
    assert isinstance(str(reg), str) and str(reg).strip()
    assert isinstance(repr(reg), str) and repr(reg).strip()


def test_utility_import_and_same_shape_padding_in_ordinary_and_oo_python(record_property):
    # Install diagnostics in each fresh process before dependency/product imports.
    # All checks live in the parent; -OO cannot remove the numerical oracle.
    script = r"""
import sys

sys.excepthook = lambda kind, error, tb: sys.stderr.write(f"{kind.__name__}: {error}\n")

import warnings

warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{filename}:{lineno}: {category.__name__}: {message}\n"
)

import json
import logging

logging.Formatter.formatException = lambda self, info: (
    f"{info[0].__name__}: {info[1]}"
)
receipt = {
    "optimize": sys.flags.optimize,
    "executable": sys.executable,
    "python_version": list(sys.version_info[:3]),
    "phase": "dependency-import",
    "warnings": [],
}

def show_warning(message, category, filename, lineno, file=None, line=None):
    receipt["warnings"].append({
        "category": category.__name__, "message": str(message),
        "filename": filename, "lineno": lineno,
    })
    (file or sys.stderr).write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = show_warning
warnings.simplefilter("always")
status = 1
try:
    import numpy as np
    # These public SciPy modules are dependencies already used by the authorized
    # utility tests. Preflight them separately from the dphtools import.
    from scipy import fft, ndimage, signal

    receipt["phase"] = "utility-import"
    from dphtools.utils import fft_pad

    receipt["phase"] = "same-shape-padding"
    data = np.array([[-3.0, 0.0, 2.5], [7.0, 4.0, -1.5]])
    result = fft_pad(data, (2, 3))
    receipt["shape"] = list(result.shape)
    receipt["values"] = result.tolist()
    receipt["phase"] = "complete"
    status = 0
except Exception as error:
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append({
            "module": tb.tb_frame.f_globals.get("__name__", ""),
            "filename": tb.tb_frame.f_code.co_filename,
            "lineno": tb.tb_lineno,
        })
        tb = tb.tb_next
    origin = frames[-1]["module"] if frames else ""
    if receipt["phase"] == "dependency-import":
        scope = "dependency"
    elif isinstance(error, ModuleNotFoundError):
        scope = "product-unavailable" if (error.name or "").startswith("dphtools") else "dependency"
    elif origin.startswith("dphtools"):
        scope = "product"
    else:
        scope = "external-or-unclassified"
    receipt["failure_scope"] = scope
    receipt["error"] = {
        "category": type(error).__name__, "message": str(error), "frames": frames,
    }
    sys.stderr.write(f"{scope} failure during {receipt['phase']}: {type(error).__name__}: {error}\n")
receipt["status"] = status
print("DPHTOOLS_RUNTIME_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
raise SystemExit(status)
"""
    receipts = []
    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    environment["MPLBACKEND"] = "Agg"
    for options, expected_optimization in [((), 0), (("-OO",), 2)]:
        command = [sys.executable, "-E", "-B", *options, "-c", script]
        entry = {"expected_optimize": expected_optimization}
        try:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            entry.update(
                returncode=process.returncode, stdout=process.stdout, stderr=process.stderr
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            entry["host_error"] = (
                "TimeoutExpired: interpreter did not complete within 60 s"
                if isinstance(error, subprocess.TimeoutExpired)
                else f"{type(error).__name__}: {error}"
            )
            # A timeout can carry partial output; preserve it for diagnosis.
            for channel in ("stdout", "stderr"):
                output = getattr(error, channel, None)
                entry[channel] = (
                    output.decode(errors="replace") if isinstance(output, bytes) else output
                )
        receipts.append(entry)

    record_property("runtime_mode_receipts", json.dumps(receipts, sort_keys=True))
    problems = []
    results = []
    for entry in receipts:
        label = f"optimize={entry['expected_optimize']}"
        if "host_error" in entry:
            problems.append(f"{label}: host/interpreter failure: {entry['host_error']}")
            continue
        lines = [
            line.removeprefix("DPHTOOLS_RUNTIME_RECEIPT ")
            for line in entry["stdout"].splitlines()
            if line.startswith("DPHTOOLS_RUNTIME_RECEIPT ")
        ]
        if len(lines) != 1:
            problems.append(f"{label}: missing/ambiguous receipt; status={entry['returncode']}")
            continue
        receipt = json.loads(lines[0])
        if receipt["optimize"] != entry["expected_optimize"]:
            problems.append(f"{label}: actual optimize={receipt['optimize']}")
        if receipt["executable"] != sys.executable:
            problems.append(f"{label}: unexpected interpreter {receipt['executable']}")
        if receipt["python_version"] != list(sys.version_info[:3]):
            problems.append(f"{label}: unexpected Python version {receipt['python_version']}")
        if entry["returncode"] != 0 or receipt["status"] != 0:
            problems.append(
                f"{label}: {receipt.get('failure_scope', 'unclassified')} failure during "
                f"{receipt['phase']}; process status={entry['returncode']}; "
                f"{receipt.get('error')}"
            )
            continue
        if receipt["phase"] != "complete" or receipt.get("shape") != [2, 3]:
            problems.append(f"{label}: operation did not return the required 2x3 array")
        if receipt.get("values") != [[-3.0, 0.0, 2.5], [7.0, 4.0, -1.5]]:
            problems.append(f"{label}: incorrect values {receipt.get('values')}")
        results.append(receipt.get("values"))
    assert not problems, "\n".join(problems)
    assert len(results) == 2 and results[0] == results[1]
