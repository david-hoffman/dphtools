"""Supplemental acceptance checks, authored after implementation.

Six analytic geometries cross ordinary and unit-circle-boundary damping.
Only the latter permits an informative numerical fitting exception. No expected
outcome, backend, exception message, or number of exceptions is prescribed.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

FIELDS = ["amps", "freqs", "damps", "phase"]
AMPLITUDE = 1.3
HELD_OUT = 19
SIGNAL_ATOL = AMPLITUDE * 1e-8


def _signal(stop, frequency, phase, damping):
    return np.array(
        [
            AMPLITUDE * math.exp(-damping * n) * math.cos(math.tau * frequency * n + phase)
            for n in range(stop)
        ],
        dtype=np.float64,
    )


def _diagnostic(error):
    """Retain caught exception diagnostics without reading source or locals."""
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append(
            {"filename": code.co_filename, "line": traceback.tb_lineno, "function": code.co_name}
        )
        traceback = traceback.tb_next
    return {"category": type(error).__name__, "message": str(error), "frames": frames}


def _assert_signal(actual, expected, sample_count, label, record_property):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    record_property(
        label + "_max_absolute_errors",
        {
            "training": float(np.max(np.abs(actual[:sample_count] - expected[:sample_count]))),
            "held_out": float(np.max(np.abs(actual[sample_count:] - expected[sample_count:]))),
            "allowed": SIGNAL_ATOL,
        },
    )
    assert_allclose(actual[:sample_count], expected[:sample_count], rtol=0, atol=SIGNAL_ATOL)
    assert_allclose(actual[sample_count:], expected[sample_count:], rtol=0, atol=SIGNAL_ATOL)


@pytest.mark.parametrize(
    "sample_count, frequency",
    [(9, 0.125), (12, 0.21875), (17, 0.375)],
    ids=["n9-f0.125", "n12-f0.21875", "n17-f0.375"],
)
@pytest.mark.parametrize("phase", [-0.8, 3.7], ids=["negative-phase", "wrapped-phase"])
@pytest.mark.parametrize("damping", [0.01, 1e-16], ids=["ordinary-control", "precision-boundary"])
def test_real_explicit_pair_is_accurate_or_reports_boundary_failure(
    sample_count, frequency, phase, damping, record_property
):
    # L=floor(N/2) gives shapes (5,4), (6,6), (9,8), all above M=2.
    # The Euler-pair circular frequency separation is >=0.25 cycles/sample.
    # Every consecutive 2x2 Hankel minor is
    # -A**2 * exp(-d*(2*j+2)) * sin(2*pi*f)**2 != 0: exact rank two.
    columns = sample_count // 2
    assert min(sample_count - columns, columns) > 2
    expected = _signal(sample_count + HELD_OUT, frequency, phase, damping)
    signal = expected[:sample_count].copy()
    original = signal.copy()
    assert np.isfinite(signal).all()

    try:
        fitted = LPSVD(signal, M=2, lfactor=0.5, removebias=False)
    except (
        ArithmeticError,
        ValueError,
        RuntimeError,
        RuntimeWarning,
        np.linalg.LinAlgError,
    ) as error:
        if damping != 1e-16:
            raise
        record_property("fit_outcome", "boundary_numerical_exception")
        record_property("boundary_diagnostic", _diagnostic(error))
        assert str(error).strip()
        return
    finally:
        assert signal.dtype == original.dtype
        assert_array_equal(signal, original)
        assert signal.tobytes() == original.tobytes()
        record_property("caller_signal_preserved", True)

    record_property("fit_outcome", "returned")
    physical = fitted[FIELDS].copy(deep=True)
    actual = physical.sort_values("freqs").to_numpy()
    assert actual.shape == (2, 4)
    assert np.isrealobj(actual)
    actual = np.asarray(actual, dtype=float)
    assert np.isfinite(actual).all()
    analytic = np.array(
        [
            [AMPLITUDE / 2, -frequency, -damping, -phase],
            [AMPLITUDE / 2, frequency, -damping, phase],
        ]
    )
    # Product-free analysis: nonzero Hankel condition <1.30, physical
    # Jacobian condition <152, first-order rounding scales <8.3e-13 for
    # parameters and <2.6e-12 for continuation. Existing coefficient
    # allowances and 1.3e-8 absolute signal error leave generous margin.
    # No relative accuracy or sign is required for sub-epsilon damping.
    assert_allclose(actual[:, :3], analytic[:, :3], rtol=1e-8, atol=1e-9)
    phase_error = (actual[:, 3] - analytic[:, 3] + math.pi) % math.tau - math.pi
    assert_allclose(phase_error, 0, rtol=0, atol=1e-8)
    record_property("analytic_coefficient_assertions_passed", True)

    # Independent coefficient evaluation: neither expectation nor this sum
    # calls the public reconstruction function. All returned rows are used.
    evaluated = np.array(
        [
            math.fsum(
                amplitude * math.exp(signed_damping * n) * math.cos(math.tau * f * n + phi)
                for amplitude, f, signed_damping, phi in actual
            )
            for n in range(sample_count + HELD_OUT)
        ]
    )
    _assert_signal(evaluated, expected, sample_count, "independent", record_property)
    record_property("independent_continuation_assertions_passed", True)

    if damping == 0.01:
        # Supplementary public evaluator check for every ordinary control.
        original_coefficients = physical.copy(deep=True)
        template = np.zeros(sample_count + HELD_OUT, dtype=np.float64)
        original_template = template.copy()
        try:
            reconstructed = reconstruct_signal(
                physical, template, ampcutoff=0, freqcutoff=0, dampcutoff=0
            )
        finally:
            assert_frame_equal(physical, original_coefficients, check_exact=True)
            assert_array_equal(template, original_template)
            assert template.dtype == original_template.dtype
            assert template.tobytes() == original_template.tobytes()
        _assert_signal(reconstructed, expected, sample_count, "public", record_property)
        record_property("ordinary_public_assertions_and_inputs_passed", True)
