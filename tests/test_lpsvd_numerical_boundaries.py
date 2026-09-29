"""Supplemental real-signal consistency at ordinary and subnormal float64 scales.

This is maintenance evidence, not a claim of original test-first history.
Numerical fitting failure is permitted at the extreme scale; a returned model
must describe the known normalized signal with finite physical coefficients.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

FIELDS = ["amps", "freqs", "damps", "phase"]
EXPECTED_TERMS = np.array(
    [
        [0.45, -0.21, -0.026, 0.65],
        [1.2, -0.07, -0.012, -0.4],
        [1.2, 0.07, -0.012, 0.4],
        [0.45, 0.21, -0.026, -0.65],
    ]
)


def _analytic_signal(sample):
    return 2.4 * np.exp(-0.012 * sample) * np.cos(math.tau * 0.07 * sample + 0.4) + (
        0.9 * np.exp(-0.026 * sample) * np.cos(math.tau * 0.21 * sample - 0.65)
    )


def _numerical_diagnostic(error):
    """Retain a caught diagnosis and frame locations without source excerpts."""
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append(
            {"filename": code.co_filename, "line": traceback.tb_lineno, "function": code.co_name}
        )
        traceback = traceback.tb_next
    return {"category": type(error).__name__, "message": str(error), "frames": frames}


@pytest.mark.parametrize("scale", [1.0, 1e-310], ids=["ordinary", "subnormal"])
def test_two_scale_real_signal_has_honest_finite_fit_or_numerical_diagnosis(
    scale, record_property
):
    sample_count, model_order, lfactor = 83, 4, 0.5
    columns = math.floor(sample_count * lfactor)
    assert min(sample_count - columns, columns) > model_order
    ordinary = _analytic_signal(np.arange(sample_count, dtype=float))
    signal = scale * ordinary
    original = signal.copy()
    assert np.isfinite(signal).all() and np.all(signal != 0)
    if scale < 1:
        assert np.all(np.abs(signal) < np.finfo(float).tiny)
        # Float64 subnormal rounding is absolute: half of the smallest
        # positive float divided by scale is about 2.47e-14 after rescaling.
        half_quantum = float(np.nextafter(0.0, 1.0)) / scale / 2
        rounding = 4 * np.finfo(float).eps * np.max(np.abs(ordinary))
        assert_allclose(signal / scale, ordinary, rtol=0, atol=half_quantum + rounding)
        record_property("normalized_input_quantization_bound", half_quantum + rounding)

    try:
        fitted = LPSVD(signal, M=model_order, lfactor=lfactor, removebias=False)
    except (
        ArithmeticError,
        ValueError,
        RuntimeError,
        RuntimeWarning,
        np.linalg.LinAlgError,
    ) as error:
        if scale == 1:
            raise
        # Accept an informative ordinary numerical/data-fitting diagnosis,
        # never NameError/AttributeError or an assertion about implementation.
        record_property("extreme_numerical_diagnosis", _numerical_diagnostic(error))
        assert str(error).strip()
        return
    finally:
        assert_array_equal(signal, original)

    physical = fitted[FIELDS].copy(deep=True)
    actual = physical.to_numpy()
    record_property("successful_main_coefficients", actual.tolist())
    assert actual.shape == (model_order, len(FIELDS))
    assert np.isfinite(actual).all()
    normalized = physical.copy(deep=True)
    normalized["amps"] = normalized["amps"] / scale
    assert np.isfinite(normalized.to_numpy()).all()

    if scale == 1:
        ordered = normalized.sort_values("freqs").to_numpy()
        assert_allclose(ordered[:, :3], EXPECTED_TERMS[:, :3], rtol=1e-8, atol=1e-9)
        phase_error = (ordered[:, 3] - EXPECTED_TERMS[:, 3] + math.pi) % math.tau - math.pi
        assert_allclose(phase_error, 0, rtol=0, atol=1e-8)

    stop = sample_count + 19
    expected = np.array(
        [
            2.4 * math.exp(-0.012 * n) * math.cos(math.tau * 0.07 * n + 0.4)
            + 0.9 * math.exp(-0.026 * n) * math.cos(math.tau * 0.21 * n - 0.65)
            for n in range(stop)
        ]
    )
    # Evaluate returned coefficients independently, so successful consistency
    # does not rely only on a round trip through two product functions.
    sample = np.arange(stop, dtype=float)
    evaluated = np.zeros(stop)
    for amplitude, frequency, damping, phase in normalized.to_numpy():
        evaluated += (
            amplitude * np.exp(damping * sample) * np.cos(math.tau * frequency * sample + phase)
        )
    assert np.isfinite(evaluated).all()
    # Independent input conditioning gives nonzero Hankel condition ~4.25,
    # quantization/gap perturbation <1.4e-13, and first-order physical
    # parameter perturbation ~1.2e-13. These allowances leave substantial
    # numerical margin while rejecting zero/underflow and incorrect signals.
    rtol, atol = (1e-8, 1e-9) if scale == 1 else (1e-7, 1e-8)
    assert_allclose(evaluated[:sample_count], expected[:sample_count], rtol=rtol, atol=atol)
    assert_allclose(evaluated[sample_count:], expected[sample_count:], rtol=rtol, atol=atol)

    original_coefficients = normalized.copy(deep=True)
    template = np.zeros(stop)
    original_template = template.copy()
    try:
        reconstructed = reconstruct_signal(
            normalized, template, ampcutoff=0, freqcutoff=0, dampcutoff=0
        )
    finally:
        assert_frame_equal(normalized, original_coefficients)
        assert_array_equal(template, original_template)
    assert reconstructed.shape == expected.shape
    assert np.isfinite(reconstructed).all()
    assert_allclose(reconstructed[:sample_count], expected[:sample_count], rtol=rtol, atol=atol)
    assert_allclose(reconstructed[sample_count:], expected[sample_count:], rtol=rtol, atol=atol)
    record_property(
        "normalized_signal_max_absolute_errors",
        {
            "independent": float(np.max(np.abs(evaluated - expected))),
            "public_training": float(
                np.max(np.abs(reconstructed[:sample_count] - expected[:sample_count]))
            ),
            "public_held_out": float(
                np.max(np.abs(reconstructed[sample_count:] - expected[sample_count:]))
            ),
        },
    )
