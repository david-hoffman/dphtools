"""Supplemental real-signal consistency near a unit-radius precision boundary.

The ordinary damping control must fit. At damping 1e-16, an informative numerical
diagnosis is allowed; a returned fit must satisfy independent analytic oracles.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

FIELDS = ["amps", "freqs", "damps", "phase"]
SAMPLE_COUNT, STOP = 6, 25


def _expected_terms(damping):
    return np.array([[0.65, -0.07, -damping, -0.4], [0.65, 0.07, -damping, 0.4]])


def _expected_signal(damping):
    # Scalar real arithmetic supplies training and 19 unseen samples. No
    # fitted coefficient or production reconstruction enters this expectation.
    return np.array(
        [1.3 * math.exp(-damping * n) * math.cos(math.tau * 0.07 * n + 0.4) for n in range(STOP)]
    )


def _assert_finite_analytic_fit(fitted, damping):
    physical = fitted[FIELDS].copy(deep=True)
    actual = physical.sort_values("freqs").to_numpy()
    assert actual.shape == (2, 4)
    assert np.isrealobj(actual)
    actual = np.asarray(actual, dtype=float)
    assert np.isfinite(actual).all()
    expected_terms = _expected_terms(damping)
    # Independent conditioning: nonzero Hankel condition <2, physical
    # Jacobian condition <214, and a conservative first-order parameter
    # rounding bound <1.1e-13. These existing real-signal allowances leave
    # substantial numerical margin and do not demand sub-epsilon damping.
    assert_allclose(actual[:, :3], expected_terms[:, :3], rtol=1e-8, atol=1e-9)
    phase_error = (actual[:, 3] - expected_terms[:, 3] + math.pi) % math.tau - math.pi
    assert_allclose(phase_error, 0, rtol=0, atol=1e-8)

    sample = np.arange(STOP, dtype=float)
    evaluated = np.zeros(STOP)
    for amplitude, frequency, signed_damping, phase in actual:
        evaluated += (
            amplitude
            * np.exp(signed_damping * sample)
            * np.cos(math.tau * frequency * sample + phase)
        )
    expected = _expected_signal(damping)
    assert np.isfinite(evaluated).all()
    # The independently calculated held-out linearized rounding bound is
    # below 9.2e-13. An absolute allowance matters near cosine zero crossings.
    assert_allclose(evaluated[:SAMPLE_COUNT], expected[:SAMPLE_COUNT], rtol=1e-8, atol=1e-9)
    assert_allclose(evaluated[SAMPLE_COUNT:], expected[SAMPLE_COUNT:], rtol=1e-8, atol=1e-9)
    return physical, expected, evaluated


def _numerical_diagnostic(error):
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append(
            {"filename": code.co_filename, "line": traceback.tb_lineno, "function": code.co_name}
        )
        traceback = traceback.tb_next
    return {"category": type(error).__name__, "message": str(error), "frames": frames}


@pytest.mark.parametrize("damping", [0.01, 1e-16], ids=["ordinary-control", "near-unit-boundary"])
def test_near_undamped_real_signal_has_correct_fit_or_boundary_diagnosis(damping, record_property):
    model_order, lfactor = 2, 0.5
    columns = math.floor(SAMPLE_COUNT * lfactor)
    assert min(SAMPLE_COUNT - columns, columns) > model_order
    sample = np.arange(SAMPLE_COUNT, dtype=float)
    signal = 1.3 * np.exp(-damping * sample) * np.cos(math.tau * 0.07 * sample + 0.4)
    original = signal.copy()
    assert np.isfinite(signal).all() and np.all(signal != 0)

    try:
        fitted = LPSVD(signal, M=model_order, lfactor=lfactor, removebias=False)
    except (
        ArithmeticError,
        ValueError,
        RuntimeError,
        RuntimeWarning,
        np.linalg.LinAlgError,
    ) as error:
        if damping != 1e-16:
            raise
        record_property("precision_boundary_numerical_diagnosis", _numerical_diagnostic(error))
        assert str(error).strip()
        return
    finally:
        assert_array_equal(signal, original)

    record_property("returned_main_coefficients", fitted[FIELDS].to_numpy().tolist())
    physical, expected, evaluated = _assert_finite_analytic_fit(fitted, damping)
    record_property(
        "independent_max_absolute_errors",
        {
            "training": float(np.max(np.abs(evaluated[:SAMPLE_COUNT] - expected[:SAMPLE_COUNT]))),
            "held_out": float(np.max(np.abs(evaluated[SAMPLE_COUNT:] - expected[SAMPLE_COUNT:]))),
        },
    )

    if damping == 0.01:
        # The ordinary control additionally exercises the documented public
        # evaluator. Both successful fits already face the independent
        # coefficient and continuation checks above, including damping zero
        # or either tiny rounding sign within the absolute damping allowance.
        original_coefficients = physical.copy(deep=True)
        template = np.zeros(STOP)
        original_template = template.copy()
        try:
            reconstructed = reconstruct_signal(
                physical, template, ampcutoff=0, freqcutoff=0, dampcutoff=0
            )
        finally:
            assert_frame_equal(physical, original_coefficients)
            assert_array_equal(template, original_template)
        assert reconstructed.shape == (STOP,)
        assert np.isfinite(reconstructed).all()
        assert_allclose(
            reconstructed[:SAMPLE_COUNT], expected[:SAMPLE_COUNT], rtol=1e-8, atol=1e-9
        )
        assert_allclose(
            reconstructed[SAMPLE_COUNT:], expected[SAMPLE_COUNT:], rtol=1e-8, atol=1e-9
        )
        record_property(
            "ordinary_public_max_absolute_error", float(np.max(abs(reconstructed - expected)))
        )
