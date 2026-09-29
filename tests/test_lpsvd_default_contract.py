"""Default LPSVD: independent exact signals, coefficient sums and continuation."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

FIELDS = ["amps", "freqs", "damps", "phase"]


@pytest.mark.parametrize(
    "sample_count, components",
    [
        (64, [(2.0, -math.log(0.99), 0.125, 0.0)]),
        (63, [(2.4, 0.025, 0.109375, 0.45), (0.9, 0.045, 0.234375, 3.6)]),
    ],
    ids=["approved-single-cosine", "odd-two-separated-cosines"],
)
@pytest.mark.parametrize("evaluation", ["independent-coefficients", "public-reconstruction"])
def test_default_automatic_fit_predicts_eight_exact_unseen_samples(
    sample_count, components, evaluation, record_property
):
    expected = np.array(
        [
            math.fsum(
                amplitude * math.exp(-damping * n) * math.cos(math.tau * frequency * n + phase)
                for amplitude, damping, frequency, phase in components
            )
            for n in range(sample_count + 8)
        ]
    )
    signal = expected[:sample_count].copy()
    original = signal.copy()
    try:
        fitted = LPSVD(signal)
    finally:
        assert_array_equal(signal, original)
    coefficients = fitted[FIELDS].to_numpy()
    assert np.isrealobj(coefficients)
    assert np.isfinite(coefficients).all()

    if evaluation == "independent-coefficients":
        # Euler's identity gives Re[A exp(d*n + i*(2*pi*f*n + phase))].
        # Every returned row participates; no automatic mode count is imposed.
        actual = np.array(
            [
                math.fsum(
                    amplitude * math.exp(damping * n) * math.cos(math.tau * frequency * n + phase)
                    for amplitude, frequency, damping, phase in coefficients
                )
                for n in range(sample_count + 8)
            ]
        )
    else:
        original_fitted = fitted.copy(deep=True)
        template = np.zeros(sample_count + 8, dtype=float)
        try:
            actual = reconstruct_signal(fitted, template)
        finally:
            assert_frame_equal(fitted, original_fitted)
            assert_array_equal(template, np.zeros(sample_count + 8))

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    amplitude_scale = sum(component[0] for component in components)
    record_property(
        "absolute_prediction_errors",
        {
            "training": float(np.max(np.abs(actual[:sample_count] - expected[:sample_count]))),
            "held_out": float(np.max(np.abs(actual[sample_count:] - expected[sample_count:]))),
            "allowed": 1e-6 * amplitude_scale,
        },
    )
    # The approved bound is absolute, including at cosine zero crossings.
    # Sum of component amplitudes bounds the exact signal's envelope.
    assert_allclose(
        actual[:sample_count], expected[:sample_count], rtol=0, atol=1e-6 * amplitude_scale
    )
    assert_allclose(
        actual[sample_count:], expected[sample_count:], rtol=0, atol=1e-6 * amplitude_scale
    )


@pytest.mark.parametrize(
    "sample_count, lfactor, order, removebias",
    [
        (12, 0.25, 4, False),
        (13, 0.75, 5, False),
        (12, 0.5, 6, True),
        (13, 0.3, 3, True),
        (13, 0.75, 4, True),
    ],
    ids=[
        "exceeds-columns",
        "exceeds-rows",
        "full-square-bias",
        "full-tall-bias",
        "full-wide-bias",
    ],
)
def test_explicit_order_impossible_for_prediction_dimensions_raises_value_error(
    sample_count, lfactor, order, removebias
):
    columns = math.floor(sample_count * lfactor)
    singular_value_count = min(sample_count - columns, columns)
    assert order > singular_value_count or (removebias and order == singular_value_count)
    signal = np.array([2 * 0.99**n * math.cos(math.pi * n / 4) for n in range(sample_count)])
    original = signal.copy()
    try:
        with pytest.raises(ValueError):
            LPSVD(signal, M=order, lfactor=lfactor, removebias=removebias)
    finally:
        assert_array_equal(signal, original)


def test_full_singular_value_order_without_bias_fits_exact_signal_and_eight_unseen_samples(
    record_property,
):
    sample_count, lfactor, model_order = 8, 0.25, 2
    columns = math.floor(sample_count * lfactor)
    assert (sample_count - columns, columns) == (6, 2)
    assert model_order == min(sample_count - columns, columns)
    # Euler's identity gives two distinct nonzero terms with poles +/-i*r,
    # r=exp(-.05). For any consecutive 2x2 Hankel minor starting at j,
    # det = y[j]*y[j+2]-y[j+1]**2 = -4*r**(2*j+2), which is nonzero.
    # Thus the prediction matrix has rank exactly 2, its full singular count.
    expected = np.array(
        [2 * math.exp(-0.05 * n) * math.cos(math.pi * n / 2 + 0.37) for n in range(16)]
    )
    signal = expected[:sample_count].copy()
    original = signal.copy()
    try:
        fitted = LPSVD(signal, M=model_order, lfactor=lfactor, removebias=False)
    finally:
        assert_array_equal(signal, original)

    coefficients = fitted.sort_values("freqs")[FIELDS].to_numpy()
    # This is the approved EXPLICIT rank-two coefficient contract; automatic
    # fits above retain their existing freedom in returned term count.
    assert coefficients.shape == (2, 4)
    assert np.isrealobj(coefficients) and np.isfinite(coefficients).all()
    analytic_terms = np.array([[1.0, -0.25, -0.05, -0.37], [1.0, 0.25, -0.05, 0.37]])
    assert_allclose(coefficients[:, :3], analytic_terms[:, :3], rtol=1e-8, atol=1e-9)
    phase_errors = np.angle(np.exp(1j * (coefficients[:, 3] - analytic_terms[:, 3])))
    assert_allclose(phase_errors, 0, rtol=0, atol=1e-8)

    evaluated = np.array(
        [
            math.fsum(
                amplitude * math.exp(damping * n) * math.cos(math.tau * frequency * n + phase)
                for amplitude, frequency, damping, phase in coefficients
            )
            for n in range(sample_count + 8)
        ]
    )
    original_fitted = fitted.copy(deep=True)
    template = np.zeros(sample_count + 8, dtype=float)
    try:
        reconstructed = reconstruct_signal(fitted, template)
    finally:
        assert_frame_equal(fitted, original_fitted)
        assert_array_equal(template, np.zeros(sample_count + 8))

    # Independent prediction conditioning is ~1.04; the first-order held-out
    # gain is ~2.58. These established explicit-coefficient tolerances and the
    # amplitude-scaled 2e-8 output allowance leave ample floating-point margin.
    for label, actual in (("independent", evaluated), ("public", reconstructed)):
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        record_property(
            label + "_full_order_absolute_errors",
            {
                "training": float(np.max(np.abs(actual[:sample_count] - expected[:sample_count]))),
                "held_out": float(np.max(np.abs(actual[sample_count:] - expected[sample_count:]))),
                "allowed": 2e-8,
            },
        )
        assert_allclose(actual[:sample_count], expected[:sample_count], rtol=0, atol=2e-8)
        assert_allclose(actual[sample_count:], expected[sample_count:], rtol=0, atol=2e-8)
