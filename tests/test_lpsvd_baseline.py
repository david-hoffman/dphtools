"""SETUP-001 L2/L3/L5: recover independently generated damped real sinusoids.

Coefficient units, phase conventions and error-estimator semantics are not
specified by the packet. Only end-to-end signal reconstruction is asserted.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal


@pytest.mark.parametrize("removebias", [False, True])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("lfactor", [1 / 3, 1 / 2])
def test_damped_sinusoids_reconstruct_original_signal(components, removebias, lfactor):
    time = np.arange(48.0)
    signal = 2 * np.exp(-0.08 * time) * np.cos(2 * np.pi * 0.125 * time + 0.2)
    if components == 2:
        signal += 0.75 * np.exp(-0.04 * time) * np.cos(2 * np.pi * 0.25 * time - 0.3)
    # Each real sinusoid has two conjugate exponential components.
    coefficients = LPSVD(signal, M=2 * components, removebias=removebias, lfactor=lfactor)
    reconstructed = reconstruct_signal(coefficients, signal)
    assert_allclose(reconstructed, signal, atol=1e-8, rtol=1e-7)


def test_lpsvd_error_populates_finite_nonnegative_fields_for_regular_noisy_fit():
    from dphtools.utils.lpsvd import calc_LPSVD_error

    sample = np.arange(64.0)
    amplitude, damping, frequency, phase = 2.0, 0.04, 0.125, 0.2
    envelope = np.exp(-damping * sample)
    angle = 2 * np.pi * frequency * sample + phase
    clean_signal = amplitude * envelope * np.cos(angle)
    noise = np.random.default_rng(7923).normal(scale=0.001, size=sample.size)
    noisy_signal = clean_signal + noise
    # The physical real-sinusoid model has four identifiable parameters. Its
    # independent analytic derivative columns have full rank on these samples.
    # Frequency is away from zero/Nyquist, amplitude is nonzero, and noise is
    # nonzero. This guards regularity without choosing uncertainty calibration.
    physical_jacobian = np.column_stack(
        (
            envelope * np.cos(angle),
            -amplitude * sample * envelope * np.cos(angle),
            -2 * np.pi * amplitude * sample * envelope * np.sin(angle),
            -amplitude * envelope * np.sin(angle),
        )
    )
    assert np.linalg.matrix_rank(physical_jacobian) == 4
    assert np.var(noise) > 0
    fitted = LPSVD(noisy_signal, M=2, removebias=True, lfactor=0.5)
    # A real sinusoid has two conjugate exponential components. LPSVD's public
    # output already includes error columns; provide its coefficient fields to
    # the routine documented to add errors, without requiring idempotence.
    coefficient_fields = ["amps", "freqs", "damps", "phase"]
    coefficients = fitted[coefficient_fields].copy()
    calc_LPSVD_error(coefficients, noisy_signal)
    error_fields = [name + "_error" for name in coefficient_fields]
    assert set(error_fields).issubset(coefficients.columns)
    errors = coefficients[error_fields].to_numpy()
    assert errors.shape == (len(fitted), len(coefficient_fields))
    assert errors.size > 0
    assert np.isfinite(errors).all()
    assert (errors >= 0).all()
    # No error magnitudes, units, noise scaling, or covariance are asserted.
