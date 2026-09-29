"""Supplemental real-signal oracles from Euler's identity and Kumaresan--Tufts.

The 1982 paper's equations (1)--(4), pp. 833--834, use M exponential terms:
https://www.math.ucdavis.edu/~saito/data/sonar/KumaresanTufts.pdf
Each A*exp(-d*n)*cos(2*pi*f*n+phi) contributes two terms with amplitudes
A/2, damping -d, frequencies +/-f, and phases +/-phi modulo 2*pi.
These are supplemental baseline tests, not pre-implementation evidence.
"""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

FIELDS = ["amps", "freqs", "damps", "phase"]
# Literal analytic coefficients, ordered by frequency; no fitted values enter.
EXPECTED_TERMS = np.array(
    [
        [0.45, -0.234375, -0.045, -3.6],
        [1.2, -0.109375, -0.025, -0.45],
        [1.2, 0.109375, -0.025, 0.45],
        [0.45, 0.234375, -0.045, 3.6],
    ]
)


def _real_signal(sample):
    """Evaluate the two physical cosines without using coefficient machinery."""
    return 2.4 * np.exp(-0.025 * sample) * np.cos(
        2 * np.pi * 0.109375 * sample + 0.45
    ) + 0.9 * np.exp(-0.045 * sample) * np.cos(2 * np.pi * 0.234375 * sample + 3.6)


@pytest.mark.parametrize(
    "lfactor, removebias",
    [
        pytest.param(1 / 3, False, id="odd-tall-no-bias"),
        pytest.param(1 / 2, True, id="odd-half-with-bias"),
        pytest.param(2 / 3, True, id="odd-wide-with-bias"),
    ],
)
def test_analytic_coefficients_and_held_out_samples(lfactor, removebias):
    sample_count, model_order = 63, 4
    # (N-L, L) is (42, 21), (32, 31), or (21, 42). All have more
    # singular values than M, including when removing noise bias.
    columns = math.floor(sample_count * lfactor)
    assert min(sample_count - columns, columns) > model_order
    signal = _real_signal(np.arange(sample_count, dtype=float))
    fitted = LPSVD(signal, M=model_order, lfactor=lfactor, removebias=removebias)

    actual = fitted.sort_values("freqs")[FIELDS].to_numpy()
    assert actual.shape == EXPECTED_TERMS.shape
    # These separated, moderately damped, noiseless components permit tight
    # float64 checks, with room for SVD/root-solving differences across builds.
    assert_allclose(actual[:, :3], EXPECTED_TERMS[:, :3], rtol=1e-8, atol=1e-9)
    phase_error = np.angle(np.exp(1j * (actual[:, 3] - EXPECTED_TERMS[:, 3])))
    assert_allclose(phase_error, 0, rtol=0, atol=1e-8)

    # Evaluate 19 unseen samples independently with scalar real math. The
    # expected continuation uses neither fitted terms nor reconstruct_signal.
    stop = sample_count + 19
    expected_tail = np.array(
        [
            2.4 * math.exp(-0.025 * n) * math.cos(math.tau * 0.109375 * n + 0.45)
            + 0.9 * math.exp(-0.045 * n) * math.cos(math.tau * 0.234375 * n + 3.6)
            for n in range(sample_count, stop)
        ]
    )
    continued = reconstruct_signal(fitted, np.zeros(stop, dtype=float))
    assert continued.shape == (stop,)
    assert_allclose(continued[sample_count:], expected_tail, rtol=1e-8, atol=1e-9)


def test_reconstruct_signal_from_analytic_coefficients():
    # Interleave the conjugate pairs. LPSVD is never called in this test.
    coefficients = pd.DataFrame(EXPECTED_TERMS[[2, 0, 3, 1]], columns=FIELDS)
    sample = np.arange(83, dtype=float)
    expected = _real_signal(sample)
    actual = reconstruct_signal(coefficients, np.zeros(sample.size, dtype=float))
    assert actual.shape == expected.shape
    # This checks only direct evaluation, so no estimation-error allowance is
    # needed. The absolute tolerance remains meaningful near cosine zeros.
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
