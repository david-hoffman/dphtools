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
