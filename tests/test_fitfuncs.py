"""Independent single-exponential fits, including the legacy fast-decay regime."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dphtools.utils.fitfuncs import exponent_fit


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "amplitude, rate, offset, stop, samples, rtol, atol",
    [(6, 0.75, 2, 5, 41, 1e-6, 1e-8), (10, 3, 5, 10, 50, 1e-3, 0)],
    ids=["ordinary-decay", "legacy-fast-decay"],
)
def test_exponent_fit_independently_generated_decay(
    sign, amplitude, rate, offset, stop, samples, rtol, atol
):
    x = np.linspace(0, stop, samples)
    data = sign * (amplitude * np.exp(-rate * x) + offset)
    parameters, covariance = exponent_fit(data, x)
    assert_allclose(parameters, [sign * amplitude, rate, sign * offset], rtol=rtol, atol=atol)
    assert covariance.shape == (3, 3)
