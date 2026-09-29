"""Postimplementation unit equivariance, without a noisy-fit accuracy oracle."""

import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils.lpsvd import LPSVD, reconstruct_signal

FIELDS = ["amps", "freqs", "damps", "phase"]


def _evaluate(terms, sample_count):
    # Euler's real exponential sum, independent of public reconstruction.
    return np.array(
        [
            math.fsum(
                amplitude * math.exp(damping * n) * math.cos(math.tau * frequency * n + phase)
                for amplitude, frequency, damping, phase in terms
            )
            for n in range(sample_count)
        ]
    )


def test_automatic_real_fit_is_equivariant_under_positive_amplitude_units(record_property):
    sample = np.arange(64, dtype=float)
    signal = 2 * np.exp(-0.025 * sample) * np.cos(math.tau * 0.125 * sample + 0.2)
    # A small, fixed, non-random full-rank perturbation: both adjacent
    # 32x32 Hankel conventions have perturbation sigma_min > .00096.
    # This is far above roundoff; no numerical-rank threshold is selected.
    signal[31:33] += 0.02
    scale, stop = 7.0, 72
    inputs = [signal, scale * signal]
    originals = [value.copy() for value in inputs]
    record_property("input_signals", [value.tolist() for value in originals])
    fits, evaluated, reconstructed = [], [], []
    for label, data in zip(["original_units", "scaled_units"], inputs):
        try:
            # All automatic-order and default bias/error algorithms remain
            # active. No requested order, private helper or mock is supplied.
            fitted = LPSVD(data)
        finally:
            assert_array_equal(data, originals[len(fits)])
        terms = fitted.sort_values("freqs")[FIELDS].to_numpy()
        record_property(label + "_coefficients", terms.tolist())
        # The independent input audit has a nonzero, isolated MDL minimum
        # (score margin >112), nonzero prediction singular values, and a
        # regular separated-cosine Jacobian. Empty/nonfinite output would
        # make unit comparisons vacuous, rather than establish invariance.
        assert terms.ndim == 2 and terms.shape[1] == 4 and len(terms) > 0
        assert np.isrealobj(terms) and np.isfinite(terms).all()
        assert np.all(terms[:, 0] > 0)
        fits.append(terms.copy())
        direct = _evaluate(terms, stop)
        assert np.isfinite(direct).all() and np.max(np.abs(direct)) > 0
        evaluated.append(direct)
        original_coefficients = fitted.copy(deep=True)
        template = np.zeros(stop)
        original_template = template.copy()
        try:
            public = np.asarray(reconstruct_signal(fitted, template))
        finally:
            assert_frame_equal(fitted, original_coefficients)
            assert_array_equal(template, original_template)
        assert public.shape == (stop,) and np.isrealobj(public) and np.isfinite(public).all()
        assert np.max(np.abs(public)) > 0
        reconstructed.append(public.copy())
        record_property(label + "_independent_evaluation", direct.tolist())
        record_property(label + "_public_reconstruction", public.tolist())

    first, second = fits
    record_property("returned_counts", [len(first), len(second)])
    assert first.shape == second.shape
    # Common scaling cancels in the MDL log geometric/arithmetic ratio and
    # in H^+ y. Exponential poles are unchanged; complex amplitudes scale.
    # Input audit: full Hankel condition <19000, physical Jacobian <421,
    # unit-conversion rounding <2e-16. A 1e-6 comparison allowance leaves
    # substantial room for numerical work; it is not a noisy accuracy bound.
    assert_allclose(second[:, 0] / scale, first[:, 0], rtol=1e-6, atol=0)
    assert_allclose(second[:, 1:3], first[:, 1:3], rtol=0, atol=1e-6)
    phase_difference = np.angle(np.exp(1j * (second[:, 3] - first[:, 3])))
    assert_allclose(phase_difference, 0, rtol=0, atol=1e-6)
    for kind, values in [("independent", evaluated), ("public", reconstructed)]:
        size = float(np.max(np.abs(values[0])))
        error = float(np.max(np.abs(values[1] / scale - values[0])))
        record_property(kind + "_relative_unit_error", error / size)
        # Relative to the returned nonzero model, including eight extra
        # sample indices. No comparison to the clean or perturbed input.
        assert_allclose(values[1] / scale, values[0], rtol=0, atol=1e-6 * size)
    for data, original in zip(inputs, originals):
        assert_array_equal(data, original)
