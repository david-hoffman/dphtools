"""Supplemental F1-F3 and L1-L2 oracles, independent of fitted output."""

import math
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import BarContainer
from matplotlib.patches import Polygon, StepPatch
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from dphtools.utils import fitfuncs
from dphtools.utils.lpsvd import estimate_model_order, reconstruct_signal


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def empirical_plot_observations(fig):
    """Read native points, vertical bars, and integer-centered step intervals."""
    for axis in fig.axes:
        for line in axis.lines:
            yield np.asarray(line.get_xydata())
        for collection in axis.collections:
            yield np.asarray(collection.get_offsets())
        for container in axis.containers:
            if isinstance(container, BarContainer) and container.orientation == "vertical":
                yield np.array(
                    [
                        [bar.get_x() + bar.get_width() / 2, bar.get_height()]
                        for bar in container.patches
                    ]
                )
        for patch in axis.patches:
            if isinstance(patch, StepPatch):
                values, edges, _ = patch.get_data()
                yield np.column_stack(((edges[:-1] + edges[1:]) / 2, values))
            elif isinstance(patch, Polygon):
                # Histograms trace the upper edge left-to-right. Ignore
                # vertical joins and the right-to-left baseline/closure.
                start, end = patch.get_xy()[:-1], patch.get_xy()[1:]
                horizontal = (end[:, 0] > start[:, 0]) & (end[:, 1] == start[:, 1])
                yield np.column_stack(
                    ((start[horizontal, 0] + end[horizontal, 0]) / 2, start[horizontal, 1])
                )


def assert_empirical_plot(fig, original, tail, density):
    """Check actual counts or relative frequencies, independent of artist type."""
    occurrence_counts = Counter(original.tolist())
    empirical = []
    for xy in empirical_plot_observations(fig):
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            continue
        x, y = xy.T
        if len(set(x)) != len(x) or not set(tail).issubset(set(x)):
            continue
        # Plotting the full observations or just their retained tail is
        # allowed. At every shown integer, y must be its actual count;
        # zero-count gaps may be present or absent. No fit curve oracle.
        if np.array_equal(x, np.floor(x)):
            if density:
                # Either full observations or the retained tail may be
                # normalized. Compare ratios only; never choose the common
                # factor, a bin-width convention, or a fitted exponent.
                for counts in (occurrence_counts, Counter(tail.tolist())):
                    if not set(counts).issubset(set(x)):
                        continue
                    expected = np.array([counts[int(value)] for value in x])
                    present = expected > 0
                    if np.all(y[present] > 0) and np.all(y[~present] == 0):
                        ratios = y[present] / expected[present]
                        empirical.append(
                            np.allclose(ratios / ratios[0], 1, rtol=1e-12, atol=1e-12)
                        )
            else:
                expected = [occurrence_counts[int(value)] for value in x]
                empirical.append(np.allclose(y, expected, rtol=0, atol=1e-12))
    assert any(empirical), "No plotted observations preserve the empirical counts/frequencies."


@pytest.mark.parametrize(
    "likelihood, observations, offset",
    [
        (fitfuncs.negloglikelihoodNB, [0, 1, 2, 2, 7, 31], 1),
        (fitfuncs.negloglikelihoodNB, [0, 0, 0], 1),
        (fitfuncs.negloglikelihoodZTNB, [1, 1, 2, 7, 31], 0),
        (fitfuncs.negloglikelihoodZTNB, [1], 0),
    ],
    ids=["nb-mixed", "nb-zero-only", "ztnb-mixed", "ztnb-singleton"],
)
def test_unit_negative_binomial_likelihood_matches_geometric_probabilities(
    likelihood, observations, offset
):
    counts = np.array(observations, dtype=np.int64)
    original = counts.copy()
    parameters = np.array([1.0, 1.0])
    # F1: P(k)=2**(-(k+1)); conditioning on nonzero counts doubles it.
    expected = sum(value + offset for value in observations) * math.log(2)
    assert_allclose(likelihood(parameters, counts), expected, rtol=1e-12, atol=1e-12)
    # Reversed noncontiguous input must retain the same multiset likelihood.
    assert_allclose(likelihood(parameters, counts[::-1]), expected, rtol=1e-12, atol=1e-12)
    if len(counts) > 1:
        split = len(counts) // 2
        separate = likelihood(parameters, counts[:split]) + likelihood(parameters, counts[split:])
        assert_allclose(separate, expected, rtol=1e-12, atol=1e-12)
    assert_array_equal(counts, original)
    assert_array_equal(parameters, [1, 1])


@pytest.mark.parametrize("explicit_axis", [False, True])
@pytest.mark.parametrize(
    "density, norm",
    [(False, False), (True, False), (True, True)],
    ids=["counts", "density", "normalized-density"],
)
def test_integer_power_law_refits_preserve_tail_and_plot_empirical_counts(
    explicit_axis, density, norm
):
    observations = np.array([9, 1, 4, 2, 4, 23, 5, 2, 7, 4, 12, 1, 9, 5, 17, 4, 2], dtype=np.int64)
    original = observations.copy()
    model = fitfuncs.PowerLaw(observations)
    # No observation equals either cutoff. Returning to 3 checks that a refit
    # does not permanently discard the data removed by the higher cutoff.
    for cutoff in (3, 6, 3):
        model.fit(xmin=cutoff, opt_max=False)
        tail = original[original > cutoff]
        assert_array_equal(np.sort(model.clipped_data), np.sort(tail))
        if explicit_axis:
            fig, ax = plt.subplots()
            model.plot(density=density, norm=norm, ax=ax)
        else:
            model.plot(density=density, norm=norm)
            fig = plt.gcf()
        fig.canvas.draw()
        assert_empirical_plot(fig, original, tail, density)
        assert_array_equal(observations, original)
        plt.close(fig)


def test_explicitly_fitted_power_law_intercept_evaluates_its_public_curve(record_property):
    observations = np.array([9, 1, 4, 2, 4, 23, 5, 2, 7, 4, 12, 1, 9, 5, 17, 4, 2], dtype=np.int64)
    original = observations.copy()
    model = fitfuncs.PowerLaw(observations)
    model.fit(xmin=3, opt_max=False)
    alpha = model.alpha
    levels = np.array([0.125, 0.5, 2.0])
    intercepts = np.array([model.intercept(value=level) for level in levels])
    assert np.isfinite(intercepts).all() and np.all(intercepts > 0)
    # x(level)=(unknown_amplitude/level)**(1/alpha). The amplitude
    # cancels in ratios, so neither count nor density units are selected.
    expected_ratios = (levels[1:] / levels[0]) ** (1 / alpha)
    record_property(
        "fitted_curve_intercept_shape",
        {
            "alpha": float(alpha),
            "levels": levels.tolist(),
            "intercepts": intercepts.tolist(),
        },
    )
    assert_allclose(intercepts[0] / intercepts[1:], expected_ratios, rtol=1e-12, atol=1e-12)
    differences = np.diff(intercepts)
    assert np.all(differences < 0) or np.all(differences > 0)
    assert_array_equal(observations, original)


@pytest.mark.parametrize("invalid_max", ["not a numeric bound", {"maximum": 20}])
def test_power_law_generator_nonnumeric_maximum_raises_an_ordinary_input_error(invalid_max):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with pytest.raises((TypeError, ValueError)):
            fitfuncs.powerlaw_prng(2.5, xmin=2, xmax=invalid_max)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.mark.parametrize(
    "spectrum, samples, expected_order",
    [
        ([1, 1, 1, 1], 80, 0),
        ([100, 1, 1, 1], 80, 1),
        ([100, 30, 1, 1, 1], 100, 2),
        ([1000, 100, 10, 1], 80, 3),
        ([12, 3, 1, 1, 1], 10, 1),
    ],
    ids=["flat-order-zero", "one-separated", "two-separated", "last-order", "n-equals-2l"],
)
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_mdl_order_matches_independent_criterion_and_is_scale_invariant(
    spectrum, samples, expected_order, scale
):
    # L1 supplies this expression. Scalar math/fsum keeps this oracle separate
    # from the array implementation and from LPSVD's automatic-order policy.
    length = len(spectrum)
    scores = []
    for order in range(length):
        tail = spectrum[order:]
        likelihood = -samples * math.fsum(math.log(value) for value in tail)
        likelihood += samples * len(tail) * math.log(math.fsum(tail) / len(tail))
        penalty = order * (2 * length - order) * math.log(samples) / 2
        scores.append(likelihood + penalty)
    ranked = sorted(range(length), key=scores.__getitem__)
    assert ranked[0] == expected_order
    assert scores[ranked[1]] - scores[ranked[0]] > 2
    # A common scale cancels: -N*(L-k)*log(c) + N*(L-k)*log(c) = 0.
    values = np.array(spectrum, dtype=float) * scale
    original = values.copy()
    assert estimate_model_order(values, N=samples, L=length) == expected_order
    assert_array_equal(values, original)


@pytest.mark.parametrize(
    "insertion, zero_frequency, sample_count",
    [(0, 0.3125, 31), (2, -0.171875, 83), (4, 0.40625, 83)],
    ids=["first-positive", "middle-negative", "last-positive"],
)
@pytest.mark.parametrize("template_dtype", [np.float64, np.complex128], ids=["real", "complex"])
def test_zero_amplitude_term_does_not_change_analytic_real_reconstruction(
    insertion, zero_frequency, sample_count, template_dtype
):
    # Euler pairs for the two physical cosines in the approved real-signal
    # packet. Neither the fitting routine nor its output supplies the oracle.
    rows = [
        [1.2, 0.109375, -0.025, 0.45],
        [0.45, -0.234375, -0.045, -3.6],
        [0.45, 0.234375, -0.045, 3.6],
        [1.2, -0.109375, -0.025, -0.45],
    ]
    rows.insert(insertion, [0.0, zero_frequency, -0.07, 1.2])
    coefficients = pd.DataFrame(
        rows, columns=["amps", "freqs", "damps", "phase"], index=[11, 23, 37, 41, 59]
    )
    original_coefficients = coefficients.copy(deep=True)
    template = np.linspace(-5, 7, sample_count).astype(template_dtype)
    original_template = template.copy()
    expected = np.array(
        [
            2.4 * math.exp(-0.025 * n) * math.cos(math.tau * 0.109375 * n + 0.45)
            + 0.9 * math.exp(-0.045 * n) * math.cos(math.tau * 0.234375 * n + 3.6)
            for n in range(sample_count)
        ]
    )
    result = reconstruct_signal(coefficients, template, ampcutoff=0, freqcutoff=0, dampcutoff=0)
    assert result.shape == expected.shape
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert_frame_equal(coefficients, original_coefficients)
    assert_array_equal(template, original_template)


def test_continuous_power_law_interior_percentiles_preserve_support_and_order(record_property):
    observations = np.array([0.6, 1.2, 1.8, 2.4, 3.2, 4.6, 6.8, 9.5, 14.0, 23.0])
    original = observations.copy()
    cutoff = 2.0
    model = fitfuncs.PowerLaw(observations)
    model.fit(xmin=cutoff, opt_max=False)
    requests = (0.2, 0.5, 0.8)
    values = np.asarray([model.percentile(value) for value in requests]).reshape(-1)
    record_property(
        "continuous_percentile_support_order",
        {"cutoff": cutoff, "requests": requests, "values": values.tolist()},
    )
    assert_array_equal(observations, original)
    # Both fractional and percentage interpretations preserve this ordering
    # and support. No quantile formula, fitted exponent, or endpoints are fixed.
    assert values.size == len(requests)
    assert np.isfinite(values).all()
    assert np.all(values >= cutoff)
    assert np.all(np.diff(values) >= 0)


def test_power_law_cutoff_above_all_observations_cannot_claim_a_successful_fit(record_property):
    observations = np.array([1, 2, 2, 4, 7, 11, 7], dtype=np.int64)
    original = observations.copy()
    model = fitfuncs.PowerLaw(observations)
    # Every observation is strictly below 20, so either endpoint-inclusion
    # convention leaves no fitting observations. No estimator is selected.
    try:
        with pytest.raises(
            (TypeError, ValueError, RuntimeError, ArithmeticError, RuntimeWarning)
        ) as caught:
            model.fit(xmin=20, opt_max=False)
            record_property("empty_tail_fit_returned_normally", True)
    finally:
        assert_array_equal(observations, original)
    # Keep an informative rejection without prescribing its exact class or
    # wording. NameError/UnboundLocalError are not ordinary fitting validation.
    assert str(caught.value).strip()
    record_property(
        "empty_tail_fit_rejection",
        {"class": type(caught.value).__name__, "message": str(caught.value)},
    )
