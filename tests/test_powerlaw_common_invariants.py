"""Common public support/state invariants, without generator/bootstrap policy."""

import random

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dphtools.utils.fitfuncs import PowerLaw


@pytest.fixture(autouse=True)
def isolate_random_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(7821)
        np.random.seed(7821)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.fixture
def fitted_discrete_model():
    # Existing public-test observations, repeated to provide 96 tail values.
    # Repetition supplies a moderate sample size, not a calibration claim.
    base = [9, 1, 4, 2, 4, 23, 5, 2, 7, 4, 12, 1, 9, 5, 17, 4, 2]
    observations = np.tile(np.array(base, dtype=np.int64), 8)
    original = observations.copy()
    model = PowerLaw(observations)
    # No observation is 3. Neither automatic cutoff selection nor endpoint
    # inclusion distinguishes the independently expected retained multiset.
    model.fit(xmin=3, opt_max=False)
    return observations, original, model


def test_generated_samples_use_integer_tail_and_preserve_fit(
    fitted_discrete_model, record_property
):
    observations, original, model = fitted_discrete_model
    fitted_parameters = np.array([model.xmin, model.C, model.alpha], copy=True)
    samples = np.asarray(model.gen_power_law())
    record_property("generated_sample_count", int(samples.size))
    # State comparisons test preservation, not the numerical fitted solution.
    assert_array_equal(observations, original)
    assert_array_equal([model.xmin, model.C, model.alpha], fitted_parameters)
    assert_array_equal(np.sort(model.clipped_data), np.sort(original[original > 3]))
    assert np.isrealobj(samples)
    assert np.isfinite(samples).all()
    assert_array_equal(samples, np.floor(samples))
    assert np.all(samples >= 3)
    # No sample-count, array-shape, upper-bound, stream or distribution rule.


def test_bootstrap_p_value_is_a_probability_and_original_fit_remains_usable(
    fitted_discrete_model, record_property
):
    observations, original, model = fitted_discrete_model
    p_value = model.calculate_p(num=5)
    record_property("p_value_shape", np.shape(p_value))
    assert np.ndim(p_value) == 0
    assert np.isrealobj(p_value)
    assert np.isfinite(p_value)
    assert 0 <= p_value <= 1
    record_property("p_value", float(p_value))
    assert_array_equal(observations, original)
    assert model.xmin == 3
    assert_array_equal(np.sort(model.clipped_data), np.sort(original[original > 3]))
    # A real later explicit fit verifies usability after the bootstrap call.
    # There are no observations at 6 either; no fitted-score oracle is selected.
    model.fit(xmin=6, opt_max=False)
    assert_array_equal(np.sort(model.clipped_data), np.sort(original[original > 6]))
    assert_array_equal(observations, original)
