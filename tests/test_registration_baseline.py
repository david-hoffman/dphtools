"""SETUP-001 L2/L5: exact synthetic point-cloud registration through public APIs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools.utils import registration

POINTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-1.0, 3.0]])


@pytest.mark.parametrize(
    "model",
    [
        registration.TranslationCPD,
        registration.RigidCPD,
        registration.SimilarityCPD,
        registration.AffineCPD,
    ],
)
@pytest.mark.parametrize("normalization", [False, True])
def test_registration_recovers_translation_and_transforms_unseen_points(model, normalization):
    displacement = np.array([0.2, -0.3])
    moving = POINTS + displacement
    reg = model(POINTS.copy(), moving)
    reg(maxiters=100, dist_tol=1e-8, normalization=normalization)
    assert_allclose(reg.transform(moving), POINTS, atol=1e-5)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0]])
    assert_allclose(reg.transform(unseen + displacement), unseen, atol=1e-5)
    assert_allclose(reg.rmse, 0, atol=1e-5)


@pytest.mark.parametrize(
    "model, matrix",
    [
        (registration.RigidCPD, [[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]]),
        (registration.SimilarityCPD, [[1.1, 0.0], [0.0, 1.1]]),
        (registration.AffineCPD, [[1.1, 0.1], [-0.05, 0.9]]),
    ],
)
@pytest.mark.parametrize("normalization", [False, True])
def test_registration_models_recover_their_documented_transformation(model, matrix, normalization):
    matrix = np.array(matrix)
    fixed = POINTS @ matrix.T + [0.2, -0.3]
    reg = model(fixed, POINTS.copy())
    reg(maxiters=100, dist_tol=1e-8, normalization=normalization)
    assert_allclose(reg.transform(POINTS), fixed, atol=1e-5)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0]])
    assert_allclose(reg.transform(unseen), unseen @ matrix.T + [0.2, -0.3], atol=1e-5)


def test_translation_registration_in_three_dimensions():
    fixed = np.column_stack((POINTS, [0.0, 1.0, -2.0, 3.0, 1.5]))
    moving = fixed + [0.2, -0.3, 0.1]
    reg = registration.TranslationCPD(fixed, moving)
    reg(maxiters=100, dist_tol=1e-8)
    assert_allclose(reg.transform(moving), fixed, atol=1e-5)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Translation", registration.TranslationCPD),
        ("translation", registration.TranslationCPD),
        ("Rigid", registration.RigidCPD),
        ("Euclidean", registration.RigidCPD),
        ("Similarity", registration.SimilarityCPD),
        ("Affine", registration.AffineCPD),
    ],
)
def test_choose_documented_models(name, expected):
    assert registration.choose_model(name) is expected


def test_choose_model_accepts_model_class():
    assert registration.choose_model(registration.AffineCPD) is registration.AffineCPD


def test_nearest_point_matches_return_corresponding_indices():
    fixed = np.array([[0.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
    moving = np.array([[4.1, 0.0], [0.1, 0.0], [20.0, 0.0]])
    fixed_indices, moving_indices = registration.closest_point_matches(fixed, moving, r=0.5)
    assert set(zip(fixed_indices, moving_indices)) == {(0, 1), (1, 0)}


def test_nearest_point_matches_no_neighbors_within_radius():
    fixed_indices, moving_indices = registration.closest_point_matches(
        np.array([[0.0, 0.0]]), np.array([[2.0, 3.0]]), r=0.1
    )
    assert len(fixed_indices) == len(moving_indices) == 0


@pytest.mark.parametrize(
    "matrix, translation",
    [
        ([[2.0, 1.0], [-0.5, 3.0]], [4.0, -2.0]),
        ([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]], [1.0, -2.0, 4.0]),
    ],
)
def test_augmented_transform_round_trip_preserves_all_coefficients(matrix, translation):
    matrix = np.array(matrix)
    translation = np.array(translation)
    augmented = registration.to_augmented(matrix, translation)
    assert augmented.shape == (len(translation) + 1,) * 2
    recovered_matrix, recovered_translation = registration.from_augmented(augmented)
    assert_array_equal(recovered_matrix, matrix)
    assert_array_equal(np.asarray(recovered_translation).ravel(), translation)


@pytest.mark.parametrize(
    "model, matrix",
    [
        (registration.TranslationCPD, np.eye(2)),
        (registration.RigidCPD, [[0, -1], [1, 0]]),
        (registration.SimilarityCPD, [[0, -2], [2, 0]]),
        (registration.AffineCPD, [[2, 1], [-0.5, 3]]),
    ],
)
def test_estimate_recovers_transform_from_known_corresponding_pairs(model, matrix):
    matrix = np.array(matrix)
    fixed = POINTS @ matrix.T + [2, -3]
    reg = model(fixed, POINTS.copy())
    reg.estimate()
    assert_allclose(reg.transform(POINTS), fixed, atol=1e-10)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0]])
    assert_allclose(reg.transform(unseen), unseen @ matrix.T + [2, -3], atol=1e-10)


def test_nearest_neighbors_pairs_dataframe_coordinates():
    import pandas as pd

    fixed = pd.DataFrame({"x0": [0.0, 4.0, 10.0], "y0": [0.0, 0.0, 0.0]})
    moving = pd.DataFrame({"x0": [4.1, 0.1, 20.0], "y0": [0.0, 0.0, 0.0]})
    fixed_matches, moving_matches = registration.nearest_neighbors(fixed, moving, r=0.5)
    assert set(zip(fixed_matches["x0"], moving_matches["x0"])) == {(0.0, 0.1), (4.0, 4.1)}
    assert_allclose(moving_matches["x0"].to_numpy() - fixed_matches["x0"].to_numpy(), 0.1)


@pytest.mark.parametrize("only2d", [False, True])
@pytest.mark.parametrize("dimension", [2, 3])
def test_registration_plot_renders_fitted_point_clouds(only2d, dimension):
    import matplotlib.pyplot as plt

    fixed = POINTS.copy() if dimension == 2 else np.column_stack((POINTS, [0, 1, -2, 3, 1.5]))
    moving = fixed + 0.2
    reg = registration.TranslationCPD(fixed, moving)
    reg(maxiters=100, dist_tol=1e-8)
    assert_allclose(reg.transform(moving), fixed, atol=1e-5)
    try:
        reg.plot(only2d=only2d)
        figure = plt.gcf()
        assert figure.axes
        assert any(axis.collections for axis in figure.axes)
        figure.canvas.draw()
    finally:
        plt.close("all")
