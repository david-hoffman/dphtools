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


@pytest.mark.parametrize("normalization", [False, True])
def test_rigid_registration_in_three_dimensions_preserves_unseen_distances(normalization):
    moving = np.column_stack((POINTS, [0.0, 1.0, -2.0, 3.0, 1.5]))
    angle = 0.08
    matrix = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    translation = np.array([0.2, -0.3, 0.1])
    reg = registration.RigidCPD(moving @ matrix.T + translation, moving.copy())
    reg(maxiters=100, dist_tol=1e-8, normalization=normalization)
    unseen = np.array([[2.5, 1.5, -0.5], [-0.5, 2.0, 4.0], [1.0, -2.0, 0.25]])
    transformed = reg.transform(unseen)
    assert_allclose(transformed, unseen @ matrix.T + translation, atol=1e-5)
    assert_allclose(
        np.linalg.norm(transformed[:, None] - transformed[None, :], axis=-1),
        np.linalg.norm(unseen[:, None] - unseen[None, :], axis=-1),
        atol=1e-5,
    )


@pytest.mark.parametrize("only2d", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_align_dataframes_recovers_displacement_and_transforms_new_points(only2d, diagnostics):
    import matplotlib.pyplot as plt
    import pandas as pd

    coordinates = np.column_stack((POINTS, [0.0, 1.0, -2.0, 3.0, 1.5]))
    displacement = np.array([0.2, -0.3, 0.1])
    fixed = pd.DataFrame(coordinates, columns=["x0", "y0", "z0"])
    moving = pd.DataFrame(coordinates + displacement, columns=fixed.columns)
    try:
        reg = registration.align(
            fixed, moving, model="translation", only2d=only2d, diagnostics=diagnostics, iters=20
        )
        dimension = 2 if only2d else 3
        unseen = np.array([[2.5, 1.5, -0.5], [-0.5, 2.0, 4.0]])[:, :dimension]
        assert_allclose(reg.transform(unseen + displacement[:dimension]), unseen, atol=1e-8)
        if diagnostics:
            assert plt.get_fignums()
            for number in plt.get_fignums():
                plt.figure(number).canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize("copy", [False, True])
def test_apply_transform_to_slab_preserves_metadata_and_copy_contract(copy):
    import pandas as pd

    coordinates = np.column_stack((POINTS, [0.0, 1.0, -2.0, 3.0, 1.5]))
    slab = pd.DataFrame(coordinates, columns=["x0", "y0", "z0"], index=[2, 4, 7, 9, 12])
    slab["label"] = list("abcde")
    original = slab.copy(deep=True)
    matrix = np.diag([1.0, 2.0, 3.0])
    translation = np.array([0.2, -0.3, 0.1])
    result = registration.apply_transform_to_slab(slab, matrix, translation, copy=copy)
    assert_allclose(result[["x0", "y0", "z0"]], coordinates * [1, 2, 3] + translation)
    assert_array_equal(result.index, original.index)
    assert_array_equal(result["label"], original["label"])
    if copy:
        pd.testing.assert_frame_equal(slab, original)
    else:
        pd.testing.assert_frame_equal(slab, result)


@pytest.mark.parametrize("model", ["translation", registration.TranslationCPD])
@pytest.mark.parametrize("limits", [0.1, (0.05, 0.1)])
def test_auto_weight_returns_a_registration_that_maps_exact_clouds(model, limits):
    displacement = np.array([0.2, -0.3])
    reg = registration.auto_weight(
        POINTS + displacement,
        POINTS.copy(),
        model,
        resolution=0.05,
        limits=limits,
        maxiters=40,
        dist_tol=1e-7,
    )
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0]])
    assert_allclose(reg.transform(unseen), unseen + displacement, atol=1e-5)
    # Every candidate can be exact here; no tie-breaking or weight is prescribed.


def test_nearest_neighbors_uses_custom_coordinates_and_transform_with_real_registration():
    import pandas as pd

    displacement = np.array([20.0, -30.0])
    reg = registration.TranslationCPD(POINTS.copy(), POINTS + displacement)
    reg.estimate()
    fixed = pd.DataFrame(POINTS, columns=["u", "v"], index=[10, 20, 30, 40, 50])
    moving = pd.DataFrame((POINTS + displacement)[[3, 0, 4, 1, 2]], columns=["u", "v"])
    fixed_matches, moving_matches = registration.nearest_neighbors(
        fixed, moving, r=1e-6, transform=reg.transform, coords=["u", "v"]
    )
    assert len(fixed_matches) == len(moving_matches) == len(POINTS)
    assert_allclose(reg.transform(moving_matches[["u", "v"]]), fixed_matches[["u", "v"]])


def test_registered_matches_preserve_correspondence_after_input_permutation():
    permutation = [3, 0, 4, 1, 2]
    moving = POINTS[permutation] + [0.2, -0.3]
    reg = registration.TranslationCPD(POINTS.copy(), moving)
    reg(maxiters=100, dist_tol=1e-8)
    fixed_indices, moving_indices = reg.matches
    assert set(zip(fixed_indices, moving_indices)) == set(zip(permutation, range(len(POINTS))))
    assert_allclose(reg.transform(moving[moving_indices]), POINTS[fixed_indices], atol=1e-6)


@pytest.mark.parametrize("initial_translation", [None, [2.0, -1.0]])
def test_propagated_commuting_translations_have_the_expected_final_mapping(initial_translation):
    displacements = np.array([[0.2, -0.3], [-0.1, 0.4], [0.5, 0.2]])
    registrations = []
    for displacement in displacements:
        reg = registration.TranslationCPD(POINTS + displacement, POINTS.copy())
        reg.estimate()
        registrations.append(reg)
    expected_translation = np.array([0.6, 0.3])
    options = {}
    if initial_translation is not None:
        initial_translation = np.asarray(initial_translation)
        options["initial"] = registration.to_augmented(np.eye(2), initial_translation)
        expected_translation = expected_translation + initial_translation
    matrices, translations = registration.propogate_transforms(registrations, **options)
    # Inspect only the endpoint. No intermediate list length or inclusion of an
    # initial pose is prescribed; all participating transformations commute.
    matrix, translation = np.asarray(matrices[-1]), np.asarray(translations[-1]).ravel()
    assert_allclose(matrix, np.eye(2), atol=1e-12)
    assert_allclose(translation, expected_translation, atol=1e-12)
    unseen = np.array([[2.5, 1.5], [-0.5, 2.0]])
    assert_allclose(unseen @ matrix + translation, unseen + expected_translation, atol=1e-12)


def test_drift_corrected_custom_tracks_register_and_match_in_three_dimensions():
    import pandas as pd

    from dphtools.utils import beads

    coordinates = ["u", "v", "w"]
    points = np.column_stack((POINTS, [0.0, 1.0, -2.0, 3.0, 1.5]))
    angle = 0.08
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    stationary = points @ rotation.T + [0.2, -0.3, 0.1]
    frames = pd.Index([10, 20, 30, 40, 50], name="frame")
    drift = np.arange(-2.0, 3.0)[:, None] * np.array([0.5, -0.25, 0.75])
    tracks = []
    for index, position in enumerate(stationary):
        # Both schedules have zero mean drift, making offset removal exact.
        rows = np.array([0, 2, 4]) if index % 2 else np.arange(5)
        track = pd.DataFrame(position + drift[rows], columns=coordinates)
        track["frame"] = frames[rows]
        track["amp"] = float(index + 1)
        tracks.append(track)
    measured = beads.calc_drift(
        tracks, coords=coordinates, frame_name="frame", frames_index=frames, weighted="amp"
    )
    assert_array_equal(measured.index, frames)
    assert_allclose(measured[coordinates], drift, atol=1e-12)
    observed_last_frame = stationary + drift[-1]
    corrected = observed_last_frame - measured.loc[50, coordinates].to_numpy()
    reg = registration.RigidCPD(points.copy(), corrected)
    reg(maxiters=100, dist_tol=1e-8, normalization=True)
    assert_allclose(reg.transform(corrected), points, atol=1e-5)
    fixed = pd.DataFrame(points, columns=coordinates)
    moving = pd.DataFrame(corrected[[3, 0, 4, 1, 2]], columns=coordinates)
    fixed_matches, moving_matches = registration.nearest_neighbors(
        fixed, moving, coords=coordinates, transform=reg.transform, r=1e-5
    )
    assert len(fixed_matches) == len(moving_matches) == len(points)
    assert_allclose(
        reg.transform(moving_matches[coordinates]), fixed_matches[coordinates], atol=1e-5
    )
