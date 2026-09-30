"""Independent S1--S5 regressions from the approved normalization contract.

Coordinates use one arbitrary distance unit. Tolerances were fixed before the
first product run; see REGISTRATION-NORMALIZATION-002-test-evidence.md.
"""

import unittest

import numpy as np

from dphtools.utils.registration import (
    AffineCPD,
    RigidCPD,
    SimilarityCPD,
    TranslationCPD,
)

# Float64 roundoff allowance for these small, well-conditioned linear maps.
ALGEBRA_RTOL = 2e-12
ALGEBRA_ATOL = 2e-12
# Original-coordinate accuracy, independent of the solver's stopping statistic.
FIT_RTOL = 1e-7
FIT_ATOL = 1e-7


def _cloud(dimension):
    """A slightly irregular grid, with unequal spreads and a nonzero mean."""
    grid = np.array(list(np.ndindex(*(3,) * dimension)), dtype=np.float64) - 1
    row = np.arange(len(grid))[:, None]
    axis = np.arange(dimension)[None, :]
    jitter = (((5 * row + 3 * axis) % 7) - 3) / 50.0
    spread = np.array([1.6, 0.9, 0.6])[:dimension]
    center = np.array([1.2, -0.7, 0.9])[:dimension]
    return (grid + jitter) * spread + center


def _rotation(dimension):
    angle = np.deg2rad(3.0)
    c, s = np.cos(angle), np.sin(angle)
    if dimension == 2:
        return np.array([[c, -s], [s, c]])
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    angle = np.deg2rad(-2.0)
    c, s = np.cos(angle), np.sin(angle)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    angle = np.deg2rad(2.0)
    c, s = np.cos(angle), np.sin(angle)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return rz @ ry @ rx


def _example(model, dimension):
    moving = _cloud(dimension)
    rotation = _rotation(dimension)
    if model is SimilarityCPD:
        linear = 1.03 * rotation
    elif model is AffineCPD:
        shear = np.eye(dimension)
        shear[0, 1] = 0.015
        if dimension == 3:
            shear[1, 2] = -0.01
        linear = rotation @ shear @ np.diag([1.04, 0.97, 1.02][:dimension])
    elif model is RigidCPD:
        linear = rotation
    elif model is TranslationCPD:
        linear = np.eye(dimension)
    else:
        raise AssertionError("Unexpected test model")
    translation = np.array([0.04, -0.03, 0.02])[:dimension]
    fixed = moving @ linear.T + translation
    held_out = np.array(
        [[-0.45, -0.25, 0.7], [1.35, -1.45, 1.6], [2.4, 0.35, 0.25]],
        dtype=np.float64,
    )[:, :dimension]
    return fixed, moving, linear, translation, held_out


class RegistrationNormalizationContract(unittest.TestCase):
    def assert_algebra_close(self, actual, expected, label):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=ALGEBRA_RTOL,
            atol=ALGEBRA_ATOL,
            err_msg=label,
        )

    def assert_fit_close(self, actual, expected, label):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=FIT_RTOL,
            atol=FIT_ATOL,
            err_msg=label,
        )

    def check_normalization(self, model, dimension):
        fixed, moving, linear, translation, held_out = _example(model, dimension)
        registration = model(fixed.copy(), moving.copy())
        registration.estimate()
        self.assert_algebra_close(registration.B, linear, "estimated linear map")
        self.assert_algebra_close(
            np.asarray(registration.translation).reshape(-1),
            translation,
            "estimated translation",
        )
        expected_held_out = held_out @ linear.T + translation
        self.assert_algebra_close(
            registration.transform(held_out),
            expected_held_out,
            "estimated held-out mapping",
        )

        registration.norm_data()
        mean_x, mean_y = fixed.mean(axis=0), moving.mean(axis=0)
        # The public docs do not define whether scale_x/scale_y are multipliers
        # or divisors. Recover effective divisors from the observed coordinate
        # charts, then independently check the chart and represented mapping.
        sx = fixed.std(axis=0) / registration.X.std(axis=0)
        sy = moving.std(axis=0) / registration.Y.std(axis=0)
        self.assertTrue(np.all(np.isfinite(sx)) and np.all(sx > 0))
        self.assertTrue(np.all(np.isfinite(sy)) and np.all(sy > 0))
        self.assert_algebra_close(registration.X, (fixed - mean_x) / sx, "normalized X")
        self.assert_algebra_close(registration.Y, (moving - mean_y) / sy, "normalized Y")

        # For column-form B and row-vector points:
        # Bn = diag(1/sx) B diag(sy); tn = (B mean_y + t - mean_x) / sx.
        expected_bn = linear * sy[None, :] / sx[:, None]
        expected_tn = (linear @ mean_y + translation - mean_x) / sx
        self.assert_algebra_close(registration.B, expected_bn, "normalized linear map")
        self.assert_algebra_close(
            np.asarray(registration.translation).reshape(-1),
            expected_tn,
            "normalized translation",
        )
        self.assert_algebra_close(
            registration.transform(registration.Y),
            registration.X,
            "normalized training-point mapping",
        )
        self.assert_algebra_close(
            registration.transform((held_out - mean_y) / sy),
            (expected_held_out - mean_x) / sx,
            "normalized held-out mapping",
        )

        registration.unnorm_data()
        self.assert_algebra_close(registration.X, fixed, "restored fixed cloud")
        self.assert_algebra_close(registration.Y, moving, "restored moving cloud")
        self.assert_algebra_close(registration.B, linear, "restored linear map")
        self.assert_algebra_close(
            np.asarray(registration.translation).reshape(-1),
            translation,
            "restored translation",
        )
        self.assert_algebra_close(registration.transform(moving), fixed, "restored training map")
        self.assert_algebra_close(
            registration.transform(held_out),
            expected_held_out,
            "restored held-out mapping",
        )

    def check_registration(self, model, dimension):
        fixed, moving, linear, translation, held_out = _example(model, dimension)
        registration = model(fixed.copy(), moving.copy())
        registration(
            tol=1e-10,
            dist_tol=1e-10,
            maxiters=200,
            init_var=None,
            weight=0,
            normalization=True,
        )
        self.assert_fit_close(registration.X, fixed, "fixed cloud in original coordinates")
        self.assert_fit_close(registration.Y, moving, "moving cloud in original coordinates")
        self.assert_fit_close(registration.B, linear, "registered linear map")
        self.assert_fit_close(
            np.asarray(registration.translation).reshape(-1),
            translation,
            "registered translation",
        )
        self.assert_fit_close(registration.transform(moving), fixed, "registered training map")
        self.assert_fit_close(registration.TY, fixed, "public transformed cloud")
        self.assert_fit_close(
            registration.transform(held_out),
            held_out @ linear.T + translation,
            "registered held-out mapping",
        )
        if model is SimilarityCPD:
            self.assert_fit_close(
                registration.B.T @ registration.B,
                1.03**2 * np.eye(dimension),
                "uniform similarity scale",
            )
            self.assertGreater(np.linalg.det(registration.B), 0)
        elif model is RigidCPD:
            self.assert_algebra_close(
                registration.B.T @ registration.B,
                np.eye(dimension),
                "rigid orthogonality",
            )
            self.assert_algebra_close(np.linalg.det(registration.B), 1.0, "proper rotation")
        elif model is TranslationCPD:
            self.assert_algebra_close(
                registration.B, np.eye(dimension), "translation-only identity"
            )

    def test_s1_similarity_normalization_2d(self):
        self.check_normalization(SimilarityCPD, 2)

    def test_s1_similarity_normalization_3d(self):
        self.check_normalization(SimilarityCPD, 3)

    def test_s1_affine_normalization_2d(self):
        self.check_normalization(AffineCPD, 2)

    def test_s1_affine_normalization_3d(self):
        self.check_normalization(AffineCPD, 3)

    def test_s2_similarity_registration_2d(self):
        self.check_registration(SimilarityCPD, 2)

    def test_s2_similarity_registration_3d(self):
        self.check_registration(SimilarityCPD, 3)

    def test_s3_affine_registration_2d(self):
        self.check_registration(AffineCPD, 2)

    def test_s3_affine_registration_3d(self):
        self.check_registration(AffineCPD, 3)

    def test_s4_rigid_registration_2d(self):
        self.check_registration(RigidCPD, 2)

    def test_s4_rigid_registration_3d(self):
        self.check_registration(RigidCPD, 3)

    def test_s5_translation_registration_2d(self):
        self.check_registration(TranslationCPD, 2)

    def test_s5_translation_registration_3d(self):
        self.check_registration(TranslationCPD, 3)
