# Registration normalization regression contract

**Version 1.0** Behavioral scope of the owner's request to solve issue #2 from PR #13.

Public references: [issue #2](https://github.com/david-hoffman/dphtools/issues/2)
and the registration signatures/docstrings in `SETUP-001-public-api.md`.
Baseline: `42e893a410c7fab978da678edc63cf09bde94b3d`.

Coordinates are finite float64 row vectors in one arbitrary, consistent distance
unit. The transformation maps moving Y to fixed X; in matrix notation
`T(Y) = Y @ B.T + translation`. Similarity means a positive uniform scale and
rotation; rigid means rotation; affine also permits unequal axis scales and shear.
Normalization is a change of coordinates, not a change to the represented mapping.
Expected answers follow from these definitions and algebraically generated point
correspondences, not from candidate output.

Use deterministic, full-rank, nondegenerate 2D/3D synthetic clouds with unequal
axis spreads and nonzero centroids. Include rotations that mix axes and a
nonidentity affine scale. The iterative registration examples should be modest,
unambiguous transforms within a local solver's basin, not a new promise of global
convergence for arbitrary rotations or outliers. Choose and justify numerical
tolerances independently from the implementation.

| Scenario | Required observable result | Expected-result source | Test mapping |
|---|---|---|---|
| S1 | Normalizing and undoing normalization preserve the mapping and point clouds for similarity and affine transforms, including held-out points. The intermediate normalized mapping is consistent with the coordinate change. | Change-of-coordinates identity and round-trip invariant | `test_s1_similarity_normalization_{2d,3d}`, `test_s1_affine_normalization_{2d,3d}` |
| S2 | Similarity registration on a rotated, uniformly scaled, translated anisotropic cloud returns the known mapping in original coordinates. | Constructed exact correspondences and similarity definition | `test_s2_similarity_registration_{2d,3d}` |
| S3 | Affine registration with rotation, unequal axis scaling, and translation returns the known mapping in original coordinates. | Constructed exact correspondences and affine definition | `test_s3_affine_registration_{2d,3d}` |
| S4 | Rigid registration preserves its rotation-only linear part and returns the known mapping on an anisotropic cloud. | Rigid-transform definition and exact correspondences | `test_s4_rigid_registration_{2d,3d}` |
| S5 | Translation registration preserves its identity linear part and returns the known translation on an anisotropic cloud. | Translation definition and exact correspondences | `test_s5_translation_registration_{2d,3d}` |

All mapped methods belong to `RegistrationNormalizationContract` in
`tests/test_registration_normalization_contract.py`; suffixes denote separate
2D and 3D variants. A supplied the mapping; B reviews its coverage and oracles.

Public interfaces: constructors `SimilarityCPD(X, Y)`, `AffineCPD(X, Y)`,
`RigidCPD(X, Y)`, `TranslationCPD(X, Y)`; `estimate()` for corresponding points;
`transform(other)`; `norm_data()` / `unnorm_data()`; and
`__call__(tol=1e-6, dist_tol=0, maxiters=1000, init_var=None, weight=0,
normalization=True)` for coherent point drift registration. Public result state
includes `X`, `Y`, `B`, `translation`, `TY`, `scale_x`, `scale_y`, `tx`, and `ty`.
Use `estimate()` to establish a transform before directly exercising normalization.
Test the real registration entry point as well as the normalization invariant.

Non-goals: new models, new invalid-input policy, changes to stopping rules,
variance/weight semantics, global convergence, dependencies, or public signatures.
Do not require a product edit if the approved baseline already satisfies the
behavior. Initially passing regression tests are valid evidence.

Checks: focused regressions, existing registration tests, and the repository's
unchanged canonical full verification with exact 100% measured owned runtime
statement and branch coverage. Default allowances: two A/B review rounds, initial
C plus one repair. No explicit monetary/time cap was supplied; metering is
unavailable. Keep work bounded to this issue and the required checks.
