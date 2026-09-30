# Role A methods and evidence: registration normalization

Approved provenance: `42e893a410c7fab978da678edc63cf09bde94b3d` (supplied identifier only; no Git/history inspection). Slice: S1–S5, five approved scenarios. This is corrective A2's handoff in A/B review window round **2 of at most 2**, following B1's rejection. B2 has not reviewed the corrected checkpoint. No monetary/time cap was supplied; monetary metering is unavailable. C's initial attempt and one repair are tracked separately by the coordinator. No allowances were replenished.

On **2026-09-30**, after A1's runs and B1's review, A2 changed only the iterative call's `init_var=0.01` to the public default `init_var=None`, then formatted this test file with Black 26.5.1. This is a **post-run, post-B1 test correction**, not original pre-run design. All fixtures, transforms, tolerances, assertions, remaining solver arguments, and scenarios are unchanged. The corrected focused baseline passes all twelve tests. The prior convergence expectation is a **test defect**, not confirmed product-red; the underlying numerical cause remains unresolved.

## Expected results and scenario mapping

All tests are in `tests/test_registration_normalization_contract.py`, class `RegistrationNormalizationContract`, using the real `dphtools.utils.registration` public interface. Each suffix `{2d,3d}` below denotes two separately discovered tests, for twelve tests total.

| Scenario | Test names | Independent expected-result source |
|---|---|---|
| S1 | `test_s1_similarity_normalization_{2d,3d}`, `test_s1_affine_normalization_{2d,3d}` | Contract S1, row-vector mapping definition, coordinate-change algebra below, and round-trip identity. `estimate()` first establishes the exact correspondence map. Check normalized/restored clouds and matrices, training points, and held-out points. |
| S2 | `test_s2_similarity_registration_{2d,3d}` | Contract S2: construct `X = Y @ (1.03 R).T + t`; independent known matrix, translation, and point predictions; `B.T @ B = 1.03**2 I`, positive determinant. |
| S3 | `test_s3_affine_registration_{2d,3d}` | Contract S3: construct `B = R @ H @ diag(1.04, 0.97[, 1.02])`, with shear `H[0,1]=0.015`, and in 3D `H[1,2]=-0.01`; check known matrix, translation, and point predictions. |
| S4 | `test_s4_rigid_registration_{2d,3d}` | Contract S4: construct `B = R`; check known mapping plus `B.T @ B = I` and `det(B) = 1`. |
| S5 | `test_s5_translation_registration_{2d,3d}` | Contract S5: construct `B = I`; check identity and known translation/point predictions. |

S2–S5 each start from a fresh constructor and call the actual iterative registration entry point, with normalization enabled. They do not seed it using `estimate()`. The corresponding-point S1 tests fill a separate gap: intermediate normalization state and coordinate conversion cannot be observed by checking only final registration results.

Let the original means be `mx`, `my`, and the effective normalization divisors be `sx`, `sy`. Recover these divisors from the per-axis standard-deviation ratios of the original and normalized clouds. Then check the entire cloud conversion, including centering and its positive diagonal scaling. This permits both isotropic and per-axis normalization without assuming the undocumented multiplier/divisor convention of the public `scale_x` and `scale_y` fields. With `Xn = (X - mx) / sx` and `Yn = (Y - my) / sy`, substitution into `X = Y @ B.T + t` gives:

```text
Bn = diag(1 / sx) @ B @ diag(sy)
tn = (B @ my + t - mx) / sx
Tn((q - my) / sy) = (T(q) - mx) / sx
```

The tests independently construct `B`, `t`, and the means. The observed cloud conversion identifies the coordinate chart; independent change-of-coordinates algebra supplies the expected intermediate mapping, including held-out points. The chart is not an oracle for the represented original-coordinate mapping. No fitted result supplies the expected original-coordinate transform. S1 does not prescribe the numerical value or storage convention of internal normalization factors.

## Fixtures and tolerances fixed before product execution

A1 recorded the fixtures and tolerances below before its first product import, probe, or test run. A2 retained them without tuning to output. The initialization correction is separately dated above and recorded below. Coordinates use one arbitrary, consistent distance unit (DU). Matrices and relative tolerances are dimensionless.

- Deterministic float64 3-by-3 and 3-by-3-by-3 grids (9 and 27 points), perturbed by a fixed modular sequence of at most 0.06 grid units; axis multipliers `[1.6, 0.9, 0.6]`, centers `[1.2, -0.7, 0.9]`, truncated in 2D. This gives full rank, unequal axis spreads, nonzero centroids, and well-separated points without random sampling.
- Rotation: 3 degrees in 2D; `Rz(3 degrees) @ Ry(-2 degrees) @ Rx(2 degrees)` in 3D. Translation: `[0.04, -0.03, 0.02]` DU, truncated in 2D. Three fixed held-out points test the represented map away from training correspondences.
- These small rotations/scales keep displacement modest relative to point separation. They are local registration examples, with no outliers, missing points, or claim of arbitrary global convergence.
- Algebra checks use `atol=2e-12`, `rtol=2e-12`. For coordinates, the absolute tolerance is in DU (or normalized coordinate units during S1); for matrices it is dimensionless. Float64 epsilon is approximately `2.22e-16`. The unperturbed centered cloud condition number is at most `1.6/0.6 = 2.67`; the bounded jitter keeps the cloud modestly conditioned (a conservative bound is below 4). Transform conditioning is near unity. An allowance of roughly `1000 * 4 * epsilon = 8.9e-13`, with additional absolute allowance for centering, fits inside `2e-12`. This is generous for roundoff yet many orders below the percent-level transform changes exercised here.
- Iterative mapping checks use `atol=1e-7`, `rtol=1e-7`, so the per-coordinate bound is `1e-7 DU + 1e-7 * abs(expected)`. This allows numerical iteration error far above algebraic roundoff while remaining far below the 0.02–0.04 DU translations and percent-level scaling. The calls retain `tol=1e-10`, `dist_tol=1e-10`, `maxiters=200`, and `weight=0`. A2 uses the documented public default `init_var=None`; A1's `init_var=0.01` convergence assumption was unsupported, as B1 established. No absolute variance or new convergence semantics are prescribed. The assertions independently check actual parameter and point error; they do not infer accuracy from a stopping flag or redefine stopping/variance semantics.
- Rigid orthogonality, proper determinant, and translation identity use the algebra tolerance: these are structural model invariants, independent of iterative correspondence accuracy.

No tolerances, fixtures, or transforms were changed after observing baseline output. A1 corrected an unsupported normalization-factor storage assumption after its first run. A2 corrected the initialization after A1's runs and B1's rejection. Both corrections and the earlier results are retained below.

## Execution evidence

A2 final test SHA256, after the initialization correction and formatting:

```text
9d00f35e7ab4df29b872ff806f809252a9a6772a0af80a0ed7eb5fc269e6aefb
```

Environment: `/private/tmp/dphtools-verify-313-jmr47yhf/bin/python`, Python 3.13.12 (Anaconda, Clang 20.1.8), NumPy 2.5.3, Black 26.5.1. Working directory: `/Users/davidhoffman/.codex/worktrees/683b/dphtools`. `MPLBACKEND=Agg`; the runner sets a fresh writable `MPLCONFIGDIR=/private/tmp/dphtools-issue2/a2-xr8mfbqc/a2-mplconfig` before product import. Bytecode writing is disabled for the focused run.

The authorized formatting commands each exited **0**:

```sh
/private/tmp/dphtools-verify-313-jmr47yhf/bin/python -m black --line-length 99 tests/test_registration_normalization_contract.py > /private/tmp/dphtools-issue2/a2-xr8mfbqc/a2-black-format.log 2>&1
/private/tmp/dphtools-verify-313-jmr47yhf/bin/python -m black --check --line-length 99 tests/test_registration_normalization_contract.py > /private/tmp/dphtools-issue2/a2-xr8mfbqc/a2-black-check.log 2>&1
```

Black reformatted one file; its subsequent check reported that the same file would be left unchanged. The coordinator-reported formatting failure was **environment/tooling**, now corrected. The coordinator reported that lint already passed; A2 did not rerun lint or inspect configuration. An abstract syntax tree comparison with the retained A1 test file confirmed that changing `init_var=0.01` to `None` is the only semantic edit. The comparison exited **0**; its result and digest are retained in `a2-change-audit.txt` in the A2 temporary directory.

A2 focused command, after formatting (exit **0**):

```sh
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 /private/tmp/dphtools-verify-313-jmr47yhf/bin/python /private/tmp/dphtools-issue2/a2-xr8mfbqc/a2-source-free-runner.py > /private/tmp/dphtools-issue2/a2-xr8mfbqc/a2-focused.log 2>&1
```

The source-free unittest runner loaded only this issue-specific test file: **12 run, 12 passed, 0 failed, 0 errors, 0 skipped**, test execution time 0.013 seconds. The SHA256 matched before and after execution; no subsequent test edits occurred.

| Scenario | 2D result | 3D result |
|---|---|---|
| S1 similarity normalization | Pass | Pass |
| S1 affine normalization | Pass | Pass |
| S2 similarity registration | Pass | Pass |
| S3 affine registration | Pass | Pass |
| S4 rigid registration | Pass | Pass |
| S5 translation registration | Pass | Pass |

The log retains Matplotlib's font-cache notice and variance-floor messages (`1e-10`) from S2 3D, S3 2D/3D, and S5 2D/3D. These did not fail the tests and are not asserted as new requirements. **No approved product defect is established by the corrected focused baseline.** Passing local examples do not prove convergence for other variance choices or arbitrary transforms.

### Historical A1 baseline and diagnostics (superseded classification)

A1 final test SHA256, rejected by B1:

```text
4c7794e7e04d287dac9a27cfd4669b0b7ce8f441f38ff79ae1d571d179dba624
```

A1 environment: `/private/tmp/dphtools-verify-313-jmr47yhf/bin/python`, Python 3.13.12 (Anaconda, Clang 20.1.8), NumPy 2.5.3. Working directory: `/Users/davidhoffman/.codex/worktrees/683b/dphtools`. `MPLBACKEND=Agg` throughout. A1's final run also set a writable `MPLCONFIGDIR`; earlier runs used Matplotlib's automatic temporary-cache fallback because the default cache directory was unwritable. This was a non-blocking environment diagnostic, not a behavioral failure.

A1 final focused command (exit **1**):

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-issue2/mplconfig /private/tmp/dphtools-verify-313-jmr47yhf/bin/python /private/tmp/dphtools-issue2/run_source_free.py > /private/tmp/dphtools-issue2/final-focused.log 2>&1
```

The custom unittest runner loaded only the new test file: **12 run, 11 passed, 1 failed, 0 errors, 0 skipped**, test execution time 0.012 seconds. Redirection preserved the command's failing exit status. A1 calculated its SHA256 after this run; A2's later correction produces the new digest above.

| Scenario | 2D result | 3D result |
|---|---|---|
| S1 similarity normalization | Pass | Pass |
| S1 affine normalization | Pass | Pass |
| S2 similarity registration | Pass | Pass |
| S3 affine registration | **Fail** | Pass |
| S4 rigid registration | Pass | Pass |
| S5 translation registration | Pass | Pass |

Failure: `test_s3_affine_registration_2d`, `AssertionError` at the registered-linear-map comparison. All four matrix elements differed. Maximum absolute element error: `0.523199925` (dimensionless).

```text
Expected B = [[ 1.0385747161, -0.0362358178],
              [ 0.0544293945,  0.9694321369]]
Actual B   = [[ 1.0337965192, -0.0911386499],
              [ 0.0355419397,  1.4926320619]]
```

Corrected classification after B1: **test defect**, because A1's required convergence with `init_var=0.01` was unsupported. A1 originally classified this as a product defect against S3; that conclusion is retracted. The mismatch is an observed result, but it is not confirmed product-red under the approved contract. The underlying numerical cause remains unresolved. The test's later translation/point assertions were not reached after the matrix failure. A1's separate public API diagnostics measured those errors without changing the regression.

The diagnostic command below exited **0** after its correction. It used fresh `AffineCPD` instances, the unchanged 2D fixture/arguments, and iteration limits of 1, 2, 5, 20, and 200 with normalization enabled/disabled. These are diagnostics, not additional scenarios or new stopping-rule requirements.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-issue2/mplconfig /private/tmp/dphtools-verify-313-jmr47yhf/bin/python /private/tmp/dphtools-issue2/probe_source_free.py
```

| Normalization | Maximum iterations | Maximum absolute B error | Maximum absolute point-coordinate error (DU) | Maximum absolute translation error (DU) |
|---|---:|---:|---:|---:|
| Enabled | 20 | 0.5231999222 | 1.408224132 | 1.267776370 |
| Enabled | 200 | 0.5231999250 | 1.408224138 | 1.267776375 |
| Disabled | 200 | 2.22e-16 | 4.44e-16 | 3.47e-17 |

With `init_var=0.01`, the normalized call returned translation `[-0.0144070687, 1.2377763751]` DU instead of `[0.04, -0.03]` DU. A1's disabled-normalization control showed an association with that combination of normalization and initialization, without proving a product defect or identifying its numerical cause. B1's additional default-initialization controls below limit this conclusion. Small-iteration diagnostic calls logged their iteration limit; several otherwise passing calls logged a variance floor of `1e-10`. Neither log is hidden or asserted as a new requirement.

Fixture-only diagnostics confirmed the conditioning assumptions:

| Dimension | Points | Centered condition number | Minimum point separation (DU) | Largest constructed displacement across models (DU) |
|---|---:|---:|---:|---:|
| 2D | 9 | 1.80378593 | 0.866367128 | 0.264145117 |
| 3D | 27 | 2.68462079 | 0.580661692 | 0.268348226 |

As a fixture sanity check, each largest displacement is less than half the minimum point separation. This alone does not establish a convergence basin for `init_var=0.01`. Cloud standard deviations are `[1.30381791, 0.72328141]` DU and `[1.31523306, 0.73327636, 0.48994069]` DU; centroids are `[1.19644444, -0.704]` DU and `[1.20118519, -0.70133333, 0.90088889]` DU. Maximum transform condition number is 1.0743182. These A1 measurements verify the fixture design; they did not change its parameters or tolerances. B1 independently confirmed full rank, anisotropy, nonzero centroids, and the centered condition numbers.

### Retained attempts and corrections

Items 1–4 are A1's retained authoring attempts within round 1. Items 5–6 record B1's rejection and A2's authorized correction in round 2 of the same review window. Prior attempt files, logs, and summaries under `/private/tmp/dphtools-issue2` were left intact. A2 did not inspect those earlier temporary artifacts except the explicitly permitted B1 summary.

1. Initial test digest `24afc9f681969c70332f89882638a503c586f858f8eb883d3244eaa7d388e431`. Command: `MPLBACKEND=Agg /private/tmp/dphtools-verify-313-jmr47yhf/bin/python /private/tmp/dphtools-issue2/run_source_free.py`. Exit 1: 12 run, 7 passed, 5 failed, 0 errors/skips. Four S1 failures were **test defects**: the tests assumed that `scale_x/scale_y` store divisors, a convention not defined by the permitted docs. Those assertions do not establish product-red evidence. The fifth failure was the unchanged S3 2D mismatch. Initial test source is retained temporarily at `/private/tmp/dphtools-issue2/test_attempt_1.py`.
2. Corrected S1 to recover effective divisors from the public normalized clouds and removed assumptions about the factor/centroid fields' storage. Digest `bffe870b68ca98de9e0ec1686d3eca141b27a01361a73e8c430338520344cec5`. Same command and environment as attempt 1. Exit 1: 12 run, 11 passed, 1 failed, 0 errors/skips. All fixtures, tolerances, transforms, expected mappings, and solver arguments remained unchanged.
3. Initial diagnostic probe exited 1 with source-free `AttributeError: 'AffineCPD' object has no attribute 'B'`, before reaching registration: it tried to inspect result state immediately after construction. This was a **probe/test defect**, not product-red evidence; the public inputs do not promise result state before estimation/registration. The corrected diagnostic inspects results only after the supported call and exits 0. Its numerical results are above.
4. Removed the test module's global warning-formatter assignment; the temporary runner already configures source-free warnings before importing the module/product. This avoids a global reporting side effect during later canonical collection. A1's final digest was `4c7794e7e04d287dac9a27cfd4669b0b7ce8f441f38ff79ae1d571d179dba624`; its final command/results are retained above. The only environment adjustment was the writable Matplotlib cache path. No numerical test behavior changed in this attempt.
5. **B1 rejected round 1**, as recorded in `/private/tmp/dphtools-issue2/B1-summary.txt`. B1 reproduced 11 passes and 1 failure (exit 1), then independently kept the 2D affine fixture, tolerances, and other arguments unchanged while comparing initializations. With `init_var=None`, both normalization enabled and disabled gave maximum matrix error `2.22e-16` and maximum training-coordinate error `4.44e-16` DU. With `init_var=0.01`, B1 reproduced the enabled-normalization mismatch (`0.5232` matrix error, `1.4082` DU point error) and the passing disabled-normalization control. Both diagnostic modes exited 0. B1 classified the unsupported convergence expectation as a **test defect** and left the numerical cause unresolved; the remaining mapping, oracle, fixture, tolerance, warning, and provenance checks were satisfactory. This did not accept the rejected checkpoint or open a new review window.
6. **A2, 2026-09-30, post-run and post-B1:** before editing, verified the rejected test digest and saved exact copies of the authorized A1 test and evidence in `/private/tmp/dphtools-issue2/a2-xr8mfbqc/a2-before-test.py` and `a2-before-evidence.md` (snapshot timestamp `2026-09-30T15:13:54.625128+00:00`). Changed only `init_var=0.01` to the public default `None`, then ran the authorized Black format/check on this file only. The syntax-tree comparison confirmed no other behavioral edit. The corrected focused suite passed 12/12, exit 0, with identical before/after digest `9d00f35e7ab4df29b872ff806f809252a9a6772a0af80a0ed7eb5fc269e6aefb`. Distinct A2 runner, formatting logs, focused log, and change audit are retained in that directory. No additional probe or authoring attempt was run by A2.

Only the issue-specific tests were run by A2. Canonical verification, existing tests, coverage configuration, and full coverage measurement were not performed; they belong to the coordinator. The tests cover the five approved local scenarios, not invalid inputs, arbitrary global convergence, noise/outliers, variance policy, or new stopping semantics. No product edits, commits, or pushes occurred. B2's independent review in a separate fresh root session is the next handoff; A2 stops here.

## Source independence

A2 read only: `AGENTS.md`; `.agents/skills/design-tests/SKILL.md`; `docs/tasks/REGISTRATION-NORMALIZATION-002-contract.md`; lines **1269–1505 only** of `docs/tasks/SETUP-001-public-api.md`; the authorized A1 artifacts `tests/test_registration_normalization_contract.py` and `docs/tasks/REGISTRATION-NORMALIZATION-002-test-evidence.md`; the independent B1 report `/private/tmp/dphtools-issue2/B1-summary.txt`; and A2's own corrected artifacts, temporary helper/snapshots, and source-free focused results. The owner-supplied role packet defines the environment and boundaries. No product source, history, unrelated existing test source, lesson entries, project/task execution records, coordinator logs, other skills/specifications, or external references were read. No delegation or memory was used. **No accidental source exposure occurred.** A1's original evidence also reported no accidental source exposure.

Before A2's product import, its temporary runner replaced `warnings.formatwarning` with a formatter that omits source text and installed a source-free exception hook and logging-exception formatter. The focused runner overrides `unittest.TextTestResult._exc_info_to_string` to print exception type/message, cause/context chains, notes, and traceback filenames/functions/line numbers by walking traceback frame metadata directly. It does not request source lines. Numerical assertion diagnostics and failing exit status are preserved. No existing tests or repository pytest configuration were collected. No reporting hooks were added to the repository test module. A1's evidence records equivalent source-free warning/traceback precautions for its earlier executions.

Source independence is a prompt-enforced practice, not an engineered isolation guarantee. No accidental product/test source exposure occurred. The baseline identifier remains supplied provenance; A did not inspect Git or independently verify the worktree revision.
