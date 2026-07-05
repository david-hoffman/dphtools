# Repository intake

Date: 2026-07-04
Base commit: `c81ffdf108c33e3571757fd0996d685c47bd7d6a`
Default branch: `main`
Current branch: `codex/agentic-harness`

## Facts

- Repository name: `david-hoffman/dphtools`.
- Remote URL: `https://github.com/david-hoffman/dphtools.git`.
- Default branch: `main`, discovered from `refs/remotes/origin/HEAD`.
- Primary language: Python.
- Secondary languages and formats: YAML, Markdown, MATLAB, Fortran, C, Jupyter Notebook.
- Package/import name: `dphtools`.
- Package layout: root package under `dphtools/`; tests under `tests/`; historical notebooks and external numerical code under `notebooks/`.
- Existing test runner: Pytest, with some `unittest.TestCase` tests.
- Existing CI workflows: `.github/workflows/ci.yml` and `.github/workflows/make_release.yml`.
- Existing release workflow: tag pushes matching `*.*.*` trigger package build and publishing to Test PyPI, PyPI, and Anaconda through repository secrets.
- Package metadata: `setup.py` plus Versioneer configuration in `setup.cfg`.
- Supported runtime versions in metadata: Python `>=3.8`.
- Existing CI runtime before Phase 0: Python 3.10 only.
- Existing CI operating systems before Phase 0: Ubuntu, macOS, and Windows.
- Runtime dependencies: NumPy, Pandas, SciPy, Matplotlib, and scikit-image from `requirements.txt`.
- Conda environment file: `environment.yml`.
- Existing linters/formatters/docs tools in CI: flake8, Black, and pydocstyle.
- Coverage configuration exists in `setup.cfg`; no committed coverage baseline was found.
- Scientific/numerical/image-processing areas are present in `dphtools/utils/` and `dphtools/display.py`.
- Historical/generated/reference files are present in `notebooks/`, including external C, Fortran, MATLAB, and text files.
- No downstream projects or consumers were documented in the repository.

## Assumptions

- The GitHub owner from the remote URL, `@david-hoffman`, is the correct initial CODEOWNER.
- Python 3.10 remains the known-good CI runtime because the existing workflow tested only Python 3.10.
- The existing OS matrix should be preserved in Phase 0 rather than narrowed.
- `docs/harness/agentic_harness_spec.md` is the local implementation brief for this harness.

## Open questions

- Which Python versions from `>=3.8` still install and pass tests on supported operating systems?
- What public API compatibility policy should the package enforce while it is marked alpha?
- Which functions require strict units, axis order, dtype, and tolerance documentation?
- Are there downstream packages, notebooks, or users that should be added to downstream smoke testing?
- Should tag-based publishing move behind protected environments and trusted publishing?

## Package and public API map

- `dphtools/__init__.py`: exposes `__version__`.
- `dphtools/display.py`: plotting, grid display, image slicing, maximum-intensity projections, color lines, and related Matplotlib helpers.
- `dphtools/utils/__init__.py`: utility exports.
- `dphtools/utils/fitfuncs.py`: exponential and power-law models and fitting helpers.
- `dphtools/utils/lm.py`: Levenberg-Marquardt-style curve fitting and likelihood helpers.
- `dphtools/utils/registration.py`: registration helpers.
- `dphtools/utils/rolling_ball.py`: rolling-ball/background utilities.
- `dphtools/utils/beads.py`, `histstats.py`, `lpsvd.py`: domain-specific numerical utilities.

## Existing tests

- Test directory: `tests/`.
- Test files: `tests/test_utils.py`, `tests/test_display.py`, `tests/test_fitfuncs.py`.
- Test style: Pytest plus `unittest.TestCase`.
- Known test features: deterministic random generator in `tests/test_utils.py`; `tests/test_fitfuncs.py` uses unseeded `np.random.randn` in setup but the noisy data is not currently asserted.
- Verification on 2026-07-04: `python -m pytest -q tests` under local Python 3.13 and NumPy 2.5.1 produced 23 passed, 12 failed, and 3 warnings. All failures were existing `split_img` tests failing on `np.product`, which NumPy 2.x removed.

## Existing CI and release workflows

- `.github/workflows/ci.yml`: previously ran formatting and tests on pull requests and filtered pushes to `main` by path.
- `.github/workflows/make_release.yml`: publishes package artifacts on version tag pushes.
- Phase 0 changes replace CI with an unfiltered required aggregate `ci-required` job and preserve release workflow semantics.

## Dependency and runtime support

- Runtime dependencies are listed in `requirements.txt`.
- Conda dependencies are listed in `environment.yml`.
- Before Phase 0 there was no dev dependency file or optional dev extra.
- Phase 0 adds `requirements-dev.txt` for existing development tools and package-check tooling without changing package metadata.

## Hardware/scientific/numerical areas

- Hardware control: not discovered.
- Firmware interfaces: not discovered.
- Scientific algorithms: discovered.
- Numerical optimization/fitting: discovered in `dphtools/utils/lm.py` and `dphtools/utils/fitfuncs.py`.
- Image processing and FFT behavior: discovered in `dphtools/display.py` and `dphtools/utils/`.
- Device-control safety limits: not applicable from discovered files.

## Fixtures and generated data

- No binary test fixtures were discovered under `tests/`.
- Historical/reference data and source files exist under `notebooks/`.
- Versioneer-generated file: `dphtools/_version.py`.

## Downstream or integration surface

- No downstream repository, integration test target, or consumer list was documented.
- The release workflow indicates published package consumers may exist, but none are named.

## Files requiring human/admin review

- `.github/`
- `.github/workflows/make_release.yml`
- `.github/workflows/ci.yml`
- `.github/dependabot.yml`
- `.github/CODEOWNERS`
- `AGENTS.md`
- `CLAUDE.md`
- `ARCHITECTURE.md`
- `QUALITY_SCORE.md`
- `scripts/agent_harness/`
- `setup.py`
- `setup.cfg`
- `requirements*.txt`
- `environment.yml`
- `conda.recipe/`
- `versioneer.py`
- `dphtools/_version.py`

## Phase 0 implementation notes

- Product package source under `dphtools/` is not changed.
- Release workflow publishing behavior is not changed.
- Claude Code project artifacts are not added because their current schema was not validated in this environment.
- Branch protection, rulesets, security settings, secrets, and environments require manual GitHub configuration.
- Current tests are not green under the local Python 3.13 and NumPy 2.5.1 environment. The compatibility failure is documented and left for a follow-up because Phase 0 must not change product behavior.
