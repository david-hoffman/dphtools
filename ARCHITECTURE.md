# ARCHITECTURE.md

Last updated: 2026-07-04

## Package Or Product Purpose

`dphtools` is a Python package for optics and image analysis.

## Public API Map

- `dphtools.__version__` exposes the Versioneer-derived package version.
- `dphtools.display` exposes plotting and image-display utilities.
- `dphtools.utils` exposes scientific and numerical utilities from modules including `fitfuncs`, `lm`, `beads`, `histstats`, `lpsvd`, `registration`, and `rolling_ball`.

The public API is not formally declared beyond importable modules and functions. Treat importable package functions as compatibility-sensitive until a narrower policy is written.

## Internal Module Map

- `dphtools/_version.py`: generated Versioneer version support. Do not edit by hand unless updating Versioneer intentionally.
- `dphtools/display.py`: Matplotlib display helpers, grid display, color-line plotting, slices, and image projections.
- `dphtools/utils/fitfuncs.py`: curve models, exponential and power-law fitting, and related statistics.
- `dphtools/utils/lm.py`: Levenberg-Marquardt-style fitting utilities and maximum-likelihood fitting support.
- `dphtools/utils/registration.py`: image registration helpers.
- `dphtools/utils/rolling_ball.py`: image background/rolling-ball utilities.
- `dphtools/utils/beads.py`, `histstats.py`, `lpsvd.py`: domain utilities for image/statistical/signal workflows.
- `notebooks/`: historical notebooks and external numerical reference code. Treat as reference material unless an issue explicitly targets it.

## Dependency Direction Rules

- Package modules may depend on NumPy, SciPy, Pandas, Matplotlib, and scikit-image, as listed in `requirements.txt`.
- Tests may use Pytest and NumPy testing helpers.
- Harness scripts must use the Python standard library unless a follow-up issue approves new dependencies.
- Product code must not depend on harness scripts.

## Data Model And File Format Conventions

No formal data model document exists yet. Current code primarily accepts NumPy arrays and array-like numeric data.

Unknowns:
- Accepted dtype policy.
- Axis-order policy.
- Binary fixture policy for future image or signal fixtures.

## Units And Coordinate-System Conventions

The repository purpose implies optics, image analysis, signal processing, and numerical fitting. Units and coordinate systems are not consistently documented yet.

Until a domain inventory is completed:
- Test names or comments must state units when units matter.
- Image tests must state shape and axis assumptions.
- FFT-centered data must call out whether data is shifted or unshifted.

## Numerical Tolerance Conventions

Existing tests use `numpy.testing.assert_allclose` and `assert_almost_equal` with explicit tolerances in some cases. No repository-wide tolerance policy existed before this harness. Use `docs/testing/numerical-tolerance-policy.md` for new work.

## Hardware Or Device Boundaries

No hardware control code, firmware interface, or device-control boundary was discovered. Scientific and image-processing code is present and should receive numerics review when changed.

## Known Fragile Areas

- Numerical optimization and fitting code in `dphtools/utils/lm.py` and `dphtools/utils/fitfuncs.py`.
- FFT, image filtering, image registration, and rolling-ball behavior.
- Versioneer-generated files and packaging metadata.
- Release workflow publishing to Test PyPI, PyPI, and Anaconda.
- Historical notebooks and bundled external C/Fortran/MATLAB reference code.

## Release Compatibility Policy

The package is marked alpha in `setup.py`. Public API changes still require a design doc, tests, and release notes because the package publishes artifacts.

Release workflow changes are high risk. They require a release issue and human/admin approval.

## Downstream Dependency Notes

No downstream projects or consumers were discovered in repository documentation. Add downstream smoke tests if consumers are later identified.
