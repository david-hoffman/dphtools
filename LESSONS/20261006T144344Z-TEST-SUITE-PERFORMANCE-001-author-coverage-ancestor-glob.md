# Coverage globs can capture a virtual environment through an ancestor name

- ID: 20261006T144344Z-TEST-SUITE-PERFORMANCE-001-author-coverage-ancestor-glob
- Date: 2026-10-06T14:43:44Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / routine author
- Status: confirmed
- Observation: `*/dphtools/**/*.py` also matched unrelated dependencies inside a checkout named `dphtools`, including coverage.py itself in a repository-local virtual environment.
- Evidence: `tests/test_verification_coverage_scope.py::test_repo_named_dphtools_does_not_instrument_environment_dependencies` failed on the original configuration and passed with anchored owned-source instrumentation. It also verifies nested source, installed package, and copied CLI measurement from outside the checkout.
- Lesson: Anchor owned source roots explicitly and test the raw measured-file set; report filtering alone can conceal costly instrumentation of unrelated code.
