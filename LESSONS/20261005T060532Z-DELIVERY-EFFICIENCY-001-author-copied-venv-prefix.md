# Verify copied child environments on the actual host

- ID: 20261005T060532Z-DELIVERY-EFFICIENCY-001-author-copied-venv-prefix
- Date: 2026-10-05T06:05:32Z
- Task/role: DELIVERY-EFFICIENCY-001 / author
- Status: confirmed
- Observation: A seeded virtual environment made from this host's standalone Python 3.10.21 passed imports and pip check, but venv.EnvBuilder copied its launcher into a child that could not locate encodings or its standard library. Symlink-based command-line venv creation had masked the defect.
- Evidence: test_installer_observer_uses_a_real_clean_environment_and_package_metadata failed in the initial environment and passed after selecting an isolated complete interpreter prefix. The new verification preflight exercises a real copied child and offline wheel installation rather than inferring success from imports.
- Lesson: Select a locked environment appropriate to the host and run preflight before expensive tests. A locally copied complete Python distribution can repair this environment defect without changing system Python, dependencies, or real installation tests.
