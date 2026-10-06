# Nested environments inherit the base interpreter's packages

- ID: 20261006T152542Z-TEST-SUITE-PERFORMANCE-001-author-nested-environment-tools
- Date: 2026-10-06T15:25:42Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: Locked tools installed only in an outer virtual environment were unavailable to real nested environments created with `system_site_packages=True`.
- Evidence: Candidate `45373d7` failed real-venv reuse tests with missing Black/Flake8/pydocstyle. Installing the unchanged hashed lock directly in the private standalone base interpreter restored both `test_real_venv_directory_aliases_allow_unchanged_command_reuse` and `test_real_venv_reuses_exact_windows_hook_and_rechecks_changed_bytes`; the five focused correction cases passed in `/tmp/dphtools-performance-corrections.log`.
- Lesson: Mirror CI's directly provisioned base interpreter for these full-suite fixtures; an outer virtual environment alone does not establish their dependency boundary.
