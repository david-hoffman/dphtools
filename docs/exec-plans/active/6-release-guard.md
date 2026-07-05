# Issue #6 Release Guard

## Decision

Passed for PR readiness, pending required human approval for protected release environments.

## Evidence

- Release workflow remains tag-triggered.
- Package build and installed-artifact smoke tests were added before publish jobs.
- TestPyPI, PyPI, and Anaconda jobs use protected environments.
- Publish jobs are not available to ordinary pull request workflows.
