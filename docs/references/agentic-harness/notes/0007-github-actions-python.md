# GitHub Actions Python

Source: https://docs.github.com/en/actions/tutorials/build-and-test-code/python

Accessed: 2026-07-04

## Project-use notes

- Python workflows commonly use `actions/setup-python` with a version matrix.
- GitHub examples show operating-system and Python-version matrices.
- Test result artifacts and Pytest are supported patterns for Python projects.

## Harness impact

- CI keeps the discovered OS matrix from the existing workflow.
- CI keeps the known-good Python 3.10 runtime in Phase 0 and documents the mismatch with package metadata.
- Future work should align declared Python `>=3.8` support with measured CI runtimes.
