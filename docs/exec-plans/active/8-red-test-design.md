# Issue #8 Red-Test Design

## Testable claim

The existing doctest suite is the red test. Utility examples must pass under current NumPy without changing the CI command.

## Oracle

The doctest expected output is the oracle. For runtime behavior, documented scalar returns should use Python scalar types where practical.

## Scope

- `dphtools/utils/__init__.py`
- No tolerance changes.
- No fixture changes.
