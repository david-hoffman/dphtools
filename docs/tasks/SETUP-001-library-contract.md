# SETUP-001: Compatibility and baseline tests

**Version 1.0** Supplemental test packet for existing library behavior.

## Approval and scope

On 2026-09-26, after the existing suite failed with current dependencies, the owner explicitly approved “compatibility maintenance and baseline tests.” Preserve public APIs and numerical intent; no new features. This extends the approved setup task. Unlimited budget applies; material scientific ambiguities remain blockers for owner intake.

Use [public signatures and docstrings](SETUP-001-public-api.md) as documentary evidence of the existing interface, together with existing authorized tests. No implementation bodies or history are provided. Assertions need independently justified outcomes, not the result of the function being tested. Empty or unclear docstrings are not permission to invent a numerical contract. Report those gaps concisely and continue independently specified behavior.

## Requirements

- L1: Existing array/image utility behavior documented by public functions and existing tests remains usable with the locked scientific dependencies. Cover useful numeric success, relevant invalid input, shape/size boundaries, and observable outputs through public calls.
- L2: Existing fitting, histogram statistics, drift/registration, signal-analysis, filtering, and visualization interfaces satisfy their documented mathematical or visual outputs. Use small deterministic synthetic examples with independently known answers. Do not infer unsupported algorithms or make broad distributional claims from random samples.
- L3: Compatibility maintenance preserves numerical intent and public signatures. NumPy scalar representation changes must not make correct values fail a behavioral test. Newly discovered wrong numerical behavior is not automatically an authorized algorithm redesign: report its contract and evidence for intake.
- L4: Keep 100% measured statement and branch coverage across owned runtime code, including unimported modules and handwritten error/demo paths. Narrow exclusions are only for generated/vendor/nonexecutable files. Do not add exclusions, skip/xfail tests, or change discovery/coverage settings. The generated Versioneer `_version.py` is the only current generated-runtime omission. Exact metrics and remaining gaps are required; coverage is not proof of mathematical correctness.
- L5: Test real public calls with owned components together. Small focused tests are useful for a genuine difficult-error or combinatorial gap. Do not mock out the algorithm whose behavior is asserted. Use controlled external boundaries, deterministic data, Matplotlib `Agg`, and bounded iteration sizes.

## Test conventions

Use `.venv/bin/python` (Python 3.13 with the universal hashed lock) locally. `.venv-delivery/bin/python` is the Python 3.10 tooling environment; its SciPy wheel cannot load on this macOS host, so that environment cannot supply local product red evidence. The remote matrix retains Python 3.10 on supported hosted runners.

You may read `tests/test_utils.py`, `tests/test_fitfuncs.py`, `tests/test_display.py`, and `tests/__init__.py`. Add cohesive tests named `tests/test_*_baseline.py`; do not edit original tests, docstrings, product code, delivery helper/tests, or infrastructure. Prefer numeric equality over brittle scalar repr. Run with `--tb=no` while blind, so failure tracebacks do not expose implementation source. You may run the library as a black box; do not inspect product source, history, other sessions, full setup notes, or lessons. Do not read coverage source listings or add tests solely to reproduce internal constants.

Each new assertion should map to documentary intent or independently known mathematics. Return a concise requirement mapping, actual pass/fail results, and unresolved contracts. These are supplemental baseline tests for existing code. Only genuine compatibility defects may be labeled reproduced regressions; do not invent red evidence for behavior-preserving work.
