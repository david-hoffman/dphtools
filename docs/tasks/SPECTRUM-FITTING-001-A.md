# SPECTRUM-FITTING-001 — A packet

Role: fresh root A, using .agents/skills/design-tests/SKILL.md.

Approved slice: SPECTRUM-FITTING-001, public contract R3, scenarios S01-S33.
Owner explicitly approved R3 and the 33-scenario exception on 2026-10-05.
One checkpoint lineage. Approved runtime baseline: a835a490373835fa01e4460ce2159365150d1946.
Behavior and expectation sources: docs/tasks/SPECTRUM-FITTING-001-contract.md only.
No fixed owner execution time/token cap was supplied; report usage if available.
Default A/B window: at most two review rounds. Initial C plus one C repair; A/B
cannot replenish that allowance. No scope expansion, PR, push, merge, or release.

Permitted inputs: this packet; the approved public contract; root AGENTS.md;
selected role skill; setup.cfg sections giving test/check conventions; approved
primary references linked by the contract. Use the existing NumPy/SciPy stack,
pytest-compatible tests, synthetic data, and the real public function.
Forbidden reads: dphtools product source, product history/diffs, existing product
tests/oracles, docs/PROJECT.md, the task execution record, other role transcripts,
and existing LESSONS entries. Reading LESSONS/README.md only is allowed if needed.
Do not use Superpowers, optional memory, delegation, or implementation helpers.
A/B restrictions are procedural, not engineered filesystem isolation.

Before any public-API probe/test, configure source-free warnings and tracebacks.
Use a Python launcher that sets warnings.formatwarning to category/message and
filename/line only, then invokes pytest with -p no:warnings --tb=no. Do not hide
warnings, diagnostic messages, failure counts, or the exit status. A useful
launcher is:

```python
import warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{category.__name__}: {message} ({filename}:{lineno})\n"
)
import pytest
raise SystemExit(pytest.main([
    "tests/test_spectrum_fitting.py", "-p", "no:warnings", "--tb=no", "-q", "-ra"
]))
```

Interpreter: /Users/davidhoffman/Documents/GitHub/dphtools/.venv/bin/python.
Set MPLBACKEND=Agg and MPLCONFIGDIR to a writable temporary directory. Importing
the existing public module is permitted without reading its source. Do not add
product stubs or weaken checks. Disclose any accidental source exposure; a later
clean run does not restore that session's blindness. Classify observed failures:
environment/tooling, test defect, product defect, or unresolved requirement.

Requested output: author tests/test_spectrum_fitting.py and a public mapping/
tolerance report at docs/tasks/SPECTRUM-FITTING-001-tests.md. Do not edit any other
tracked file. Tests must cover S01-S33 through the public entry point and trace
numerical expectations to the contract, primary definitions, or invariants.
Choose justified tolerances independently, including discriminating wrong
amplitude/width/model/background/covariance values. Keep oracle helpers simple.
Return changed paths, scenario mapping, tolerance rationale, exact baseline
command/result and failure classification, input-exposure disclosure, and any
contract blocker. No test tuning from candidate implementation/output is allowed.
Completion: meaningful tests and handoff, or a specific contract blocker.
