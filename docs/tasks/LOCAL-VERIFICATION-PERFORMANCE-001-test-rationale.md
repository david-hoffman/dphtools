# LOCAL-VERIFICATION-PERFORMANCE-001 test rationale

Role A changes only fixture preparation in `tests/test_verification.py`.
The approved P1/P2 timing conditions and all existing regression requirements
remain unchanged. No numerical tests, data, tolerances, or assertions are edited.

| Consolidated setup and purpose | Retained evidence |
| --- | --- |
| The `verifier` fixture's identical Black stand-in preflight distinguishes broken fixture execution from verifier failures. | `_external_tool_preflight` runs the same real `python -m black fixture-probe` subprocess once per module, with the same environment construction, working-directory convention, 20 s timeout, exit-status assertion, recorded-tool assertion, and record cleanup. All 15 test functions / 48 parameterized cases still request `verifier`, which depends on this preflight. |

Each case still constructs a new `VerificationCommand` in its own `tmp_path`.
The module preflight uses a separate `tmp_path_factory` directory. No repository,
tool file, environment dictionary, report, call record, or coverage-data path is
shared with a test. Tests that rewrite tools or configuration keep private files.
The constructor and all existing non-fixture helpers remain unchanged.

All test bodies and parameter declarations remain unchanged. They retain usage
errors, fast/full operation inventories, failure continuation, invalid-report
rejection, report freshness, real lint/docstring/coverage checks, nested coverage,
undecodable output, inert import, and denied process launch. Every subprocess in
those tests remains real and unchanged. The sole reduction is repeated fixture
preflight execution: 48 probes become one, removing 47 redundant launches. One
additional private fixture directory is prepared for that probe. Hook and doctor
fixtures, including their probes and local Git boundaries, remain unchanged.

Both local affected-suite attempts passed 48 tests, with zero failures, errors,
or skips. External elapsed times were 21.169 s before and 20.594 s after the edit.
The difference is 0.575 s (about 2.7% of this subset). This single sequential pair
is descriptive; it does not establish a reliable speedup or P1/P2 acceptance.
No full benchmark was run. The coordinator retains cold/warm full measurement,
clean-install checks, and exact global/per-package coverage validation.

The commands used `/private/tmp/dphtools-verification-env/bin/python -m pytest
tests/test_verification.py --tb=no --junitxml=reports/performance-A/<attempt>-pytest.xml`.
All six specified numerical-library thread settings and `BLACK_NUM_WORKERS` were
`1`. Before execution, the existing `SOURCEFREE` fixture text configured source-free
line caching, exception rendering, and warning formatting in the runner and a
temporary `sitecustomize.py` inherited by Python children. Coverage configuration
remained opaque and unchanged. No runtime source exposure was observed.

JUnit XML, complete process logs, commands, environment settings, elapsed times,
allowed-input SHA256 revisions, and preservation checks are retained under
`reports/performance-A/`. The baseline and candidate manifests record all seven
permitted input files; the candidate manifest also records this rationale's hash.
The preservation check confirms unchanged syntax trees for all test functions,
parameters, and existing non-fixture helpers/classes, plus byte-identical hook and
delivery test files. Role A did not commit. Independent role B review is next.
