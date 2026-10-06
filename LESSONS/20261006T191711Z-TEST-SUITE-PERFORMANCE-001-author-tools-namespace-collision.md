# Coverage tools namespace collision

- ID: 20261006T191711Z-TEST-SUITE-PERFORMANCE-001-author-tools-namespace-collision
- Date: 2026-10-06T19:17:11Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: A correct owned-file coverage report hid extra tracing: the broad tools include also measured pandas core/tools in disposable installed environments.
- Evidence: Candidate f3bdc912daef4b48b5d50984d748ee9e8408d541, CI run https://github.com/david-hoffman/dphtools/actions/runs/37514279032 macOS raw data contained 175 such paths with executed arcs. The strengthened test_repo_named_dphtools_does_not_instrument_environment_dependencies failed on a nested dependency tools file, then passed with explicit helper-root registration; original owned/copied assertions remained.
- Lesson: Inspect raw measured paths as well as filtered owned reports. Register copied roots before coverage expands and serializes its include patterns; keep owned descendants measured and add a regression for namespace collisions. Timing effects remain unproven.
