### 2026-09-26-SETUP-001-A-warning-source | Blind test inputs
- Status: confirmed
- Observation: supplemental A's early public LPSVD probe printed four implementation assignment lines through Python's default warning renderer, despite avoiding source reads and pytest tracebacks.
- Evidence: root Codex session `01a0e001-1b12-7c53-88b7-c2a5565951dc` disclosed the exposure; its local run log `.delivery-runs/A-library.jsonl` and handoff record the warning output and subsequent source-free formatting. The supplemental session is explicitly not claimed perfectly blind.
- Lesson (returned by A): During blind public-API test authoring, Python warnings can print implementation source lines even when pytest uses --tb=no. Configure a source-free warning formatter before black-box probes and test execution; retain warning categories and messages, and disclose any accidental source exposure.
