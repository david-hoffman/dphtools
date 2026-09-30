# Diagnostic launchers must preserve the canonical import environment

- ID: 20260930T190945Z-RELEASE-AUTOMATION-001-setup-diagnostic-runner-import-path
- Date: 2026-09-30T19:09:45Z
- Task/role: RELEASE-AUTOMATION-001 / setup coordinator
- Status: confirmed
- Observation: A source-free pytest launcher under reports/ replaced the working-directory Python import entry with its own script directory. Package-qualified test-helper imports then failed although the canonical module invocation makes the repository importable.
- Evidence: reports/release-automation/roles/A-resume3-events.jsonl records collection failures; inserting Path.cwd() in the ignored launcher allowed source-free collection of all 272 then-authored cases. The later baseline collected 288 cases, and tests/test_delivery.py passed all 39 cases. No canonical discovery or runtime setting changed.
- Lesson: Preserve the canonical module invocation's import environment when adding diagnostic wrappers. Classify wrapper import failures as tooling before changing product code or declaring product-red evidence.
