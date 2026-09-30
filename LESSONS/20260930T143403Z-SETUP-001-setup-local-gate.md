### 2026-09-28-SETUP-001-setup-local-gate | Verify before pushing
- Status: confirmed
- Observation: recording known local coverage failures did not prevent repeated pushes of candidates that necessarily failed the same hosted gate.
- Evidence: candidate `8ca6cf2` had 514 passing local tests but only 1553/1822 statements and 335/436 branches; Actions run `36440054784` repeated those exact coverage failures on all three platforms. The owner required local success before pushing on 2026-09-28.
- Lesson: share one deterministic verification command between local checks and CI, require local success before pushing, and use ordinary hooks for early feedback. CI verifies a locally passing candidate; it is not the place to discover already-known failures.
