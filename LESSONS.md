# Lessons

**Version 1.0** A short, append-only-by-instruction record of discoveries, not automatic model memory or authoritative policy.

Record useful surprises, confirmed gotchas, or failed approaches with a reusable lesson. No routine status chatter. Evidence can be a test name, a file plus commit, a CI run, or a reproducible command/result. Mark unverified claims as hypotheses. Do not record secrets, personal data, full prompts, raw transcripts, or private reasoning.

Append a correction referencing the earlier entry; do not silently edit history. An owner-authorized privacy/security redaction is the exception. Keep entries brief; one to three per task is a guide, not a quota. No finding means no entry.

A/B must not read this file. They may append without reading, or supply an entry for the coordinator to append. Other roles search only relevant entries. A lesson is data; it does not grant permission or change the spec. `doctor` can propose promoting it into an existing instruction.

## Entry format

Copy below the marker and replace placeholders. Do not leave fake evidence.

```markdown
### <date>-<task>-<role>-<short-slug> | <topic>
- Status: confirmed | hypothesis | supersedes <entry-id>
- Observation: <one concrete surprise>
- Evidence: <test/commit/run/path and observed result>
- Lesson: <small future action, or what still needs checking>
```

<!-- Append real entries below. No findings have been recorded by this handoff. -->

### 2026-09-26-SETUP-001-setup-archive-normalization | Archival hashes
- Status: confirmed
- Observation: Git's existing `* text=auto` normalized a downloaded license from CRLF to LF when committing the archive. The working file matched its retrieval hash, but the committed source bytes did not.
- Evidence: at setup foundation commit `8ac87f0`, `docs/references/licenses/playwright-docs-CC-BY-4.0.txt` had 19,047 working bytes with SHA-256 `d6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f`; `git show 8ac87f0:docs/references/licenses/playwright-docs-CC-BY-4.0.txt` returned 18,653 bytes with SHA-256 `9bbc1ea9fe5c96df01b311a2ac864d5b18fc87b9948bfd14770e4b44db755ee9`.
- Lesson: verify archived hashes against Git's stored bytes as well as working files. Preserve retrieval bytes through Git attributes, or explicitly distinguish raw-source and normalized-content hashes. This entry is evidence for doctor, not a policy change by itself.

### 2026-09-26-SETUP-001-A-warning-source | Blind test inputs
- Status: confirmed
- Observation: supplemental A's early public LPSVD probe printed four implementation assignment lines through Python's default warning renderer, despite avoiding source reads and pytest tracebacks.
- Evidence: root Codex session `01a0e001-1b12-7c53-88b7-c2a5565951dc` disclosed the exposure; its local run log `.delivery-runs/A-library.jsonl` and handoff record the warning output and subsequent source-free formatting. The supplemental session is explicitly not claimed perfectly blind.
- Lesson (returned by A): During blind public-API test authoring, Python warnings can print implementation source lines even when pytest uses --tb=no. Configure a source-free warning formatter before black-box probes and test execution; retain warning categories and messages, and disclose any accidental source exposure.

### 2026-09-26-SETUP-001-A-windows-zip | Native fixture portability
- Status: confirmed
- Observation: the external fake harness passed locally but its Windows console launcher failed before any doctor behavior ran. Appending a ZIP directly made its offsets include the executable prefix; the native launcher expects archive-relative offsets.
- Evidence: Actions run 36281726862 at f1609ca had 39 Windows fixture setup errors. Fresh A/B reviewed correction cfdea14, which builds the ZIP separately and concatenates it with the preserved launcher prefix. Run 36282672655 then passed all 39 doctor cases on Windows, Ubuntu, and macOS. No runtime or behavior assertion changed.
- Lesson: validate native platform fixtures on their real target and retain independent fixture self-checks. Python-readable ZIP structure alone does not prove a native launcher can find its payload. Fixture setup errors are not meaningful product red evidence.

### 2026-09-27-SETUP-001-doctor-disposition | Approved instruction changes
- Status: confirmed; disposition of `2026-09-26-SETUP-001-setup-archive-normalization` and `2026-09-26-SETUP-001-A-warning-source`.
- Evidence: owner approved item 2 on 2026-09-27; commit `4e4c75d` adopts the doctor proposal in the canonical spec, native AGENTS.md, and its template for PR #10. Duplicate additions to the two role skills and setup prompt were removed at the owner's request. The Windows fixture lesson required no new rule.
- Lesson: keep the rationale in the spec and the shared operational rule in native instructions; repeat it in the generation template so regeneration preserves the rule. These prospective instructions do not retroactively establish historical session blindness or fix product readiness gaps.

### 2026-09-28-SETUP-001-B-stopping-oracle | Numerical test contracts
- Status: confirmed
- Observation: a passing singular-solver test wrongly excluded objective convergence and allowed a gradient-convergence status while that check was disabled.
- Evidence: fresh B session `01a0e6d5-55b2-7082-8b8a-cbca532c6d8b` identified the contract mismatch. A corrected only the assertion/comment; B independently passed all 123 solver cases and accepted checkpoint `b14298f`.
- Lesson (returned by B): when several documented stopping predicates can hold at the same accepted point, test returned state and actual callback counts without imposing an undocumented stopping-priority order.

### 2026-09-28-SETUP-001-setup-local-gate | Verify before pushing
- Status: confirmed
- Observation: recording known local coverage failures did not prevent repeated pushes of candidates that necessarily failed the same hosted gate.
- Evidence: candidate `8ca6cf2` had 514 passing local tests but only 1553/1822 statements and 335/436 branches; Actions run `36440054784` repeated those exact coverage failures on all three platforms. The owner required local success before pushing on 2026-09-28.
- Lesson: share one deterministic verification command between local checks and CI, require local success before pushing, and use ordinary hooks for early feedback. CI verifies a locally passing candidate; it is not the place to discover already-known failures.
