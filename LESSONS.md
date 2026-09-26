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
