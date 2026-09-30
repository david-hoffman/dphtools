### 2026-09-26-SETUP-001-setup-archive-normalization | Archival hashes
- Status: confirmed
- Observation: Git's existing `* text=auto` normalized a downloaded license from CRLF to LF when committing the archive. The working file matched its retrieval hash, but the committed source bytes did not.
- Evidence: at setup foundation commit `8ac87f0`, `docs/references/licenses/playwright-docs-CC-BY-4.0.txt` had 19,047 working bytes with SHA-256 `d6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f`; `git show 8ac87f0:docs/references/licenses/playwright-docs-CC-BY-4.0.txt` returned 18,653 bytes with SHA-256 `9bbc1ea9fe5c96df01b311a2ac864d5b18fc87b9948bfd14770e4b44db755ee9`.
- Lesson: verify archived hashes against Git's stored bytes as well as working files. Preserve retrieval bytes through Git attributes, or explicitly distinguish raw-source and normalized-content hashes. This entry is evidence for doctor, not a policy change by itself.
