# Keep blind-role coordination flags free of implementation results

- ID: 20261001T005634Z-RELEASE-AUTOMATION-001-coordinator-blind-role-flags-format
- Date: 2026-10-01T00:56:34Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Supersedes: 20261001T005601Z-RELEASE-AUTOMATION-001-coordinator-blind-role-flags
- Observation: An A6 operational flag included prior implementation regression counts and a C handoff identity; only the completion bit was needed. No owned source, source-bearing trace data or C report/conversation contents were supplied, but extra implementation-status metadata exceeded the intended narrow input boundary. This entry also corrects the preceding entry's metadata format.
- Evidence: Local `reports/release-automation/roles/C2-regression-ended-original.json` retains the original fields; `coordinator-barrier-disclosure.md` records them for fresh B6 assessment. The current flag contains only `finished=true`. Installed AGENTS excludes implementation-bearing task state from A/B inputs; LESSONS/README.md supplies this format.
- Lesson: Give blind roles only the operational bit needed to proceed. Retain and disclose accidental extra status metadata. Read the format guide before writing a lesson; correct an existing entry with a new superseding file.
