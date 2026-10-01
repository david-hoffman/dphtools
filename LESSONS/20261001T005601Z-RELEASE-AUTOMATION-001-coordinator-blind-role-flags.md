# Keep blind-role coordination flags free of implementation results

- Task: RELEASE-AUTOMATION-001
- Role: coordinator
- Date: 2026-10-01
- Status: confirmed

## Observation

An operational completion flag for blind test author A6 included prior implementation regression counts and a C handoff identity. Only the completion bit was needed. No owned source, source-bearing trace data or C report/conversation contents were supplied, but the extra implementation-status metadata exceeded the intended narrow input boundary.

## Evidence

The original local flag is retained in `reports/release-automation/roles/C2-regression-ended-original.json`. `reports/release-automation/roles/coordinator-barrier-disclosure.md` records the fields and correction for independent B6 assessment. The current flag contains only `finished=true`. The installed AGENTS instruction restricts A/B to approved public inputs and excludes implementation-bearing task state.

## Lesson

Keep coordination signals separate from check evidence. Give blind roles only the operational bit needed to proceed. Retain and disclose accidental extra status metadata rather than silently replacing it or calling it source exposure.
