# Doctor prompt

**Version 1.0** This is the procedure for `delivery doctor`, using the existing `review-work` skill in doctor mode. Before the command is implemented, give this prompt to a fresh coding session. It may write a documentation patch; it must not claim a command ran when it did not.

```text
Run doctor under docs/agentic-software-delivery-v1.0/DELIVERY-SYSTEM-SPEC.md section 8.
Check the working tree and stop
rather than overwrite unrelated/uncommitted work or interrupt an active delivery task.
Read relevant project instructions, recent LESSONS.md entries, and concrete CI/test
or repository evidence. Search narrowly; do not replay all sessions or recrawl sources.

Classify failures under spec section 4: environment/tooling failure, test defect,
product defect, or unresolved requirement. Then distinguish an instruction gap from
an execution error under an existing rule. Verify lessons before generalizing them.
A green rerun does not prove a previous failure harmless. Hypotheses remain hypotheses.

For a confirmed spec gap, create a small docs branch and actually edit the existing
docs/agentic-software-delivery-v1.0/DELIVERY-SYSTEM-SPEC.md plus directly affected
instructions. Replace or remove text before adding more. Keep shared operational
instructions in AGENTS.md and role-specific differences in skills; synchronize their
generation templates. No new framework, skill swarm, or parallel spec. No gap means
no change. With --check, report only and do not write files.

Do not change runtime code, tests, workflows, thresholds, or repository settings.
Do not rewrite desired behavior to match a bug or silently approve architecture drift.
Those need separate scoped work. Test a new procedural instruction on one relevant
example when practical. Explain what was and was not verified.

Present the exact diff, evidence/lesson IDs, and expected improvement. Wait for my
approval before committing, pushing, or merging. Then use a normal Git commit/PR;
append a LESSONS.md disposition linking it. Do not change an in-flight task's rules.
All delivery documents stay version 1.0; Git tracks their revisions.
Stop at the approved budget. No additional reviewer or autonomous retry loop.
```
