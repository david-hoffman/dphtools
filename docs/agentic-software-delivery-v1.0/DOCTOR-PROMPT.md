# Doctor prompt

**Version 1.0** This is the procedure for `delivery doctor`, using the existing `review-work` skill in doctor mode. Before the command is implemented, give this prompt to a fresh coding session. It may write a documentation patch; it must not claim a command ran when it did not.

```text
Run doctor under docs/agentic-software-delivery-v1.0/DELIVERY-SYSTEM-SPEC.md section 8.
Check the working tree and stop
rather than overwrite unrelated/uncommitted work or interrupt an active delivery task.
Read relevant project instructions, recent entries in LESSONS/, and concrete CI/test
or repository evidence. Search narrowly; do not replay all sessions or recrawl sources.
When release automation is installed, compare its source/version/destination policy,
verification and approval boundary, artifact identity, recovery, and completion checks
with project instructions and available hosted evidence. Report unapplied setup and
drift; workflow text alone does not prove protected approval or registry trust.

Classify failures under spec section 4: environment/tooling failure, test defect,
product defect, or unresolved requirement. Then distinguish an instruction gap from
an execution error under an existing rule. Verify lessons before generalizing them.
A green rerun does not prove a previous failure harmless. Hypotheses remain hypotheses.
Use the compact Current state metrics (scenario count, author/reviewer launches,
specialist A/B rounds/C repairs when applicable, spend when known) to assess overhead.
Check routine author/fresh-reviewer routing against genuine scientific/safety risk,
focused PR checks, the highest owner-approved tier, conservative base/candidate
classification, whole-domain/test closure, fresh full when required, and complete
required-platform evidence for the sealed selected plan. Check honest scope reporting
and that ci-required always runs; omitted domains and cached results are not fresh proof.
Workflow text cannot establish actual ci-required protection. An evidenced patch may propose tuning
the default slice size; do not tune it autonomously, reset consumed budgets, or lower
the verification/coverage merge gate. An authorized move of duplicated local work
to verified required CI preserves acceptance; missing protection must stay visible.

For a confirmed spec gap, create a small docs branch and actually edit the existing
docs/agentic-software-delivery-v1.0/DELIVERY-SYSTEM-SPEC.md plus directly affected
instructions. Replace or remove text before adding more. Keep shared operational
instructions in AGENTS.md and role-specific differences in skills; synchronize their
generation templates. No new framework, skill swarm, or parallel spec. No gap means
no change. With --check, report only and do not write files.

Do not change runtime code, tests, workflows, acceptance thresholds, or repository settings.
Do not dispatch a release, approve publication, create a remote tag, or publish artifacts.
Do not rewrite desired behavior to match a bug or silently approve architecture drift.
Those need separate scoped work. Test a new procedural instruction on one relevant
example when practical. Explain what was and was not verified.

Present the exact diff, evidence/lesson IDs, and expected improvement. Wait for my
approval before committing, pushing, or merging. Then use a normal Git commit/PR;
add a new timestamped lesson file in LESSONS/ linking the disposition to that commit
and the earlier entry. Do not change an in-flight task's rules.
All delivery documents stay version 1.0; Git tracks their revisions.
Stop at the approved budget. No additional reviewer or autonomous retry loop.
```
