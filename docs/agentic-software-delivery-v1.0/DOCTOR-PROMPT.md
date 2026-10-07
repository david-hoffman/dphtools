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
focused/cheap PR checks, risk-based full local runs, and complete required platform CI.
Workflow text cannot establish actual ci-required protection. An evidenced patch may propose tuning
the default slice size; do not tune it autonomously, reset consumed budgets, or activate
a changed verification/coverage merge gate. An authorized move of duplicated local work
to verified required CI preserves acceptance; missing protection must stay visible.

For coverage friction, use spec sections 5.1–5.2 and the task's pinned project policy.
Classify missing approved behavior/risk, unnecessary implementation complexity,
unresolved semantics, measurement faults, or a selected-target policy conflict before
proposing correction. An uncovered branch alone does not establish a new requirement
or justify another A/B cycle. Preserve blind source-free specialist test handoffs,
authorized edit ownership, spending and remaining review/repair allowances.
Do not lower a gate through a failing task's repair. An evidenced policy mismatch may
justify a separate amendment with independent policy review and explicit owner patch
approval. Reconcile affected instructions and authorized checks before activation;
active tasks retain pinned gates unless explicitly migrated, without resetting allowances.
Approved behavior/risk evidence and each selected measured target remain binding;
unselected metrics are advisory, while missing required measurements block acceptance.

For a confirmed spec gap, create a small docs branch and actually edit the existing
docs/agentic-software-delivery-v1.0/DELIVERY-SYSTEM-SPEC.md plus directly affected
instructions. Replace or remove text before adding more. Keep shared operational
instructions in AGENTS.md and role-specific differences in skills; synchronize their
generation templates. No new framework, skill swarm, or parallel spec. No gap means
no change. With --check, report only and do not write files.

Do not change runtime code, tests, workflows, active acceptance thresholds, or repository settings.
Doctor may propose policy text; it cannot activate that proposal or modify installed checks.
Do not dispatch a release, approve publication, create a remote tag, or publish artifacts.
Do not rewrite desired behavior to match a bug or silently approve architecture drift.
Those need separate scoped work. Test a new procedural instruction on one relevant
example when practical. Explain what was and was not verified.

Present the exact diff, evidence/lesson IDs, and expected improvement. Obtain independent
policy review and my explicit patch approval before activation, committing, pushing,
or merging. Reuse actual authorization already supplied. Then use a normal Git commit/PR;
add a new timestamped lesson file in LESSONS/ linking the disposition to that commit
and the earlier entry. Changing an in-flight task's rules needs explicit owner migration
approval; preserve prior windows, spending and repair use.
All delivery documents stay version 1.0; Git tracks their revisions.
Stop at the approved budget. Independent policy review is required; do not add other
reviewers or an autonomous retry loop.
```
