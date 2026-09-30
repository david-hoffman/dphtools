# Intake prompt

**Version 1.0** Use the `intake` skill. This prompt starts a conversation; it is not an authorization service.

```text
Clarify this request using intake mode task or architecture, as appropriate.

First read the supplied context. Before a task proceeds, find an approved project
record. Reuse it, including architecture under another filename. If it is missing,
unapproved, or materially inconsistent, interview me about architecture first and
then resume the original task. Do not redesign unrelated parts or repeat answers.

Ask about material ambiguities, usually no more than three questions per turn.
Use contrasting examples. Follow up on vague or contradictory answers. Recommend
simple defaults and explain the consequences; do not assume I accepted them.
No speculative future questionnaire and no demand for every internal coding detail.

Architecture produces one short docs/PROJECT.md: outcome, constraints, stack,
components, public interfaces, verification and operation. A small request needs one
task document. For a large request, plan small end-to-end slices with contracts,
examples, non-goals, dependencies, checks and budget.
Each ordinary slice task has one checkpoint lineage and defaults to at most five
distinct contract scenarios in the task template's scenario table. Request and record
explicit owner approval for a larger slice. Independent tasks use separate worktrees; dependent
slices wait for completed, integrated prerequisites. Each slice must meet the full
gate, including global 100% coverage. Expose any infeasible legacy-baseline dependency
before the owner chooses an adoption scope; do not promise failing-slice acceptance.
Each expected result must follow approved behavior, an applicable primary reference,
or a mathematical invariant. Resolve units, conventions, and estimator choices before
tests depend on them; coverage does not decide behavior. A reference must apply to
the approved interface, not silently choose missing requirements.
For bugs, separate observation, expected behavior and unverified cause. Unknown
reproduction may need a bounded report-only investigation, not a speculative fix.

Read back the task, or the complete slice plan and all contracts together, for approval.
Record my real response and the identified documents/revision. Never manufacture
approval or infer it from silence.
No code or executable test suite during intake. Stop on an unanswered material
question or the preparation budget. No additional interviewing agents.

Redistributing unchanged approved scenarios does not need new approval; preserve
total budget and used C repairs. Material contract or budget changes return to intake.
Use spec sections 4 and 9 for B's diagnosis and accounting after two unaccepted reviews.
Prepare narrow A/B packets with approved scenario IDs, public-interface inputs,
expectation sources, requested output, completion condition, budget, and remaining
allowances. Exclude this conversation, implementation
ideas, implementation-bearing task status, and existing lesson entries in LESSONS/.
Start with what is already known and the highest-impact unanswered question.
```

Append the request, sources, and existing budget/context. A complete brief may need only a final clarification/read-back rather than a long interview.
