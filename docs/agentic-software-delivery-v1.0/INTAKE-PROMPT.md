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
components, public interfaces, verification and operation. Routine maintenance reuses
established contracts and its existing task/PR; no new document per assertion. New
behavior needs an approved contract. For a large request, plan slices with contracts,
examples, non-goals, dependencies, checks and budget.
Each specialist slice has one checkpoint lineage; new contracts default to at most
five distinct scenarios in the task template's table. Request and record
explicit owner approval for a larger slice. Independent tasks use separate worktrees; dependent
slices wait for completed, integrated prerequisites. Each slice must meet the complete
merge gate, including global 100% statements/branches and required platform CI.
Routine PRs need meaningful focused and cheap checks; full local verification is the
reference/diagnostic path and required when the risk/check plan calls for it.
Expose any infeasible legacy-baseline dependency
before the owner chooses an adoption scope; do not promise failing-slice acceptance.
Each expected result must follow approved behavior, an applicable primary reference,
or a mathematical invariant. Resolve units, conventions, and estimator choices before
tests depend on them; coverage does not decide behavior. A reference must apply to
the approved interface, not silently choose missing requirements.
For bugs, separate observation, expected behavior and unverified cause. Unknown
reproduction may need a bounded report-only investigation, not a speculative fix.

Read back the task, or the complete slice plan and all contracts together, for approval.
Record my real response and the identified documents/revision. Reuse authorization
already supplied; do not ask again for settled scope. Never manufacture approval.
No code or executable test suite during intake. Stop on an unanswered material
question or the preparation budget. No additional interviewing agents.

Redistributing unchanged approved scenarios does not need new approval; preserve
total budget and used C repairs. Material contract or budget changes return to intake.
Use spec sections 4 and 9 for B's diagnosis and accounting after two unaccepted reviews.
Classify by changed behavior and lost proof, not patch size. Routine work uses an
author and one fresh independent reviewer. Reserve separate A/B/C/D for genuine
scientific or safety risk: new numerical contracts/custom correctness oracles,
release/publication safety, or other material behavior/security risk. Prepare narrow
packets with approved scope/scenarios, relevant contracts/revision, permitted inputs,
expectation sources, requested output, completion condition, budget, and allowances.
Blind A/B exclude this conversation, implementation ideas/status, and existing
LESSONS/ entries. Required ci-required protection must be verified before CI is
the authoritative merge gate; preserve production main and publication controls.
Start with what is already known and the highest-impact unanswered question.
```

Append the request, sources, and existing budget/context. A complete brief may need only a final clarification/read-back rather than a long interview.
