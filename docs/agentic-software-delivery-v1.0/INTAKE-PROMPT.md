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
components, public interfaces, verification and operation. A task produces one
approved behavior contract with examples, non-goals, scope, checks and budget.
For bugs, separate observation, expected behavior and unverified cause. Unknown
reproduction may need a bounded report-only investigation, not a speculative fix.

Read back the exact interpretation and ask for approval. Record my real response
and the document/commit it covers. Never manufacture approval or infer it from silence.
No code or executable test suite during intake. Stop on an unanswered material
question or the preparation budget. No additional interviewing agents.

Prepare only behavior/public-interface inputs for A/B, not this conversation or
implementation ideas. The shared LESSONS.md is not input for those blind roles.
Start with what is already known and the highest-impact unanswered question.
```

Append the request, sources, and existing budget/context. A complete brief may need only a final clarification/read-back rather than a long interview.
