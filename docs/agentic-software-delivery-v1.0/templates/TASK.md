# <task-id>: <observable outcome>

**Version 1.0** Git versions revisions. This record is not technical enforcement.

## Contract

- Slice plan/reference, slice ID, prerequisite tasks, and intended baseline (if part of a larger request):
- Project/interface references:
- Purpose and non-goals:
- R1: <observable requirement>
- R2: <observable requirement>
- Inputs, outputs, errors, and permissions:
- Expected-result sources: approved behavior, applicable primary references, or mathematical invariants; units, conventions, estimators, and any unresolved limits.
- Allowed scope and applicable checks: highest approved verification tier, comparison base/candidate, complete owned domains/test closure, promotion reasons, and meaningful focused checks:
- Route: routine author/fresh independent reviewer, or specialist A/B/C/D; justify genuine scientific/safety risk by changed behavior and lost proof. Reuse established contracts where applicable:
- Release impact and draft release notes, including compatibility/migration needs; record none when applicable:
- Preparation/execution budget within the plan, risk, and inherited review/repair allowances:

| Scenario ID | Approved input/context and observable success, error, or boundary outcome | Expectation source | Test mapping supplied by author and independently reviewed (A/B for specialists) |
|---|---|---|---|
| S1 | <one distinct contract example> | <approved behavior/reference/invariant> | <test references after authoring> |

Default: at most five scenarios; record an explicit size exception if approved. Count these cases, not test functions/assertions. For a large request, the slice plan at `docs/tasks/<plan-id>.md` links every slice contract, dependencies, outcomes, budgets, and completion conditions for one approval read-back. Do not invent a separately unapproved contract in a role packet.

The author supplies mapping/expectation evidence; the independent reviewer checks it. Specialist A supplies the mapping and numerical tolerance rationale, recorded without changing approved behavior; B verifies cases and that tolerances accept legitimate variation while rejecting plausible errors. Routine maintenance may link the established contract instead of creating a new table/document for each assertion.

For a bug: observed versus expected behavior, environment, reproduction evidence, and hypotheses. Unknown causes do not authorize a speculative fix.

## Current state

Use this section or a pointer to one current-state section in the linked PR/discussion. Final candidate identity and verification results belong outside the candidate's tracked tree, in the PR or conversation; prepare the pointer before verification. Do not change the verified commit to record its own hash/results or keep competing status copies.

- Status: draft / approved / blocked / done.
- Actual owner approval references and identified contract revision; affected changes needing renewed approval:
- Approved scope/scenarios, route and rationale; specialist test checkpoint where applicable, including parent/replaced task or checkpoint:
- Author/independent-reviewer references and outcomes; specialist A/B/C/D references when applicable:
- Exact candidate commit:
- Latest applicable checks: selected tier/domains/test closure, base/candidate, command, environment/platform, result, exact required-domain coverage (global/per-package for full), and evidence/CI/PR links:
- Baseline evidence: intended failure for a claimed bug/missing feature, or initially passing approved existing behavior; classified failures and observing role:
- Metrics: scenarios=<count>; author/reviewer launches=<count>; specialist A/B rounds=<current/prior or not applicable>; C repairs=<used/remaining or not applicable>; task/plan budget=<used/remaining or unavailable>.
- If review rounds restart, an accepted checkpoint reopens, or a slice splits: evidence and authorized reason, B's diagnosis/resolution where required, approval if behavior changed, prior rounds, scenario redistribution, and remaining budget/C repair allocation:
- Blockers and pending owner decisions:
- Next action, responsible role, narrow packet, and completion condition:

Replace superseded status here; Git preserves history. Link reports instead of repeating them. Material contract changes need approval; unchanged-scenario subdivision follows the plan's approval. Renamed work or revised checkpoints never erase attempts or spent budget. Specialist C repair use carries over; acceptance closes its A/B review window, and later authorized corrections/restarts follow specification section 9.

Supply narrow packets to all roles; specialist A/B receive only approved behavior/public interfaces and necessary revision identifiers, not implementation details from this section or full conversations. Routine PRs require meaningful focused checks and the highest required project tier; uncertainty requires full. Merge requires complete required-platform evidence for the sealed selected plan and exact whole-domain coverage (fresh global/per-package for full), fresh independent review of an unchanged candidate, verified native required-check protection, and owner authorization. Keep logs outside Git and report host limitations honestly. Only lesson entries in `LESSONS/` are append-only; record discoveries there, not in a duplicate task log.
