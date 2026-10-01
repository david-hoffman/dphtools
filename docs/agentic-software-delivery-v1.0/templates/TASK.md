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
- Allowed scope and applicable checks:
- Release impact and draft release notes, including compatibility/migration needs; record none when applicable:
- Preparation/execution budget within the plan, risk, and inherited review/repair allowances:

| Scenario ID | Approved input/context and observable success, error, or boundary outcome | Expectation source | Test mapping supplied by A and checked by B |
|---|---|---|---|
| S1 | <one distinct contract example> | <approved behavior/reference/invariant> | <test references after authoring> |

Default: at most five scenarios; record an explicit size exception if approved. Count these cases, not test functions/assertions. For a large request, the slice plan at `docs/tasks/<plan-id>.md` links every slice contract, dependencies, outcomes, budgets, and completion conditions for one approval read-back. Do not invent a separately unapproved contract in a role packet.

A's handoff supplies the mapping and numerical tolerance rationale; the coordinator records them here without changing approved behavior. B verifies that each case is exercised and that tolerances accept legitimate variation while rejecting plausible errors.

For a bug: observed versus expected behavior, environment, reproduction evidence, and hypotheses. Unknown causes do not authorize a speculative fix.

## Current state

Use this section or a pointer to one current-state section in the linked PR/discussion. Final candidate identity and verification results belong outside the candidate's tracked tree, in the PR or conversation; prepare the pointer before verification. Do not change the verified commit to record its own hash/results or keep competing status copies.

- Status: draft / approved / blocked / done.
- Actual owner approval references and identified contract revision; affected changes needing renewed approval:
- Approved slice/scenarios and reviewed test checkpoint, including parent/replaced task or checkpoint:
- A/B/C/D session/review references and completion outcomes:
- Exact candidate commit:
- Latest applicable checks: command, candidate, environment, result, exact coverage, evidence/CI/PR links:
- Baseline evidence: intended failure for a claimed bug/missing feature, or initially passing approved existing behavior; classified failures and observing role:
- Metrics: scenarios=<count>; A/B rounds=<used in current two-round window, prior windows linked>; C repairs=<used/remaining>; task/plan budget=<used/remaining or unavailable>.
- If review rounds restart, an accepted checkpoint reopens, or a slice splits: evidence and authorized reason, B's diagnosis/resolution where required, approval if behavior changed, prior rounds, scenario redistribution, and remaining budget/C repair allocation:
- Blockers and pending owner decisions:
- Next action, responsible role, narrow packet, and completion condition:

Replace superseded status here; Git preserves its history. Link reports instead of repeating them. A material contract change needs actual approval; unchanged-scenario subdivision follows the plan's approval. Revised checkpoints or renamed work do not erase attempts, spent budget, or C repair use. Acceptance closes an A/B review window; later authorized corrections and restarts after two nonacceptances follow specification section 9's distinct routes.

Supply A/B only their approved behavior/public-interface packet and necessary revision identifiers, not implementation details from this section or full conversations. Only lesson entries in `LESSONS/` are append-only; record each reusable discovery in a new timestamped file there, not in a duplicate task log.
