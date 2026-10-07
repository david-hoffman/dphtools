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
- Approved project coverage policy and committed local revision; inherited choice and any explicitly approved policy change/active-task migration:
- Relevant high-risk paths, required observable outcomes, and unresolved risk decisions:
- Allowed scope and applicable checks:
- Route: routine author/fresh independent reviewer, or specialist A/B/C/D; justify genuine scientific/safety risk by changed behavior and lost proof. Reuse established contracts where applicable:
- Release impact and draft release notes, including compatibility/migration needs; record none when applicable:
- Preparation/execution budget within the plan, risk, and inherited review/repair allowances:

| Scenario ID | Approved input/context and observable success, error, or boundary outcome | Expectation source | Test mapping supplied by author and independently reviewed (A/B for specialists) |
|---|---|---|---|
| S1 | <one distinct contract example> | <approved behavior/reference/invariant> | <test references after authoring> |

Default: at most five scenarios; record an explicit size exception if approved. Count these cases, not test functions/assertions. For a large request, the slice plan at `docs/tasks/<plan-id>.md` links every slice contract, dependencies, outcomes, budgets, and completion conditions for one approval read-back. Do not invent a separately unapproved contract in a role packet.

The author supplies mapping/expectation and risk evidence; the independent reviewer audits omissions and records a short risk disposition. Specialist A supplies the mapping and numerical tolerance rationale, recorded without changing approved behavior; B verifies cases and that tolerances accept legitimate variation while rejecting plausible errors. Routine maintenance may link the established contract instead of creating a new table/document for each assertion. Accepted tests are the reviewed baseline, not proof of every possible behavior or immunity to a concrete omission found later.

Resolve an undefined request for “complete coverage” before test design under specification section 5.1. Inherit settled project choices without another interview. Behavior coverage and independent risk review always apply to the approved scope; numerical line/statement/branch targets apply only as selected. Selected targets retain exact thresholds, scope, exclusions, limits, and complete required reports; unselected metrics are advisory. Existing installed gates remain binding until independent policy review and explicit owner approval of the identified change and its application. Active-task migration must be explicit and retains prior attempts, spending, rounds, and repair allowances.

For a bug: observed versus expected behavior, environment, reproduction evidence, and hypotheses. Unknown causes do not authorize a speculative fix.

## Current state

Use this section or a pointer to one current-state section in the linked PR/discussion. Final candidate identity and verification results belong outside the candidate's tracked tree, in the PR or conversation; prepare the pointer before verification. Do not change the verified commit to record its own hash/results or keep competing status copies.

- Status: draft / approved / blocked / done.
- Actual owner approval references and identified contract revision; affected changes needing renewed approval:
- Approved scope/scenarios, route and rationale; specialist test checkpoint where applicable, including parent/replaced task or checkpoint:
- Author/independent-reviewer references and outcomes; specialist A/B/C/D references when applicable:
- Exact candidate commit:
- Latest applicable checks: command, candidate, environment, result, exact coverage, evidence/CI/PR links:
- Requirement/scenario-to-test mapping and independent risk-audit disposition; concrete omissions and their resolution, or why runtime obligations are unaffected for non-behavioral work:
- Baseline evidence: intended failure for a claimed bug/missing feature, or initially passing approved existing behavior; classified failures and observing role:
- Metrics: scenarios=<count>; author/reviewer launches=<count>; specialist A/B rounds=<current/prior or not applicable>; C repairs=<used/remaining or not applicable>; task/plan budget=<used/remaining or unavailable>.
- If review rounds restart, an accepted checkpoint reopens, or a slice splits: evidence and authorized reason, B's diagnosis/resolution where required, approval if behavior changed, prior rounds, scenario redistribution, and remaining budget/C repair allocation:
- Blockers and pending owner decisions:
- Next action, responsible role, narrow packet, and completion condition:

Replace superseded status here; Git preserves history. Link reports instead of repeating them. Material contract changes need approval; unchanged-scenario subdivision follows the plan's approval. Renamed work or revised checkpoints never erase attempts or spent budget. Specialist C repair use carries over; acceptance closes its A/B review window, and later authorized corrections/restarts follow specification section 9.

For a coverage finding, record the concrete missing outcome/risk or selected metric shortfall and its classification under specification section 5.2: missing behavior/risk tests, unnecessary implementation complexity, unresolved semantics, measurement defect, or policy conflict. State evidence and why existing tests miss a claimed omission before commissioning test correction; a bare uncovered branch does not automatically return to A. Route each finding to its authorized owner within existing scope and allowances.

Supply narrow packets to all roles; specialist A/B receive only approved behavior/public interfaces, coverage policy/relevant risks, permitted test evidence, and necessary revision identifiers, not implementation details from this section, copied implementation reports, coverage-line maps, or full conversations. Use source-free public-contract reproductions for implementation-derived omissions. Routine PRs require meaningful focused/cheap checks; full local verification follows the risk/check plan. Merge requires full required platform verification and the approved coverage policy, fresh independent review of an unchanged candidate, verified native required-check protection, and owner authorization. Keep logs outside Git and report host limitations honestly. Only lesson entries in `LESSONS/` are append-only; record discoveries there, not in a duplicate task log.
