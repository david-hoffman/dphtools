---
name: review-work
description: Review tests (B), review a passing candidate (D), or improve the spec from evidence (doctor). Use one mode per fresh session.
metadata:
  version: "1.0"
  status: released
---

# Review work

Follow installed repository `AGENTS.md` for shared session, access, verification, budget, and reporting rules. Use one mode per packet.

## Tests — B

Use A's approved behavior/public-interface inputs and proposed tests.

Check contract clarity, observable assertions, expected-result sources, negative/boundary behavior, and requirement coverage. A test oracle is the rule deciding whether a result is correct. Review custom numerical/plotting decision logic, including parsers, observers, tolerances, units, and allowed outcomes. Use small distinguishing examples to check both acceptance of legitimate alternatives and rejection of plausible wrong results. Do not normalize malformed output into validity or impose an undocumented choice. Missing requirements return to intake; test corrections return to A.

Complete B with acceptance of an identified test revision or concrete defects/contract gaps. Accepted tests need valid baseline evidence (regression evidence for refactors) and a recorded local checkpoint before C. Use the failure routes in specification section 4; a setup failure is not product-red evidence.

## Candidate — D

Read the approved task, exact candidate, test checkpoint, and actual check evidence, not C's conversation. Form findings before consulting current-task implementation lessons.

Inspect behavior, security, public boundaries, and test/workflow changes. Compare the tested commit with the proposed merge. Missing/skipped/incomplete results are not success. Check the real product result, not only a summary or compilation.

Include simplification in this review: unnecessary wrappers, duplication, dependencies, and speculative features. Stay within the task. No extra simplification agent. Complete D with acceptance of the exact passing candidate or concrete findings and evidence; do not fix the candidate and then approve it. Route findings under specification section 4. Product corrections use fresh C within the recorded repair allowance, renewed checks, and fresh D. Reuse applicable check evidence; rerun when changed inputs or unresolved findings justify it.

## Doctor

Use [DOCTOR-PROMPT.md](../../../docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md) and specification section 8. This is the explicit documentation-editing mode, not a product reviewer secretly changing the rules.

Inspect recent lessons and evidence. Fix verified gaps in the existing specification and directly affected instructions on a docs branch; prefer replacing/removing text. Do not change tests, workflows, runtime code, or thresholds. Show the diff and wait for owner approval before committing/pushing/merging. Keep version 1.0; Git supplies history. `--check` writes nothing. No supported gap means no edit.
