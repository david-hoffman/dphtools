---
name: review-work
description: Independently review a routine candidate, specialist tests (B), specialist candidate (D), or delivery docs (doctor).
metadata:
  version: "1.0"
  status: released
---

# Review work

Follow installed repository `AGENTS.md` for routing, shared session, access, verification, budget, and reporting rules. Use one mode per narrow packet. Routine candidate review needs one fresh independent session; A/B/C/D are reserved for genuine scientific or safety risk.

## Tests — B

Use A's approved behavior/public-interface inputs, approved coverage policy, relevant risks, and proposed tests.

Check contract clarity, expected-result sources, and the scenario-to-test mapping in the task's Contract table. Review custom oracle logic, including parsers, observers, units, and allowed outcomes. Check A's tolerance rationale in both directions: legitimate numerical variation must pass and plausible wrong values must fail. Do not normalize malformed output into validity or impose an undocumented choice.

Independently audit missing approved outcomes and relevant high-impact risks, including valid inputs, reachable failures, state changes, and boundaries. Record a short requirement/test and risk disposition. A's mapping alone does not establish adequacy. Accepted tests are the reviewed baseline, not proof of every possible behavior. A bare uncovered line or branch is not a test defect; require the classified public-contract omission and why current tests miss it before commissioning a correction. Blind packets must exclude implementation reports and coverage-line maps.

Reject tests that merely execute code to raise coverage, negative cases that pass through an unrelated rejection, fixtures that supply the purported production result, and names claiming unexercised behavior. Require a meaningful correctness check, not a particular assertion syntax. Behavior-preserving refactors should not break tests unless the supposed refactor violates an independently approved interface/structural contract.

Complete B with acceptance of an identified test revision/mapping or concrete defects/contract gaps. After the second nonacceptance in a review window, diagnose the blocker: send a specific ambiguity and recommended clarification to intake, or propose genuine end-to-end split points for excessive scope. Confirm the diagnosed problem is resolved before restarting rounds under specification section 9; otherwise report the blocker. Do not add another reviewer or model-escalation path.

Accepted tests need valid baseline evidence and a recorded checkpoint before C. Existing-code tests may pass initially without mutation testing or manufactured failure. Classify observed failures under specification section 4; a setup failure is not product-red evidence.

## Candidate — routine reviewer or specialist D

Read approved scope/established contracts, the exact candidate, and actual applicable check evidence, not the author's conversation. Specialist D also receives the reviewed test checkpoint. Form findings before consulting current-task implementation lessons.

Inspect behavior, security, public boundaries, and test/workflow changes. Audit concrete missed approved outcomes and reachable high-impact risks beyond the accepted baseline; record the short risk disposition. Compare the tested commit with the proposed merge. Missing/skipped/incomplete required results are not success. Check approved behavior/risk evidence and every selected measured target under `docs/PROJECT.md`, including exact metrics, scope, exclusions, limits, and complete required reports separately per platform/package. Never union platforms to hide a gap. Unselected metrics cannot become acceptance gates; an unmet selected target still blocks. Check the real product result, not only a summary or compilation. Flag production seams created solely for tests; investigate test-only local callers without treating externally used public APIs as dead code.

Classify coverage findings under specification section 5.2 before correction: missing behavior/risk tests, unnecessary complexity, unresolved semantics, measurement defects, or policy conflicts. State the missing approved outcome or concrete reachable risk, evidence, and why tests miss it before commissioning test work. Specialist corrections use fresh blind A/B; routine corrections use the authorized author and fresh independent review. Give blind A/B a source-free public-contract reproduction, never copied implementation reports or coverage-line maps; preserve supplemental labels. Route safe simplification to its authorized author, new meaning to intake, and measurement defects to authorized infrastructure work. An irreducible selected-target conflict needs an owner policy decision. Policy changes require fresh independent policy review and explicit owner approval of the identified patch before activation; active-task migration must be explicit and retains history, spending, rounds, and repair allowances. Diagnosis does not establish readiness or waive a failing selected target.

Include simplification in this review: unnecessary wrappers, duplication, dependencies, speculative features, and redundant guards or control flow added solely for a percentage. Required validation and real invariants remain obligations. Stay within the task. No extra simplification agent. Complete review with acceptance of the exact candidate under the applicable checks or concrete findings and evidence; identify pending platform CI rather than calling it passed. Do not fix the candidate and then approve it. Route findings under specification section 4. Routine corrections return to the authorized author and need renewed independent review; specialist corrections use fresh C within the recorded allowance and fresh D. Reuse applicable evidence; rerun when changed inputs or unresolved findings justify it. Confirm full Linux/macOS/Windows results and verified `ci-required` protection before merge; workflow text or local success cannot establish that gate.

## Doctor

Use [DOCTOR-PROMPT.md](../../../docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md) and specification section 8. This is the explicit documentation-editing mode, not a product reviewer secretly changing the rules.

Inspect recent lessons, task metrics, and evidence. Complete with an evidenced patch to the existing instructions or a report of no supported gap; `--check` writes nothing. Scenario counts and outcomes may support an owner-approved change to default slice size. Doctor cannot silently change an agreed coverage target or make an in-flight task use new rules; coverage-policy changes need independent policy review and explicit owner patch approval under specification section 5.1. Runtime, test, workflow, and configured gate changes require separate authorized scope. Follow the dedicated prompt's edit/approval limits and keep Version 1.0.
