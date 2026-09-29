---
name: review-work
description: Review tests (B), review a passing candidate (D), or improve the spec from evidence (doctor). Use one mode per fresh session.
metadata:
  version: "1.0"
  status: released
---

# Review work

## Tests — B

Use A's approved behavior/public-interface inputs and proposed tests. Do not read implementation, its history/conversation, or LESSONS.md.

Check contract clarity, real observable assertions, independent expected values, negative/boundary behavior, and requirement coverage. Prefer meaningful E2E tests over mock-only or duplicate unit tests. Name a plausible wrong behavior the suite should detect. Return corrections to A. Do not implement or silently choose missing requirements.

Accepted tests still need meaningful failing evidence and a recorded Git test checkpoint before C.

## Candidate — D

Read the approved task, exact candidate, test checkpoint, and actual check evidence, not C's conversation. Form findings before consulting current-task implementation lessons.

Inspect behavior, security, public boundaries, and test/workflow changes. Compare the tested commit with the proposed merge. Missing/skipped/incomplete results are not success. Check the real product result, not only a summary or compilation.

Include simplification in this review: unnecessary wrappers, duplication, dependencies, and speculative features. Stay within the task. No extra simplification agent. Report concrete findings and evidence; do not fix the candidate and then approve it. Corrections use fresh C and renewed checks/D.

## Doctor

Use [DOCTOR-PROMPT.md](../../../docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md) and specification section 8. This is the explicit documentation-editing mode, not a product reviewer secretly changing the rules.

Inspect recent lessons and evidence. Fix verified gaps in the existing specification and directly affected instructions on a docs branch; prefer replacing/removing text. Do not change tests, workflows, runtime code, or thresholds. Show the diff and wait for owner approval before committing/pushing/merging. Keep version 1.0; Git supplies history. `--check` writes nothing. No supported gap means no edit.

## Every mode

Keep results short and evidence-linked. Stop on material ambiguity or budget exhaustion. Append useful findings to LESSONS.md, except blind mode must append without reading or return the entry for someone else to append. Do not claim these prompt restrictions are mechanically enforced.
