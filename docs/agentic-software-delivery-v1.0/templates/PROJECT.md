# Project

**Version 1.0** Git versions revisions. Replace this template with real decisions; placeholders do not establish approval.

## Purpose and non-goals
Who uses it, the first useful outcome, and what is out of scope.

## Constraints
Deployment, integrations, data/access, compatibility, operational needs, budget, and unresolved material questions.

## Stack and structure
Detected versus proposed choices, evidence, responsibilities, and public boundaries. Keep few components. Link existing architecture rather than duplicating it. Optional diagram only when useful.

## Public interfaces
Link canonical schemas/contracts, including units, conventions, and estimator choices where applicable. Distinguish observed behavior from approved behavior and proposed changes. Record unresolved decisions before tests depend on them.

## Adoption baseline and scope
Summarize or link the read-only inventory: existing failures, exact measured coverage and scope, environment/tooling problems, and unresolved behavior. Identify the revision, commands, and environment; mark stale or blocked measurements. Separate delivery-tooling installation from product remediation, with effort and owner decisions. Expose any legacy baseline dependency that prevents a small slice satisfying full verification and the approved coverage policy; record the owner's adoption decision, including any approved larger slice. Installation approval alone does not authorize repairs or establish readiness.

## Coverage policy
Record the actual owner choice and reason after explaining specification section 5.1: approved behavior with independent risk review, optionally with selected line, statement, branch, or combined measured targets. Behavior coverage is the recommendation for ordinary application work, not assumed approval. Lines and statements are distinct metrics; 100% of a selected metric is valid. Identify the approval and committed local policy revision. For selected metrics, record exact tools/commands, thresholds, runtime/package/platform scope, aggregation, exclusions, and reporting limits. Default measurement scope is instrumentable owned runtime globally and per package on each required platform, including never-imported files and relevant subprocesses/server/browser code; narrower scope requires explicit approval. Mark unselected metrics advisory or unused. Tasks inherit the choice and identify relevant risks; do not repeat a settled interview.

Existing installed gates remain binding until the identified policy patch and its application receive fresh independent policy review and explicit owner approval. Record upstream provenance and local customizations. A policy edit does not change configured checks or establish success. Preserve active tasks' original gates unless the owner explicitly approves migration, retaining history, spending, attempts, rounds, and repair allowances.

## Verify and operate
Canonical focused/cheap and full reference commands, build/lint/type/test/E2E/selected coverage steps, supported targets, dependency installation, test data, report locations, and recovery/release instructions. Routine PR opening/reopening (draft included) and updates require meaningful focused plus cheap checks on the exact candidate; full local checks apply when the approved risk/check plan requires them. Merge requires independent review of the unchanged candidate, complete required platform verification, approved behavior/risk evidence, and every selected measured target. Inspect exact metrics and complete required reports separately per platform/package; unselected metrics are advisory and missing required measurement remains a gap. Rely on a stable aggregate such as `ci-required` only after actual native required-check protection is read back and verified; missing access/protection is a reported blocker. Preserve unrelated branch and release controls. Pre-PR backups may retain incomplete checkpoints. Record supported fresh-session/doctor commands and host constraints; do not invent tools or override mandatory host progress updates.

Record routine author/fresh-reviewer routing and the genuine scientific/safety risks requiring separate A/B/C/D. Reuse established contracts and narrow context. Keep one Current state record, logs outside Git, concise result/evidence pointers, and native CI completion/auto-merge only after protection and owner merge authorization.

For an installed release process, record the owner-selected source branch, version/channel policy, destinations, preparation/status commands, required platforms, and clean artifact-installation checks. Identify the protected publication environment, trusted workflow/registry identities, retention and partial-publication recovery steps, completion checks, and unapplied external setup. An owner request permits preparation; explicit publication approval follows full verification of the frozen bundle and is separate from merge approval. Publish retained files with verified hashes, never a rebuild. Do not infer a next version or release authorization from a merge.

## Approval and next slice
Reference the owner's actual approval of the identified record. Link the approved end-to-end slice plan and contracts, remaining decisions, and prerequisite tasks. Architecture approval is not task approval.
