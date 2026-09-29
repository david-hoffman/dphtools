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
Summarize or link the read-only inventory: existing failures, exact measured statement/branch coverage and scope, environment/tooling problems, and unresolved behavior. Identify the revision, commands, and environment; mark stale or blocked measurements. Separate delivery-tooling installation from product remediation, with effort and owner decisions. Expose any legacy baseline dependency that prevents a small slice reaching full verification and global 100% coverage; record the owner's adoption decision, including any approved larger slice. Installation approval alone does not authorize repairs or establish readiness.

## Verify and operate
Canonical full verification command and its build/lint/type/test/E2E/coverage steps, supported targets, dependency installation, test data setup, report locations, and recovery/release instructions. This command must pass on the exact candidate before opening/reopening a PR (draft included) or pushing an update to an open PR. Pre-PR pushes may back up failing checkpoints; CI repeats verification on configured platforms. Record the actual fresh-session and doctor invocation when installed. Do not invent tools.

## Approval and next slice
Reference the owner's actual approval of the identified record. Link the approved end-to-end slice plan and contracts, remaining decisions, and prerequisite tasks. Architecture approval is not task approval.
