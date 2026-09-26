# SETUP-001: Install and demonstrate the delivery workflow

**Version 1.0** Status: approved setup in progress; readiness unproven.

## Contract

- Project: [approved architecture baseline](../PROJECT.md). Canonical spec: `docs/agentic-software-delivery-v1.0/DELIVERY-SYSTEM-SPEC.md`.
- R1: Install four canonical Codex skills and native repository instructions; retain one live spec, root lessons, Version 1.0, product docs, and sound existing tooling.
- R2: Repair/adapt ordinary CI, lock dependencies, establish real public-entry-point tests and exact 100% statement/branch requirements globally and per package. Report failing/unsupported baseline measurement without weakening acceptance.
- R3: Provide `delivery doctor` as one native Codex invocation using `review-work` doctor mode and `DOCTOR-PROMPT.md`. Default mode produces only an evidenced docs patch, then waits for owner approval. `--check` reports without edits. No scheduler, controller, or custom permission enforcement.
- R4: Demonstrate genuinely fresh A/B, reviewed meaningful red tests and a Git checkpoint, fresh C, green checks and fresh D, native CI failure blocking and a normal passing PR where actually achieved. Missing evidence remains a gap, not a simulated success.
- R5: Preserve required sources once, append an honest lesson, and demonstrate a real doctor spec edit. Doctor's exact patch requires approval before commit/push/merge.
- Non-goals: product features, stack migration, stronger-than-prompt restrictions, automatic releases, or fabricated coverage/session evidence.
- Allowed setup changes: delivery docs/skills, root instructions/lessons, minimal launcher, tests and CI/dependency infrastructure. Product defects or undefined scientific behavior return to scoped intake; no speculative fixes.
- A/B use only [the doctor contract](SETUP-001-contract.md), selected skill, authorized tests/fixtures, and test conventions. This execution record is not blind-role input.
- Budget: unlimited preparation/execution, explicitly approved by owner; failed attempts count in reported work. No reliable dollar metering. At most one implementation repair by default. Stop for a material owner decision or an external blocker.

## Owner approval

On 2026-09-26 the owner replied to the exact five-step setup plan and architecture baseline: “I approve this plan, but with unlimited budget, only come back to me with a real blocker to solve.” This file records that approval, rather than claiming its later bytes or future numerical requirements had been separately approved. Doctor's narrow command interface below is a setup implementation choice within the requested default/check behavior. Material scope changes still require intake.

## Execution record

- Starting branch: `codex/agentic-software-delivery`; product baseline `c81ffdf`.
- Read-only discovery: canonical spec read once; no approved prior architecture found. Existing Python architecture routed through intake and owner approval above. Empty-project route is documented, not an executed empty-project demonstration.
- Initial local CI contains duplicate `run` keys; no current coverage invocation or threshold. Prior ignored XML reports 275/1761 lines and 47/416 branches; this is stale evidence, not the setup baseline.
- GitHub read on 2026-09-26: public `david-hoffman/dphtools`, default `main`, admin access available. Existing protection requires `ci-required`, current branches, one approval, code-owner review, last-push approval, conversation resolution, and administrator enforcement; force pushes/deletion disabled. No settings changed during discovery.
- Role/checkpoint evidence is recorded below as executed. No passing whole-repository candidate, PR, or real doctor demonstration is claimed until its actual evidence is recorded.

### Approved compatibility maintenance

The owner additionally approved “compatibility maintenance and baseline tests” on 2026-09-26 after being shown 15 failures with current dependencies. Scope: preserve public APIs and numerical intent; repair compatibility and add tests for existing documented behavior; no new features. Ambiguous scientific contracts still return to intake.

Fresh current-environment baseline: 28 passed, 15 failed. Twelve failures reach removed `np.product`; three are NumPy scalar representations in doctests. The earlier Python 3.10/SciPy 1.15.3 attempt failed collection because macOS 27 rejected a binary extension; it is environment failure, not meaningful red evidence. Formatting, critical lint, and docstrings passed; type checks found existing errors. Build produced a wheel and source distribution without publishing.

Role A started as root Codex session `01a0dfe9-f17c-7aa3-8bf9-b783349a94ae`, with memory/delegation disabled. It paused for a CLI-feature-name clarification. Native queueing did not unblock the active process; the process was interrupted and the same A role resumed with verified flags, without changing its blind inputs. It authored 37 command cases and reported 37 expected feature-absence failures, no fixture errors/skips. This is new-helper absence, not a historical product bug. Final A turn reported 410,763 input tokens (368,768 cached), 19,886 output tokens; no reliable dollar total is available. The initial interrupted attempt also consumed resources not totaled here.

### Reviewed doctor test checkpoint

- B was fresh root session `01a0dff5-057c-7292-8ea6-39e5f1bf6329`. It requested four corrections: default-mode intent, unsupported transport flags, unreadable regular prompt files, and valid punctuation after prompt paths. A corrected its tests within the same role; B rereviewed and accepted. The formatter changed layout before B's acceptance. These same-role continuations are not claimed as additional independent sessions.
- Test checkpoint: `80ff2a1` (`tests/test_delivery.py`, 39 cases), committed before implementation. A, B, and the coordinator all observed 39 scaffold failures with no fixture errors or skips. The root rerun used `.venv/bin/python -m pytest -q tests/test_delivery.py --tb=no`. Windows fixture execution is still pending CI.
- B's initially read-only sandbox could not create test temporary files; the authorized test command obtained temporary-file access. This is not a claim of custom filesystem isolation. A/B received no implementation/history/lesson inputs.
- Latest native usage reports: A correction turn reported 807,973 input / 28,852 output tokens; B rereview reported 391,063 input / 7,759 output tokens. These are raw harness-reported counts, not a dollar estimate or an asserted sum across attempts.
- The supplemental library A session is independently fresh: `01a0e001-1b12-7c53-88b7-c2a5565951dc`. It receives only the approved library packet, extracted public signatures/docstrings, and existing tests. Its tests cannot be called original test-first evidence for pre-existing code.

### Verification infrastructure evidence

- Black 26.5.1 and uv 0.12.19 replace versions flagged by `pip-audit`. The final universal hashed lock audit reported no known vulnerabilities. A provisional pip-tools lock omitted Windows-only dependencies/build-tool hashes; it was replaced before adoption with one uv-generated universal lock. No competing lock or locking tool is installed as repository policy.
- The baseline report now includes previously blanket-excluded handwritten branches and explicitly names never-imported owned files. It reports 434/1817 statements and 57/432 branches, including the three-line unimplemented launcher. Generated `_version.py` is omitted at measurement and reporting. The 100% gate fails as intended; none of these values establish readiness.
- A copied-script coverage probe confirmed that subprocess/copy paths combine back to the canonical helper. The probe only establishes measurement plumbing, not helper functionality or a green product suite.
- All 20 retained archive/source/license/rendered files match their manifest hashes in both the working tree and Git after commit `3b2131f`. Lesson `2026-09-26-SETUP-001-setup-archive-normalization` records the observed failure that motivated the fix.
- Seven owned-code type-check errors remain after recognizing SciPy's dynamic import boundary as a documented static-checking gap. No product/type fixes have been claimed yet.
- Actionlint 1.7.12 validated `.github/workflows/ci.yml` with no findings. Its downloaded Darwin ARM64 binary matched the published SHA-256 `aba9ced2dee8d27fecca3dc7feb1a7f9a52caefa1eb46f3271ea66b6e0e6953f`; the binary/log remain local ignored diagnostics, not additional repository infrastructure. This validates workflow syntax, not a remote passing run.
- Supplemental library A disclosed that one black-box signal-analysis probe printed four implementation lines through Python warning formatting. It did not deliberately open source; later probes removed warning source excerpts. This limits the supplemental session's blindness and must not be described as perfect isolation. The separate doctor A/B evidence above is unaffected.

### Supplemental test review and pre-implementation baseline

- A initially delivered 196 supplemental cases: 178 passed, 18 failed, no skips/xfails. Its isolated library measurement was 1252/1814 statements and 246/432 branches; this was not a whole-repository gate run. The raw final author-turn usage was 2,814,661 input / 48,283 output tokens. The same-role doctest continuation reported 3,271,245 input / 53,168 output tokens; these reports may be cumulative and are not summed as cost.
- Fresh B session `01a0e01d-78bc-7ce1-bba2-43e30a35914d` independently reproduced 178 passes and 18 failures. It required corrections to an unspecified zero-truncated-Poisson estimator assumption, undefined fitted-uncertainty expectations, and a background-filter test that permitted a no-op result. It also suggested a nonconstant zero-sigma Gaussian boundary. Corrections returned to A; supplemental acceptance/checkpoint remains pending.
- B accepted seven representation-independent doctest comparisons in `scale`, `mode`, and `slice_maker`, and the formatter's single blank-line removal in `tests/test_fitfuncs.py`. A's extracted doctest run passed all 12 steps after correction; no product implementation change was used to satisfy them.
- The coordinator ran all proposed tests and doctests before implementation: 209 passed, 69 failed, no skips, 30 warnings. The failures include 39 intended doctor-scaffold failures, 12 existing tiling compatibility cases, and 18 supplemental cases (one subsequently returned to contract intake). Canonical pre-C coverage measured 1270/1817 statements and 256/432 branches: 547 statements and 176 branches missing. The exact 100% gate failed. Local evidence: `.delivery-runs/pre-c-tests.txt`, `reports/pre-c-pytest.xml`, `reports/pre-c-coverage.json`.
- Critical lint and docstring checks pass for this provisional state. Black identifies only the known extra blank line in `dphtools/__init__.py`; it has not yet been changed. There is still no passing whole-repository candidate.
- B identified missing documented solver capabilities separately from dependency replacements. A scope decision was sent to the owner for `pyls`, vector/matrix weighted `ls`, numerical Jacobians, row-oriented derivatives, and verified mathematical/API repairs. The original compatibility approval is not recorded as approval of that pending extension.
