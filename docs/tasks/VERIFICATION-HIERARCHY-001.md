# Verification hierarchy

Owner-approved infrastructure maintenance. This task starts at `a835a490373835fa01e4460ce2159365150d1946` on `origin/codex-main`. It replaces the abandoned full-suite optimization task. It does not reuse PR #21's fixture patch or its candidate evidence.

## Contract

The owner explicitly approved all six scenarios and the change from fresh global coverage on every PR to exact whole-domain coverage when that domain runs. The highest applicable trigger wins. Timing projections are planning evidence, not deadlines or achieved results.

| ID | Trigger | Required behavior |
| --- | --- | --- |
| S1 | Local iteration or eligible prose | Fresh formatting, critical lint, and docstrings; meaningful focused tests during development. Build inputs, executable examples, and delivery policy promote to stronger verification. |
| S2 | Understood library impact | Quality, types, audit, build, real clean installations of the invocation's exact wheel and source distribution, every library test/doctest, exact complete library coverage and report validation. |
| S3 | Doctor-only impact | S2 plus all doctor tests and the shared release CLI tests needed to measure the whole delivery entry. |
| S4 | Release machinery with unchanged shared infrastructure | Quality/preparation, every library and release test, shared doctor closure, real current-artifact installations, exact complete required-domain coverage and report validation. Relevant PRs run release tests without a tag. |
| S5 | Shared infrastructure or uncertain effects | Fresh complete canonical verification and exact global/per-package statements and branches. Verification, CI, coverage, packaging, requirements, manifests, README, Versioneer, shared fixtures/helpers, mixed domains, add/delete/rename, unknown files, unavailable/changed base, and unclear import/initialization/dependency/version/install/subprocess effects require this tier. |
| S6 | Release preparation | Complete required-platform full verification, frozen metadata/hashes, and real clean installations of the exact retained artifacts. Preserve main-only sources and separate manual preparation/publication approval. |

No scientific behavior, tolerance, dependency lock, coverage exclusion, required case removal, skip, evidence cache, publication, merge, remote runner, or Actions dispatch is authorized. Ordinary automatic PR CI remains separate platform evidence. Commands and ownership are in [LOCAL-VERIFICATION-CONTRACT](LOCAL-VERIFICATION-CONTRACT.md) and `tools/verification-domains.json`.

## Route and evidence plan

Routine author plus fresh independent reviewer. This changes verification cadence under an explicit existing correctness contract; it adds no numerical oracle or publication authority. Review actual behavior, ownership, expectation sources, coverage instrumentation, reporting, conservative fallback, scope sealing and CI aggregation. Repair the existing preparation reader/canonical-receipt mismatch within the same established full-report contract; current full and collection must clean-install their own build pair. No publication authority or approval behavior changes. This is format/verification integration maintenance, with the existing exact global report validator reused rather than a new correctness oracle.

Use this attached worktree only, branch `codex/verification-hierarchy`, with the existing locked `/private/tmp/dphtools-python13-isolated/bin/python`. Keep one local test worker; numerical-library and Black thread/worker limits remain 1. All commands run sequentially. Time command launch through final exit, including preparation, quality, build, installations, tests and reporting. Record the frozen interpreter/lock/resource/cache plan before source edits. Keep reports and every attempt under `reports/verification-hierarchy/`, outside Git. Preserve validated reports before deleting completed generated fixtures. Do not treat historical timing/coverage receipts as changed-candidate proof.

Historical baseline evidence is under the primary checkout's `reports/local-verification-performance/`; PR #21's retirement evidence is under `reports/storage-cleanup/retired-verification-pr-21/`. Three exact-baseline retained report sets have been revalidated by hash. A fresh local baseline full run and at least three final-candidate runs of each new selective command supplement those historical measurements. This S5 implementation requires its own fresh final-candidate full run. Linux/Windows local timings and complete end-to-end release preparation remain unavailable.

The inherited projections are 3–10 s plus focused tests for S1, 80–110 s for S2, 90–120 s for S3, and 15.7–16.0 min for S4. Historical complete full runs took 1280.941 s and 1306.662 s warm, 1311.454 s controlled tool cold. Changing cadence does not meet the abandoned 300 s complete-command objective. Report actual scoped totals and scope limits separately.

Release impact: no library API, numerical result, package dependency, destination or approval changes. Draft release notes: add complete domain verification commands and conservative PR planning; canonical verification installs its own wheel/source pair; preparation validates the current full receipt format. Changed release-helper bytes require a newly frozen bundle under the existing release policy. This task does not prepare or approve a publication bundle.

## Current state

- Approval: all six scenarios, scoped coverage/cadence, local implementation and a new PR against `codex-main` are authorized. No merge or publication.
- Route: routine infrastructure author; fresh independent review required.
- Candidate: branch `codex/verification-hierarchy`; the frozen commit and its own results will be recorded in the conversation/PR and external evidence, outside the candidate tree.
- Checkpoint: no specialist checkpoint; use approved contract and meaningful existing tests.
- Checks: frozen-candidate results, receipts, timing tables and limitations are recorded in [the external final evidence](../../reports/verification-hierarchy/final-results.json), outside this tracked pointer. Before freezing, baseline full passed 1,831 tests with exact complete owned coverage in 1,243.7 s; focused checks passed 278 tests before the last classifier repair, then 78 supported-syntax cases passed with complete classifier coverage. These preliminary results do not certify the frozen candidate. All attempts are retained outside Git.
- Review: one fresh independent reviewer identified six classifier/receipt findings; source repairs provisionally closed them. Exact-candidate acceptance and remaining platform limits are recorded with the external final evidence.
- Blockers: no implementation decision blocked. Required platform results and actual branch-protection readback are not established by local results; no merge is authorized.
- Next action: follow the external final evidence disposition. Opening and attaching a new PR requires the frozen candidate's full result, three runs per selective mode and independent review. Subsequent merge/publication still needs its own required evidence and authorization.
- Metrics: scenarios 6; author launches 2 (root plus policy drafting assistance), read-only analyst launches 1; reviewer launches 1; specialist A/B/C/D not applicable; no new time/dollar cap supplied; elapsed command time retained in external records; dollar/token metering unavailable.
