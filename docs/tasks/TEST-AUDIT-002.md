# Test audit: verification

## Authorization and candidate pointer

The owner authorized adding the linked OpenClaw test-audit workflow to ASD,
running separate audit and repair agents until complete, and opening a PR.
Non-test code edits are forbidden. Instruction changes are the supporting skill
and ASD pointers; runtime, tooling, workflows, discovery, and coverage policy
remain outside repair scope. No merge or publication is authorized.

Route: routine maintenance under the established local verification contract.
The two discovery lanes and separate author produced the test repairs below.
A fresh independent reviewer must inspect the candidate and applicable evidence.
No new scenario/behavior contract or specialist checkpoint is introduced.

Base: `b995f5843edbb9247cdb3d9c0b1b08c97010e02a` on `codex-main`.
Branch: `codex/test-audit-verification`. This is the prepared task pointer:
the authoritative Current state, final candidate identity, commands/environment,
focused and required results, independent review, platform status, blockers,
next action, and launch/budget metrics are recorded in the conversation or
linked PR, outside this tracked candidate. This document preserves approved
scope and pre-edit evidence rather than a competing mutable status copy.

The owner supplied no spending cap; the routine review/repair loop runs until
acceptance or a real blocker. Specialist allowances are not reset or used.
Skill frontmatter validation passed using a disposable YAML dependency without
changing the locked verification environment. A confirmed metrics product defect
is explicitly deferred below because runtime code repair is forbidden.

## Scope and method

Apply [.agents/skills/test-audit](../../.agents/skills/test-audit/SKILL.md).
Upstream text was read at `e09f3ecd491db23f4fdc6e79b4349808d010f394`
(SHA-256 `fc38d2a34ade41960355d951b018dea1a0dfc89faca28914ba2c9a45d007fa3f`).
The original inert archive and R10 setup provenance remain unchanged. This is
a local adaptation; OpenClaw campaign/Node/landing skills are not installed.

The coherent batch is `tests/test_verification*.py` and directly used test
support, including [TEST-AUDIT-001's named follow-ups](TEST-AUDIT-001.md#named-follow-ups-not-edits-in-this-batch).
Auditors inspected complete tests, runtime owners/entry points/callers/callees,
overlaps, CI routing, applicable installed dependency sources/types, and history.
Source reading was confined to routine discovery; blind A/B restrictions remain.

Full pre-edit candidate fields and retained false positives are recorded outside
Git in [core evidence](/private/tmp/dphtools-test-audit-002-core.md) and
[boundary evidence](/private/tmp/dphtools-test-audit-002-boundaries.md).
Locations below refer to the base, not shifted candidate lines.

| Finding | Evidence and authorized test repair | Retained proof |
|---|---|---|
| Fixture coverage rereads | `test_verification.py:515` checks facts authored by its external-tool fixture. Remove redundant rereads. | Real coverage instrumentation, never-imported files and all invalid-report cases. |
| Duplicate lint launches | `test_verification.py:908` directly reruns the same dependency already reached by the real fast command. Remove direct probes. | Public fast success and precise F821/file rejection. |
| Incidental total order | `test_verification.py:503,702,781,1200` fixes independent operations to one sequence. Reuse inventory/dependency observations. | No missing/duplicate operations; quality/build/install/test/report prerequisites; receipts matching actual calls; live streaming. |
| Vacuous identity assertion | `test_verification.py:1392` OR is already forced true by changed tags. Remove it and its unused local. | Actual Git/tag/history/dirty/version/artifact observations and changed-input reuse proof. |
| Foreign identity masked by seal | `test_verification_shards.py:283` changes run/platform but retains a stale digest. Reseal to reach semantic identity rejection. | Separate byte-tamper cases and explicit intended-guard diagnostics. |
| Duplicate-index case masks its claim | `test_verification_shards.py:530` reaches foreign execution first. Remove the redundant parameter/branch. | Real CLI duplicate-directory rejection and remaining semantic/execution cases. |
| Invalid inventory masked by path fallback | `test_verification_scope.py:200` changes the inventory file itself. Establish invalid inventory in fixture base, then change only prose and require disclosed inventory failure. | Valid-inventory prose control and all closure/fallback cases. |
| Dynamic prefix masked by new call | `test_verification_scope.py:293` introduces an unsupported call. Change only a literal while retaining unchanged helper/hook prefixes. | Existing ordinary literal control and unsupported-call tests. |
| Dead support | Unused hook full observer branch, shard resealer and scope imports. Reuse support where meaningful; remove the remainder. | Every hook process/platform/backup test and complete receipt/measurement cases. |

Real installation, receipt reuse, Versioneer, lock/startup bytes, hook platform
behavior, sharded parent/child coverage, lifecycle, report integrity, and metrics
parser tests stay. A suspicious pattern alone does not justify deletion.
No runtime seam is removed; no threshold, exclusion, skip, or discovery rule changes.

## Deferred product issue

At the base, `tools/verification_metrics.py:88` accepts old receipt modes but
rejects `library`, `doctor`, and `release`. Minimal temporary orchestration
fixtures invoked the unchanged public verifier for each mode. Each returned 0
with a complete passed version 2.0 receipt and the requested actual mode.
The unchanged metrics CLI rejected each with exit 1 and
`metrics: Unknown receipt mode`.

[Probe results](/private/tmp/dphtools-test-audit-002-metrics-probe/results.json)
retain exact receipts and logs. These controlled external-tool fixtures confirm
CLI/receipt compatibility only; they are not real product coverage or a successful
domain/full gate. No branch test is removed or introduced to conceal this issue.
Repairing that runtime code requires a separately authorized task.

## Validation and release impact

Run focused modified-owner/sibling tests, targeted Black and `git diff --check`,
then `python tools/verification.py pr --base origin/codex-main` on the exact
committed candidate. Added instruction/task files and verification changes
require Tier 5: fresh full verification, exact 100% statements/branches globally
and per package. Hosted Linux/macOS/Windows evidence stays pending until observed.

No package version, public behavior, or publication change. Draft release note:
verification tests retain stronger process proof, isolate intended rejection
guards, and ASD gains a supporting test-audit workflow.
