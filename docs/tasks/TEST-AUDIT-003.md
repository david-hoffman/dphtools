# Entire-suite test audit

## Authorization and candidate pointer

The owner requested a test audit of the entire suite and removal of as many
tests as possible by combination or deletion. This authorizes tests and their
supporting task evidence. Product, tooling, workflows, discovery, coverage policy,
and delivery instructions remain unchanged. No push, PR, merge, or publication
is requested.

Route: routine maintenance using the installed test-audit and implement-task
skills. Existing observable contracts and independent expected values are reused.
No new scientific contract or custom correctness oracle is introduced. Lost
meaningful numerical or release-safety proof is a blocker, not a deletion reason.
No new contract scenarios or specialist checkpoint are introduced.

Base: `04cbac9b59fa695218ddbafb6978529eafd8a2bc` (`origin/codex-main` at intake).
Branch: `codex/test-audit-suite`. This is the prepared task pointer. The single
Current state, exact final candidate identity, counts, commands/environment,
focused and required verification, review, blockers, next action, and metrics
are recorded in the conversation outside the tracked candidate. Raw logs and
detailed pre-edit ledgers remain outside Git. No spending cap was supplied.

## Audit and retained proof

The baseline collects 1,932 tests across a 61-module test inventory. Independent discovery
covers 26 numerical modules, 16 array/display/public-boundary modules, 12
release/doctor modules, and seven verification modules, plus used support.
Auditors inspect complete tests, their owners and non-test callers, history,
overlapping proof, and relevant approved public contracts before changing tests.
Externally usable library APIs remain supported even without in-repository callers.
The eight source/inherited doctests (binning, scaling, radial profile, mode, slices,
padding and parity, and prime factors) remain as executable public documentation. Removing
them would change product documentation beyond this tests-only audit.

Detailed pre-edit evidence:

- [Numerical ledger](/private/tmp/dphtools-test-audit-003-numerical.md).
- [Array/display ledger](/private/tmp/dphtools-test-audit-003-arrays.md).
- [Release/doctor ledger](/private/tmp/dphtools-test-audit-003-release.md).
- [Verification combinations](/private/tmp/dphtools-test-audit-003-verification.md).
- [Verification discovery](/private/tmp/dphtools-test-audit-003-verification-discovery.md).

Each candidate records its baseline location, actual detectable regression,
callers/history, stronger retained proof, editing limits, risk, and focused checks.
Numerical combinations preserve independent expectations, signs, parameter
regimes, error policies, tolerances, state fields, and successful/failed outcomes.
Repeated private ordinary ZTP roots are removed while the impractical near-one
sample regime remains. Legacy exponential sign and fast-decay cases move to
independently generated observations. Repeated LPSVD fits serve both retained
evaluators. Power-law seed replay moves into stronger sampling checks, and exact
duplicate fit/refit/control fixtures give way to fuller state/likelihood checks.

Release reductions remove fixture self-comparisons, existence-only probes,
weaker duplicate installation/recovery executions, and repeated unchanged
successful controls. Real installed-helper measurement, package structure,
credential stripping, transport adapters, intended guard diagnostics, retained
artifact identity, and recovery/finalization ordering remain. Successful helper
lookup and conflicting helper rejection remain in one lifecycle test.

Verification combinations reuse the first successful fast/full command for
directory uniqueness and stale-result rejection. All original receipt seals,
report bytes, command/cwd/interpreter observations, operation counts and
dependency edges remain. Distinct platform, nested-process, integrity, startup,
classification, measurement, streaming, and negative-guard scenarios remain.

No test module is removed or renamed: the approved ownership inventory is outside
scope. Empty inventory placeholders are avoided; retained modules contain useful
tests or used support, and the pre-existing display stub remains unchanged.
A test-count change does not by itself demonstrate runtime improvement.

## Validation and release impact

Freeze edits before running focused owner/sibling tests, targeted Black, and
`git diff --check`. Then classify/run the exact candidate with
`python tools/verification.py pr --base 04cbac9b59fa695218ddbafb6978529eafd8a2bc`.
Spanning domains and verification tests require Tier 5: canonical fresh full
verification and exact 100% owned statements/branches globally and per package,
including subprocesses and never-imported files. No exclusions, skips, weakened
thresholds, or omitted required domains count as success. A valid failing test
is classified and remains visible; this tests-only task cannot repair production.

Use a fresh independent routine reviewer on the exact frozen candidate and actual
results. Author repairs require renewed independent review. Host verification
does not establish the Linux/macOS/Windows merge matrix or native branch protection.
No merge is attempted here.

No version, public product behavior, or publication change. Draft release note:
redundant tests and repeated setup are consolidated while distinct behavior,
numerical regimes, and release/verification safeguards remain covered.
