# Coverage Policy

## Current Phase

Phase 3 is enabled. Total line and branch coverage may not decrease below the committed baseline without a waiver issue.

## Definitions

- Line coverage: executable lines run by tests.
- Branch coverage: meaningful branches tested.
- Diff coverage: changed executable lines covered by tests.
- Behavior coverage: public behavior asserted, not merely executed.

## Staged Policy

- Phase 0: record baseline where possible.
- Phase 1: fail if coverage decreases without a linked waiver issue.
- Phase 2: require clean-context metadata for product source changes.
- Phase 3: require non-decreasing total coverage and diff coverage for changed product code.

## Exclusions

Coverage exclusions must be explicit and justified. Existing coverage configuration omits Versioneer-generated `_version.py`.

## Baseline

`docs/agent-harness/coverage-baseline.json` records the current total line and branch coverage baseline.

Current baseline:

- Line coverage: 15.56%
- Branch coverage: 11.30%

Updating the committed baseline requires an explicit coverage-baseline issue.

## Diff Coverage

`scripts/agent_harness/diff_coverage_gate.py --enabled` checks changed product lines under `dphtools/` against `coverage.xml`. Changed executable lines must be covered unless a waiver issue is recorded.
