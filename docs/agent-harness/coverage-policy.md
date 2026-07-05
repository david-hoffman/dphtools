# Coverage Policy

## Current Phase

Phase 0 records coverage where possible. It does not fail on legacy gaps.

## Definitions

- Line coverage: executable lines run by tests.
- Branch coverage: meaningful branches tested.
- Diff coverage: changed executable lines covered by tests.
- Behavior coverage: public behavior asserted, not merely executed.

## Staged Policy

- Phase 0: record baseline where possible.
- Phase 1: fail if coverage decreases without a linked waiver issue.
- Phase 2: require diff coverage for changed product code after tooling is stable.
- Phase 3: ratchet total coverage toward a documented target.

## Exclusions

Coverage exclusions must be explicit and justified. Existing coverage configuration omits Versioneer-generated `_version.py`.

## Baseline

`docs/agent-harness/coverage-baseline.json` is initialized with unknown values. Run `make coverage` after installing development dependencies to produce local measurements. Updating the committed baseline requires an explicit coverage-baseline issue.
