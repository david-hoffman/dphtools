# Phase 3 coverage ratchet evidence

Date: 2026-07-05
Repository: `david-hoffman/dphtools`

## Facts

- Test suite passes locally after issue #5 compatibility fix.
- Coverage baseline is recorded in `docs/agent-harness/coverage-baseline.json`.
- Total line coverage baseline: 15.56%.
- Total branch coverage baseline: 11.30%.
- Coverage non-decrease enforcement is enabled.
- Diff coverage enforcement is enabled for changed executable product lines under `dphtools/`.

## Commands

```bash
python -m pytest --cov=dphtools --cov-branch --cov-report=term-missing --cov-report=xml tests
python scripts/agent_harness/coverage_gate.py
python scripts/agent_harness/diff_coverage_gate.py --enabled
```

## Waiver Policy

Coverage decreases require a waiver note under:

```text
docs/agent-harness/coverage-waivers/<issue-number>.md
```

and the gate must be run with the matching `--waiver-issue` argument.

## Known Gaps

- Baseline coverage is low because large public modules have little or no direct test coverage.
- Diff coverage is line-based and depends on coverage.py XML plus a resolvable git base.
- Coverage.py may store filenames relative to package directories; the diff coverage gate normalizes those paths before comparing changed product lines.
- Mutation and property testing remain not enabled.
