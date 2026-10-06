# A narrow timing margin needs a repeat

- ID: 20261006T203831Z-TEST-SUITE-PERFORMANCE-001-author-timing-margin-repeat
- Date: 2026-10-06T20:38:31Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: A complete reviewed 299.668912-second hosted log enclosure was followed by a passing same-tree run whose verifier receipt alone took 300.159588 seconds.
- Evidence: [reviewed run](https://github.com/david-hoffman/dphtools/actions/runs/37523179737), [PR repeat](https://github.com/david-hoffman/dphtools/actions/runs/37525821216), retained receipt-timings.json and source-match.json under /private/tmp/dphtools-performance-ci-37525821216/. Every correctness gate passed; the second internal interval missed the performance contract independently of diagnostic shutdown.
- Lesson: Preserve the repeat as performance evidence and refresh stale advisory costs when worker tails diverge. Replaying fixed historical costs can motivate rebalancing, but cannot prove new hosted timing or bound future host variation.
