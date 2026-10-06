# Nested quality commands multiply process pools

- ID: 20261006T195148Z-TEST-SUITE-PERFORMANCE-001-author-nested-quality-pools
- Date: 2026-10-06T19:51:48Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: Locked Black and Flake8 independently size process pools to available CPUs, even when launched inside bounded test workers. Bounding test workers alone does not bound nested quality processes.
- Evidence: Locked Black 26.5.1 concurrency implementation and Flake8 7.4.1 checker implementation; `/private/tmp/dphtools-quality-paired-s5zvc5xf/report.json` records three native covered real-fast pairs with unchanged fresh runtime identity. All six pass the original 60-second deadline; median wall time changes from 2.109 to 1.910 seconds and coverage families from 26 to four. Original CLI checks and `/private/tmp/dphtools-performance-quality-bound-real-fast/checks.json` pass explicit `--workers 1` / `--jobs 1`.
- Lesson: Bound inner quality pools explicitly when the outer test layer already parallelizes. Native evidence does not identify hosted timeout causes or promise hosted savings; real preflight creation does not execute these quality tools.
