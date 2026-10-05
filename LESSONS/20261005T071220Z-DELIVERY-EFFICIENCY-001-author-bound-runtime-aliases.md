# Bind internal runtime aliases without admitting external inputs

- ID: 20261005T071220Z-DELIVERY-EFFICIENCY-001-author-bound-runtime-aliases
- Date: 2026-10-05T07:12:20Z
- Task/role: DELIVERY-EFFICIENCY-001 / author
- Status: confirmed
- Observation: Rejecting every directory symlink made conservative reuse ineligible for standard Python layouts. CPython's macOS framework installs Headers as an alias of include/python3.10; ordinary Linux virtual environments also create lib64 aliases.
- Evidence: CI run 74's macOS shard reported Runtime imports cannot be completely identified. The [CPython 3.10.11 install rule](https://github.com/python/cpython/blob/v3.10.11/Makefile.pre.in#L1789) creates the framework alias. The bounded-alias tests first failed and then passed, including genuine venv CLI reuse while retaining lib64 and an internal Headers alias.
- Lesson: Permit an alias only when its resolved target stays inside a completely hashed canonical runtime prefix. Bind the alias path and target as well as the target bytes; changing bytes, renaming, or same-content retargeting must invalidate identity, and external aliases must still decline reuse.
