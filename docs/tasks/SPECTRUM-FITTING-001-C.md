# SPECTRUM-FITTING-001 — C packet

Role: fresh root specialist C, .agents/skills/implement-task/SKILL.md.
Owner approved public contract R3 and S01-S33 as one slice on 2026-10-05.
Public behavior: [contract](SPECTRUM-FITTING-001-contract.md).
Architecture/check commands: docs/PROJECT.md and root AGENTS.md.

Reviewed test checkpoint: 1f9050ccd1d7e6b0ae86fd9eabc3600fcf29a268.
Tests SHA-256: 4f762faf63839e4301038a91d02abee9b2f6c8748dc671d19a48aec1ffb674d2.
Mapping/report SHA-256: 0f9e2b90b7f6d96a016073bc2dd0807c3888a9d257f781c568d097f221d23674.
B2 accepted all behavior/oracles/tolerances; B3 accepted Black-only correction
with exact AST equality. Original window2/2 and format window1/2 are closed;
prior attempts/spending remain retained. The public tests/report are permitted;
do not load A/B private conversations or logs.

Approved runtime baseline: a835a490373835fa01e4460ce2159365150d1946.
Accepted tests on unchanged runtime collect263 cases: 1 failure/262 errors,
exit1, because the new public spectrum_fit entry point is absent. This is feature
absence, not an existing bug reproduction. No numerical assertion was reached.
Baseline evidence: /private/tmp/dphtools-spectrum-delivery/accepted-baseline.log
and historical public test report; B3 reproduced the formatted baseline.

Allowed edits: additive spectrum_fit at dphtools.utils.fitfuncs, new fitting
helpers/modules under dphtools/utils if needed, and their docstrings. Preserve
all existing public signatures/behavior. No tests/fixtures/snapshots, workflows,
coverage/discovery/type settings, dependencies, skills, delivery rules, task
records, or unrelated product interfaces may change. Do not add test-only seams.
No Superpowers, optional memory, or delegation. No push, PR, merge, or release.

Use the isolated locked interpreter:
/private/tmp/dphtools-spectrum-delivery/venv/bin/python
Python3.13.12; existing NumPy2.5.3/SciPy1.18.1. It passed canonical preflight,
including real nested installation. Never install into the shared primary .venv.
Use MPLBACKEND=Agg and MPLCONFIGDIR=/private/tmp/dphtools-spectrum-delivery/mplconfig.

Implement the smallest approved change, then meaningful focused tests and fast.
Commit only authorized product paths with normal hooks before full so the exact
candidate is immutable. Set DPHTOOLS_PYTHON to the isolated interpreter for Git.
Run canonical full on that commit:

```sh
MPLCONFIGDIR=/private/tmp/dphtools-spectrum-delivery/mplconfig /private/tmp/dphtools-spectrum-delivery/venv/bin/python tools/verification.py full
```

Retain full log at /private/tmp/dphtools-spectrum-delivery/C-full.log and verifier
receipts. No exclusions/skips/check weakening. Require exact global/per-file100%
owned statements/branches, including child processes and never-imported files.
If a check fails, classify environment/tooling, test defect, product defect, or
unresolved requirement before repair. Test defects/coverage gaps return to the
coordinator; C may not modify tests or evade a valid check. An unclassified cause
requires bounded diagnosis. Do not repeatedly run expensive full checks when a
focused failure already blocks them.

This is initial C. One subsequent fresh C repair is available; no new time/token
cap was supplied, report measurable usage and unavailable billing. Preserve
allowances. Finish with candidate hash/changed paths, exact focused/fast/full
commands/results/coverage, unchanged test-hash confirmation, and any classified
blocker. You cannot approve your own candidate. D/platform checks remain pending.
