# Test audit: image splitting

## Current state

The owner requested the OpenClaw test-audit skill on a new branch from
`codex-main`. Branch `codex/test-audit` starts at
`3f7253cf330e98d9cfdd03047fea0e0586889123`. This is an authorized test-maintenance
audit using the requested skill's source-reading workflow; it does not change
product behavior or the installed delivery policy. The owner subsequently
authorized committing, pushing, opening a PR into `codex-main`, and merging
after CI passes. Final verification and independent-review results belong in
the conversation or linked PR, outside the candidate tree.

Metrics: one existing split/crop contract; two test functions / twelve collected
cases removed; no new scenarios; no A/B product-test checkpoint or C repair;
no spending cap supplied and no dollar metering available.

## Method and scope

[Requested skill](https://github.com/openclaw/openclaw/blob/main/.agents/skills/test-audit/SKILL.md).
Discovery was read-only and parallel across numerical, array/display, delivery,
and release tests. Select one coherent owner batch, not a subsystem campaign.
Use dphtools' pytest and canonical verification commands in place of OpenClaw's
Vitest/changed-check wrappers. Run the skill's independent autoreview after edits.

## Evidence recorded before editing

Locations below identify the baseline revision, not shifted post-edit lines.

| Candidate | Actual failure detection | Stronger retained proof |
| --- | --- | --- |
| `tests/test_utils.py:244::test_split_img` | Exceptions or wrong output shape for a 4096 x 1024 uninitialized image and 32 x 32 tiles; no pixel checks. | `tests/test_utils_baseline.py:214::test_split_tiles_preserve_values_without_assuming_tile_order` checks shape and independently enumerated tile contents on a rectangular image with rectangular tiles. |
| `tests/test_utils.py:262::test_split_img_random` (11 cases) | Exceptions during cropping/splitting; returned tiles are discarded. | `tests/test_utils_baseline.py:224::test_crop_for_split_retains_pixels_and_divisible_shape` checks crop dimensions, source pixels, and all split pixels; the 2-D and 3-D tile tests at lines 214 and 329 check independent tile contents. |

Both tests originated with the imported helper in `53a52a2` (2022-07-25), then
changed for rectangular inputs in `0585e8e`. They provided early shape/smoke
coverage. `acb291f` added the stronger baselines and repaired generalized N-D
splitting and square-grid reassembly. No historical performance or large-image
capacity contract was found for these probes.

Owners read in full: `dphtools/utils/__init__.py::split_img`,
`crop_image_for_split`, and sibling `combine_img`. No checked-in non-test callers
were found, including notebooks. These remain documented public library APIs in
`docs/tasks/SETUP-001-public-api.md`; no production deletion is justified.
The installed locked NumPy 2.2.6 `column_stack` implementation,
`ndarray.reshape`/`transpose` types, and their documented semantics were inspected.
The owned code has no large-size threshold or separate large-input path.

Deletion unlocked: the two tests, their `testdata` allocation, duplicate module
RNG initialization, and unused `split_img`/`crop_image_for_split` imports.
No production code or production seam is removed. The remaining split/combine
round trips separately protect reassembly, including rectangular and single tiles.

Risk: the obsolete parameter table advanced a shared random generator at import
time. Removing it changes deterministic inputs to other legacy utility tests.
Remove that coupling rather than preserving dummy draws; run the complete
utility files. Retained tests cover every split/crop statement and loop path;
compare measured line/branch sets before and after to verify this expectation.
Large-allocation smoke coverage is deliberately removed; it is not a benchmark.

## Validation plan and baseline

Use Python 3.10.21 in `.venv-delivery`, installed from `requirements-dev.lock`
with hash verification. Before editing, this focused command passed 100 tests:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dphtools-matplotlib .venv-delivery/bin/python -m pytest tests/test_utils.py tests/test_utils_baseline.py --cov=dphtools.utils --cov-report=json:/tmp/dphtools-utils-before.json --cov-fail-under=0 -q
```

Repeat with `dphtools-utils-after.json` and compare covered lines and branches.
The focused command's threshold override permits partial-suite measurement only;
the repository's full 100% statement/branch gate remains unchanged. Then run
`python tools/verification.py full` with the locked interpreter, targeted Black,
and `git diff --check`. The full command includes all fast checks. Inspect
`git diff --numstat` and report tests/support separately from production/tooling.
CI owns the additional macOS and Windows runs; Linux proof does not establish them.

## Retained false positives

- Split/combine round trips independently guard reconstruction. Independent tile
  content tests prevent compensating split/combine errors from being the sole oracle.
- The rolling-ball module process test protects its real executable entry point.
- The angle keyword regression protects the original public `mat_b=` parameter.
- The near-one ZTP scalar test covers a numerical regime that would require
  approximately 10^13 observations through the sample API; ordinary private-helper
  duplication does not justify removing this distinct resource boundary.
- Real release installation, transport, child measurement, and safety controls
  protect behavior beyond fixture self-checks. Doctor invocation and real Git hook
  tests likewise protect process and lifecycle contracts.

## Named follow-ups, not edits in this batch

- **Exponential fitting:** consolidate the two legacy `TestExponentFit` cases into
  the independent-data table. Preserve their fast-decay regime (`rate*x_max=30`,
  versus 3.75 in the existing independent case) before deleting the old module.
- **Ordinary ZTP roots:** review the mean-2/mean-20 private-kernel replay against
  the public conditional-likelihood tests; retain the distinct near-one case.
- **Verification fixtures:** remove assertions on fixture-authored coverage JSON
  and duplicate direct Flake8 probes where retained public verifier calls prove
  the same contract. Replace incidental total-order assertions with the existing
  inventory/dependency-order observer, preserving meaningful ordering.
- **Release fixtures:** review the structural fixture self-comparison and copied
  probe-file existence test against retained real manifest/probe execution proof.

Discovery is not a claim that every test in these areas has been cleared.
