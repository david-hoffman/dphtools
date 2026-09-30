# Setup prompt

**Version 1.0** Supply the whole package to a fresh coding-harness session at the target monorepo, then use this prompt. Nothing here claims that a `delivery` command is already installed.

```text
Set up the Agentic Software Delivery System described in DELIVERY-SYSTEM-SPEC.md.
Read that file once. It supersedes the earlier conversation drafts.

Use ONE monorepo, existing coding-harness sessions, four skills, ordinary GitHub
Actions, and Git. Do not build a custom controller, separate control repository,
GitHub App, immutable test store, permission enforcement, or agent swarm.
Restrictions on editing reviewed tests/workflows and append-only lessons are prompts.

Start read-only. Inventory existing failures, exact measured statement/branch coverage,
environment/tooling problems, and unresolved behavior before proposing remediation.
Identify the revision, environment, and commands; label stale evidence and blocked
measurement. Use existing checks in an isolated usable environment without changing
product/tooling files or running setup scripts for discovery. Preserve sound tooling.
Find usable approved architecture; otherwise route to architecture intake. An empty
repo needs an owner-approved minimal project record before choosing a stack. Existing
code is evidence, not automatically the intended behavior. Repeated setup must not
overwrite working choices. Do not implement product features during setup.

Show the small setup plan and obtain approval. Separate delivery-tooling installation
from existing-product remediation, with effort, measurement gaps, and owner decisions.
Installation approval does not authorize product repairs or resolve missing behavior.
Expose legacy baseline dependencies that make small slices unable to pass the full
gate, including global 100% coverage. Let the owner choose adoption scope case by case,
including an explicitly larger slice where needed; do not weaken readiness criteria.
Adapt paths and instructions without creating two live specs or overwriting the
product README. Create root LESSONS/ with a README.md format guide for one timestamped
Markdown file per lesson, following specification section 7. Install the four skills in the
selected harness's supported location, with
one canonical copy of each. Generate or reconcile AGENTS.md as the single operational
home for shared instructions; skills contain role-specific differences. Add a thin
native bridge only when required. Record actual launch and check commands.

Infer languages/frameworks. Research suitable native formatting/lint/type/test tools.
Create or adapt ordinary CI and minimal test infrastructure as authorized setup work.
Prefer real end-to-end/public-entry-point tests, smaller tests only for useful gaps.
Retain 100% measured statement/branch coverage and report unsupported measurement.
Use GITHUB-SETUP.md; apply settings only with permission or give exact owner actions.

Implement delivery doctor as a thin invocation of review-work in doctor mode using
DOCTOR-PROMPT.md. No daemon or scheduled LLM loop. Default behavior writes an evidenced
spec/instruction patch on a docs branch, then waits for owner approval; --check only
reports. Keep every delivery document at version 1.0. Git versions edits.

Use spec sections 3–4 and the task template for the jointly approved slice plan,
five-scenario default, fresh root roles, numerical oracle review, initially passing
existing-code tests, and B's diagnosis after two unaccepted reviews. Independent tasks
use separate worktrees; dependent slices wait for completed, integrated prerequisites.
Keep the scenario/round/repair/spend line in Current state under section 9; preserve
history and consumed budget/repairs when work is renamed or split.

Demonstrate valid baseline evidence and a reviewed test checkpoint, fresh C,
full local verification passing on the exact candidate, fresh D, and a normal PR.
Apply section 4's gate before opening/reopening a PR (drafts included) or pushing an
update to an open PR. Pre-PR pushes may back up failing checkpoints. Use fast generic
hooks and the full command at submission, not a custom controller or backup branch.
Known failures, including incomplete coverage, block submission.
CI repeats verification on its configured platforms. Inspect native protections and
reuse existing failure-blocking evidence; do not submit a known failure to create it.
Add one honest lesson and demonstrate doctor editing the spec.
Label setup evidence honestly; do not fabricate independent sessions
or active protections. Preserve required source references once; do not make routine
agents reread the archive. Stop at the approved budget and report remaining gaps.
```
