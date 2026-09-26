# GitHub setup

**Version 1.0** One monorepo, ordinary Actions, native branch protection. No external control repository or custom GitHub App. These settings are instructions until applied and checked in the actual repository.

## Implementer

Identify the repository, default branch, existing CI, account capabilities, and the owner's available permissions. Preserve useful existing workflows. Propose the minimal settings diff before applying it. If access is missing, provide the exact remaining owner steps and say they are unapplied.

Create or adapt one ordinary required job, preferably `verify`. Run the project's canonical checks, including end-to-end tests and combined coverage. Do not add a duplicate workflow just to obtain that name. Use the real existing job name when appropriate. All required targets must run; no path filter or conditional should silently omit them. Avoid error suppression and use native tool failures for empty discovery, missing reports, and insufficient coverage.

Pin third-party actions/dependencies, use only needed CI permissions, and keep production secrets out of tests. For fork contributions, do not run untrusted proposed code with privileged credentials. Save useful failure reports using ordinary Actions artifacts; no evidence service.

## Owner: configure the default branch

Use the repository's branch protection or branch ruleset settings for `main` (or the actual default branch). The exact controls available depend on repository visibility, account plan, and permissions. GitHub's [protected-branch documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) describes availability and settings.

1. Require changes through pull requests and require the actual CI job(s). Run the workflow once if needed for its check name to be available. Require branches to be current with the base branch before merging.
2. Disallow ordinary force pushes, branch deletion, and bypass. Apply the rule to administrators where supported. Do not create fake human reviewer accounts or a mandatory approval count that a solo owner cannot satisfy; independent agent review is documented in the task/PR, not proven by GitHub identity.
3. Confirm the rule is active. A deliberately failing test on a temporary PR should prevent an ordinary merge. Restore/fix it and verify the expected passing result. Do not test by force-pushing or deleting the real main branch.

GitHub accepts successful, skipped, or neutral required-check conclusions. The workflow should run required validation rather than skip the job. If several jobs are genuinely necessary, require all applicable checks rather than accepting an incomplete aggregate. See the same [GitHub documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## What this does not enforce

Tests and workflows live beside product code. Agents are instructed not to weaken them, but no custom file restriction or check-source authority prevents that. Reviewer D inspects the diff; native CI checks the submitted configuration. A workflow deletion/change or an agent ignoring instructions can defeat the intended process. Do not claim otherwise.

The first tests may intentionally fail before implementation; keep that work on the task branch. Initial test infrastructure and workflows are authorized setup work. Later workflow changes require a separate owner-approved maintenance task, not an implementer “fixing CI” inside an ordinary feature.

Record the effective branch rule and required check names in the repository's setup notes. When a protection feature is unavailable, report the gap rather than building a substitute platform or saying merging is protected. Owner-driven manual review remains a weaker operating choice, not equivalent enforcement.

## dphtools configuration observed during setup

Read on 2026-09-26 with `gh api repos/david-hoffman/dphtools/branches/main/protection`. Repository: `david-hoffman/dphtools`, public, default branch `main`; the authenticated account has admin access. No settings have been changed by setup.

The existing rule requires `ci-required`, up-to-date branches, one approving review, code-owner review, approval after the last push by someone other than its pusher, conversation resolution, and enforcement for administrators. Force pushes and deletion are disabled. No repository rulesets were returned.

The adapted ordinary `ci` workflow preserves the required `ci-required` name. Its unconditional job fails unless the entire `verify` matrix succeeds. The matrix retains Linux, macOS, and Windows; pinned runner families make changes explicit. The required job has no path filter, no conditional bypass, and no failure suppression. This is an ordinary aggregate of actual configured CI, not independent certification of workflows or agent roles.

Owner verification actions after the first setup PR run:

1. Open repository **Settings → Branches → main → Edit** (or the corresponding active branch rule). Keep **Require a pull request**, **Require status checks**, `ci-required`, and **Require branches to be up to date** enabled.
2. Keep administrator enforcement, force-push prohibition, deletion prohibition, and conversation resolution enabled. Preserve the existing human-review settings unless you explicitly decide to change them.
3. Confirm `ci-required` is present on the PR and fails when any matrix target fails. Inspect the actual PR's merge state. A protected-branch settings read by itself is not the failure-blocking demonstration.
4. A PR authored by the owner still needs an eligible real reviewer under the current one-approval/last-push rule. If the repository is operated solo and no such reviewer exists, explicitly authorize changing the approval count to zero and disabling code-owner/last-push approval requirements while retaining required PRs and CI. These changes are not applied or presumed approved. Do not fabricate reviewer identities or treat an agent report as a GitHub human approval.

Read-only verification commands:

```sh
gh api repos/david-hoffman/dphtools/branches/main/protection
gh api repos/david-hoffman/dphtools/rulesets
gh pr checks PR_NUMBER
gh pr view PR_NUMBER --json url,mergeStateStatus,statusCheckRollup,reviewDecision
```

The setup task records actual PR/run links when they exist. Merge and production release remain explicit owner actions; setup does not merge, push release tags, or invoke the existing publishing workflow.
