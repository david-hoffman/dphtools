# GitHub setup

**Version 1.0** One monorepo, ordinary Actions, native branch protection. No external control repository or custom GitHub App. These settings are instructions until applied and checked in the actual repository.

## Implementer

Identify the repository, default branch, existing CI, account capabilities, and the owner's available permissions. Preserve useful existing workflows. Propose the minimal settings diff before applying it. If access is missing, provide the exact remaining owner steps and say they are unapplied.

Create or adapt one ordinary required job, preferably `verify`. Run the project's canonical checks, including end-to-end tests and combined coverage. Do not add a duplicate workflow just to obtain that name. Use the real existing job name when appropriate. All required targets must run; no path filter or conditional should silently omit them. Avoid error suppression and use native tool failures for empty discovery, missing reports, and insufficient coverage.

Before opening or reopening a PR (including a draft), or pushing an update to an open PR, run full canonical verification against the exact candidate and record its commit, environment, command, and result. Confirm the tested tree matches the commit and remains unchanged through submission. Any known failure, including incomplete coverage, blocks submission. CI repeats verification on configured platforms; classify newly discovered failures under specification section 4 before repair.

Before a PR exists, ordinary branch pushes may back up failing checkpoints; record their incomplete status. No dedicated backup branch or controller is needed. Use fast generic hooks and run the full command at submission. Closing a PR or moving a branch never permits unverified submission; opening or reopening the PR still requires the full gate.

Pin third-party actions/dependencies, use only needed CI permissions, and keep production secrets out of tests. For fork contributions, do not run untrusted proposed code with privileged credentials. Save useful failure reports using ordinary Actions artifacts; no evidence service.

## Owner: configure the PR target branch

Use the repository's branch protection or branch ruleset settings for the approved PR target. The exact controls available depend on repository visibility, account plan, and permissions. GitHub's [protected-branch documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) describes availability and settings.

1. Require changes through pull requests and require the actual CI job(s). Run the workflow once if needed for its check name to be available. Require branches to be current with the base branch before merging.
2. Disallow ordinary force pushes, branch deletion, and bypass. Apply the rule to administrators where supported. Do not create fake human reviewer accounts or a mandatory approval count that a solo owner cannot satisfy; independent agent review is documented in the task/PR, not proven by GitHub identity.
3. Confirm the rule is active and inspect required checks and merge state on a real PR. Reuse existing failed-run/protection evidence when available. Do not submit a known failure solely to demonstrate blocking. Configuration inspection alone does not demonstrate an observed blocked merge; report missing behavioral evidence plainly.

GitHub accepts successful, skipped, or neutral required-check conclusions. The workflow should run required validation rather than skip the job. If several jobs are genuinely necessary, require all applicable checks rather than accepting an incomplete aggregate. See the same [GitHub documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Owner: activate an approved release process

Installing workflow files does not activate their external protections or approve a release. Record the actual configuration and any unavailable controls in the project/setup evidence. Keep the owner-selected release source separate from the PR target.

1. Configure the publication environment to require owner approval and restrict its eligible source to the project's trusted release branch/workflow. Check default-branch and workflow-dispatch behavior. Required full verification and retained-artifact installation checks must precede the approval stage; ordinary merges and tag pushes must not publish.
2. Bind each registry's publishing identity to the exact repository, top-level workflow, and protected environment. Prefer short-lived trusted-publisher credentials where supported. Restrict token and identity-token permissions to the jobs that need them; candidate execution and post-upload installation checks receive no publication credentials.
3. Confirm artifact retention and exact originating run/artifact selection support the documented recovery interval. The approval summary identifies source/version, destinations, notes, required results, and file hashes. Publication consumes those files without rebuilding and stops on identity or registry conflicts. Missing artifacts require new preparation and approval.
4. Record setup inspection separately from observed hosted behavior. A rehearsal or production publication requires its own explicit owner approval of a prepared bundle. Do not publish, create a remote release tag, change registry settings, or delete old credentials merely to demonstrate installation. Report remaining owner steps; do not claim enforcement from workflow text or local mocks.

## What this does not enforce

Tests and workflows live beside product code. Agents are instructed not to weaken them, but no custom file restriction or check-source authority prevents that. Reviewer D inspects the diff; native CI checks the submitted configuration. A workflow deletion/change or an agent ignoring instructions can defeat the intended process. Do not claim otherwise.

The first tests may fail before implementation; they may be backed up before a PR exists. Tests of existing approved behavior may pass initially without mutation or artificial failure. Neither case changes the full submission gate. Initial test infrastructure and workflows are authorized setup work. Later workflow changes require a separate owner-approved maintenance task, not an implementer “fixing CI” inside an ordinary feature.

Record the effective branch rule, required check names, and evidence links in the setup task's concise Current state section. When a protection feature is unavailable, report the gap rather than building a substitute platform or saying merging is protected. Owner-driven manual review remains a weaker operating choice, not equivalent enforcement.

Preserve existing human-review settings unless the owner explicitly approves changing them. A solo owner may lack an eligible reviewer under existing approval rules; disclose that limitation rather than fabricate reviewer identities or treat an agent report as GitHub human approval. Merge and production release remain explicit owner actions.
