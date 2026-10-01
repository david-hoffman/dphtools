# A single-maintainer publication gate can prevent its own approval

- ID: 20260930T183238Z-RELEASE-AUTOMATION-001-setup-solo-approval
- Date: 2026-09-30T18:32:38Z
- Task/role: RELEASE-AUTOMATION-001 / infrastructure preparation
- Status: confirmed
- Observation: The existing PyPI environment has one required reviewer and prevents self-review. A release dispatched by that reviewer cannot receive their approval. It also lacks a deployment branch restriction.
- Evidence: Read-only GitHub environment API inspection on 2026-09-30 returned one reviewer, `prevent_self_review: true`, `can_admins_bypass: false`, and `deployment_branch_policy: null`. GitHub's deployment-environment documentation states that preventing self-review disallows approval by the initiator.
- Lesson: Inspect actual environment settings when installing release automation. For an owner-dispatched, owner-approved process, permit that owner to approve while retaining the reviewer requirement and administrator-bypass restriction; explicitly restrict the release branch. Report these as unapplied activation changes until configured, rather than claiming workflow YAML establishes them.
