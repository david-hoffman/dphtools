# Selected source reading notes

**Version 1.0**

Original sources read during setup on September 26, 2026 UTC; R10 added on
September 29, 2026 UTC without rereading the earlier sources. Exact source selection, retrieval
times, versions, hashes, and reading limitations are in
[manifest.json](manifest.json). These are original summaries, not substitutes
for upstream originals or additions to the delivery specification.

## Adopted sources

R1–R6: Read the complete pinned pstack README and five selected skill texts.
The useful ideas are observable behavioral tests, direct checks of actual
outputs, simpler structures, concise evidence-linked discoveries, and focused
instruction improvements. Do not import the plugin's model choices, swarms,
controller playbooks, unattended merging, transcript mining, automatic backlog
filing, or repeated review loops. Its test heuristics are not universal bans on
absence assertions or property tests. Its logging helper and reviewer templates
were not selected, read, or installed. [pstack source](https://github.com/cursor/plugins/tree/ecc249f1e306fc64ddf83c7bed16cacf7c2239db/pstack).

R7, L1, L2: Read the complete pinned source and rendered substantive text for
GitHub protected branches, branch-rule management, and required-check
troubleshooting. Native checks and owner-configured branch rules can support
review, but skipped jobs can satisfy a required check. Workflow filters,
dependency failures, trigger events, and check names affect whether validation
actually runs. Available settings and bypass behavior need inspection in the
real repository; this research applies no settings.
[Protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches),
[rule management](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule),
[required checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).

R8: Read the complete pinned Playwright MDX source and rendered substantive
article. The retained testing ideas are visible behavior, independent state,
stable public interactions, and controlled service boundaries. Browser-specific
installation, generators, sharding, and trace configuration are outside this
Python library setup. Bounded assertion waiting does not justify rerunning a
failed suite until it passes. Linked tutorials and image/video assets were not
selected. [Best practices](https://playwright.dev/docs/best-practices).

R9: Read the complete pinned claude-trace README. It describes request logging,
a local viewer, and optional model-powered indexing with extra token use.
That differs from a concise shared lesson record. No implementation was read,
no tool was installed, and no private logs or credentials were accessed.
The repository license endpoint returned 404; the only license found in its
tree belongs to another application. Preserve metadata and this summary, not
the full README. [Selected README](https://github.com/badlogic/lemmy/blob/92e4ba60328bb9e6d756f18bd5c1e2f166768a61/apps/claude-trace/README.md).

R10: Read the complete pinned OpenClaw test-audit skill and MIT license. Adapt
its behavioral-refactor criterion, rejection of unnecessary test-only production
interfaces, and checks for coverage-only execution, wrong-reason negative
results, fixtures providing the claimed output, and overstated test names.
Retain independent contracts and legitimate external public callers; a local
call-count search alone does not establish dead library code. Our adaptation
requires meaningful correctness checks, not a particular assertion keyword.
Approved exception or completion behavior can be checked without inventing
another observable result. The source also cautions against automatically
deleting existing tests that look coupled to implementation.

Only these judgment criteria are adopted. Its implementation-reading audit
workflow is incompatible with blind test roles. Campaign instructions, linked
skills, commands, and production implementations were not read, installed, or
executed. It provides no basis for choosing numerical tolerances. The exact
skill bytes and license notice are retained under inert `.txt` filenames.
[Pinned skill](https://github.com/openclaw/openclaw/blob/80930af448ebabc84174146b56bc106d37fab3b4/.agents/skills/test-audit/SKILL.md).

## Original articles

O1: Read the complete skills article prose. It describes focused reusable
skills, useful trigger descriptions, progressive disclosure, setup context,
verification, and accumulating practical gotchas. These support a small set of
maintained skills and evidence-based lessons. Hooks, marketplaces, usage
tracking, and Claude-specific persistence are not required here. The original
URL redirects to the selected June 3, 2026 article at `claude.dev`. Image alt
text was available; image pixels were not inspected.
[Selected article](https://claude.dev/blog/lessons-from-building-claude-code-how-we-use-skills/).

O2: Read the complete dynamic-workflows article prose. It presents substantial
parallel orchestration and verification with potentially higher token use.
Its controller machinery, fan-out, auto mode, model defaults, and continuing
loops are excluded. The useful contrast is that independent verification can
be retained without installing that orchestration system. No performance claim
is adopted. [Article](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code).

O3: Read the complete Symphony article narrative before and after the embedded
service specification. The narrative describes tracker-driven continuous agent
execution, isolated workspaces, and documented workflow policy. This setup
retains explicit work intent and repository-local process instructions; it does
not adopt the daemon, tracker control plane, restart loop, or custom controller.
The embedded 1,363-line controller specification and linked implementation were
deliberately excluded and are not claimed fully read. Direct bytes returned
HTTP 403; the narrative was read through the web tool.
[Article](https://openai.com/index/open-source-codex-orchestration-symphony/).

O4: Read the complete substantive harness-engineering article, including its
repository-tree example. The useful ideas are discoverable local knowledge,
a short instruction entry point, reproducible validation, and feeding observed
failures back into instructions. Its architecture, minimal merge gates,
automerging, custom enforcement, and recurring cleanup agents are not imported.
The article reports one team's experience, not proof this setup works. Direct
bytes returned HTTP 403; article text was read through the web tool.
[Article](https://openai.com/index/harness-engineering/).

## Mechanism-specific supporting reading

S1: Read the complete Agent Skills specification in pinned source and rendered
form. It defines required name/description frontmatter and optional supporting
files. Its experimental tool field is not permission enforcement. No reference
validator or source scripts were downloaded or executed.
[Specification](https://agentskills.io/specification).

S2: Read the complete substantive AGENTS.md page. It describes a dedicated
Markdown instruction entry point that complements a product README and can
point to build/test conventions. No example stack or migration command was
adopted. The site's mapping to its MIT-licensed repository revision was not
established, so only metadata and this summary are preserved.
[AGENTS.md](https://agents.md/).

The two GitHub links above were followed because concrete settings and check
behavior matter to the adopted mechanism. Unrelated implementation links were
not recursively crawled. Earlier LangChain, Open Code Review, SoL-Pi, C4, and
Markdown-diagram references remain linked in the package register; this pass
does not claim to have read or archived them.
