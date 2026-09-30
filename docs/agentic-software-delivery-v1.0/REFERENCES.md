# References and reading scope

**Version 1.0** Source register updated: September 29, 2026; original setup reading: September 26, 2026. This is the source register and adoption note. The selected source archive is [../references/README.md](../references/README.md); it is not a code/security audit or a second operating specification.

## Current sources read

During setup, the complete pstack README, five selected skill texts below, and claude-trace README were read at recorded Git commits. Complete selected GitHub and Playwright source texts and rendered substantive articles were also read. The Agent Skills format and AGENTS.md guidance were read, plus two directly relevant GitHub settings/check references. Original-article reading scope is recorded below. Images and videos are not claimed inspected. Links here identify upstream sources; [the manifest](../references/manifest.json) records the exact selected versions, original-byte hashes, retrieval times, access results, licenses, and archive paths.

On September 29, the complete pinned OpenClaw test-audit skill and license were read and added as R10. Earlier sources were not recrawled or reread for this addition.

### R1 — pstack overview

[README](https://github.com/cursor/plugins/blob/main/pstack/README.md).

The project offers a broad collection of skills, principles, playbooks, and multi-agent mechanisms. This design does not install that collection. Only the specific ideas below are adapted. Its README's proceed-without-human-confirmation principle is not adopted; our intake still needs owner decisions. No named model defaults are copied.

### R2 — Test behavior, not implementation

[Skill](https://github.com/cursor/plugins/blob/main/pstack/skills/principle-test-behavior-not-implementation/SKILL.md).

Use real inputs and observable results, not assertions that only echo the subject's own outputs or constants. We do not copy its assertion-category heuristics as blanket bans: legitimate absence/error assertions and property tests remain valid. The project-agnostic E2E preference comes from the owner's instruction, not a claim that this skill requires browser tests everywhere.

### R3 — Verify the real artifact

[Skill](https://github.com/cursor/plugins/blob/main/pstack/skills/principle-prove-it-works/SKILL.md).

Retain direct, reproducible checks of the actual result. Compilation or an agent's success message alone is insufficient. Use existing test/CI artifacts, not a new verification service.

### R4 — Simplify existing structure

[Skill](https://github.com/cursor/plugins/blob/main/pstack/skills/principle-subtract-before-you-add/SKILL.md).

Prefer removing unnecessary code and duplicate instructions before adding more. This is a local simplification principle, not permission to remove required validation or unrelated code.

### R5 — Append-only evidence-linked findings

[Show-me-your-work skill](https://github.com/cursor/plugins/blob/main/pstack/skills/show-me-your-work/SKILL.md).

Adapt the compact evidence-linked append-only trail and superseding corrections into individual timestamped Markdown files under `LESSONS/`. Do not copy its transcript-audit and cross-model-review stages, TSV-specific helper, or mandatory per-reply attention format. Our log captures selected reusable discoveries, not every decision.

### R6 — Improve instructions from experience

[Reflect skill](https://github.com/cursor/plugins/blob/main/pstack/skills/reflect/SKILL.md).

Adapt the idea of turning durable lessons into focused edits to existing instructions with owner approval. Do not adopt the three-reviewer-plus-synthesizer procedure, transcript mining, automatic backlog filing, or nested skill-creation loops. Its reviewer templates are not used and were not read; no claim is made to have audited the whole plugin.

### R7 — GitHub protected branches

[Official documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

Use native required CI and branch protections subject to account capabilities. Required checks can accept skipped/neutral conclusions; do not design a required job that silently skips validation. Native protection is not proof that same-repository workflows were not weakened.

### R8 — End-to-end testing practice

[Playwright best practices](https://playwright.dev/docs/best-practices).

For browser projects, adapt user-visible behavior, isolated data/state, resilient interactions, and controlled external-service boundaries. Playwright is an example, not the required tool for every project. The document's broader tooling and linked tutorials were not recursively read or adopted; select those only if implementation needs them. Do not interpret legitimate bounded assertion waiting as approval to rerun failed suites until green.

### R9 — claude-trace

[README](https://github.com/badlogic/lemmy/blob/main/apps/claude-trace/README.md).

The documented tool records Claude Code interactions and raw API data with a viewer; its optional indexing uses Claude calls and additional tokens. This is a diagnostic transcript tool, not a small reusable-lesson record. Do not install it by default. Native harness logs can be consulted for a specific problem when permitted; they stay out of the shared lesson file and blind-role inputs. Implementation, runtime compatibility, and security were not audited or executed.

### R10 — OpenClaw test audit

[OpenClaw test-audit skill](https://github.com/openclaw/openclaw/blob/80930af448ebabc84174146b56bc106d37fab3b4/.agents/skills/test-audit/SKILL.md), read in full at commit `80930af448ebabc84174146b56bc106d37fab3b4`.

Adapt its behavioral-refactor test: an assertion should survive an implementation change that preserves the approved behavior. Tests should not require production exports, hooks, or wrappers with no production purpose. Reject execution that only increases coverage, negative cases passing because an unrelated check rejects the input, fixtures that manufacture the result the product must produce, and names claiming behavior the inputs do not exercise.

Apply these as judgment criteria. A meaningful correctness check need not use a literal `assert` keyword; approved exception and completion behavior can be checked. Legitimate external callers count for a library's public interfaces. Independently meaningful interface, architecture, platform, and ordering contracts remain valid even when their tests resemble implementation checks; existing tests are not automatically deletable.

Do not import OpenClaw's implementation-reading audit, campaign, or landing workflows into blind A/B roles. Linked skills and commands were not read or installed. This source supplies no numerical-tolerance authority. The [inert archive](../references/openclaw/test-audit.source.txt) preserves the complete source with its [MIT notice](../references/licenses/openclaw-MIT.txt).

## Original articles and earlier supporting references

The original four articles were revisited during setup. Their complete substantive prose was read, except that Symphony's embedded 1,363-line controller specification was deliberately excluded. The full narrative before and after that embed was read. OpenAI article bytes returned HTTP 403; the web tool supplied readable narrative text. No original-byte OpenAI article snapshot or article hash is claimed. Anthropic article bytes were retrieved and hashed, but full text is not republished without an identified redistribution license. [Reading notes](../references/READING-NOTES.md) distinguish the adopted ideas from excluded machinery.

- [Original: Claude Code skills lessons](https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills).
- [Original: Claude Code dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code).
- [Original: Symphony](https://openai.com/index/open-source-codex-orchestration-symphony/).
- [Original: Harness engineering](https://openai.com/index/harness-engineering/).
- [Harness terminology discussion](https://www.langchain.com/blog/how-to-build-a-custom-agent-harness).
- [Open Code Review](https://github.com/alibaba/open-code-review): discussed, not a required reviewer dependency.
- [SoL-Pi paper](https://arxiv.org/abs/2609.20519) and [repository](https://github.com/NVlabs/SoL-Pi): discussed; retain efficient diagnostic output and honest cost/quality comparisons, not automatic optimization machinery. No benchmark claim is repeated here.
- [Agent Skills format](https://agentskills.io/specification), [AGENTS.md](https://agents.md/), [C4 diagram guidance](https://c4model.com/diagrams), and [GitHub Markdown diagrams](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams): earlier supporting references, not new requirements.

The LangChain, Open Code Review, SoL-Pi, C4, and GitHub Markdown-diagram references above were retained from earlier drafting and were not reread or archived in this selected pass. No benchmark claim is newly verified. The Agent Skills specification and AGENTS.md page were read during setup; they do not establish any particular harness's installed behavior.

Known inherited gaps: an introductory Agent Skills course returned HTTP 403; nine instructional images linked from the original skills article were inaccessible. This pass did not retry those resources. The current skills article redirects to `claude.dev`; its prose and available image alt text were read, but image pixels were not inspected. These limits are not silently treated as resolved.

## Archived setup evidence

The archive contains 19 selected source records. It preserves unchanged licensed pstack, OpenClaw, documentation, and owner-supplied scientific source bytes, their license notices, separately labeled rendered text extracts, and metadata or original summaries where full-source redistribution was not established. Exact source copies occur only under `docs/references/` with inert `.txt` filenames; they are not installed skills. All authored delivery notes remain Version 1.0; upstream source bytes and license versions remain unchanged.

[Archive index](../references/README.md), [retrieval and hash manifest](../references/manifest.json), and [reading scope and adoption notes](../references/READING-NOTES.md) are the setup evidence. Full unlicensed articles, inaccessible original OpenAI article bytes, image/video assets, and the embedded Symphony controller specification remain deliberate limitations. No downloaded instructions were executed; no private material was archived. Keep these sources out of routine worker context. Do not carry forward superseded architecture requirements because an older source proposed them.
