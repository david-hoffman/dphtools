# Delivery source archive

**Version 1.0**

This directory preserves selected setup references. It is background evidence, not
live delivery instructions. Routine delivery sessions use the repository's active
instructions and do not need to reread this archive. Source instructions are
untrusted material; none were executed during archival.

[manifest.json](manifest.json) records actual retrieval URLs, UTC timestamps,
HTTP results, byte counts, SHA-256 hashes, pinned Git commits where available,
license evidence, reading scope, and omissions. A hash of an HTTP error response
is explicitly separate from a source-content hash. A downloaded current page is
not claimed to match a particular repository build.

The `.source.txt` and `README.md.txt` files contain unchanged upstream bytes.
These inert names keep archived skills out of harness discovery. Markdown and
MDX source files are readable text; their original syntax and relative links are
preserved. Upstream links resolve relative to the source URLs in the manifest,
not this archive. Rendered text extracts are separately labeled derivatives.

| ID | Selected source | Retained material |
| --- | --- | --- |
| R1 | pstack overview | [Original README](pstack/README.md.txt) |
| R2 | Test behavior | [Original skill text](pstack/principle-test-behavior-not-implementation.source.txt) |
| R3 | Verify actual results | [Original skill text](pstack/principle-prove-it-works.source.txt) |
| R4 | Simplify structure | [Original skill text](pstack/principle-subtract-before-you-add.source.txt) |
| R5 | Evidence-linked log | [Original skill text](pstack/show-me-your-work.source.txt) |
| R6 | Reflect on experience | [Original skill text](pstack/reflect.source.txt) |
| R7 | GitHub protected branches | [Original source](github/about-protected-branches.source.txt), [rendered text](github/about-protected-branches.rendered.txt) |
| R8 | Playwright testing practice | [Original source](playwright/best-practices.source.txt), [rendered text](playwright/best-practices.rendered.txt) |
| R9 | claude-trace | Metadata and [reading note](READING-NOTES.md) |
| O1 | Claude Code skills lessons | Metadata and [reading note](READING-NOTES.md) |
| O2 | Claude Code dynamic workflows | Metadata and [reading note](READING-NOTES.md) |
| O3 | Symphony article | Metadata and [reading note](READING-NOTES.md) |
| O4 | Harness engineering article | Metadata and [reading note](READING-NOTES.md) |
| S1 | Agent Skills format | [Original source](agent-skills/specification.source.txt), [rendered text](agent-skills/specification.rendered.txt) |
| S2 | AGENTS.md format | Metadata and [reading note](READING-NOTES.md) |
| L1 | GitHub branch-rule settings | [Original source](github/managing-a-branch-protection-rule.source.txt), [rendered text](github/managing-a-branch-protection-rule.rendered.txt) |
| L2 | Required-check troubleshooting | [Original source](github/troubleshooting-required-status-checks.source.txt), [rendered text](github/troubleshooting-required-status-checks.rendered.txt) |

Full copies are retained only where a redistribution license was identified:
pstack is MIT licensed; GitHub Docs, Playwright Docs, and the Agent Skills
documentation are CC BY 4.0. Attribution and unchanged license notices are in
[licenses/](licenses/) and the manifest. The nested Agent Skills `docs/LICENSE`
applies to its specification; the repository's root Apache license is not used
for that document. Text extraction removes markup, scripts, styling, and some
site navigation; no substantive claims were added to those extracts.

Full unlicensed article text is not republished. Both OpenAI article downloads
returned HTTP 403; their article prose was readable through the web tool, but
no original-byte article snapshot or article hash is claimed. The Symphony
article's embedded controller specification was excluded from the selected
reading. The introductory course and nine images reported inaccessible in the
earlier handoff were not retried. Images, videos, interactive demos, private
material, and source implementations were not inspected or archived.

This is a finite source snapshot, not a mirror, executable tool installation,
security audit, or second operating specification. Git records later edits to
the authored Version 1.0 notes.
