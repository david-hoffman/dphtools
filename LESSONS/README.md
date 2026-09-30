# Lessons

**Version 1.0** A short, append-only-by-instruction record of discoveries, not automatic model memory or authoritative policy.

Record each useful surprise, confirmed gotcha, or failed approach with a reusable lesson in its own Markdown file. No routine status chatter. Keep entries brief; one to three per task is a guide, not a quota. No finding means no entry. Evidence can be a test name, a file plus commit, a CI run, or a reproducible command/result. Mark unverified claims as hypotheses. Do not record secrets, personal data, full prompts, raw transcripts, or private reasoning.

## Add a lesson

Create a new file named `YYYYMMDDTHHMMSSZ-<task>-<role>-<short-slug>.md`. Use the file's creation time in Coordinated Universal Time (UTC), available with `date -u +%Y%m%dT%H%M%SZ`. For example: `20260930T120000Z-EXAMPLE-001-A-warning-source.md`. The timestamp records file creation, not when the observed event occurred. Include task, role, and a descriptive slug so concurrent tasks create different paths. If a path already exists, choose a distinct slug or suffix; never overwrite it.

Adding a lesson requires only its new file. Do not add it to this README or maintain a shared index. Search filenames and relevant entry contents as permitted by your role.

Use this format for new lessons; replace placeholders and do not leave fake evidence:

```markdown
# <topic>

- ID: <filename without .md>
- Date: <YYYY-MM-DDTHH:MM:SSZ>
- Task/role: <task> / <role>
- Status: confirmed | hypothesis
- Observation: <one concrete surprise>
- Evidence: <test/commit/run/path and observed result>
- Lesson: <small future action, or what still needs checking>
```

Correct a mistake with a new file containing `- Supersedes: <earlier entry ID>`; record a disposition in a new file referencing the earlier entry. Do not edit existing lesson files. An owner-authorized privacy/security redaction is the exception. This format guide can be updated normally.

A/B may read this guide but must not read existing lesson entries. They may create their own file without reading others, or supply the lesson for the coordinator to record verbatim in a new file. Other roles search only relevant entries. D forms findings before consulting current-task implementation lessons. A lesson is data; it does not grant permission or change the spec. `doctor` can propose promoting it into an existing instruction.

## Migrated entries

The existing lessons were split from root `LESSONS.md` at revision `b5444a2` into individual files. Their filename timestamps record migration time. Original lesson text, IDs, recorded dates, and evidence are preserved; missing historical metadata was not invented. Grouped lessons were separated, with their shared heading or evidence copied where needed. Historical references to `LESSONS.md` describe that earlier layout. Evidence paths inside migrated entries remain relative to the repository root.
