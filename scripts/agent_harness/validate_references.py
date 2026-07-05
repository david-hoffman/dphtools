"""Validate agentic harness reference manifest and notes."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = ROOT / "docs/references/agentic-harness/sources.yml"

REQUIRED_URLS = {
    "https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills",
    "https://claude.com/blog/introducing-dynamic-workflows-in-claude-code",
    "https://openai.com/index/open-source-codex-orchestration-symphony/",
    "https://openai.com/index/harness-engineering/",
    "https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches",
    "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks",
    "https://docs.github.com/en/actions/tutorials/build-and-test-code/python",
    "https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference",
    "https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/manage-your-dependency-security",
    "https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments",
}

REQUIRED_FIELDS = [
    "id",
    "title",
    "publisher",
    "url",
    "accessed_at",
    "note_path",
    "snapshot_path",
    "snapshot_sha256",
    "terms_note",
]


def clean_value(value: str) -> str | None:
    value = value.strip()
    if value == "null":
        return None
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def parse_manifest(text: str) -> list[dict[str, str | None]]:
    entries: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None
    key_value = re.compile(r"^\s{4}([A-Za-z0-9_-]+):\s*(.*)$")
    for line in text.splitlines():
        if line.startswith("  - id:"):
            if current:
                entries.append(current)
            current = {"id": clean_value(line.split(":", 1)[1])}
            continue
        if current is None:
            continue
        match = key_value.match(line)
        if match:
            key, value = match.groups()
            current[key] = clean_value(value)
    if current:
        entries.append(current)
    return entries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    if not SOURCES_PATH.is_file():
        print(f"Reference manifest missing: {SOURCES_PATH.relative_to(ROOT)}")
        return 1

    entries = parse_manifest(SOURCES_PATH.read_text(encoding="utf-8"))
    if not entries:
        errors.append("sources.yml has no sources")

    urls = {entry.get("url") for entry in entries}
    for url in REQUIRED_URLS:
        if url not in urls:
            errors.append(f"missing required URL: {url}")

    seen_ids: set[str] = set()
    for entry in entries:
        source_id = entry.get("id") or "<missing id>"
        if source_id in seen_ids:
            errors.append(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)

        for field in REQUIRED_FIELDS:
            if field not in entry:
                errors.append(f"{source_id}: missing field {field}")

        note_path = entry.get("note_path")
        if not note_path:
            errors.append(f"{source_id}: note_path is empty")
        else:
            note_file = ROOT / note_path
            if not note_file.is_file():
                errors.append(f"{source_id}: missing note file {note_path}")
            else:
                note_text = note_file.read_text(encoding="utf-8")
                if "Harness impact" not in note_text:
                    errors.append(f"{source_id}: note must include a Harness impact section")

        snapshot_path = entry.get("snapshot_path")
        snapshot_sha = entry.get("snapshot_sha256")
        no_snapshot_reason = entry.get("no_snapshot_reason")
        if snapshot_path:
            snapshot_file = ROOT / snapshot_path
            if not snapshot_file.is_file():
                errors.append(f"{source_id}: snapshot file missing: {snapshot_path}")
            elif not snapshot_sha:
                errors.append(f"{source_id}: snapshot_sha256 required when snapshot_path is set")
            elif file_sha256(snapshot_file) != snapshot_sha:
                errors.append(f"{source_id}: snapshot checksum mismatch")
        elif not no_snapshot_reason:
            errors.append(f"{source_id}: no_snapshot_reason required when snapshot_path is null")

    if errors:
        print("Reference validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Reference validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
