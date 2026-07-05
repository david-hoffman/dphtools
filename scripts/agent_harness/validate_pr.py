"""Validate local or GitHub pull request evidence for the harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PR_TEMPLATE = ROOT / ".github/pull_request_template.md"

REQUIRED_TEMPLATE_SECTIONS = [
    "## Linked issue",
    "## Change summary",
    "## Phase",
    "## Agent workflow evidence",
    "## Tests and commands",
    "## Coverage",
    "## Scientific, hardware, or numerical impact",
    "## Public API impact",
    "## Downstream impact",
    "## Release impact",
    "## Human/admin decisions needed",
]


def check_template(errors: list[str]) -> None:
    if not PR_TEMPLATE.is_file():
        errors.append("missing .github/pull_request_template.md")
        return
    text = PR_TEMPLATE.read_text(encoding="utf-8")
    for section in REQUIRED_TEMPLATE_SECTIONS:
        if section not in text:
            errors.append(f"PR template missing section: {section}")


def check_ci_event(errors: list[str]) -> None:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_name != "pull_request":
        print("No pull_request GitHub event; skipping PR body checks.")
        return
    if not event_path:
        errors.append("GITHUB_EVENT_PATH is missing for pull_request event")
        return

    payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request") or {}
    body = pull_request.get("body") or ""
    title = pull_request.get("title") or ""

    if "Closes #" in body and "Closes #\n" in body:
        errors.append("PR body still contains the placeholder linked issue")
    if "Closes #" not in body and "Fixes #" not in body and "Refs #" not in body:
        errors.append("PR body must link an issue with Closes #, Fixes #, or Refs #")
    for section in REQUIRED_TEMPLATE_SECTIONS[1:]:
        if section not in body:
            errors.append(f"PR body missing template section: {section}")
    if not title.strip():
        errors.append("PR title is empty")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run local file-level PR checks.")
    parser.add_argument(
        "--ci", action="store_true", help="Run GitHub event PR checks when available."
    )
    args = parser.parse_args()

    errors: list[str] = []
    check_template(errors)
    if args.ci:
        check_ci_event(errors)
    elif args.local:
        print("Local mode: GitHub PR body checks are not available.")
    else:
        print("No mode selected; running local template checks only.")

    if errors:
        print("PR validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PR validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
