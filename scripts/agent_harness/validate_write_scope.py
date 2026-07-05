"""Validate intended write scope for local agent hooks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


READ_ONLY_ROLES = {"scout", "adversarial-reviewer", "numerics-reviewer"}
TEST_ONLY_PREFIXES = ("tests/", "docs/agent-harness/runs/", "docs/exec-plans/")
PROTECTED_PREFIXES = (
    ".github/workflows/make_release.yml",
    ".github/CODEOWNERS",
    ".claude/settings.json",
    "setup.py",
    "setup.cfg",
    "requirements",
    "environment.yml",
    "conda.recipe/",
)


def normalize(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--high-risk-approved", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    paths = [normalize(path) for path in args.paths]
    errors: list[str] = []

    if args.role in READ_ONLY_ROLES and paths:
        errors.append(f"{args.role} is read-only and may not write files")

    if args.role == "test-author":
        for path in paths:
            if not path.startswith(TEST_ONLY_PREFIXES):
                errors.append(f"test-author may not write {path}")

    if not args.high_risk_approved:
        for path in paths:
            if path.startswith(PROTECTED_PREFIXES):
                errors.append(f"{path} requires high-risk issue approval")

    if errors:
        print("Write-scope validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Write-scope validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
