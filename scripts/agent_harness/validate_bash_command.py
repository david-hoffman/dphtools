"""Reject high-risk shell commands for local agent hooks."""

from __future__ import annotations

import argparse
import re
import shlex
import sys


SECRET_PATTERNS = [
    re.compile(r"\b[A-Za-z_]*(TOKEN|SECRET|PASSWORD|KEY)[A-Za-z_]*\b"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+", help="Command string or argv to inspect.")
    parser.add_argument("--allow-release", action="store_true")
    parser.add_argument("--allow-destructive", action="store_true")
    args = parser.parse_args()

    command = " ".join(args.command)
    tokens = shlex.split(command)
    errors: list[str] = []

    if tokens[:3] == ["git", "push", "origin"] and len(tokens) >= 4 and tokens[3] == "main":
        errors.append("direct push to default branch main is blocked")
    if tokens[:2] == ["git", "push"] and any(token in {"--force", "-f"} for token in tokens):
        errors.append("force push requires explicit human approval")
    if not args.allow_destructive and tokens[:2] == ["rm", "-rf"]:
        target = tokens[2] if len(tokens) > 2 else ""
        if not (target.startswith("/tmp/") or target.startswith("/private/tmp/")):
            errors.append("destructive rm -rf is only allowed under safe temp directories")
    if not args.allow_release and (
        "twine upload" in command
        or "gh-action-pypi-publish" in command
        or "anaconda upload" in command
    ):
        errors.append("release publication commands require release approval")
    for pattern in SECRET_PATTERNS:
        if pattern.search(command) and ("echo" in tokens or "printenv" in tokens):
            errors.append("command may expose secret-like environment values")

    if errors:
        print("Command validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Command validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
