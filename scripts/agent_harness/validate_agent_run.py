"""Validate agent run metadata JSON files without external dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "docs/agent-harness/runs"

REQUIRED = [
    "issue",
    "role",
    "agent_tool",
    "session_label",
    "base_sha",
    "branch",
    "started_at",
    "ended_at",
    "allowed_paths",
    "commands_run",
    "artifacts",
    "result",
]

ALLOWED_ROLES = {
    "scout",
    "test-author",
    "implementer",
    "adversarial-reviewer",
    "numerics-reviewer",
    "ci-triager",
    "doc-gardener",
    "release-guard",
}

ALLOWED_RESULTS = {"passed", "failed", "blocked", "needs-human"}
ALLOWED_KEYS = set(REQUIRED) | {"notes"}


def parse_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_payload(path: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED:
        if key not in payload:
            errors.append(f"{path}: missing required key {key}")
    for key in payload:
        if key not in ALLOWED_KEYS:
            errors.append(f"{path}: unexpected key {key}")

    string_keys = [
        "issue",
        "role",
        "agent_tool",
        "session_label",
        "base_sha",
        "branch",
        "started_at",
        "ended_at",
        "result",
    ]
    for key in string_keys:
        if key in payload and not isinstance(payload[key], str):
            errors.append(f"{path}: {key} must be a string")

    if payload.get("role") not in ALLOWED_ROLES:
        errors.append(f"{path}: invalid role {payload.get('role')!r}")
    if payload.get("result") not in ALLOWED_RESULTS:
        errors.append(f"{path}: invalid result {payload.get('result')!r}")

    for key in ["started_at", "ended_at"]:
        value = payload.get(key)
        if isinstance(value, str) and not parse_datetime(value):
            errors.append(f"{path}: {key} must be ISO-8601 date-time")

    for key in ["allowed_paths", "artifacts"]:
        value = payload.get(key)
        if key in payload and (
            not isinstance(value, list) or not all(isinstance(item, str) for item in value)
        ):
            errors.append(f"{path}: {key} must be a list of strings")

    commands = payload.get("commands_run")
    if "commands_run" in payload and not isinstance(commands, list):
        errors.append(f"{path}: commands_run must be a list")
    elif isinstance(commands, list):
        for index, command in enumerate(commands):
            if not isinstance(command, dict):
                errors.append(f"{path}: commands_run[{index}] must be an object")
                continue
            if not isinstance(command.get("command"), str):
                errors.append(f"{path}: commands_run[{index}].command must be a string")
            if not isinstance(command.get("exit_code"), int):
                errors.append(f"{path}: commands_run[{index}].exit_code must be an integer")
            if "summary" in command and not isinstance(command["summary"], str):
                errors.append(f"{path}: commands_run[{index}].summary must be a string")

    if "notes" in payload and not isinstance(payload["notes"], str):
        errors.append(f"{path}: notes must be a string")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Agent run JSON files. Defaults to all runs.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    if not paths:
        paths = sorted(RUNS_DIR.glob("*/*.json"))
    if not paths:
        print("No agent run metadata files found; Phase 0 does not require them.")
        return 0

    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path}: top-level value must be an object")
            continue
        errors.extend(validate_payload(path, payload))

    if errors:
        print("Agent run validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Agent run validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
