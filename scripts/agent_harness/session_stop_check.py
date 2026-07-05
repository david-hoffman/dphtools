"""Run lightweight session-end harness checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> int:
    print("+", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    commands = [
        [sys.executable, "scripts/agent_harness/validate_harness.py"],
        [sys.executable, "scripts/agent_harness/validate_references.py"],
        [sys.executable, "scripts/agent_harness/validate_pr.py", "--local"],
    ]
    for command in commands:
        exit_code = run(command)
        if exit_code:
            return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
