"""Format touched Python files with Black when paths are supplied."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    python_files = [
        str((ROOT / path).resolve())
        for path in args.paths
        if path.endswith(".py") and (ROOT / path).is_file()
    ]
    if not python_files:
        print("No Python files supplied; nothing to format.")
        return 0

    command = [sys.executable, "-m", "black", "-l", "99", *python_files]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
