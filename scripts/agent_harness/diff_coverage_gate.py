"""Require changed product lines to be covered when diff coverage is enabled."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COVERAGE_XML = ROOT / "coverage.xml"
PRODUCT_PREFIXES = ("dphtools/",)
WAIVERS_DIR = ROOT / "docs/agent-harness/coverage-waivers"


def run_git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None
    return result.stdout


def choose_base(explicit_base: str | None) -> str | None:
    if explicit_base:
        return explicit_base
    env_base = None
    if "GITHUB_BASE_REF" in os_environ():
        env_base = f"origin/{os_environ()['GITHUB_BASE_REF']}"
    for candidate in [env_base, "origin/main", "main", "HEAD^"]:
        if not candidate:
            continue
        if run_git(["git", "rev-parse", "--verify", candidate]) is not None:
            return candidate
    return None


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def changed_product_lines(base: str) -> dict[str, set[int]]:
    output = run_git(["git", "diff", "--unified=0", f"{base}...HEAD", "--", "dphtools"])
    if output is None:
        output = run_git(["git", "diff", "--unified=0", base, "--", "dphtools"])
    if not output:
        return {}

    changed: dict[str, set[int]] = {}
    current_file: str | None = None
    new_line = 0
    hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
    for line in output.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file.startswith(PRODUCT_PREFIXES):
                changed.setdefault(current_file, set())
            continue
        match = hunk_re.match(line)
        if match:
            new_line = int(match.group("start"))
            continue
        if current_file is None or not current_file.startswith(PRODUCT_PREFIXES):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changed[current_file].add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            new_line += 1
    return {path: lines for path, lines in changed.items() if lines}


def coverage_lines() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    root = ET.parse(COVERAGE_XML).getroot()
    executable: dict[str, set[int]] = {}
    covered: dict[str, set[int]] = {}
    for class_node in root.findall(".//class"):
        filename = class_node.attrib.get("filename", "")
        if not filename.startswith(PRODUCT_PREFIXES):
            continue
        executable.setdefault(filename, set())
        covered.setdefault(filename, set())
        for line_node in class_node.findall("./lines/line"):
            number = int(line_node.attrib["number"])
            hits = int(line_node.attrib.get("hits", "0"))
            executable[filename].add(number)
            if hits > 0:
                covered[filename].add(number)
    return executable, covered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--base", help="Git base ref for diff coverage.")
    parser.add_argument(
        "--waiver-issue", help="Issue number for an approved diff coverage waiver."
    )
    args = parser.parse_args()

    if not args.enabled:
        print("Diff coverage enforcement is disabled.")
        return 0
    if args.waiver_issue and (WAIVERS_DIR / f"{args.waiver_issue}.md").is_file():
        print(f"Diff coverage waiver issue {args.waiver_issue} is present.")
        return 0
    if not COVERAGE_XML.is_file():
        print("coverage.xml is required for diff coverage enforcement.")
        return 1

    base = choose_base(args.base)
    if base is None:
        print("Could not determine git base for diff coverage.")
        return 1

    changed = changed_product_lines(base)
    if not changed:
        print("No changed product lines found for diff coverage.")
        return 0

    executable, covered = coverage_lines()
    misses: list[str] = []
    checked = 0
    for path, lines in changed.items():
        executable_lines = executable.get(path, set())
        covered_lines = covered.get(path, set())
        for line in sorted(lines):
            if line not in executable_lines:
                continue
            checked += 1
            if line not in covered_lines:
                misses.append(f"{path}:{line}")

    if misses:
        print("Changed executable product lines are not covered:")
        for miss in misses:
            print(f"- {miss}")
        return 1

    print(f"Diff coverage gate passed for {checked} changed executable product line(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
