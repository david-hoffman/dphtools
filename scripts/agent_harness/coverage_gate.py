"""Phase-aware coverage gate."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/agent-harness/coverage-baseline.json"
COVERAGE_XML = ROOT / "coverage.xml"
WAIVERS_DIR = ROOT / "docs/agent-harness/coverage-waivers"
EPSILON = 0.01


def read_coverage_xml() -> tuple[float, float] | None:
    if not COVERAGE_XML.is_file():
        return None
    root = ET.parse(COVERAGE_XML).getroot()
    line_rate = float(root.attrib.get("line-rate", "0")) * 100
    branch_rate = float(root.attrib.get("branch-rate", "0")) * 100
    return line_rate, branch_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--waiver-issue", help="Issue number for an approved coverage decrease waiver."
    )
    args = parser.parse_args()

    if not BASELINE.is_file():
        print(f"Coverage baseline missing: {BASELINE.relative_to(ROOT)}")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = read_coverage_xml()
    if current is None:
        if baseline.get("enforced"):
            print("coverage.xml is required when coverage enforcement is enabled")
            return 1
        print("coverage.xml not found; Phase 0 coverage enforcement is disabled.")
        return 0

    line_coverage, branch_coverage = current
    print(f"Current line coverage: {line_coverage:.2f}%")
    print(f"Current branch coverage: {branch_coverage:.2f}%")

    if args.update_baseline:
        baseline.update(
            {
                "line_coverage": round(line_coverage, 2),
                "branch_coverage": round(branch_coverage, 2),
                "updated_at": date.today().isoformat(),
            }
        )
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print("Coverage baseline updated.")
        return 0

    if not baseline.get("enforced"):
        print("Coverage enforcement disabled for Phase 0.")
        return 0

    baseline_line = baseline.get("line_coverage")
    baseline_branch = baseline.get("branch_coverage")
    waiver_path = WAIVERS_DIR / f"{args.waiver_issue}.md" if args.waiver_issue else None
    waiver_exists = bool(waiver_path and waiver_path.is_file())
    if baseline_line is not None and line_coverage + EPSILON < float(baseline_line):
        if waiver_exists:
            print(f"Line coverage decreased with approved waiver issue {args.waiver_issue}.")
            return 0
        print(f"Line coverage decreased from {baseline_line:.2f}% to {line_coverage:.2f}%")
        return 1
    if baseline_branch is not None and branch_coverage + EPSILON < float(baseline_branch):
        if waiver_exists:
            print(f"Branch coverage decreased with approved waiver issue {args.waiver_issue}.")
            return 0
        print(f"Branch coverage decreased from {baseline_branch:.2f}% to {branch_coverage:.2f}%")
        return 1

    print("Coverage gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
