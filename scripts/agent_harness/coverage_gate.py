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
    if baseline_line is not None and line_coverage < float(baseline_line):
        print(f"Line coverage decreased from {baseline_line:.2f}% to {line_coverage:.2f}%")
        return 1
    if baseline_branch is not None and branch_coverage < float(baseline_branch):
        print(f"Branch coverage decreased from {baseline_branch:.2f}% to {branch_coverage:.2f}%")
        return 1

    print("Coverage gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
