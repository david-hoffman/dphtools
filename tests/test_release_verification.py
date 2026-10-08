"""Preparation accepts complete canonical receipts and rejects scoped evidence."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from . import test_release_version as version_cli
from .test_release_bundle import generate
from .test_release_support import Bundle, PLATFORMS, STEPS, rejected, write_json

worker = version_cli.worker
FULL_STEPS = (
    *STEPS,
    "interpreter",
    "locked-dependencies",
    "imports",
    "venv",
    "nested-venv",
    "nested-pip",
    "nested-import",
    "preflight",
    "wheel-artifacts",
    "clean-install",
    "coverage-data",
)


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def modern_reports(worker, bundle):
    root = worker[0].parents[1]
    sources = sorted(
        {
            p.relative_to(root).as_posix()
            for directory in ("dphtools", "tools")
            for p in (root / directory).rglob("*.py")
            if p.relative_to(root).as_posix() != "dphtools/_version.py"
        }
        | {"tools/delivery"}
    )
    counts = dict(
        num_statements=1,
        covered_lines=1,
        missing_lines=0,
        excluded_lines=0,
        num_branches=0,
        covered_branches=0,
        missing_branches=0,
    )
    coverage = {
        "meta": {"branch_coverage": True},
        "files": {
            path: {"summary": counts, "missing_lines": [], "missing_branches": []}
            for path in sources
        },
        "totals": {k: v * len(sources) for k, v in counts.items()},
    }
    for platform in PLATFORMS:
        directory = bundle.reports / platform
        report = json.loads((directory / "checks.json").read_text())
        scope = {
            "name": "full",
            "coverage_claim": "global",
            "domains": ["library", "doctor", "release", "verification"],
            "measured_sources": sources,
            "unvalidated_sources": [],
            "unvalidated_test_paths": [],
        }
        identity = {
            "scope": scope,
            "fixture": "structural canonical report; not numerical proof",
            "environment": {
                "system": {
                    "ubuntu-24.04": "Linux",
                    "macos-15": "Darwin",
                    "windows-2025": "Windows",
                }[platform],
                "platform": report["platform"],
                "executable": report["python"],
            },
        }
        report.update(
            document_version="2.0",
            complete=True,
            outcome="passed",
            scope=scope,
            coverage_complete=True,
            identity=identity,
            identity_digest=seal(identity),
            steps=[],
        )
        for name in FULL_STEPS:
            log = directory / (name + ".log")
            log.write_text("Retained fixture boundary: " + name + "\n")
            report["steps"].append(
                {
                    "name": name,
                    "command": ["fixture", name],
                    "returncode": 0,
                    "state": "passed",
                    "input_identity": identity,
                    "input_digest": seal(identity),
                    "log": {
                        "path": log.name,
                        "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    },
                }
            )
        report["receipt_digest"] = seal(report)
        write_json(directory / "checks.json", report)
        write_json(directory / "coverage.json", coverage)
        (directory / "coverage.xml").write_text(
            f'<coverage lines-valid="{len(sources)}" lines-covered="{len(sources)}" branches-valid="0" branches-covered="0"/>'
        )
        (directory / "pytest.xml").write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="fixture"/></testsuite>'
        )


def reseal(path, change):
    report = json.loads(path.read_text())
    report.pop("receipt_digest")
    change(report)
    report["receipt_digest"] = seal(report)
    write_json(path, report)


def test_canonical_full_report_version_two_is_accepted_for_preparation(worker, tmp_path):
    bundle = Bundle(tmp_path / "bundle")
    assert generate(worker, bundle).returncode == 0, "Legacy full control"
    bundle.generated.unlink()
    modern_reports(worker, bundle)
    result = generate(worker, bundle)
    assert result.returncode == 0, version_cli.diagnostic(result)


@pytest.mark.parametrize(
    "defect",
    [
        "scoped",
        "incomplete",
        "failed",
        "unvalidated",
        "identity",
        "seal",
        "step",
        "step-input",
        "log",
        "coverage",
        "missing-report",
    ],
)
def test_preparation_rejects_scoped_partial_changed_or_incomplete_proofs(worker, tmp_path, defect):
    bundle = Bundle(tmp_path / "bundle")
    modern_reports(worker, bundle)
    assert generate(worker, bundle).returncode == 0, "Valid current canonical control"
    bundle.generated.unlink()
    directory = bundle.reports / PLATFORMS[0]
    path = directory / "checks.json"
    if defect == "log":
        (directory / "tests.log").write_text("Changed log")
    elif defect == "coverage":
        coverage = json.loads((directory / "coverage.json").read_text())
        coverage["files"].pop(next(iter(coverage["files"])))
        write_json(directory / "coverage.json", coverage)
    elif defect == "missing-report":
        (directory / "coverage.xml").unlink()
    elif defect == "seal":
        report = json.loads(path.read_text())
        report["receipt_digest"] = "foreign"
        write_json(path, report)
    else:

        def change(report):
            if defect == "scoped":
                report["scope"]["coverage_claim"] = "scoped"
            elif defect == "incomplete":
                report["complete"] = False
            elif defect == "failed":
                report["outcome"] = "failed"
            elif defect == "unvalidated":
                report["scope"]["unvalidated_sources"] = ["tools/release.py"]
            elif defect == "identity":
                report["identity_digest"] = "foreign"
            elif defect == "step":
                report["steps"][0]["state"] = "reused"
            else:
                report["steps"][0]["input_digest"] = "foreign"

        reseal(path, change)
    rejected(generate(worker, bundle))


@pytest.mark.parametrize("defect", ["copied-linux", "platform-string", "executable"])
def test_current_full_receipts_require_distinct_consistent_platform_evidence(
    worker, tmp_path, defect
):
    bundle = Bundle(tmp_path / "bundle")
    modern_reports(worker, bundle)
    assert generate(worker, bundle).returncode == 0, "Valid distinct-platform control"
    bundle.generated.unlink()
    if defect == "copied-linux":
        for target in PLATFORMS[1:]:
            shutil.rmtree(bundle.reports / target)
            shutil.copytree(bundle.reports / PLATFORMS[0], bundle.reports / target)
    else:

        def change(report):
            key = "platform" if defect == "platform-string" else "executable"
            report["identity"]["environment"][key] = "foreign"
            report["identity_digest"] = seal(report["identity"])

        reseal(bundle.reports / PLATFORMS[0] / "checks.json", change)
    result = generate(worker, bundle)
    rejected(result)
    assert "Foreign or inconsistent platform evidence" in version_cli.diagnostic(result)
