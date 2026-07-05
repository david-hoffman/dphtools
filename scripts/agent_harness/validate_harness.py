"""Validate Phase 0 agent harness files."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "ARCHITECTURE.md",
    "QUALITY_SCORE.md",
    "Makefile",
    "requirements-dev.txt",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/agent_task.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/release_checklist.yml",
    ".github/workflows/ci.yml",
    "docs/generated/README.md",
    "docs/generated/repo-intake.md",
    "docs/agent-harness/README.md",
    "docs/agent-harness/workflow.md",
    "docs/agent-harness/branch-protection.md",
    "docs/agent-harness/clean-context-protocol.md",
    "docs/agent-harness/agent-run-schema.json",
    "docs/agent-harness/enforcement.json",
    "docs/agent-harness/coverage-policy.md",
    "docs/agent-harness/coverage-baseline.json",
    "docs/agent-harness/metadata-policy.md",
    "docs/agent-harness/test-quality-rubric.md",
    "docs/agent-harness/review-rubric.md",
    "docs/agent-harness/implementation-notes.md",
    "docs/testing/README.md",
    "docs/testing/fixture-policy.md",
    "docs/testing/numerical-tolerance-policy.md",
    "docs/testing/oracle-policy.md",
    "docs/product-specs/README.md",
    "docs/references/agentic-harness/README.md",
    "docs/references/agentic-harness/sources.yml",
    "scripts/agent_harness/__init__.py",
    "scripts/agent_harness/validate_harness.py",
    "scripts/agent_harness/validate_references.py",
    "scripts/agent_harness/validate_agent_run.py",
    "scripts/agent_harness/validate_pr.py",
    "scripts/agent_harness/validate_write_scope.py",
    "scripts/agent_harness/validate_bash_command.py",
    "scripts/agent_harness/prove_red_tests.py",
    "scripts/agent_harness/coverage_gate.py",
    "scripts/agent_harness/diff_coverage_gate.py",
    "scripts/agent_harness/fixture_audit.py",
    "scripts/agent_harness/downstream_smoke.py",
    "scripts/agent_harness/format_touched.py",
    "scripts/agent_harness/session_stop_check.py",
]

REQUIRED_MAKE_TARGETS = [
    "bootstrap",
    "test-fast",
    "coverage",
    "harness-check",
    "package",
    "check",
]

QUALITY_FIELDS = [
    "Last updated:",
    "Default branch:",
    "Default branch commit:",
    "Required gate present:",
    "Required gate unskipped:",
    "Line coverage:",
    "Branch coverage:",
    "`AGENTS.md` current:",
    "Reference manifest valid:",
    "Branch protection configured:",
]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def check_required_files(errors: list[str]) -> None:
    for relpath in REQUIRED_FILES:
        if not (ROOT / relpath).is_file():
            errors.append(f"missing required file: {relpath}")


def check_agents(errors: list[str]) -> None:
    path = ROOT / "AGENTS.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > 250:
        errors.append(f"AGENTS.md is {len(lines)} lines; maximum is 250")
    required_snippets = [
        "docs/generated/repo-intake.md",
        "docs/agent-harness/workflow.md",
        "docs/agent-harness/clean-context-protocol.md",
        "make harness-check",
        "ci-required",
    ]
    text = "\n".join(lines)
    for snippet in required_snippets:
        if snippet not in text:
            errors.append(f"AGENTS.md missing required snippet: {snippet}")


def check_claude(errors: list[str]) -> None:
    path = ROOT / "CLAUDE.md"
    if path.exists() and "AGENTS.md" not in path.read_text(encoding="utf-8"):
        errors.append("CLAUDE.md must point to AGENTS.md")


def check_makefile(errors: list[str]) -> None:
    path = ROOT / "Makefile"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for target in REQUIRED_MAKE_TARGETS:
        if not re.search(rf"^{re.escape(target)}\s*:", text, flags=re.MULTILINE):
            errors.append(f"Makefile missing target: {target}")


def check_quality_score(errors: list[str]) -> None:
    path = ROOT / "QUALITY_SCORE.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for field in QUALITY_FIELDS:
        if field not in text:
            errors.append(f"QUALITY_SCORE.md missing field: {field}")


def check_ci(errors: list[str]) -> None:
    path = ROOT / ".github/workflows/ci.yml"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if re.match(r"\s*paths(-ignore)?\s*:", line):
            errors.append(f"ci.yml must not use path filters; found line {lineno}: {line.strip()}")
    for snippet in [
        "pull_request:",
        "push:",
        "workflow_dispatch:",
        "contents: read",
        "ci-required:",
        "needs: [harness-validate, lint, tests, package]",
        "validate_harness.py",
        "validate_references.py",
        "validate_pr.py --ci",
    ]:
        if snippet not in text:
            errors.append(f"ci.yml missing required snippet: {snippet}")


def check_gitignore(errors: list[str]) -> None:
    path = ROOT / ".gitignore"
    if not path.exists():
        errors.append(".gitignore missing")
        return
    text = path.read_text(encoding="utf-8")
    if ".claude/worktrees/" not in text:
        errors.append(".gitignore must ignore .claude/worktrees/")


def check_claude_artifacts(errors: list[str]) -> None:
    claude_dir = ROOT / ".claude"
    if not claude_dir.exists():
        notes = read_text("docs/agent-harness/implementation-notes.md")
        if "not added because" not in notes:
            errors.append("missing implementation note explaining absent .claude artifacts")
        return

    required_skills = [
        "repo-intake",
        "reference-snapshot",
        "write-red-tests",
        "implement-to-tests",
        "adversarial-review",
        "scientific-numerics-review",
        "ci-triage",
        "coverage-gap-hunt",
        "fixture-audit",
        "doc-gardener",
        "release-guard",
    ]
    skills_dir = claude_dir / "skills"
    if skills_dir.exists():
        for skill in required_skills:
            if not (skills_dir / skill / "SKILL.md").is_file():
                errors.append(f".claude skills enabled but missing: {skill}/SKILL.md")

    required_agents = [
        "scout.md",
        "test-author.md",
        "implementer.md",
        "adversarial-reviewer.md",
        "numerics-reviewer.md",
        "ci-triager.md",
        "doc-gardener.md",
    ]
    agents_dir = claude_dir / "agents"
    if agents_dir.exists():
        for agent in required_agents:
            if not (agents_dir / agent).is_file():
                errors.append(f".claude agents enabled but missing: {agent}")


def check_branch_protection_docs(errors: list[str]) -> None:
    text = read_text("docs/agent-harness/branch-protection.md")
    for snippet in ["ci-required", "Direct pushes blocked", "Force pushes blocked"]:
        if snippet not in text:
            errors.append(f"branch-protection.md missing: {snippet}")


def check_enforcement_config(errors: list[str]) -> None:
    path = ROOT / "docs/agent-harness/enforcement.json"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for snippet in [
        '"phase": 2',
        '"clean_context_metadata_enforced": true',
        '"coverage_non_decrease_enforced": false',
    ]:
        if snippet not in text:
            errors.append(f"enforcement.json missing required Phase 2 setting: {snippet}")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_agents(errors)
    check_claude(errors)
    check_makefile(errors)
    check_quality_score(errors)
    check_ci(errors)
    check_gitignore(errors)
    check_claude_artifacts(errors)
    check_branch_protection_docs(errors)
    check_enforcement_config(errors)

    if errors:
        print("Harness validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Harness validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
