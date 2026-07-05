"""Validate local or GitHub pull request evidence for the harness."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PR_TEMPLATE = ROOT / ".github/pull_request_template.md"
ENFORCEMENT = ROOT / "docs/agent-harness/enforcement.json"
RUNS_DIR = ROOT / "docs/agent-harness/runs"
RED_TEST_PROOFS = ROOT / "docs/agent-harness/red-test-proofs"
TEST_AMENDMENTS = ROOT / "docs/agent-harness/test-amendments"

REQUIRED_TEMPLATE_SECTIONS = [
    "## Linked issue",
    "## Change summary",
    "## Phase",
    "## Agent workflow evidence",
    "## Tests and commands",
    "## Coverage",
    "## Scientific, hardware, or numerical impact",
    "## Public API impact",
    "## Downstream impact",
    "## Release impact",
    "## Human/admin decisions needed",
]

LINKED_ISSUE_RE = re.compile(r"\b(?:Closes|Fixes|Refs)\s+#(?P<number>\d+)\b", re.I)
PLACEHOLDERS = ["Closes #\n", "<explain>", "Paste exact commands"]

PRODUCT_PREFIXES = ("dphtools/",)
TEST_PREFIXES = ("tests/",)
NUMERICAL_PREFIXES = (
    "dphtools/display.py",
    "dphtools/utils/",
)
HIGH_RISK_PREFIXES = (
    ".github/",
    "AGENTS.md",
    "CLAUDE.md",
    "ARCHITECTURE.md",
    "QUALITY_SCORE.md",
    "docs/agent-harness/",
    "docs/references/",
    "docs/testing/",
    "scripts/agent_harness/",
    "setup.py",
    "setup.cfg",
    "requirements",
    "environment.yml",
    "conda.recipe/",
)
RELEASE_PREFIXES = (".github/workflows/make_release.yml", "conda.recipe/")
CI_PREFIXES = (".github/workflows/",)


def read_enforcement() -> dict[str, Any]:
    if not ENFORCEMENT.is_file():
        return {"phase": 0, "clean_context_metadata_enforced": False}
    return json.loads(ENFORCEMENT.read_text(encoding="utf-8"))


def check_template(errors: list[str]) -> None:
    if not PR_TEMPLATE.is_file():
        errors.append("missing .github/pull_request_template.md")
        return
    text = PR_TEMPLATE.read_text(encoding="utf-8")
    for section in REQUIRED_TEMPLATE_SECTIONS:
        if section not in text:
            errors.append(f"PR template missing section: {section}")


def extract_linked_issues(body: str) -> list[str]:
    issues: list[str] = []
    for match in LINKED_ISSUE_RE.finditer(body):
        number = match.group("number")
        if number not in issues:
            issues.append(number)
    return issues


def issue_exists(issue_number: str, errors: list[str]) -> None:
    if not os.environ.get("GITHUB_REPOSITORY"):
        return
    issue = github_issue_metadata(issue_number)
    if issue is None:
        errors.append(f"linked issue #{issue_number} could not be read")


def github_issue_metadata(issue_number: str) -> dict[str, Any] | None:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        return None
    url = f"https://api.github.com/repos/{repository}/issues/{issue_number}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else None
    except urllib.error.HTTPError as exc:
        print(f"GitHub issue metadata fetch failed for #{issue_number}: HTTP {exc.code}")
    except (json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"GitHub issue metadata fetch failed for #{issue_number}: {exc}")
    return None


def section_content(body: str, section: str) -> str:
    marker = f"## {section}"
    start = body.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    next_section = body.find("\n## ", start)
    if next_section == -1:
        return body[start:].strip()
    return body[start:next_section].strip()


def label_names(pull_request: dict[str, Any]) -> set[str]:
    labels = pull_request.get("labels") or []
    return {label.get("name", "") for label in labels if isinstance(label, dict)}


def issue_label_names(issue: dict[str, Any]) -> set[str]:
    labels = issue.get("labels") or []
    return {label.get("name", "") for label in labels if isinstance(label, dict)}


def changed_files_from_git(base_ref: str | None) -> list[str]:
    candidates: list[list[str]] = []
    if base_ref:
        candidates.append(["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"])
        candidates.append(["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    candidates.append(["git", "diff", "--name-only", "HEAD^...HEAD"])

    for command in candidates:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            continue
        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if files:
            return files
    return []


def startswith_any(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def load_run_metadata(issue: str) -> list[dict[str, Any]]:
    paths = sorted((RUNS_DIR / issue).glob("*.json"))
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload["_path"] = str(path.relative_to(ROOT))
            payloads.append(payload)
    return payloads


def roles_present(payloads: list[dict[str, Any]]) -> set[str]:
    return {
        payload["role"]
        for payload in payloads
        if payload.get("result") == "passed" and isinstance(payload.get("role"), str)
    }


def red_test_proof_exists(issue: str, payloads: list[dict[str, Any]]) -> bool:
    if (RED_TEST_PROOFS / f"{issue}.md").is_file():
        return True
    for payload in payloads:
        if payload.get("role") != "test-author":
            continue
        artifacts = payload.get("artifacts") or []
        if any("red-test" in artifact or "red_test" in artifact for artifact in artifacts):
            return True
        for command in payload.get("commands_run") or []:
            if (
                "prove_red_tests.py" in command.get("command", "")
                and command.get("exit_code") == 0
            ):
                return True
    return False


def check_role_scopes(issue: str, payloads: list[dict[str, Any]], errors: list[str]) -> None:
    amendment_exists = (TEST_AMENDMENTS / f"{issue}.md").is_file()
    for payload in payloads:
        role = payload.get("role")
        allowed_paths = payload.get("allowed_paths") or []
        if role in {"scout", "adversarial-reviewer", "numerics-reviewer"} and allowed_paths:
            errors.append(f"{payload.get('_path')}: {role} must be read-only")
        if role == "test-author":
            for path in allowed_paths:
                if startswith_any(path, PRODUCT_PREFIXES):
                    errors.append(
                        f"{payload.get('_path')}: test-author may not write product source"
                    )
        if role == "implementer" and not amendment_exists:
            for path in allowed_paths:
                if startswith_any(path, TEST_PREFIXES):
                    errors.append(
                        f"{payload.get('_path')}: implementer may not write tests without a test amendment"
                    )


def required_product_roles(product_changed: bool, numerical_changed: bool) -> set[str]:
    if not product_changed:
        return set()

    roles = {"test-author", "implementer", "adversarial-reviewer"}
    if numerical_changed:
        roles.add("numerics-reviewer")
    return roles


def required_pr_roles(labels: set[str], changed_files: list[str]) -> set[str]:
    roles: set[str] = set()
    if "risk:high" in labels:
        roles.add("scout")
        roles.add("adversarial-reviewer")
    if any(startswith_any(path, CI_PREFIXES) for path in changed_files):
        roles.add("ci-triager")
    if any(startswith_any(path, RELEASE_PREFIXES) for path in changed_files):
        roles.add("release-guard")
    return roles


def check_pr_body(body: str, errors: list[str]) -> list[str]:
    for section in REQUIRED_TEMPLATE_SECTIONS:
        if section not in body:
            errors.append(f"PR body missing template section: {section}")
    for placeholder in PLACEHOLDERS:
        if placeholder in body:
            errors.append(f"PR body still contains placeholder text: {placeholder}")

    issues = extract_linked_issues(body)
    if not issues:
        errors.append("PR body must link an issue with Closes #, Fixes #, or Refs #")
        return []
    for issue in issues:
        issue_exists(issue, errors)
    return issues


def check_labels(labels: set[str], changed_files: list[str], errors: list[str]) -> None:
    risk_labels = labels & {"risk:low", "risk:medium", "risk:high"}
    if len(risk_labels) != 1:
        errors.append("PR must have exactly one risk label")

    high_risk_changed = any(startswith_any(path, HIGH_RISK_PREFIXES) for path in changed_files)
    if high_risk_changed and "risk:high" not in labels:
        errors.append(
            "high-risk governance, packaging, release, or harness changes require risk:high"
        )


def check_human_decision(
    body: str, changed_files: list[str], labels: set[str], errors: list[str]
) -> None:
    high_risk_changed = any(startswith_any(path, HIGH_RISK_PREFIXES) for path in changed_files)
    if "risk:high" not in labels and not high_risk_changed:
        return
    content = section_content(body, "Human/admin decisions needed")
    if not content:
        errors.append("risk:high changes require a human/admin decision note in the PR body")


def check_single_issue_phase2_evidence(
    issue: str,
    labels: set[str],
    changed_files: list[str],
    product_changed: bool,
    numerical_changed: bool,
) -> list[str]:
    errors: list[str] = []
    product_changed = any(startswith_any(path, PRODUCT_PREFIXES) for path in changed_files)
    numerical_changed = any(startswith_any(path, NUMERICAL_PREFIXES) for path in changed_files)
    if not product_changed:
        return []

    payloads = load_run_metadata(issue)
    present = roles_present(payloads)
    needed = required_product_roles(product_changed, numerical_changed)
    missing = sorted(needed - present)
    if missing:
        errors.append(
            f"missing passed agent run metadata for issue #{issue}: {', '.join(missing)}"
        )
    if not red_test_proof_exists(issue, payloads):
        errors.append(f"missing red-test proof for issue #{issue}")
    check_role_scopes(issue, payloads, errors)
    return errors


def check_phase2_evidence(
    issues: list[str],
    labels: set[str],
    changed_files: list[str],
    errors: list[str],
) -> None:
    product_changed = any(startswith_any(path, PRODUCT_PREFIXES) for path in changed_files)
    numerical_changed = any(startswith_any(path, NUMERICAL_PREFIXES) for path in changed_files)
    if not product_changed:
        return
    if not issues:
        return

    issue_errors: dict[str, list[str]] = {}
    for issue in issues:
        candidate_errors = check_single_issue_phase2_evidence(
            issue, labels, changed_files, product_changed, numerical_changed
        )
        if not candidate_errors:
            return
        issue_errors[issue] = candidate_errors

    errors.append("no linked issue has complete clean-context metadata for product source changes")
    for issue, candidate_errors in issue_errors.items():
        errors.append(f"issue #{issue} evidence problems:")
        errors.extend(f"  {error}" for error in candidate_errors)


def check_pr_role_evidence(
    issues: list[str],
    labels: set[str],
    changed_files: list[str],
    errors: list[str],
) -> None:
    needed = required_pr_roles(labels, changed_files)
    if not needed:
        return

    payloads: list[dict[str, Any]] = []
    for issue in issues:
        issue_payloads = load_run_metadata(issue)
        payloads.extend(issue_payloads)
        check_role_scopes(issue, issue_payloads, errors)

    present = roles_present(payloads)
    missing = sorted(needed - present)
    if missing:
        errors.append("missing passed PR-level agent run metadata: " + ", ".join(missing))


def check_ci_event(errors: list[str]) -> None:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_name != "pull_request":
        print("No pull_request GitHub event; skipping PR body checks.")
        return
    if not event_path:
        errors.append("GITHUB_EVENT_PATH is missing for pull_request event")
        return

    payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request") or {}
    body = pull_request.get("body") or ""
    title = pull_request.get("title") or ""
    base_ref = (pull_request.get("base") or {}).get("ref")
    labels = label_names(pull_request)
    pr_number = pull_request.get("number")
    if pr_number:
        current_pr = github_issue_metadata(str(pr_number))
        if current_pr:
            body = current_pr.get("body") or body
            title = current_pr.get("title") or title
            labels = issue_label_names(current_pr) or labels
    changed_files = changed_files_from_git(base_ref)

    if not title.strip():
        errors.append("PR title is empty")
    issues = check_pr_body(body, errors)
    check_labels(labels, changed_files, errors)
    check_human_decision(body, changed_files, labels, errors)

    enforcement = read_enforcement()
    if enforcement.get("clean_context_metadata_enforced"):
        check_pr_role_evidence(issues, labels, changed_files, errors)
        check_phase2_evidence(issues, labels, changed_files, errors)

    if not changed_files:
        errors.append("could not determine changed files for PR validation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run local file-level PR checks.")
    parser.add_argument(
        "--ci", action="store_true", help="Run GitHub event PR checks when available."
    )
    args = parser.parse_args()

    errors: list[str] = []
    check_template(errors)
    if args.ci:
        check_ci_event(errors)
    elif args.local:
        print("Local mode: GitHub PR body checks are not available.")
    else:
        print("No mode selected; running local template checks only.")

    if errors:
        print("PR validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PR validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
