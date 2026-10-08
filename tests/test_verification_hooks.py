"""Real Git hook entry points in isolated repositories, with local-only remotes."""

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from .test_verification import FULL_SEQUENCE, ROOT, VerificationCommand, _detail, _executable
from .test_verification import _sequence
from .test_verification import TOOL
from .test_verification import _prove_executable_runtime
from .test_verification import _check_manifest

PYTHON_LAUNCHER = r"""
import json
import os
from pathlib import Path
import subprocess
import sys
with Path(os.environ["VERIFICATION_TEST_PYTHONS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:], "launcher": sys.argv[0]}) + "\n")
returncode = subprocess.call([sys.executable, *sys.argv[1:]])
if os.environ.get("VERIFICATION_TEST_PYTHON_EXIT"):
    Path(os.environ["VERIFICATION_TEST_PYTHON_EXIT"]).write_text(str(returncode), encoding="utf-8")
sys.exit(returncode)
"""

GIT_LAUNCHER = r"""
import json
import os
from pathlib import Path
import subprocess
import sys
with Path(os.environ["VERIFICATION_TEST_GIT_CALLS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")
if set(sys.argv[1:]) & {"push", "fetch", "pull", "clone", "ls-remote"}:
    print("Fixture prevented a hook from contacting a remote", file=sys.stderr)
    sys.exit(125)
sys.exit(subprocess.call([os.environ["VERIFICATION_TEST_REAL_GIT"], *sys.argv[1:]]))
"""


class HookRepository(VerificationCommand):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.git_executable = shutil.which("git")
        assert self.git_executable, "Git is required for the hook contract"
        self.python_record = tmp_path / "python-calls.jsonl"
        self.git_record = tmp_path / "hook-git-calls.jsonl"
        self.env.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_TERMINAL_PROMPT="0",
            GIT_ALLOW_PROTOCOL="file",
            GIT_AUTHOR_NAME="Fixture",
            GIT_COMMITTER_NAME="Fixture",
            GIT_AUTHOR_EMAIL="fixture@example.invalid",
            GIT_COMMITTER_EMAIL="fixture@example.invalid",
            VERIFICATION_TEST_REAL_GIT=self.git_executable,
            VERIFICATION_TEST_PYTHONS=str(self.python_record),
            VERIFICATION_TEST_GIT_CALLS=str(self.git_record),
        )
        self.env.pop("DPHTOOLS_PYTHON", None)
        # Remove inherited repository redirection before making disposable repositories.
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            self.env.pop(key, None)
        self.default_python = _executable(self.bin / "python", PYTHON_LAUNCHER)
        self.selected_python = _executable(self.bin / "chosen python", PYTHON_LAUNCHER)
        _executable(self.bin / "git", GIT_LAUNCHER)
        _prove_executable_runtime(self)
        hook_dir = self.repo / ".githooks"
        hook_dir.mkdir()
        for name in ("pre-commit", "pre-push"):
            shutil.copy2(ROOT / ".githooks" / name, hook_dir / name)
        (self.repo / ".gitignore").write_text(
            "reports/\ndist/\n.coverage*\n__pycache__/\n", encoding="utf-8"
        )
        template = tmp_path / "empty git template"
        template.mkdir()
        self.git("init", "--initial-branch=main", f"--template={template}")
        self.git("add", ".")
        self.git("commit", "-m", "Disposable fixture base")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()
        self.tracked = self.repo / "tracked.txt"
        self.tracked.write_text("committed\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "-m", "Disposable fixture head")
        self.head = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("config", "core.hooksPath", ".githooks")
        self.remote = tmp_path / "local bare remote"
        self.git("init", "--bare", f"--template={template}", str(self.remote))
        self.git("remote", "add", "fixture", str(self.remote))

    def git(self, *args, success=True):
        result = subprocess.run(
            [self.git_executable, *args],
            cwd=self.repo,
            env=self.env,
            input="",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        if success:
            assert result.returncode == 0, f"Git fixture/setup failed: {_detail(result)}"
        return result

    def hook(self):
        return self.git("hook", "run", "pre-commit", success=False)

    def push(self, *refspecs):
        return self.git("push", "fixture", *refspecs, success=False)

    def remote_refs(self):
        result = self.git(
            "--git-dir", str(self.remote), "for-each-ref", "--format=%(refname) %(objectname)"
        )
        return result.stdout

    def python_calls(self):
        if not self.python_record.exists():
            return []
        return [
            json.loads(line)
            for line in self.python_record.read_text(encoding="utf-8").splitlines()
        ]

    def assert_no_hook_remote_calls(self):
        if self.git_record.exists():
            calls = [
                json.loads(line)
                for line in self.git_record.read_text(encoding="utf-8").splitlines()
            ]
            assert not any(
                set(args) & {"push", "fetch", "pull", "clone", "ls-remote"} for args in calls
            ), calls


@pytest.fixture
def hook_repo(tmp_path):
    repo = HookRepository(tmp_path)
    yield repo
    repo.assert_no_hook_remote_calls()


@pytest.mark.parametrize("selection", ["default", "explicit"])
def test_pre_commit_checks_staged_content_with_selected_interpreter(hook_repo, selection):
    if selection == "explicit":
        hook_repo.env["DPHTOOLS_PYTHON"] = str(hook_repo.selected_python)
    hook_repo.tracked.write_text("staged change\n", encoding="utf-8")
    hook_repo.git("add", "tracked.txt")
    before = hook_repo.git("write-tree").stdout
    result = hook_repo.hook()
    assert result.returncode == 0, _detail(result)
    assert _sequence(hook_repo.calls()) == ["black", "flake8", "pydocstyle"]
    calls = hook_repo.python_calls()
    assert len(calls) == 1 and calls[0]["argv"][-1] == "fast", calls
    expected = hook_repo.selected_python if selection == "explicit" else hook_repo.default_python
    assert Path(calls[0]["launcher"]) == expected
    assert hook_repo.git("write-tree").stdout == before
    assert hook_repo.tracked.read_text(encoding="utf-8") == "staged change\n"


@pytest.mark.parametrize("state", ["unstaged", "partially-staged", "tracked-deletion"])
def test_pre_commit_rejects_unstaged_tracked_content(hook_repo, state):
    if state == "partially-staged":
        hook_repo.tracked.write_text("staged\n", encoding="utf-8")
        hook_repo.git("add", "tracked.txt")
    if state == "tracked-deletion":
        hook_repo.tracked.unlink()
    else:
        hook_repo.tracked.write_text("not staged\n", encoding="utf-8")
    result = hook_repo.hook()
    assert result.returncode != 0, _detail(result)
    assert result.stdout + result.stderr
    assert hook_repo.calls() == [], "Reject incoherent tracked content before verification"


def test_pre_commit_rejects_failed_fast_checks(hook_repo):
    hook_repo.env["VERIFICATION_TEST_FAIL"] = "black"
    result = hook_repo.hook()
    assert result.returncode != 0, _detail(result)
    assert _sequence(hook_repo.calls()) == ["black", "flake8", "pydocstyle"]


@pytest.mark.parametrize("selection", ["default", "explicit"])
def test_pre_push_checks_head_once_for_multiple_refs(hook_repo, selection):
    """S1-S2: both interpreter routes run the real fast verifier once for all refs."""
    if selection == "explicit":
        hook_repo.env["DPHTOOLS_PYTHON"] = str(hook_repo.selected_python)
    _assert_clean_head(hook_repo)
    result = hook_repo.push("HEAD:refs/heads/one", "HEAD:refs/heads/two")
    assert result.returncode == 0, _detail(result)
    _assert_fast_invocation(hook_repo, selection)
    _check_manifest(hook_repo, result, "fast")
    assert hook_repo.remote_refs().splitlines() == [
        f"refs/heads/one {hook_repo.head}",
        f"refs/heads/two {hook_repo.head}",
    ]
    _assert_clean_head(hook_repo)


def _assert_clean_head(hook_repo):
    assert hook_repo.git("rev-parse", "HEAD").stdout.strip() == hook_repo.head
    assert hook_repo.git("status", "--porcelain", "--untracked-files=all").stdout == ""


def _assert_check_operations(calls, mode):
    observed = _sequence(calls)
    expected = FULL_SEQUENCE if mode == "full" else ["black", "flake8", "pydocstyle"]
    # Reuse the operation inventory, not its incidental total ordering.
    assert Counter(observed) == Counter(expected), observed
    if mode == "full":
        dependencies = [
            ("build", "pip"),
            ("pip", "coverage:run"),
            ("coverage:erase", "coverage:run"),
            ("coverage:run", "coverage:combine"),
            ("coverage:combine", "coverage:json"),
            ("coverage:combine", "coverage:xml"),
            ("coverage:combine", "coverage:report"),
        ]
        for before, after in dependencies:
            assert observed.index(before) < observed.index(after), (before, after, observed)


def _assert_fast_invocation(hook_repo, selection="default"):
    _assert_check_operations(hook_repo.calls(), "fast")
    calls = hook_repo.python_calls()
    assert len(calls) == 1 and len(calls[0]["argv"]) == 2, calls
    assert calls[0]["argv"][1] == "fast", calls
    assert (hook_repo.repo / calls[0]["argv"][0]).resolve() == hook_repo.entry.resolve()
    expected = hook_repo.selected_python if selection == "explicit" else hook_repo.default_python
    assert Path(calls[0]["launcher"]) == expected


@pytest.mark.parametrize("state", ["unstaged", "staged", "untracked"])
def test_pre_push_rejects_dirty_tree_without_updating_remote(hook_repo, state):
    """S5-S7: each dirty state is observed before a real push is rejected."""
    _assert_clean_head(hook_repo)
    path = hook_repo.repo / "untracked.txt" if state == "untracked" else hook_repo.tracked
    path.write_text("dirty\n", encoding="utf-8")
    if state == "staged":
        hook_repo.git("add", "tracked.txt")
    expected = {
        "unstaged": " M tracked.txt",
        "staged": "M  tracked.txt",
        "untracked": "?? untracked.txt",
    }
    assert hook_repo.git("status", "--porcelain", "--untracked-files=all").stdout.rstrip() == (
        expected[state]
    )
    before = hook_repo.remote_refs()
    result = hook_repo.push("HEAD:refs/heads/main")
    assert result.returncode != 0, _detail(result)
    assert hook_repo.remote_refs() == before
    assert hook_repo.calls() == []
    assert hook_repo.python_calls() == [], "Reject dirty content before starting the verifier"
    diagnostic = (result.stdout + result.stderr).lower()
    assert any(
        word in diagnostic for word in ("clean", "dirty", "changes", "untracked", "staged")
    ), _detail(result)


@pytest.mark.parametrize("multiple", [False, True], ids=["single-ref", "multiple-refs"])
def test_pre_push_rejects_any_non_head_source_revision(hook_repo, multiple):
    """S8: one non-HEAD source prevents every requested ref update."""
    _assert_clean_head(hook_repo)
    assert hook_repo.base != hook_repo.head
    assert hook_repo.git("cat-file", "-t", hook_repo.base).stdout.strip() == "commit"
    refspecs = [f"{hook_repo.base}:refs/heads/old"]
    if multiple:
        refspecs.insert(0, "HEAD:refs/heads/current")
    before = hook_repo.remote_refs()
    result = hook_repo.push(*refspecs)
    assert result.returncode != 0, _detail(result)
    assert hook_repo.remote_refs() == ""
    assert hook_repo.remote_refs() == before
    assert hook_repo.calls() == []
    assert hook_repo.python_calls() == [], "Reject non-HEAD revisions before verification"
    assert "head" in (result.stdout + result.stderr).lower(), _detail(result)


def test_pre_push_allows_deletion_alongside_head_revision(hook_repo):
    """S9: deletion refs are exempt from the source-equals-HEAD requirement."""
    seed = hook_repo.push("HEAD:refs/heads/delete-me")
    assert seed.returncode == 0, _detail(seed)
    assert hook_repo.remote_refs().strip() == f"refs/heads/delete-me {hook_repo.head}"
    hook_repo.record.unlink()
    hook_repo.python_record.unlink()
    _assert_clean_head(hook_repo)
    result = hook_repo.push(":refs/heads/delete-me", "HEAD:refs/heads/keep-me")
    assert result.returncode == 0, _detail(result)
    _assert_fast_invocation(hook_repo)
    _check_manifest(hook_repo, result, "fast")
    assert len(hook_repo.python_calls()) == 1
    assert hook_repo.remote_refs().strip() == f"refs/heads/keep-me {hook_repo.head}"
    _assert_clean_head(hook_repo)


def test_pre_push_full_only_failure_allows_real_local_backup(hook_repo):
    """S3: the same full-only fault fails full but permits a fast-checked backup."""
    hook_repo.env["VERIFICATION_TEST_FAIL"] = "pip_audit"
    _assert_clean_head(hook_repo)
    full = hook_repo.run("full")
    assert full.returncode == 1, _detail(full)
    # The former all-later-tools expectation is replaced by the authorized DAG:
    # audit failure preserves independent cheap results and blocks costly dependents.
    assert _sequence(hook_repo.calls()) == ["black", "flake8", "pydocstyle", "mypy", "pip_audit"]
    full_report = _check_manifest(hook_repo, full, "full", "pip_audit")
    receipt = json.loads((full_report / "checks.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is True and receipt["outcome"] == "failed"
    steps = {step["name"]: step for step in receipt["steps"]}
    assert steps["audit"]["state"] == "failed" and steps["audit"]["returncode"] == 23
    assert all(
        steps[name]["state"] == "passed"
        for name in ("preflight", "format", "lint", "docstrings", "types")
    )
    blocked = {
        "build",
        "wheel-artifacts",
        "clean-install",
        "install",
        "coverage-erase",
        "tests",
        "coverage-data",
        "coverage-combine",
        "coverage-json",
        "coverage-xml",
        "coverage-report",
        "report-validation",
    }
    assert {name for name, step in steps.items() if step["state"] == "blocked"} == blocked
    assert all(
        steps[name]["returncode"] is None and steps[name]["blocking_reasons"] for name in blocked
    )
    assert "audit" in steps["build"]["dependencies"]
    assert "audit: failed" in steps["build"]["blocking_reasons"]
    assert "build: blocked" in steps["wheel-artifacts"]["blocking_reasons"]
    assert "wheel-artifacts: blocked" in steps["clean-install"]["blocking_reasons"]
    assert "clean-install: blocked" in steps["install"]["blocking_reasons"]
    assert "coverage-erase: blocked" in steps["tests"]["blocking_reasons"]
    assert hook_repo.remote_refs() == ""
    assert hook_repo.python_calls() == [], "The full demonstration is separate from the hook"
    _assert_clean_head(hook_repo)
    hook_repo.record.unlink()

    # Keep the same repository, interpreter, tool stand-ins, and failure setting.
    # This disposable file-only remote has no PR service or open PR.
    result = hook_repo.push("HEAD:refs/heads/backup")
    assert result.returncode == 0, _detail(result)
    _assert_fast_invocation(hook_repo)
    fast_report = _check_manifest(hook_repo, result, "fast")
    assert fast_report != full_report
    assert hook_repo.remote_refs().strip() == f"refs/heads/backup {hook_repo.head}"
    _assert_clean_head(hook_repo)


def test_pre_push_fast_failure_rejects_real_local_push(hook_repo):
    """S4: a real fast-verifier failure blocks the push without updating refs."""
    hook_repo.env["VERIFICATION_TEST_FAIL"] = "black"
    _assert_clean_head(hook_repo)
    before = hook_repo.remote_refs()
    result = hook_repo.push("HEAD:refs/heads/main")
    assert result.returncode != 0, _detail(result)
    _assert_fast_invocation(hook_repo)
    _check_manifest(hook_repo, result, "fast", "black")
    assert hook_repo.remote_refs() == ""
    assert hook_repo.remote_refs() == before
    _assert_clean_head(hook_repo)


@pytest.mark.parametrize("hook", ["pre-commit", "pre-push"])
def test_hooks_fail_visibly_when_selected_interpreter_is_missing(hook_repo, hook):
    """S10 and retained pre-commit behavior: a missing interpreter fails visibly."""
    _assert_clean_head(hook_repo)
    hook_repo.env["DPHTOOLS_PYTHON"] = str(hook_repo.bin / "missing interpreter")
    assert not Path(hook_repo.env["DPHTOOLS_PYTHON"]).exists()
    before = hook_repo.remote_refs()
    result = hook_repo.hook() if hook == "pre-commit" else hook_repo.push("HEAD:refs/heads/main")
    assert result.returncode != 0, _detail(result)
    assert "missing interpreter" in result.stdout + result.stderr, _detail(result)
    assert hook_repo.calls() == []
    assert hook_repo.remote_refs() == ""
    assert hook_repo.remote_refs() == before
    assert hook_repo.python_calls() == []


# This code runs only in an external-tool stand-in, inside a disposable repository.
# The actual verifier and hooks are never replaced or imported.
STATE_CHANGE = r"""
import json
import os
from pathlib import Path
import subprocess

def git(*args):
    result = subprocess.run([os.environ["VERIFICATION_TEST_REAL_GIT"], *args],
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=20, check=True)
    return result.stdout.strip()

mutation = os.environ["VERIFICATION_TEST_STATE_CHANGE"]
before = {"head": git("rev-parse", "HEAD"), "index": git("write-tree")}
if mutation in ("tracked", "staged"):
    Path("tracked.txt").write_text("changed during verification\n", encoding="utf-8")
    if mutation == "staged":
        git("add", "tracked.txt")
elif mutation == "head":
    # A new commit with the identical tree isolates HEAD drift from dirty files,
    # without invoking the pre-commit hook recursively.
    tree = git("rev-parse", "HEAD^{tree}")
    replacement = git("commit-tree", tree, "-p", before["head"], "-m", "Fixture HEAD drift")
    git("update-ref", "HEAD", replacement, before["head"])
elif mutation == "untracked":
    Path("arrived-during-check.txt").write_text("new file\n", encoding="utf-8")
else:
    raise ValueError("Unknown fixture mutation: " + mutation)
after = {"head": git("rev-parse", "HEAD"), "index": git("write-tree")}
Path(os.environ["VERIFICATION_TEST_STATE_RECEIPT"]).write_text(
    json.dumps({"mutation": mutation, "before": before, "after": after}), encoding="utf-8"
)
"""


def _arrange_state_change(hook_repo, mutation):
    receipt = hook_repo.record.with_name("state-change.json")
    status = hook_repo.record.with_name("verifier-exit.txt")
    hook_repo.env.update(
        VERIFICATION_TEST_STATE_CHANGE=mutation,
        VERIFICATION_TEST_STATE_RECEIPT=str(receipt),
        VERIFICATION_TEST_PYTHON_EXIT=str(status),
    )
    (hook_repo.modules / "pydocstyle.py").write_text(STATE_CHANGE + TOOL, encoding="utf-8")
    return receipt, status


def _assert_successful_check(hook_repo, mode, status):
    assert status.is_file(), "The real verifier did not finish through the interpreter fixture"
    assert status.read_text(encoding="utf-8") == "0", "Verifier failed before stale-state check"
    calls = hook_repo.python_calls()
    assert len(calls) == 1 and calls[0]["argv"][-1] == mode, calls
    _assert_check_operations(hook_repo.calls(), mode)
    reports = hook_repo.reports()
    assert len(reports) == 1
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    assert data["mode"] == mode
    assert data["steps"] and all(step["returncode"] == 0 for step in data["steps"])


@pytest.mark.parametrize("mutation", ["tracked", "staged"])
def test_pre_commit_rejects_content_changed_during_successful_checks(hook_repo, mutation):
    hook_repo.tracked.write_text("staged before verification\n", encoding="utf-8")
    hook_repo.git("add", "tracked.txt")
    index_before = hook_repo.git("write-tree").stdout.strip()
    receipt, status = _arrange_state_change(hook_repo, mutation)
    result = hook_repo.hook()

    _assert_successful_check(hook_repo, "fast", status)
    change = json.loads(receipt.read_text(encoding="utf-8"))
    assert change["before"] == {"head": hook_repo.head, "index": index_before}
    assert change["after"]["head"] == hook_repo.head
    assert hook_repo.tracked.read_text(encoding="utf-8") == "changed during verification\n"
    unstaged = hook_repo.git("diff", "--quiet", success=False)
    if mutation == "staged":
        assert change["after"]["index"] != index_before
        assert unstaged.returncode == 0, "Staged-change fixture must have no unstaged changes"
    else:
        assert change["after"]["index"] == index_before
        assert unstaged.returncode == 1, "Tracked-change fixture must retain its unstaged edit"
    assert result.returncode != 0, f"Hook accepted stale {mutation} content: {_detail(result)}"


@pytest.mark.parametrize("mutation", ["tracked", "head", "untracked"])
def test_pre_push_rejects_state_changed_during_successful_checks(hook_repo, mutation):
    """S11-S13: a successful fast check cannot authorize changed content or HEAD."""
    assert hook_repo.git("status", "--porcelain", "--untracked-files=all").stdout == ""
    index_before = hook_repo.git("write-tree").stdout.strip()
    receipt, status = _arrange_state_change(hook_repo, mutation)
    result = hook_repo.push("HEAD:refs/heads/main")

    _assert_successful_check(hook_repo, "fast", status)
    _assert_fast_invocation(hook_repo)
    change = json.loads(receipt.read_text(encoding="utf-8"))
    assert change["before"] == {"head": hook_repo.head, "index": index_before}
    assert change["after"]["index"] == index_before
    if mutation == "head":
        assert change["after"]["head"] != hook_repo.head
        assert hook_repo.git("status", "--porcelain", "--untracked-files=all").stdout == ""
    else:
        assert change["after"]["head"] == hook_repo.head
        if mutation == "tracked":
            assert hook_repo.tracked.read_text(encoding="utf-8") == "changed during verification\n"
            assert hook_repo.git("diff", "--quiet", success=False).returncode == 1
        else:
            assert hook_repo.git("diff", "--quiet", success=False).returncode == 0
            assert hook_repo.git("ls-files", "--others", "--exclude-standard").stdout.strip() == (
                "arrived-during-check.txt"
            )
    assert result.returncode != 0, f"Hook accepted stale {mutation} state: {_detail(result)}"
    assert hook_repo.remote_refs() == "", "A stale result updated the temporary bare remote"


def test_pre_push_rejects_index_changed_during_successful_checks(hook_repo):
    """S14: a restaged edit leaves no unstaged diff but invalidates the checked index."""
    assert hook_repo.git("status", "--porcelain", "--untracked-files=all").stdout == ""
    index_before = hook_repo.git("write-tree").stdout.strip()
    refs_before = hook_repo.remote_refs()
    receipt, status = _arrange_state_change(hook_repo, "staged")
    result = hook_repo.push("HEAD:refs/heads/main")

    _assert_successful_check(hook_repo, "fast", status)
    _assert_fast_invocation(hook_repo)
    change = json.loads(receipt.read_text(encoding="utf-8"))
    assert change["before"] == {"head": hook_repo.head, "index": index_before}
    assert change["after"]["head"] == hook_repo.head
    assert change["after"]["index"] != index_before
    assert hook_repo.git("diff", "--quiet", success=False).returncode == 0
    refs_after = hook_repo.remote_refs()
    assert (
        result.returncode != 0
    ), f"Hook accepted a stale staged index; bare refs={refs_after!r}: {_detail(result)}"
    assert refs_after == refs_before, "A stale staged result updated the temporary bare remote"
