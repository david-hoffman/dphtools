"""Approved trigger hierarchy through real Git and verifier entry points."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verification_scope import classify, scope


class Repository:
    """A private real Git repository with explicit domain ownership."""

    def __init__(self, directory):
        self.root = directory / "repository"
        self.root.mkdir()
        for name in ("verification.py", "verification_inputs.py", "verification_scope.py"):
            self.write("tools/" + name, (ROOT / "tools" / name).read_text())
        self.write("dphtools/__init__.py", '"""Fixture package."""\n')
        self.write("dphtools/core.py", '"""Core."""\ndef value():\n    return 42\n')
        self.write("dphtools/never.py", '"""Never imported."""\nVALUE = 17\n')
        self.write("tools/delivery", '"""Shared entry."""\n')
        self.write("tools/release.py", '"""Release."""\ndef main():\n    return 0\n')
        for name in ("library", "doctor", "release", "verification"):
            self.write(f"tests/test_{name}.py", "def test_case():\n    assert True\n")
        self.write("tests/conftest.py", '"""Shared fixture configuration."""\n')
        self.write("docs/notes.md", "Plain notes.\n")
        self.write("docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md", "Doctor prompt.\n")
        self.write("README.md", "Packaging input.\n")
        self.write(".gitignore", "reports/\n__pycache__/\n")
        self.mapping = {
            "runtime": {
                "doctor": ["tools/delivery"],
                "release": ["tools/release.py"],
                "verification": [
                    "tools/verification.py",
                    "tools/verification_inputs.py",
                    "tools/verification_scope.py",
                ],
            },
            "tests": {
                d: [f"tests/test_{d}.py"] for d in ("library", "doctor", "release", "verification")
            },
            "support": {"tests/conftest.py": "shared"},
            "shared_tests": {
                "doctor": ["tests/test_release.py"],
                "release": ["tests/test_doctor.py"],
            },
            "prose": ["docs/notes.md"],
        }
        self.save_mapping()
        self.git("init")
        self.git("config", "user.name", "Verification fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.base = self.commit()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def save_mapping(self):
        self.write("tools/verification-domains.json", json.dumps(self.mapping))

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.root, text=True, stderr=subprocess.PIPE
        ).strip()

    def commit(self):
        self.git("add", ".")
        self.git(
            "-c", "core.hooksPath=" + str(self.root / "absent-hooks"), "commit", "-m", "fixture"
        )
        return self.git("rev-parse", "HEAD")

    def plan(self, base=None, *options):
        result = subprocess.run(
            [
                sys.executable,
                str(self.root / "tools/verification.py"),
                "classify",
                "--base",
                base or self.base,
                *options,
            ],
            cwd=self.root.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)


@pytest.fixture
def repository(tmp_path):
    return Repository(tmp_path)


@pytest.mark.parametrize(
    "path,content,name",
    [
        ("dphtools/core.py", '"""Core."""\ndef value():\n    return 43\n', "library"),
        ("tests/test_library.py", "def test_case():\n    assert 1 == 1\n", "library"),
        ("tests/test_doctor.py", "def test_case():\n    assert 1 == 1\n", "doctor"),
        ("tests/test_release.py", "def test_case():\n    assert 1 == 1\n", "release"),
        ("tools/release.py", '"""Release."""\ndef main():\n    return 1\n', "release"),
        ("docs/notes.md", "Changed plain notes.\n", "fast"),
    ],
)
def test_highest_applicable_trigger_and_disclosed_complement(repository, path, content, name):
    repository.write(path, content)
    head = repository.commit()
    result = repository.plan()
    assert result["name"] == name
    assert result["classification"]["base_sha"] == repository.base
    assert result["classification"]["head_sha"] == head
    if name == "fast":
        assert result["coverage_claim"] == "none" and result["measured_sources"] == []
    else:
        assert result["coverage_claim"] == "scoped"
        assert "dphtools/never.py" in result["measured_sources"]
        assert "tools/verification.py" in result["unvalidated_sources"]
        assert "tests/test_verification.py" in result["unvalidated_test_paths"]
        if name in ("doctor", "release"):
            assert "tests/test_doctor.py" in result["selected_test_paths"]
            assert "tests/test_release.py" in result["selected_test_paths"]
            assert "tools/delivery" in result["measured_sources"]


@pytest.mark.parametrize(
    "path,content",
    [
        ("README.md", "New package description.\n"),
        ("docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md", "Changed policy prompt.\n"),
        ("tools/delivery", '"""Changed shared entry."""\n'),
        ("tests/conftest.py", '"""Changed fixtures."""\n'),
        ("docs/notes.md", "Example:\n```python\nprint(42)\n```\n"),
        ("dphtools/core.py", "import sys\ndef value():\n    return 42\n"),
        ("dphtools/core.py", '"""Core."""\ndef value(x=1):\n    return 42\n'),
        ("dphtools/core.py", '"""Core."""\ndef value():\n    return subprocess.run([])\n'),
        ("dphtools/core.py", '"""Core."""\ndef value():\n    return thing.__version__\n'),
        ("dphtools/core.py", "not valid Python!\n"),
    ],
)
def test_shared_packaging_examples_and_uncertain_effects_promote_to_full(
    repository, path, content
):
    repository.write(path, content)
    repository.commit()
    result = repository.plan()
    assert result["name"] == "full" and result["coverage_claim"] == "global"
    assert result["unvalidated_sources"] == result["unvalidated_test_paths"] == []


@pytest.mark.parametrize(
    "operation", ["add", "delete", "rename", "unknown-runtime", "unknown-test"]
)
def test_changed_file_inventory_never_silently_selects_a_suite(repository, operation):
    if operation == "add":
        repository.write("dphtools/new.py", "VALUE = 42\n")
    elif operation == "delete":
        (repository.root / "dphtools/core.py").unlink()
    elif operation == "rename":
        (repository.root / "dphtools/core.py").rename(repository.root / "dphtools/renamed.py")
    elif operation == "unknown-runtime":
        repository.write("tools/new.py", "VALUE = 42\n")
    else:
        repository.write("tests/test_new.py", "def test_new():\n    assert True\n")
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_mixed_domains_and_uncommitted_changes_require_full(repository):
    repository.write("dphtools/core.py", '"""Core."""\ndef value():\n    return 43\n')
    assert repository.plan()["name"] == "full"
    repository.write("tests/test_doctor.py", "def test_case():\n    assert 2 == 2\n")
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_missing_unavailable_and_nonancestor_base_require_full(repository):
    assert classify(repository.root, None)["name"] == "full"
    assert repository.plan("missing-ref")["name"] == "full"
    repository.git("checkout", "-b", "other-base")
    repository.write("docs/notes.md", "Other base.\n")
    other = repository.commit()
    repository.git("checkout", "--detach", repository.base)
    assert repository.plan(other)["name"] == "full"
    assert repository.plan()["name"] == "full", "An empty diff is not understood impact"


@pytest.mark.parametrize("defect", ["duplicate", "absent", "invalid-json"])
def test_invalid_domain_inventory_falls_back_and_is_visible(repository, defect):
    if defect == "duplicate":
        repository.mapping["tests"]["doctor"].append("tests/test_library.py")
        repository.save_mapping()
    elif defect == "absent":
        repository.mapping["tests"]["library"].append("tests/absent.py")
        repository.save_mapping()
    else:
        repository.write("tools/verification-domains.json", "invalid JSON")
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_explicit_scope_includes_nested_never_imported_sources_and_unknown_fallback(repository):
    repository.write("dphtools/nested/never.py", "VALUE = 42\n")
    selected = scope(repository.root, "library")
    assert "dphtools/nested/never.py" in selected["measured_sources"]
    assert selected["coverage_claim"] == "scoped"
    repository.write("tools/unassigned.py", "VALUE = 42\n")
    assert scope(repository.root, "library")["coverage_claim"] == "global"


def test_classification_output_is_data_and_changes_with_base(repository):
    repository.write("docs/notes.md", "Changed notes.\n")
    head = repository.commit()
    output = repository.root / "reports/plan.json"
    result = repository.plan(None, "--output", str(output))
    assert json.loads(output.read_text()) == result and result["name"] == "fast"
    assert repository.plan(head)["name"] == "full"


def test_scope_failure_is_nonzero_without_a_false_receipt(repository):
    (repository.root / "tools/verification-domains.json").unlink()
    result = subprocess.run(
        [sys.executable, str(repository.root / "tools/verification.py"), "library"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "scope failed" in result.stderr
    assert not list(repository.root.glob("reports/**/checks.json"))


@pytest.mark.parametrize(
    "prefix",
    [
        "import subprocess as sp\n",
        "from subprocess import run as launch\n",
        "from . import helper\n",
    ],
)
def test_existing_shared_or_unknown_imports_require_full_even_through_aliases(repository, prefix):
    repository.write("dphtools/core.py", prefix + "def value():\n    return 42\n")
    repository.base = repository.commit()
    repository.write("dphtools/core.py", prefix + "def value():\n    return 43\n")
    repository.commit()
    assert repository.plan()["name"] == "full"


@pytest.mark.parametrize("prefix", ["import math\n", "from math import sin\n"])
def test_unchanged_known_scientific_imports_permit_understood_body_edits(repository, prefix):
    repository.write("dphtools/core.py", prefix + "def value():\n    return 42\n")
    repository.base = repository.commit()
    repository.write("dphtools/core.py", prefix + "def value():\n    return 43\n")
    repository.commit()
    assert repository.plan()["name"] == "library"


@pytest.mark.parametrize(
    "prefix", ["import pytest\n@pytest.fixture\n", "from pytest import fixture\n@fixture\n"]
)
def test_fixture_body_edits_are_shared_infrastructure(repository, prefix):
    repository.write("tests/test_library.py", prefix + "def example():\n    return 42\n")
    repository.base = repository.commit()
    repository.write("tests/test_library.py", prefix + "def example():\n    return 43\n")
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_body_identical_comment_edits_do_not_invent_changed_effects(repository):
    repository.write(
        "dphtools/core.py", '"""Core."""\n# A comment.\ndef value():\n    return 42\n'
    )
    repository.commit()
    assert repository.plan()["name"] == "library"


@pytest.mark.parametrize(
    "prefix",
    [
        "def helper():\n    return __import__('subprocess')\n",
        "def __getattr__(name):\n    return 42\n",
    ],
)
def test_unchanged_dynamic_helpers_or_import_hooks_require_full(repository, prefix):
    repository.write("dphtools/core.py", prefix + "def value():\n    return 42\n")
    repository.base = repository.commit()
    repository.write("dphtools/core.py", prefix + "def value():\n    return helper()\n")
    repository.commit()
    assert repository.plan()["name"] == "full"


@pytest.mark.parametrize(
    "prefix", ["import pytest\n@pytest.fixture\n", "from pytest import fixture as fx\n@fx\n"]
)
def test_release_and_doctor_fixture_edits_require_full(repository, prefix):
    repository.write("tests/test_release.py", prefix + "def prepared():\n    return 42\n")
    repository.base = repository.commit()
    repository.write("tests/test_release.py", prefix + "def prepared():\n    return 43\n")
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_changed_test_imports_and_explicit_shared_helpers_require_full(repository):
    repository.write(
        "tests/test_doctor.py", "import subprocess\ndef test_case():\n    assert True\n"
    )
    repository.commit()
    assert repository.plan()["name"] == "full"
    repository.mapping["shared_inputs"] = ["tools/release.py"]
    repository.save_mapping()
    repository.base = repository.commit()
    repository.write("tools/release.py", '"""Release."""\ndef main():\n    return 1\n')
    repository.commit()
    assert repository.plan()["name"] == "full"


@pytest.mark.parametrize(
    "suffix",
    [
        "VALUE = value()\n",
        "VALUE = value\n",
        "value()\n",
        "VALUE: int = 17\n",
        "class Initialized:\n    VALUE = value()\n",
        "def other(argument=value()):\n    return argument\n",
        "def other(argument=value):\n    return 0\n",
        "def other() -> value():\n    return 0\n",
        "def other() -> Unknown:\n    return 0\n",
        "@value\ndef other():\n    return 0\n",
        "def generic[T: value()]():\n    return 0\n",
        "VALUE, OTHER = (1, 2)\n",
    ],
)
def test_changed_function_used_in_module_initialization_or_uncertain_header_requires_full(
    repository, suffix
):
    repository.write("dphtools/core.py", "def value():\n    return 42\n" + suffix)
    repository.base = repository.commit()
    repository.write("dphtools/core.py", "def value():\n    return 43\n" + suffix)
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_plain_async_body_edit_with_literal_module_constant_remains_understood(repository):
    before = "VALUE = 17\nasync def value(argument: int = 2) -> int:\n    return 42\n"
    repository.write("dphtools/core.py", before)
    repository.base = repository.commit()
    repository.write("dphtools/core.py", before.replace("return 42", "return 43"))
    repository.commit()
    assert repository.plan()["name"] == "library"


@pytest.mark.parametrize(
    "path", ["tests/test_doctor.py", "tests/test_release.py", "tools/release.py"]
)
def test_domain_filename_cannot_narrow_existing_process_or_policy_boundaries(repository, path):
    before = "import subprocess\ndef operation():\n    return subprocess.run(['main'])\n"
    repository.write(path, before)
    repository.base = repository.commit()
    repository.write(path, before.replace("'main'", "'codex-main'"))
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_production_doctor_approval_policy_edit_requires_full(repository):
    path = "docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md"
    before = (ROOT / path).read_text()
    assert "approval before committing, pushing, or merging." in before
    repository.write(path, before)
    repository.base = repository.commit()
    repository.write(path, before.replace("approval before", "approval after"))
    repository.commit()
    assert repository.plan()["name"] == "full"


def test_fast_plan_cannot_collect_nonexistent_test_shards(repository):
    repository.write("docs/notes.md", "Changed notes.\n")
    repository.commit()
    result = subprocess.run(
        [
            sys.executable,
            str(repository.root / "tools/verification.py"),
            "collect",
            "--base",
            repository.base,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "Tier 1 uses fast directly" in result.stderr
    assert not list(repository.root.glob("reports/**/checks.json"))


@pytest.mark.parametrize(
    "prefix,expression",
    [
        ("import numpy as np\n", "np.ctypeslib.load_library('unverified', '.')"),
        ("from numpy.f2py.f2py2e import run_compile\n", "run_compile()"),
        ("import numpy.ctypeslib as native\n", "43"),
        ("from numpy import *\n", "43"),
        ("", "obj.trigger"),
        ("", "obj"),
        ("", "argument[0]"),
        ("", "abs(argument)"),
    ],
)
def test_unproven_calls_attributes_import_apis_and_receivers_require_full(
    repository, prefix, expression
):
    before = prefix + "def value(argument=0):\n    return 42\n"
    repository.write("dphtools/core.py", before)
    repository.base = repository.commit()
    repository.write("dphtools/core.py", before.replace("return 42", "return " + expression))
    repository.commit()
    assert repository.plan()["name"] == "full"


@pytest.mark.parametrize(
    "before,after",
    [
        (
            "def value(flag):\n    if flag:\n        import matplotlib\n",
            "def value(flag):\n    if not flag:\n        import matplotlib\n",
        ),
        (
            "def value(argument):\n    return 42\n",
            "def value(argument):\n    with argument:\n        return 42\n",
        ),
        (
            "def helper(unknown):\n    return 42\ndef value():\n    return 42\n",
            "def helper(unknown):\n    return 42\ndef value():\n    return unknown\n",
        ),
        (
            "def value(argument=0):\n    return 42\n",
            "def value(argument=0):\n    assert True, argument\n    return 42\n",
        ),
        (
            "def value(argument=0):\n    return 42\n",
            "def value(argument=0):\n    assert argument\n    return 42\n",
        ),
        ("def value():\n    return 42\n", "def value():\n    42\n    return 43\n"),
        ("def value():\n    return 42\n", "def value():\n    x = 42\n    return x\n"),
    ],
)
def test_unsupported_implicit_execution_and_lexical_names_cannot_narrow(repository, before, after):
    repository.write("dphtools/core.py", before)
    repository.base = repository.commit()
    repository.write("dphtools/core.py", after)
    repository.commit()
    assert repository.plan()["name"] == "full"


@pytest.mark.parametrize(
    "body",
    [
        '    """Literal docstring."""\n    pass\n    assert 1 == 1, "literal message"\n    return [1, 2]\n',
        "    return\n",
        "    return (1 + 2, -3, {'key': [1, 2]})\n",
    ],
)
def test_supported_literal_only_bodies_remain_scoped(repository, body):
    header = "def value(first: 'int' = 2, *args, second=None, **kwargs) -> int:\n"
    repository.write("dphtools/core.py", header + "    return 42\n")
    repository.base = repository.commit()
    repository.write("dphtools/core.py", header + body)
    repository.commit()
    assert repository.plan()["name"] == "library"
