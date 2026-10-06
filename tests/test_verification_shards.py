"""Real collection, isolated execution, and byte-bound aggregation contracts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verification
import verification_shards as shards_runtime


class ShardedCommand:
    """Run the public verifier against a tiny real parent/child test package."""

    def __init__(self, directory):
        self.root = directory / "fixture repository"
        self.entry = self.root / "runner/verification.py"
        self.entry.parent.mkdir(parents=True)
        for name in (
            "verification.py",
            "verification_inputs.py",
            "verification_shards.py",
            "verification_parallel.py",
        ):
            shutil.copy2(ROOT / "tools" / name, self.entry.parent)
        package = self.root / "dphtools"
        package.mkdir()
        (package / "__init__.py").write_text(
            '"""Parent doctest.\n\n>>> VALUE\n42\n"""\nVALUE = 42\n'
        )
        (package / "child.py").write_text('"""Child-only measured code."""\nVALUE = 17\n')
        (package / "_version.py").write_text(
            'def get_versions():\n    return {"version": "1.0"}\n'
        )
        (self.root / "tools").mkdir()
        (self.root / "tools/delivery").write_text('"""Empty owned fixture."""\n')
        (self.root / "tests").mkdir()
        (self.root / "tests/test_real.py").write_text(
            "import os, subprocess, sys\nimport dphtools\n\n"
            "def test_parent():\n    assert dphtools.VALUE == 42\n\n"
            "def test_child():\n    subprocess.run([sys.executable, '-c', "
            "'import dphtools.child; assert dphtools.child.VALUE == 17'], check=True)\n"
        )
        (self.root / "setup.cfg").write_text(
            "[coverage:run]\nbranch = True\nparallel = True\npatch = subprocess\n"
            "include = dphtools/**/*.py\nomit = dphtools/_version.py\n"
            "[coverage:report]\nfail_under = 100\nexclude_lines =\n    (?!x)x\n"
            "partial_branches =\n    (?!x)x\n"
        )
        (self.root / "requirements-dev.lock").write_text(
            "coverage==" + __import__("coverage").__version__ + "\n"
        )
        (self.root / ".gitignore").write_text("reports/\n__pycache__/\n*.egg-info/\n")
        tools = directory / "controlled quality tools"
        tools.mkdir()
        for name in ("black", "flake8", "pydocstyle", "mypy", "pip_audit"):
            (tools / (name + ".py")).write_text("# Real child quality boundary.\n")
        (tools / "build.py").write_text(
            "import pathlib, sys, zipfile\n"
            "output = pathlib.Path(sys.argv[sys.argv.index('--outdir') + 1])\n"
            "output.mkdir(parents=True)\n"
            "with zipfile.ZipFile(output / 'dphtools-1.0-py3-none-any.whl', 'w') as wheel:\n"
            "    wheel.write('dphtools/__init__.py', 'dphtools/__init__.py')\n"
            "    wheel.writestr('dphtools-1.0.dist-info/METADATA', "
            "'Metadata-Version: 2.1\\nName: dphtools\\nVersion: 1.0\\n')\n"
            "    wheel.writestr('dphtools-1.0.dist-info/WHEEL', "
            "'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')\n"
            "    wheel.writestr('dphtools-1.0.dist-info/RECORD', '')\n"
        )
        self.env = dict(os.environ, PYTHONPATH=str(tools), PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("PYTEST_ADDOPTS", None)
        self.env.update(GITHUB_RUN_ID="fixture-run", GITHUB_RUN_ATTEMPT="1")
        for arguments in (
            ["init"],
            ["config", "user.name", "Shard fixture"],
            ["config", "user.email", "fixture@example.invalid"],
            ["add", "."],
            ["commit", "-m", "fixture"],
            ["tag", "1.0"],
        ):
            result = subprocess.run(["git", *arguments], cwd=self.root, capture_output=True)
            assert result.returncode == 0, result.stderr
        self.collect = self.root / "reports/collect"
        self.manifest = self.collect / "manifest.json"

    def run(self, mode, directory, *arguments):
        if "--workers" not in arguments:
            arguments = (*arguments, "--workers", "1")
        return subprocess.run(
            [sys.executable, str(self.entry), mode, "--report-dir", str(directory), *arguments],
            cwd=self.root.parent,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def shard(self, index, directory=None):
        directory = directory or self.root / f"reports/shard-{index}"
        result = self.run(
            "shard", directory, "--manifest", str(self.manifest), "--shard-index", str(index)
        )
        return directory, result

    def aggregate(self, directory, shards):
        return self.run(
            "aggregate", directory, "--manifest", str(self.manifest), "--shards", *map(str, shards)
        )


@pytest.fixture(scope="module")
def completed_shards(tmp_path_factory):
    command = ShardedCommand(tmp_path_factory.mktemp("real-shards"))
    collected = command.run("collect", command.collect)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    shards = []
    for index in (0, 1):
        directory, result = command.shard(index)
        assert result.returncode == 0, result.stdout + result.stderr
        shards.append(directory)
    return command, shards


def test_real_collection_is_partitioned_once_and_parent_child_coverage_aggregates(
    completed_shards,
):
    command, shards = completed_shards
    manifest = json.loads(command.manifest.read_text())
    nodes = manifest["nodes"]
    assert len(nodes) == 3 and any(node.startswith("dphtools/") for node in nodes)
    assigned = [node for shard in manifest["assignments"] for node in shard]
    assert sorted(assigned) == nodes and len(set(assigned)) == len(assigned)
    executed = []
    for directory in shards:
        receipt = json.loads((directory / "shard.json").read_text())
        executed.extend(receipt["nodes"])
        assert list(directory.glob(".coverage*"))
        assert (directory / "install/dphtools/__init__.py").is_file()
    assert sorted(executed) == nodes
    result = command.aggregate(command.root / "reports/aggregate", shards)
    assert result.returncode == 0, result.stdout + result.stderr
    coverage = json.loads((command.root / "reports/aggregate/coverage.json").read_text())
    assert coverage["files"]["dphtools/child.py"]["executed_lines"] == [2]
    assert coverage["totals"]["missing_lines"] == 0


def test_collection_node_ids_remain_repository_relative_under_ancestor_configuration(tmp_path):
    """A surrounding pytest project must not prefix this repository's frozen IDs."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    command = ShardedCommand(tmp_path)
    result = command.run("collect", command.collect)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(command.manifest.read_text())
    assert manifest["nodes"] == [
        "dphtools/__init__.py::dphtools",
        "tests/test_real.py::test_child",
        "tests/test_real.py::test_parent",
    ]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "report", "coverage", "run", "platform"]
)
def test_aggregate_rejects_incomplete_foreign_and_changed_receipts(
    completed_shards, tmp_path, fault
):
    command, original = completed_shards
    shards = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for source, target in zip(original, shards):
        shutil.copytree(source, target)
    if fault == "missing":
        shards.pop()
    elif fault == "duplicate":
        shards[1] = shards[0]
    elif fault in ("report", "coverage"):
        artifact = (
            shards[0] / "pytest.xml" if fault == "report" else next(shards[0].glob(".coverage*"))
        )
        with artifact.open("ab") as stream:
            stream.write(b"tampered")
    else:
        receipt = json.loads((shards[0] / "shard.json").read_text())
        receipt["identity"]["run"]["id"] = "old" if fault == "run" else "fixture-run"
        if fault == "platform":
            receipt["identity"]["environment"]["system"] = "Foreign"
        (shards[0] / "shard.json").write_text(json.dumps(receipt))
    result = command.aggregate(tmp_path / "aggregate", shards)
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "aggregate/checks.json").read_text())
    assert receipt["outcome"] == "failed"
    assert not (tmp_path / "aggregate/coverage.json").exists()


def test_shard_rejects_current_collection_change(completed_shards, tmp_path):
    command, _ = completed_shards
    extra = command.root / "tests/test_added.py"
    extra.write_text("def test_added():\n    assert True\n")
    try:
        _, result = command.shard(0, tmp_path / "changed")
        assert result.returncode == 1, result.stdout + result.stderr
        assert not (tmp_path / "changed/shard.json").exists()
    finally:
        extra.unlink()


@pytest.mark.parametrize("behavior", ["skip", "failure"])
def test_real_skipped_or_failed_execution_cannot_produce_success(tmp_path, behavior):
    command = ShardedCommand(tmp_path)
    test = command.root / "tests/test_real.py"
    test.write_text(
        "import pytest\ndef test_behavior():\n"
        + ("    pytest.skip('visible skip')\n" if behavior == "skip" else "    assert False\n")
    )
    result = command.run("collect", command.collect)
    assert result.returncode == 0, result.stdout + result.stderr
    statuses = [command.shard(index)[1].returncode for index in (0, 1)]
    assert 1 in statuses


@pytest.fixture(scope="module")
def interface_shards(tmp_path_factory):
    """Exercise the same production orchestration interface used by the CLI."""
    command = ShardedCommand(tmp_path_factory.mktemp("shard-interface"))
    command.collect = command.root / "reports/interface-collect"
    command.manifest = command.collect / "manifest.json"
    previous = dict(os.environ)
    os.environ.update(command.env)
    try:
        arguments = SimpleNamespace(manifest=None, durations=[])
        run = verification.VerificationRun(command.root, "collect", command.collect)
        shards_runtime.run_sharded(run, arguments)
        assert run.finish() == 0
        directories = []
        for index in (0, 1):
            directory = command.root / f"reports/interface-shard-{index}"
            run = verification.VerificationRun(command.root, "shard", directory)
            arguments = SimpleNamespace(manifest=command.manifest, shard_index=index)
            shards_runtime.run_sharded(run, arguments)
            assert run.finish() == 0
            directories.append(directory)
    finally:
        os.environ.clear()
        os.environ.update(previous)
    return command, directories


def interface_aggregate(command, directory, inputs, monkeypatch):
    """Return actual production aggregation outcomes, without replacing any tool."""
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = verification.VerificationRun(command.root, "aggregate", directory)
    arguments = SimpleNamespace(manifest=command.manifest, shards=inputs)
    shards_runtime.run_sharded(run, arguments)
    return run.finish()


def test_production_aggregate_interface_combines_real_parent_and_child_data(
    interface_shards, tmp_path, monkeypatch
):
    command, inputs = interface_shards
    assert interface_aggregate(command, tmp_path / "aggregate", inputs, monkeypatch) == 0


def _reseal(path, mutate):
    value = json.loads(path.read_text())
    value.pop("digest", None)
    mutate(value)
    shards_runtime.write_json(path, shards_runtime.sealed(value))


@pytest.mark.parametrize(
    "fault",
    [
        "manifest-digest",
        "foreign-input",
        "foreign-manifest",
        "index-type",
        "index-range",
        "index-duplicate",
        "count",
        "nodes",
        "missing-step",
        "failed-step",
        "missing-report",
        "duplicate-report",
        "collection",
        "durations",
        "missing-coverage",
        "extra-coverage",
        "junit-count",
        "malformed-json",
        "malformed-xml",
        "missing-fields",
        "foreign-node",
        "skipped",
        "failed-exit",
        "missing-execution",
        "duplicate-execution",
        "duration-type",
        "duration-negative",
        "duration-infinite",
        "coverage-tamper",
        "absent-artifact",
        "path-escape",
        "checks-absent",
        "checks-incomplete",
        "checks-failed",
        "checks-identity",
        "checks-digest",
        "checks-prefix",
        "checks-final",
        "checks-returncode",
        "checks-step-digest",
        "checks-last-log",
        "checks-command-tamper",
        "checks-duration-tamper",
        "checks-seal-absent",
        "checks-version",
    ],
)
def test_production_aggregate_rejects_semantic_receipt_faults(
    interface_shards, tmp_path, monkeypatch, fault
):
    command, original = interface_shards
    inputs = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for source, target in zip(original, inputs):
        shutil.copytree(source, target)
    receipt_path = inputs[0] / "shard.json"
    receipt = json.loads(receipt_path.read_text())
    execution = inputs[0] / "execution.json"
    if fault.startswith("checks-"):
        checks_path = inputs[0] / "checks.json"
        checks = json.loads(checks_path.read_text())
        if fault == "checks-absent":
            checks_path.unlink()
        elif fault == "checks-last-log":
            (inputs[0] / checks["steps"][-1]["log"]["path"]).write_text("changed")
        else:
            if fault == "checks-incomplete":
                checks["complete"] = False
            elif fault == "checks-failed":
                checks["outcome"] = "failed"
            elif fault == "checks-identity":
                checks["identity"]["coverage_settings"] = "foreign"
            elif fault == "checks-digest":
                checks["identity_digest"] = "foreign"
            elif fault == "checks-prefix":
                checks["steps"].pop(0)
            elif fault == "checks-final":
                checks["steps"][-1]["name"] = "foreign"
            elif fault == "checks-returncode":
                checks["steps"][-1]["returncode"] = 1
            elif fault == "checks-command-tamper":
                checks["steps"][-1]["command"] = ["unobserved command"]
            elif fault == "checks-duration-tamper":
                checks["steps"][-1]["duration_seconds"] += 42
            elif fault == "checks-seal-absent":
                checks.pop("receipt_digest")
            elif fault == "checks-version":
                checks["document_version"] = "unknown"
            else:
                checks["steps"][-1]["input_digest"] = "foreign"
            if fault not in (
                "checks-command-tamper",
                "checks-duration-tamper",
                "checks-seal-absent",
            ):
                checks.pop("receipt_digest", None)
                checks["receipt_digest"] = shards_runtime.digest(checks)
            checks_path.write_text(json.dumps(checks))
    elif fault in (
        "collection",
        "foreign-node",
        "skipped",
        "failed-exit",
        "missing-execution",
        "duplicate-execution",
        "duration-type",
        "duration-negative",
        "duration-infinite",
    ):
        actual = json.loads(execution.read_text())
        if fault == "collection":
            actual["nodes"] = ["foreign"]
        elif fault == "foreign-node":
            actual["executions"][0]["node"] = "foreign"
        elif fault == "skipped":
            actual["executions"][0]["outcome"] = "skipped"
        elif fault == "failed-exit":
            actual["exitstatus"] = 1
        elif fault == "missing-execution":
            actual["executions"].pop()
        elif fault == "duplicate-execution":
            actual["executions"].append(actual["executions"][0])
        else:
            actual["executions"][0]["duration"] = {
                "duration-type": "invalid",
                "duration-negative": -1,
                "duration-infinite": float("inf"),
            }[fault]
        shards_runtime.write_json(execution, actual)
        for record in receipt["artifacts"]:
            if record["path"] == "execution.json":
                record["sha256"] = shards_runtime.file_hash(execution)
    elif fault == "manifest-digest":
        receipt["digest"] = "wrong"
    elif fault == "foreign-input":
        receipt["identity"]["run"]["attempt"] = "previous"
    elif fault == "foreign-manifest":
        receipt["manifest"] = "foreign"
    elif fault.startswith("index-"):
        receipt["index"] = {"index-type": "0", "index-range": 2, "index-duplicate": 1}[fault]
        if fault == "index-duplicate":
            receipt["nodes"] = json.loads(command.manifest.read_text())["assignments"][1]
    elif fault == "count":
        receipt["count"] = 3
    elif fault == "nodes":
        receipt["nodes"] = ["foreign"]
    elif fault == "missing-step":
        receipt["steps"].pop()
    elif fault == "failed-step":
        receipt["steps"][0]["state"] = "failed"
    elif fault == "missing-report":
        receipt["artifacts"] = [
            record for record in receipt["artifacts"] if record["path"] != "pytest.xml"
        ]
    elif fault == "duplicate-report":
        receipt["artifacts"].append(receipt["artifacts"][0])
    elif fault == "durations":
        receipt["durations"][receipt["nodes"][0]] += 1
    elif fault == "missing-coverage":
        receipt["artifacts"] = [
            record for record in receipt["artifacts"] if not record["path"].startswith(".coverage")
        ]
    elif fault == "extra-coverage":
        (inputs[0] / ".coverage.unrecorded").write_bytes(b"foreign")
    elif fault in ("junit-count", "malformed-xml"):
        junit = inputs[0] / "pytest.xml"
        junit.write_text("<testsuites/>" if fault == "junit-count" else "<broken")
        for record in receipt["artifacts"]:
            if record["path"] == "pytest.xml":
                record["sha256"] = shards_runtime.file_hash(junit)
    elif fault == "malformed-json":
        receipt_path.write_text("{")
    elif fault == "missing-fields":
        receipt.pop("index")
    elif fault == "coverage-tamper":
        next(inputs[0].glob(".coverage*")).write_bytes(b"foreign")
    elif fault == "absent-artifact":
        execution.unlink()
    else:
        receipt["artifacts"][0]["path"] = "../outside"
    if fault not in ("manifest-digest", "malformed-json"):
        receipt.pop("digest", None)
        shards_runtime.write_json(receipt_path, shards_runtime.sealed(receipt))
    elif fault == "manifest-digest":
        shards_runtime.write_json(receipt_path, receipt)
    assert interface_aggregate(command, tmp_path / "aggregate", inputs, monkeypatch) == 1


@pytest.mark.parametrize(
    "nodes,durations",
    [
        ([], {}),
        (["a", "a"], {}),
        (["a"], {}),
        (["a", "b"], {"a": -1}),
        (["a", "b"], {"a": "1"}),
        (["a", "b"], {"a": float("inf")}),
    ],
)
def test_partition_rejects_invalid_collections_and_durations(nodes, durations):
    with pytest.raises(ValueError):
        shards_runtime.partition(nodes, durations)


def test_partition_uses_recorded_durations_and_preserves_exact_ownership():
    result = shards_runtime.partition(["a", "b", "c", "d"], {"a": 9, "b": 7, "c": 2, "d": 1})
    assert result == [["a", "d"], ["b", "c"]]


def test_real_pytest_plugin_records_full_collection_and_rejects_disagreement(
    tmp_path, monkeypatch
):
    test = tmp_path / "test_plugin_contract.py"
    test.write_text("def test_one():\n    assert True\ndef test_two():\n    assert True\n")
    output = tmp_path / "events.json"
    recorder = shards_runtime.NodeRecorder({"output": str(output)})
    assert pytest.main([str(test), "-q", "--confcutdir", str(tmp_path)], plugins=[recorder]) == 0
    actual = json.loads(output.read_text())
    assert len(actual["nodes"]) == 2
    assert set(shards_runtime.validate_execution(actual, actual["nodes"])) == set(actual["nodes"])
    mismatch = shards_runtime.NodeRecorder({"output": str(output), "expected": ["foreign"]})
    assert pytest.main([str(test), "-q", "--confcutdir", str(tmp_path)], plugins=[mismatch]) != 0
    empty = tmp_path / "test_empty_contract.py"
    empty.write_text("# no tests\n")
    recorder = shards_runtime.NodeRecorder({"output": str(output)})
    assert pytest.main([str(empty), "-q", "--confcutdir", str(tmp_path)], plugins=[recorder]) != 0


def test_pytest_plugin_is_inert_without_explicit_configuration(monkeypatch):
    monkeypatch.delenv("VERIFICATION_SHARD_CONFIG", raising=False)
    shards_runtime.pytest_configure(None)


def test_real_pytest_module_plugin_configuration_and_selection(tmp_path, monkeypatch):
    test = tmp_path / "test_configured_plugin.py"
    test.write_text("def test_one():\n    assert True\ndef test_two():\n    assert True\n")
    configuration = tmp_path / "configuration.json"
    output = tmp_path / "events.json"
    shards_runtime.write_json(configuration, {"output": str(output)})
    monkeypatch.setenv("VERIFICATION_SHARD_CONFIG", str(configuration))
    assert (
        pytest.main([str(test), "-q", "--confcutdir", str(tmp_path)], plugins=[shards_runtime])
        == 0
    )
    nodes = shards_runtime.read_json(output)["nodes"]
    shards_runtime.write_json(
        configuration, {"output": str(output), "expected": nodes, "assigned": nodes[:1]}
    )
    assert (
        pytest.main([str(test), "-q", "--confcutdir", str(tmp_path)], plugins=[shards_runtime])
        == 0
    )
    actual = shards_runtime.read_json(output)
    assert actual["nodes"] == nodes
    assert set(shards_runtime.validate_execution(actual, nodes[:1])) == set(nodes[:1])


def test_collect_accepts_sealed_real_durations(interface_shards, tmp_path, monkeypatch):
    command, inputs = interface_shards
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    directory = command.root / "reports/duration-collect"
    run = verification.VerificationRun(command.root, "collect", directory)
    arguments = SimpleNamespace(
        manifest=directory / "manifest.json",
        durations=[
            ROOT / "tools/verification-durations.json",
            inputs[0] / "shard.json",
            inputs[1] / "shard.json",
        ],
    )
    shards_runtime.run_sharded(run, arguments)
    assert run.finish() == 0
    manifest = shards_runtime.read_json(directory / "manifest.json")
    recorded = shards_runtime.read_json(ROOT / "tools/verification-durations.json")["durations"]
    for source in inputs:
        recorded.update(shards_runtime.read_json(source / "shard.json")["durations"])
    assert manifest["assignments"] == shards_runtime.partition(manifest["nodes"], recorded)


@pytest.mark.parametrize(
    "fault",
    [
        "checks-absent",
        "checks-incomplete",
        "checks-mode",
        "quality-identity",
        "quality-gate",
        "quality-failed",
        "manifest-digest",
        "manifest-count",
        "manifest-duplicate",
        "assignment-duplicate",
        "assignment-missing",
        "malformed-shape",
    ],
)
def test_manifest_rejects_unclosed_collection_and_invalid_partitions(
    interface_shards, tmp_path, monkeypatch, fault
):
    command, _ = interface_shards
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    directory = tmp_path / "collection"
    shutil.copytree(command.collect, directory)
    manifest = shards_runtime.read_json(directory / "manifest.json")
    if fault.startswith("checks-"):
        checks = directory / "checks.json"
        if fault == "checks-absent":
            checks.unlink()
        else:
            value = shards_runtime.read_json(checks)
            if fault == "checks-incomplete":
                value["complete"] = False
            else:
                value["mode"] = "full"
            value.pop("receipt_digest", None)
            value["receipt_digest"] = shards_runtime.digest(value)
            shards_runtime.write_json(checks, value)
    elif fault.startswith("quality-"):
        quality = shards_runtime.read_json(directory / "quality.json")
        if fault == "quality-identity":
            quality["identity"]["run"]["id"] = "foreign"
        elif fault == "quality-gate":
            quality["steps"] = [step for step in quality["steps"] if step["name"] != "preflight"]
        else:
            quality["steps"][0]["state"] = "failed"
        quality.pop("digest")
        shards_runtime.write_json(directory / "quality.json", shards_runtime.sealed(quality))
        manifest["quality"]["sha256"] = shards_runtime.file_hash(directory / "quality.json")
    elif fault == "manifest-count":
        manifest["assignments"].pop()
    elif fault == "manifest-duplicate":
        manifest["nodes"].append(manifest["nodes"][0])
    elif fault == "assignment-duplicate":
        manifest["assignments"][0].append(manifest["assignments"][1][0])
    elif fault == "assignment-missing":
        manifest["assignments"][0].pop()
    elif fault == "manifest-digest":
        manifest["digest"] = "foreign"
    elif fault == "malformed-shape":
        manifest = []
    if fault not in ("manifest-digest", "malformed-shape"):
        manifest.pop("digest")
        manifest = shards_runtime.sealed(manifest)
    shards_runtime.write_json(directory / "manifest.json", manifest)
    run = verification.VerificationRun(command.root, "shard", tmp_path / "worker")
    shards_runtime.run_sharded(
        run, SimpleNamespace(manifest=directory / "manifest.json", shard_index=0)
    )
    assert run.finish() == 1
    assert not (tmp_path / "worker/shard.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("shard",),
        ("aggregate",),
        ("shard", "--manifest", "missing.json"),
        ("shard", "--manifest", "missing.json", "--shard-index", "2"),
        ("aggregate", "--manifest", "missing.json"),
    ],
)
def test_shard_cli_rejects_missing_or_invalid_inputs_before_creating_reports(arguments):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/verification.py"), *arguments],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


@pytest.mark.parametrize("mode", ["shard", "aggregate"])
def test_public_shard_entry_blocks_execution_for_absent_manifest(tmp_path, mode):
    arguments = (
        ["--shard-index", "0"] if mode == "shard" else ["--shards", str(tmp_path / "absent-shard")]
    )
    report = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/verification.py"),
            mode,
            "--report-dir",
            str(report),
            "--manifest",
            str(tmp_path / "absent.json"),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    checks = json.loads((report / "checks.json").read_text())
    assert checks["complete"] is True and checks["outcome"] == "failed"
    assert (
        next(step for step in checks["steps"] if step["name"] == "collection")["state"]
        == "blocked"
    )
