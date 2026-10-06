"""Real local workers preserve execution ownership, isolation, and child coverage."""

import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

import pytest

from .test_verification_shards import ShardedCommand

DRIVER = """
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'runner'))
from verification import VerificationRun, process_coverage
from verification_parallel import parallel_test_step
root, directory, workers = sys.argv[1:]
run = VerificationRun(Path(root), 'full', Path(directory))
run.module('coverage-erase', ['coverage', 'erase'], dependencies=())
parallel_test_step(run, int(workers))
fresh = any(path.is_file() for path in run.directory.glob('.coverage*'))
run.run_step('coverage-data', action=lambda: 0, dependencies=('coverage-erase',),
             reason=None if fresh else 'No fresh worker coverage')
process_coverage(run)
sys.exit(run.finish())
"""

OVERLAPPING_TESTS = """
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import pytest

@pytest.mark.parametrize('number', [0, 1])
def test_workers(number, tmp_path):
    shared = Path(os.environ['PARALLEL_TEST_BARRIER'])
    start = time.monotonic()
    (shared / ('ready-' + str(number))).write_text(str(start))
    deadline = start + 15
    while len(list(shared.glob('ready-*'))) != 2:
        assert time.monotonic() < deadline, 'Workers failed to overlap'
        time.sleep(0.01)
    subprocess.run([sys.executable, '-c',
                    'import dphtools.child; assert dphtools.child.VALUE == 17'], check=True)
    (shared / ('result-' + str(number) + '.json')).write_text(json.dumps({
        'start': start, 'end': time.monotonic(), 'basetemp': str(tmp_path),
        'temporary': tempfile.gettempdir(), 'coverage': os.environ['COVERAGE_FILE']}))
"""


class ParallelCommand(ShardedCommand):
    """Exercise production test orchestration without reinstalling the host package."""

    def __init__(self, directory):
        super().__init__(directory)
        helper = Path(__file__).resolve().parents[1] / "tools/verification_parallel.py"
        if helper.exists():
            import shutil

            shutil.copy2(helper, self.entry.parent)
        self.driver = self.root / "parallel-driver.py"
        self.driver.write_text(DRIVER, encoding="utf-8")
        self.report = self.root / "reports/parallel"

    def parallel(self, workers=2):
        return subprocess.run(
            [sys.executable, str(self.driver), str(self.root), str(self.report), str(workers)],
            cwd=self.root.parent,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )


def test_local_workers_overlap_with_private_temp_and_real_child_coverage(tmp_path):
    command = ParallelCommand(tmp_path)
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    command.env["PARALLEL_TEST_BARRIER"] = str(barrier)
    (command.root / "tests/test_real.py").write_text(OVERLAPPING_TESTS)
    result = command.parallel()
    assert result.returncode == 0, result.stdout + result.stderr
    observations = [json.loads(path.read_text()) for path in sorted(barrier.glob("result-*.json"))]
    assert len(observations) == 2
    assert max(item["start"] for item in observations) < min(item["end"] for item in observations)
    for field in ("basetemp", "temporary", "coverage"):
        assert len({item[field] for item in observations}) == 2
    for item in observations:
        for field in ("basetemp", "temporary"):
            temporary = Path(item[field]).resolve()
            assert not temporary.is_relative_to(command.root.resolve())
            assert not temporary.exists()
    coverage = json.loads((command.report / "coverage.json").read_text())
    assert coverage["files"]["dphtools/child.py"]["executed_lines"] == [2]
    cases = list(ElementTree.parse(command.report / "pytest.xml").iter("testcase"))
    assert len(cases) == 3
    executions = json.loads((command.report / "tests.json").read_text())
    assert len(executions["nodes"]) == 3
    assert len(executions["executions"]) == 9
    assert len(list(command.report.glob("worker-*/checks.json"))) == 2


@pytest.mark.parametrize("behavior", ["skip", "failure"])
def test_local_worker_failure_retains_fresh_diagnostic_coverage(tmp_path, behavior):
    command = ParallelCommand(tmp_path)
    temporary_record = tmp_path / "worker-temporary.json"
    command.env["PARALLEL_FAILURE_TEMP"] = str(temporary_record)
    test = command.root / "tests/test_real.py"
    test.write_text(
        "import json, os, tempfile, pytest, dphtools.child\n"
        "from pathlib import Path\n"
        "def test_behavior(tmp_path):\n"
        "    Path(os.environ['PARALLEL_FAILURE_TEMP']).write_text(json.dumps(\n"
        "        {'basetemp': str(tmp_path), 'temporary': tempfile.gettempdir()}))\n"
        + ("    pytest.skip('visible skip')\n" if behavior == "skip" else "    assert False\n")
        + "def test_success():\n    assert dphtools.child.VALUE == 17\n"
    )
    result = command.parallel()
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads((command.report / "checks.json").read_text())
    assert next(step for step in receipt["steps"] if step["name"] == "tests")["state"] == "failed"
    assert (command.report / "coverage.json").is_file()
    assert (command.report / "pytest.xml").is_file()
    for temporary in json.loads(temporary_record.read_text()).values():
        temporary = Path(temporary).resolve()
        assert not temporary.is_relative_to(command.root.resolve())
        assert not temporary.exists()


def test_failed_collection_starts_no_workers(tmp_path):
    command = ParallelCommand(tmp_path)
    (command.root / "tests/test_real.py").write_text("This is invalid syntax\n")
    result = command.parallel()
    assert result.returncode == 1, result.stdout + result.stderr
    assert not list(command.report.glob("worker-*"))
    assert not (command.report / "coverage.json").exists()


def test_requested_workers_are_bounded_by_real_collection(tmp_path):
    command = ParallelCommand(tmp_path)
    result = command.parallel(8)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(command.report.glob("worker-*"))) == 3


def test_parallel_shard_cli_receipts_keep_the_existing_ci_partition(tmp_path):
    command = ParallelCommand(tmp_path)
    result = command.run("collect", command.collect)
    assert result.returncode == 0, result.stdout + result.stderr
    directories = []
    for index in (0, 1):
        directory = command.root / f"reports/shard-{index}"
        result = command.run(
            "shard",
            directory,
            "--manifest",
            str(command.manifest),
            "--shard-index",
            str(index),
            "--workers",
            "2",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        directories.append(directory)
    result = command.aggregate(command.root / "reports/aggregate", directories)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(command.manifest.read_text())
    assert len(manifest["assignments"]) == 2
    assert sum(len(group) for group in manifest["assignments"]) == 3


def test_worker_child_exit_status_remains_authoritative_after_successful_reports(tmp_path):
    command = ParallelCommand(tmp_path)
    (command.root / "tests/test_real.py").write_text(
        "import atexit, os, coverage, dphtools.child\n"
        "def fail_after_reports(collector):\n"
        "    collector.save()\n"
        "    os._exit(23)\n"
        "def test_success():\n"
        "    assert dphtools.child.VALUE == 17\n"
        "    atexit.register(fail_after_reports, coverage.Coverage.current())\n"
    )
    result = command.parallel()
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Invocation is missing, failed, or incomplete" in result.stdout
    assert (command.report / "coverage.json").is_file()


@pytest.mark.parametrize("subset", [False, True])
def test_parallel_interface_keeps_full_collection_and_exact_phase_reports(
    tmp_path, monkeypatch, subset
):
    command = ParallelCommand(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verification import VerificationRun
    from verification_parallel import parallel_test_step
    from verification_shards import read_json, sealed, test_step, validate_execution, write_json

    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    write_json(command.root / "tools/verification-durations.json", sealed({"durations": {}}))
    run = VerificationRun(command.root, "test-interface", command.report)
    run.module("coverage-erase", ["coverage", "erase"], dependencies=())
    test_step(run, "collection", {}, dependencies=("coverage-erase",))
    nodes = read_json(run.directory / "collection.json")["nodes"]
    selected = [node for node in nodes if node.startswith("tests/")] if subset else nodes
    settings = {"expected": nodes, "assigned": selected} if subset else None
    parallel_test_step(
        run,
        2,
        name="execution",
        settings=settings,
        dependencies=("collection",),
    )
    assert run.finish() == 0
    actual = read_json(run.directory / "execution.json")
    assert actual["nodes"] == nodes
    assert set(validate_execution(actual, selected)) == set(selected)
    assert len(list(ElementTree.parse(run.directory / "pytest.xml").iter("testcase"))) == len(
        selected
    )


@pytest.mark.parametrize("behavior", ["skip", "failure", "missing-junit", "malformed-junit"])
def test_parallel_interface_retains_failed_worker_measurement(tmp_path, monkeypatch, behavior):
    command = ParallelCommand(tmp_path)
    test = command.root / "tests/test_real.py"
    if behavior in ("skip", "failure"):
        test.write_text(
            "import pytest, dphtools.child\n"
            "def test_behavior():\n"
            + ("    pytest.skip('visible skip')\n" if behavior == "skip" else "    assert False\n")
        )
    elif behavior == "missing-junit":
        test.write_text(
            "import os, coverage, dphtools.child\n"
            "def test_interrupted():\n"
            "    coverage.Coverage.current().save()\n"
            "    os._exit(23)\n"
        )
    else:
        (command.root / "tests/conftest.py").write_text(
            "import os, pytest\nfrom pathlib import Path\n"
            "@pytest.hookimpl(trylast=True)\n"
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    if not session.config.option.collectonly:\n"
            "        (Path(os.environ['COVERAGE_FILE']).parent / 'pytest.xml').write_text('<broken')\n"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verification import VerificationRun
    from verification_parallel import parallel_test_step

    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = VerificationRun(command.root, "test-interface", command.report)
    run.module("coverage-erase", ["coverage", "erase"], dependencies=())
    parallel_test_step(run, 2)
    assert run.finish() == 1
    assert next(step for step in run.steps if step["name"] == "tests")["state"] == "failed"
    assert list(run.directory.glob(".coverage.worker-*"))


@pytest.mark.parametrize("fault", ["selection", "shape", "duration-shape", "failed-collection"])
def test_parallel_interface_rejects_invalid_inputs_before_execution(tmp_path, monkeypatch, fault):
    command = ParallelCommand(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verification import VerificationRun
    from verification_parallel import parallel_test_step
    from verification_shards import read_json, sealed, test_step, write_json

    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = VerificationRun(command.root, "test-interface", command.report)
    run.module("coverage-erase", ["coverage", "erase"], dependencies=())
    test_step(run, "collection", {}, dependencies=("coverage-erase",))
    nodes = read_json(run.directory / "collection.json")["nodes"]
    settings = {"expected": nodes, "assigned": ["foreign"]}
    if fault == "shape":
        settings = {"expected": None}
    elif fault == "duration-shape":
        write_json(command.root / "tools/verification-durations.json", sealed({}))
        settings = {"expected": nodes}
    elif fault == "failed-collection":
        (command.root / "tests/test_real.py").write_text("This is invalid syntax\n")
        settings = None
    parallel_test_step(run, 2, settings=settings)
    assert run.finish() == 1
    assert not list(run.directory.glob("worker-*"))


@pytest.mark.parametrize("workers", [0, 9, True, "2"])
def test_invalid_worker_counts_are_rejected(workers):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verification_parallel import parallel_test_step

    with pytest.raises(ValueError, match="Workers must be between 1 and 8"):
        parallel_test_step(None, workers)


def test_serial_worker_preserves_original_command_path(tmp_path, monkeypatch):
    command = ParallelCommand(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verification import VerificationRun
    from verification_parallel import parallel_test_step

    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = VerificationRun(command.root, "test-interface", command.report)
    run.module("coverage-erase", ["coverage", "erase"], dependencies=())
    parallel_test_step(run, 1)
    assert run.finish() == 0
    assert [step["name"] for step in run.steps] == ["coverage-erase", "tests", "tests-validation"]
    assert not list(run.directory.glob("worker-*"))


def test_serial_worker_rejects_a_node_removed_after_recorded_collection(tmp_path):
    command = ParallelCommand(tmp_path)
    (command.root / "tests/test_real.py").write_text(
        "import dphtools.child\n"
        "def test_parent():\n    assert dphtools.child.VALUE == 17\n"
        "def test_child():\n    assert dphtools.child.VALUE == 17\n"
    )
    (command.root / "tests/conftest.py").write_text(
        "import pytest\n"
        "@pytest.hookimpl(hookwrapper=True, tryfirst=True)\n"
        "def pytest_collection_modifyitems(items):\n"
        "    yield\n"
        "    items[:] = [item for item in items if not item.nodeid.endswith('::test_child')]\n"
    )
    result = command.parallel(workers=1)
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads((command.report / "checks.json").read_text())
    assert next(step for step in receipt["steps"] if step["name"] == "tests")["state"] == "passed"
    assert (
        next(step for step in receipt["steps"] if step["name"] == "tests-validation")["state"]
        == "failed"
    )
    assert "Missing or duplicate node execution" in result.stdout
    execution = json.loads((command.report / "tests.json").read_text())
    assert len(execution["nodes"]) == 3
    assert len(execution["executions"]) == 6
    assert len(list(ElementTree.parse(command.report / "pytest.xml").iter("testcase"))) == 2
    assert (command.report / "coverage.json").is_file()
    assert not list(command.report.glob("worker-*"))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "seal",
        "collection",
        "junit",
        "coverage",
        "durations",
        "receipt",
        "receipt-seal",
        "identity",
        "assignment",
        "worker-count",
        "empty-assignment",
    ],
)
def test_merge_rejects_changed_worker_execution(tmp_path, fault):
    command = ParallelCommand(tmp_path)
    result = command.parallel()
    assert result.returncode == 0, result.stdout + result.stderr
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verification_parallel import merge_workers
    from verification_shards import read_json, sealed, write_json

    workers = sorted(command.report.glob("worker-*"))
    nodes = read_json(command.report / "collection.json")["nodes"]
    assignments = [read_json(path / "tests-config.json")["assigned"] for path in workers]
    path = workers[0] / "tests.json"
    record = read_json(path)
    record.pop("digest")
    if fault == "missing":
        record["executions"].pop()
    elif fault == "duplicate":
        record["executions"].append(record["executions"][0])
    elif fault == "collection":
        record["nodes"] = []
    elif fault == "junit":
        (workers[0] / "pytest.xml").write_text("<testsuites/>")
    elif fault == "coverage":
        for raw in workers[0].glob(".coverage*"):
            raw.unlink()
    elif fault == "durations":
        record["durations"] = {}
    elif fault in ("receipt", "receipt-seal", "identity"):
        receipt = read_json(workers[0] / "checks.json")
        if fault == "identity":
            receipt["identity"]["environment"]["machine"] = "foreign-machine"
            from verification_inputs import digest

            receipt["identity_digest"] = digest(receipt["identity"])
        else:
            receipt["complete"] = False
        if fault in ("receipt", "identity"):
            from verification_inputs import digest

            receipt["receipt_digest"] = digest(
                {k: v for k, v in receipt.items() if k != "receipt_digest"}
            )
        write_json(workers[0] / "checks.json", receipt)
    elif fault == "assignment":
        assignments[1] = assignments[0]
    elif fault == "worker-count":
        assignments.pop()
    elif fault == "empty-assignment":
        assignments[0] = []
    if fault not in (
        "junit",
        "coverage",
        "receipt",
        "receipt-seal",
        "identity",
        "assignment",
        "worker-count",
        "empty-assignment",
    ):
        write_json(path, record if fault == "seal" else sealed(record))
    (workers[0] / ".coverage-directory").mkdir()
    with pytest.raises(ValueError):
        merge_workers(
            command.report,
            workers,
            assignments,
            nodes,
            read_json(command.report / "checks.json")["identity"],
        )
