"""Exercise the pilot summarizer through its real public command."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "tools/verification_metrics.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_receipt(path, value):
    value["receipt_digest"] = digest(
        {key: item for key, item in value.items() if key != "receipt_digest"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")


def record(task_id="task", cohort="candidate", group="maintenance", **values):
    return {"task_id": task_id, "cohort": cohort, "comparable_group": group, **values}


def invoke(tmp_path, records):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(ENTRY), str(path)], text=True, capture_output=True, check=False
    )


def receipt(tmp_path, name="run", states=("passed", "passed", "passed")):
    original = receipt(tmp_path, name + "-original") if "reused" in states else None
    directory = tmp_path / name
    directory.mkdir()
    identity = {"check_version": "2.0", "inputs": {"source.py": "source-hash"}}
    steps = []
    for step_name, state in zip(("format", "lint", "docstrings"), states):
        log = directory / (step_name + ".log")
        log.write_text(state, encoding="utf-8")
        steps.append(
            {
                "name": step_name,
                "command": ["python", "-m", step_name],
                "state": state,
                "returncode": {"passed": 0, "reused": None, "failed": 1, "blocked": None}[state],
                "duration_seconds": 2.0,
                "dependencies": [],
                "blocking_reasons": ["environment blocked"] if state == "blocked" else [],
                "input_identity": identity,
                "input_digest": digest(identity),
                "log": {"path": log.name, "sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
            }
        )
        if state == "reused":
            original_path, original_value = original
            prior = next(step for step in original_value["steps"] if step["name"] == step_name)
            steps[-1]["provenance"] = {
                "receipt": str(original_path),
                "receipt_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                "command": prior["command"],
                "input_digest": prior["input_digest"],
                "log_sha256": prior["log"]["sha256"],
                "duration_seconds": prior["duration_seconds"],
            }
    failed = [step["name"] for step in steps if step["state"] not in ("passed", "reused")]
    value = {
        "document_version": "2.0",
        "mode": "fast",
        "complete": True,
        "outcome": "failed" if failed else "passed",
        "platform": "test-platform",
        "python": "python",
        "identity": identity,
        "identity_digest": digest(identity),
        "failed": failed,
        "steps": steps,
    }
    path = directory / "checks.json"
    write_receipt(path, value)
    return path, value


def test_one_real_task_retains_unavailable_measurements_and_incomplete_pilot(tmp_path):
    result = invoke(tmp_path, [record(agent_launches=2, receipts=[])])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["pilot"] == {
        "status": "incomplete",
        "comparable_tasks": 0,
        "required_tasks": 10,
        "remaining_tasks": 10,
    }
    assert report["cohorts"]["candidate"]["metrics"]["model_tokens"] == {
        "known": 0,
        "missing": 1,
        "total": None,
        "mean": None,
    }
    assert report["cohorts"]["candidate"]["metrics"]["executed_checks"]["mean"] is None
    assert report["cohorts"]["candidate"]["metrics"]["agent_launches"]["mean"] == 2


def test_only_paired_groups_are_compared_and_ten_unique_tasks_complete_pilot(tmp_path):
    tasks = [
        record(
            str(index),
            "reference" if index < 5 else "candidate",
            time_to_pr_seconds=100 if index < 5 else 40,
            model_tokens=None if index == 0 else 200,
        )
        for index in range(10)
    ]
    tasks.append(record("unpaired", group="other", time_to_pr_seconds=900))
    first = invoke(tmp_path, tasks)
    second = invoke(tmp_path, list(reversed(tasks)))
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["pilot"]["status"] == "complete"
    assert report["pilot"]["comparable_tasks"] == 10
    assert len(report["comparisons"]) == 1
    comparison = report["comparisons"][0]
    assert comparison["estimated"] is True
    assert comparison["metrics"]["time_to_pr_seconds"]["mean_delta"] == -60
    assert comparison["metrics"]["model_tokens"]["reference"]["missing"] == 1
    assert comparison["metrics"]["time_to_merge_seconds"]["mean_delta"] is None


def test_controlled_comparison_and_empty_input(tmp_path):
    tasks = [
        record("a", "reference", controlled_comparison=True),
        record("b", controlled_comparison=True),
    ]
    report = json.loads(invoke(tmp_path, tasks).stdout)
    assert report["comparisons"][0]["estimated"] is False
    empty = json.loads(invoke(tmp_path, []).stdout)
    assert empty["pilot"]["comparable_tasks"] == 0
    assert empty["cohorts"]["reference"]["tasks"] == 0


def test_complete_receipts_count_actual_execution_reuse_repetition_and_time(tmp_path):
    first, _ = receipt(tmp_path)
    second, _ = receipt(tmp_path, "retry", ("failed", "blocked", "reused"))
    third, _ = receipt(tmp_path, "later", ("reused", "passed", "passed"))
    result = invoke(
        tmp_path, [record(receipts=[str(first.relative_to(tmp_path)), str(second), str(third)])]
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)["cohorts"]["candidate"]["metrics"]
    assert metrics["executed_checks"]["total"] == 6
    assert metrics["reused_checks"]["total"] == 2
    assert metrics["blocked_checks"]["total"] == 1
    assert metrics["repeated_checks"]["total"] == 3
    assert metrics["check_seconds"]["total"] == 18
    assert metrics["executed_check_seconds"]["total"] == 12
    assert metrics["reused_check_seconds"]["total"] == 4
    assert metrics["blocked_check_seconds"]["total"] == 2
    assert metrics["repeated_check_seconds"]["total"] == 6


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan"), "unavailable"])
def test_invalid_numbers_cannot_be_summarized_as_measurements(tmp_path, value):
    result = invoke(tmp_path, [record(model_tokens=value)])
    assert result.returncode == 1
    assert not result.stdout
    assert "metrics:" in result.stderr


@pytest.mark.parametrize(
    "tasks",
    [
        {},
        [None],
        [record("same"), record("same")],
        [record("")],
        [record(cohort="other")],
        [record(group="")],
        [record(controlled_comparison=1)],
        [record(receipts="checks.json")],
        [record(receipts=[False])],
    ],
)
def test_invalid_task_structure_is_rejected(tmp_path, tasks):
    result = invoke(tmp_path, tasks)
    assert result.returncode == 1
    assert not result.stdout


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(complete=False),
        lambda value: value.update(outcome="failed"),
        lambda value: value.update(identity_digest="wrong"),
        lambda value: value.update(document_version="1.0"),
        lambda value: value.update(mode="foreign"),
        lambda value: value.update(identity=[]),
        lambda value: value.update(steps=[]),
        lambda value: value.update(failed=["format"]),
        lambda value: value["steps"][0].update(input_digest="wrong"),
        lambda value: value["steps"][0].pop("command"),
        lambda value: value["steps"][0].update(command=[]),
        lambda value: value["steps"][0].update(input_identity=[]),
        lambda value: value["steps"][0].update(returncode=True),
        lambda value: value["steps"][0].update(duration_seconds=-1),
        lambda value: value["steps"][0].update(state="incomplete"),
        lambda value: value["steps"][0].update(dependencies=["absent"]),
        lambda value: value["steps"][0].update(dependencies=[None]),
        lambda value: value["steps"][0].update(blocking_reasons=["hidden blocker"]),
        lambda value: value["steps"][1].update(name="format"),
        lambda value: value["steps"][0]["log"].update(path="../outside.log"),
        lambda value: value["steps"][0]["log"].update(sha256="changed"),
    ],
)
def test_partial_inconsistent_or_tampered_receipts_are_rejected(tmp_path, change):
    path, value = receipt(tmp_path)
    change(value)
    write_receipt(path, value)
    result = invoke(tmp_path, [record(receipts=[str(path)])])
    assert result.returncode == 1
    assert not result.stdout


def test_receipt_paths_cannot_double_count_one_invocation(tmp_path):
    path, _ = receipt(tmp_path)
    result = invoke(
        tmp_path, [record("a", receipts=[str(path)]), record("b", receipts=[str(path)])]
    )
    assert result.returncode == 1
    assert "Duplicate receipt" in result.stderr


def test_actual_internal_steps_and_different_inputs_are_not_repeated(tmp_path):
    first, _ = receipt(tmp_path)
    second, value = receipt(tmp_path, "changed")
    value["steps"][0]["command"] = None
    value["steps"][0]["input_identity"] = {"changed": "inputs"}
    value["steps"][0]["input_digest"] = digest(value["steps"][0]["input_identity"])
    value["steps"][1]["dependencies"] = ["format"]
    write_receipt(second, value)
    result = invoke(tmp_path, [record(receipts=[str(first), str(second)])])
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)["cohorts"]["candidate"]["metrics"]
    assert metrics["executed_checks"]["total"] == 6
    assert metrics["repeated_checks"]["total"] == 2


def test_cli_missing_files_json_errors_and_import_is_inert(tmp_path):
    missing = subprocess.run(
        [sys.executable, str(ENTRY), str(tmp_path / "absent.json")],
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 1
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ENTRY), str(invalid)], capture_output=True, check=False
    )
    assert result.returncode == 1
    imported = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(ENTRY)!r})"],
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0
    assert not imported.stdout and not imported.stderr


@pytest.mark.parametrize("missing", [True, False])
def test_receipt_seal_prevents_silent_metadata_edits(tmp_path, missing):
    path, value = receipt(tmp_path)
    if missing:
        value.pop("receipt_digest")
    else:
        value["steps"][0]["duration_seconds"] = 100
    path.write_text(json.dumps(value), encoding="utf-8")
    result = invoke(tmp_path, [record(receipts=[str(path)])])
    assert result.returncode == 1
    assert not result.stdout


@pytest.mark.parametrize(
    "change",
    [
        lambda step: step.update(returncode=0),
        lambda step: step.pop("provenance"),
        lambda step: step["provenance"].update(receipt_sha256="different"),
        lambda step: step["provenance"].update(command=["invented", "execution"]),
        lambda step: step["provenance"].update(input_digest="foreign"),
        lambda step: step["provenance"].update(log_sha256="changed"),
        lambda step: step["provenance"].update(duration_seconds=999),
    ],
)
def test_reused_steps_require_null_exit_and_matching_original_provenance(tmp_path, change):
    path, value = receipt(tmp_path, states=("passed", "passed", "reused"))
    change(value["steps"][-1])
    write_receipt(path, value)
    result = invoke(tmp_path, [record(receipts=[str(path)])])
    assert result.returncode == 1
    assert not result.stdout


def test_reuse_cannot_measure_resealed_incomplete_original_evidence(tmp_path):
    path, value = receipt(tmp_path, states=("passed", "passed", "reused"))
    provenance = value["steps"][-1]["provenance"]
    original = Path(provenance["receipt"])
    prior = json.loads(original.read_text())
    prior["complete"] = False
    write_receipt(original, prior)
    provenance["receipt_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    write_receipt(path, value)
    result = invoke(tmp_path, [record(receipts=[str(path)])])
    assert result.returncode == 1
    assert not result.stdout
