"""Summarize explicit maintenance-task measurements without inventing missing data."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

METRICS = (
    "time_to_pr_seconds",
    "time_to_merge_seconds",
    "agent_launches",
    "runner_seconds",
    "model_tokens",
    "defects",
    "executed_checks",
    "reused_checks",
    "blocked_checks",
    "repeated_checks",
    "check_seconds",
    "executed_check_seconds",
    "reused_check_seconds",
    "blocked_check_seconds",
    "repeated_check_seconds",
)


def require(condition, message):
    """Reject malformed measurements rather than emit a plausible summary."""
    if not condition:
        raise ValueError(message)


def number(value):
    """Keep unavailable values separate from finite nonnegative measurements."""
    require(
        value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
        "Expected a finite nonnegative number or null",
    )
    return value


def digest(value):
    """Use the verifier's stable JSON representation for receipt identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_provenance(step):
    """Bind a reuse measurement to an actual successful original invocation."""
    provenance = step["provenance"]
    path = Path(provenance["receipt"])
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == provenance["receipt_sha256"], "Original receipt differs"
    )
    original = json.loads(raw)
    require(
        original["complete"] is True
        and original["outcome"] == "passed"
        and all(item["state"] == "passed" for item in original["steps"]),
        "Reuse requires complete originally executed evidence",
    )
    matching = [item for item in receipt_steps(path, original) if item["name"] == step["name"]]
    require(len(matching) == 1, "Original check is absent")
    prior = matching[0]
    require(
        provenance["command"] == prior["command"] == step["command"]
        and provenance["input_digest"] == prior["input_digest"] == step["input_digest"]
        and provenance["log_sha256"] == prior["log"]["sha256"]
        and number(provenance["duration_seconds"]) == prior["duration_seconds"],
        "Original check provenance differs",
    )


def receipt_steps(path, value=None):
    """Read complete receipts and validate their internal identities and retained logs."""
    if value is None:
        value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Receipt must be an object")
    require(
        value["receipt_digest"]
        == digest({key: item for key, item in value.items() if key != "receipt_digest"}),
        "Receipt seal differs",
    )
    require(value["document_version"] == "2.0" and value["complete"] is True, "Incomplete receipt")
    require(
        value["mode"] in ("preflight", "fast", "full", "collect", "shard", "aggregate"),
        "Unknown receipt mode",
    )
    require(isinstance(value["identity"], dict) and value["identity"], "Missing receipt identity")
    require(value["identity_digest"] == digest(value["identity"]), "Receipt identity differs")
    require(
        all(isinstance(value[key], str) and value[key] for key in ("mode", "python", "platform")),
        "Missing receipt environment",
    )
    steps = value["steps"]
    require(isinstance(steps, list) and steps, "Missing receipt steps")
    states = {}
    for step in steps:
        require(isinstance(step, dict), "Step must be an object")
        name, state = step["name"], step["state"]
        require(isinstance(name, str) and name and name not in states, "Invalid or duplicate step")
        require(state in ("passed", "failed", "blocked", "reused"), "Unknown step state")
        command = step["command"]
        require(
            command is None
            or (
                isinstance(command, list)
                and command
                and all(isinstance(part, str) for part in command)
            ),
            "Invalid step command",
        )
        require(
            isinstance(step["input_identity"], dict) and step["input_identity"],
            "Missing step identity",
        )
        require(step["input_digest"] == digest(step["input_identity"]), "Step identity differs")
        require(number(step["duration_seconds"]) is not None, "Missing step duration")
        dependencies, reasons = step["dependencies"], step["blocking_reasons"]
        require(
            isinstance(dependencies, list) and isinstance(reasons, list), "Invalid dependencies"
        )
        require(
            all(isinstance(item, str) for item in [*dependencies, *reasons]),
            "Invalid dependency details",
        )
        require(all(item in states for item in dependencies), "Missing step dependency")
        blocked = reasons or any(states[item] not in ("passed", "reused") for item in dependencies)
        code = step["returncode"]
        require(
            (state == "blocked" and code is None and blocked)
            or (state == "reused" and code is None and not blocked)
            or (
                state in ("passed", "failed")
                and not blocked
                and type(code) is int
                and ((code == 0) == (state == "passed"))
            ),
            "Step state, blockers, and return code differ",
        )
        if state == "reused":
            validate_provenance(step)
        log = path.parent / step["log"]["path"]
        require(
            log.resolve().is_relative_to(path.parent.resolve()), "Log escaped receipt directory"
        )
        require(
            hashlib.sha256(log.read_bytes()).hexdigest() == step["log"]["sha256"],
            "Retained log differs",
        )
        states[name] = state
    failed = [name for name, state in states.items() if state not in ("passed", "reused")]
    require(
        value["failed"] == failed and value["outcome"] == ("failed" if failed else "passed"),
        "Receipt outcome differs",
    )
    return steps


def summary(tasks):
    """Report sample availability beside each total and mean."""
    metrics = {}
    for metric in METRICS:
        values = [task["metrics"][metric] for task in tasks if task["metrics"][metric] is not None]
        total = math.fsum(values) if values else None
        metrics[metric] = {
            "known": len(values),
            "missing": len(tasks) - len(values),
            "total": total,
            "mean": total / len(values) if values else None,
        }
    return {"tasks": len(tasks), "metrics": metrics}


def summarize(records, base):
    """Compare only groups with explicit reference and candidate measurements."""
    require(isinstance(records, list), "Task records must be a list")
    tasks, ids, seen_receipts = [], set(), set()
    for record in records:
        require(isinstance(record, dict), "Task record must be an object")
        task_id, group = record["task_id"], record["comparable_group"]
        require(
            isinstance(task_id, str) and task_id and task_id not in ids,
            "Invalid or duplicate task ID",
        )
        require(isinstance(group, str) and group, "Missing comparable group")
        require(record["cohort"] in ("reference", "candidate"), "Unknown cohort")
        controlled = record.get("controlled_comparison", False)
        require(type(controlled) is bool, "controlled_comparison must be boolean")
        ids.add(task_id)
        metrics = {metric: number(record.get(metric)) for metric in METRICS[:6]}
        metrics.update({metric: None for metric in METRICS[6:]})
        paths = record.get("receipts", [])
        require(isinstance(paths, list), "Receipts must be a list")
        if paths:
            metrics.update({metric: 0 for metric in METRICS[6:]})
            executed = set()
            for supplied in paths:
                require(isinstance(supplied, str) and supplied, "Invalid receipt path")
                path = (base / supplied).resolve()
                require(path not in seen_receipts, "Duplicate receipt invocation")
                seen_receipts.add(path)
                for step in receipt_steps(path):
                    state = step["state"]
                    metrics["check_seconds"] += step["duration_seconds"]
                    if state in ("passed", "failed"):
                        metrics["executed_checks"] += 1
                        metrics["executed_check_seconds"] += step["duration_seconds"]
                        key = (step["name"], step["input_digest"])
                        metrics["repeated_checks"] += int(key in executed)
                        metrics["repeated_check_seconds"] += (
                            int(key in executed) * step["duration_seconds"]
                        )
                        executed.add(key)
                    else:
                        category = "reused" if state == "reused" else "blocked"
                        metrics[category + "_checks"] += 1
                        metrics[category + "_check_seconds"] += step["duration_seconds"]
        tasks.append({**record, "metrics": metrics, "controlled_comparison": controlled})
    tasks.sort(key=lambda task: task["task_id"])
    cohorts = {
        cohort: summary([task for task in tasks if task["cohort"] == cohort])
        for cohort in ("reference", "candidate")
    }
    comparisons, comparable = [], 0
    for group in sorted({task["comparable_group"] for task in tasks}):
        selected = [task for task in tasks if task["comparable_group"] == group]
        grouped = {
            cohort: summary([task for task in selected if task["cohort"] == cohort])
            for cohort in cohorts
        }
        if not all(value["tasks"] for value in grouped.values()):
            continue
        comparable += len(selected)
        differences = {}
        for metric in METRICS:
            reference = grouped["reference"]["metrics"][metric]
            candidate = grouped["candidate"]["metrics"][metric]
            differences[metric] = {
                "reference": reference,
                "candidate": candidate,
                "mean_delta": (
                    candidate["mean"] - reference["mean"]
                    if reference["known"] and candidate["known"]
                    else None
                ),
            }
        comparisons.append(
            {
                "comparable_group": group,
                "estimated": not all(task["controlled_comparison"] for task in selected),
                "metrics": differences,
            }
        )
    return {
        "document_version": "1.0",
        "pilot": {
            "status": "complete" if comparable >= 10 else "incomplete",
            "comparable_tasks": comparable,
            "required_tasks": 10,
            "remaining_tasks": max(0, 10 - comparable),
        },
        "cohorts": cohorts,
        "comparisons": comparisons,
    }


def main():
    """Read explicit task records and print a deterministic JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    arguments = parser.parse_args()
    try:
        records = json.loads(arguments.records.read_text(encoding="utf-8"))
        result = summarize(records, arguments.records.resolve().parent)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"metrics: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
