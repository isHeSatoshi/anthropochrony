import json
from types import SimpleNamespace

import pytest

from anthropochrony.artifacts import write_json
from anthropochrony.gates import GateError
from anthropochrony.workflow import measurement, measurement_gate


def test_full_measurement_cannot_launch_before_pilot_gate(tmp_path):
    for gate in ("G1", "G2"):
        write_json(tmp_path / f"results/gates/{gate}.json", {"passed": True})
    with pytest.raises(GateError, match="G3"):
        measurement(tmp_path, SimpleNamespace(phase="full", run="base"))
    assert not (tmp_path / "artifacts/budget.json").exists()


def test_pilot_requires_100_tasks_k8_and_reports_family_pass_at_one(tmp_path):
    tasks = [{"task_id": str(i), "family": "FUNC" if i < 50 else "LIB"} for i in range(100)]
    states = [
        {
            "task_id": task["task_id"],
            "done": True,
            "first_pass": task["family"] == "FUNC",
            "attempts": [{"finish_reason": "stop", "exec": {"status": "pass"}}],
        }
        for task in tasks
        for _ in range(8)
    ]
    write_json(
        tmp_path / "data/measure/invariance.json",
        {
            "passed": True,
            "samples_tasks": 100,
            "reference_k": 8,
            "condition_k": 2,
            "comparisons": {name: {"median_abs_log10_token_ratio": 0} for name in ("slow", "fast")},
        },
    )
    measurement_gate(tmp_path, states, tasks)
    gate = json.loads((tmp_path / "results/gates/G3.json").read_text())
    assert gate["passed"]
    assert gate["evidence"]["pass_at_1_by_family"] == {"FUNC": 1, "LIB": 0}
    with pytest.raises(GateError, match="pilot_100_tasks_k8"):
        measurement_gate(tmp_path, states[:-1], tasks)


def test_invariance_checks_each_extreme_and_rejects_a_stale_pass_flag():
    from anthropochrony.measure.invariance import invariance_metrics, invariance_passed

    tasks = [{"task_id": str(i), "split": "val"} for i in range(100)]

    def traces(k, tokens, speeds):
        return [
            {
                "task_id": task["task_id"],
                "k": index,
                "model_sha": "fixture",
                "done": True,
                "card": {"S": speeds[0], "P": speeds[1]},
                "attempts": [{"o_tokens": tokens}],
            }
            for task in tasks
            for index in range(k)
        ]

    reference = traces(8, 100, (96, 26000))
    slow = traces(2, 100, (2, 20))
    fast = traces(2, 120, (2500, 50000))
    result = invariance_metrics(tasks, reference, slow, fast)
    assert result["comparisons"]["slow"]["passed"]
    assert not result["comparisons"]["fast"]["passed"]
    assert not result["passed"]
    result["passed"] = True
    assert not invariance_passed(result)
    passed = invariance_metrics(tasks, reference, slow, traces(2, 104, (2500, 50000)))
    assert invariance_passed(passed)
    with pytest.raises(ValueError, match="K=2"):
        invariance_metrics(tasks, reference, slow[:-1], fast)
