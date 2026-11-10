import json

import pytest

from anthropochrony.artifacts import file_hash, write_json
from anthropochrony.budget import Budget, BudgetError
from anthropochrony.curate.pipeline import curate
from anthropochrony.labels import build_labels, canonical_answer
from anthropochrony.labels.speeds import sample_cards
from anthropochrony.parse import parse
from anthropochrony.release import redact_record
from anthropochrony.tasks.splits import assign_splits, duplicate_groups, verify_splits
from anthropochrony.timemodel import SpeedCard


def task(task_id="a", family="FUNC", split="train"):
    return {
        "task_id": task_id,
        "family": family,
        "split": split,
        "source": "local-fixture",
        "source_revision": "fixture-v1",
        "license": "Apache-2.0",
        "ref_tokens": 30,
        "task_turn": f"Implement function {task_id} for a unique problem.",
        "reference_solution": f"def {task_id}(x):\n    return x+1",
    }


def trace(task_id, index, passed=True):
    return {
        "run_id": f"{task_id}-{index}",
        "task_id": task_id,
        "first_pass": passed,
        "n_attempts": 1,
        "harness_error": False,
        "attempts": [
            {
                "u_tokens": 100,
                "o_tokens": 200,
                "exec": {"wall_s": 3, "status": "pass" if passed else "fail"},
            }
        ],
    }


def test_labels_hand_computed_and_train_only_shrinkage():
    tasks = [task("train", split="train"), task("test", split="test")]
    traces = [trace("train", i, passed=False) for i in range(8)]
    traces += [trace("test", i, passed=True) for i in range(8)]
    labels, _ = build_labels(tasks, traces, [SpeedCard("c", 20, 100)])
    test_label = next(r for r in labels if r["task_id"] == "test")
    # 100/100 + 200/20 + 3 s execution, plus one call's adopted latency.
    expected = 14 + 0.967
    assert test_label["T_med_s"] == pytest.approx(expected)
    assert test_label["label_version"] == "v2-latency"
    assert test_label["PFP"] == 0.8  # Train-only prior is zero, not test's perfect rate.
    assert test_label["T_q10_s"] <= expected / 1.25
    assert test_label["T_q90_s"] >= expected * 1.25
    answer = canonical_answer(test_label)
    assert parse(answer).point_s == 15  # Canonical answers keep two significant figures.


def test_training_cards_deterministic_and_exclude_rounded_holdout():
    config = {
        "profiles": {"all": {"S": [2, 2500], "ratio": [4, 300]}},
        "heldout_decode_band": [150, 260],
    }
    first = sample_cards("a", config, n=1000)
    assert first == sample_cards("a", config, n=1000)
    assert all(not 150 <= c.S <= 260 for c in first)


def test_group_splits_no_class_leak_and_order_independent():
    tasks = [task(f"f{i}", family="LIB") for i in range(40)] + [task("class", family="CLASS")]
    for i, row in enumerate(tasks):
        row["task_turn"] = f"Unique item {i} " + " ".join(f"token{i}_{j}" for j in range(10))
        row["reference_solution"] = f"def example_{i}(): return {i}"
        row["ref_tokens"] = i + 5
    tasks.append(dict(tasks[-1], task_id="class-duplicate", family="FUNC"))
    groups = duplicate_groups(tasks)
    kept, dropped = assign_splits(tasks, groups)
    verify_splits(kept)
    assert any(r["task_id"] == "class-duplicate" for r in dropped)
    assert all(r["split"] == "test_ood" for r in kept if r["family"] == "CLASS")
    reversed_kept, _ = assign_splits(list(reversed(tasks)))
    assert {t["task_id"]: t["split"] for t in kept} == {
        t["task_id"]: t["split"] for t in reversed_kept
    }


def test_budget_hard_cap_and_single_job_limit(tmp_path):
    budget = Budget(tmp_path)
    with pytest.raises(BudgetError):
        budget.reserve("too-large", 5.01)
    for i in range(6):
        budget.reserve(str(i), 5)
    assert budget.total() == 30
    with pytest.raises(BudgetError):
        budget.reserve("overflow", 0.01)
    budget.update("0", actual_usd=1)
    assert budget.total() == 26


def test_artifact_sidecar_hash(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"value": 1}, config={"seed": 0})
    sidecar = json.loads(path.with_name("result.json.manifest.json").read_text())
    assert sidecar["artifact_sha256"] == file_hash(path)
    assert all(
        k in sidecar
        for k in (
            "git_sha",
            "input_hashes",
            "config_hash",
            "environment_lock_hash",
            "hardware",
            "job_id",
            "started_at",
            "ended_at",
        )
    )


def test_real_curatorkit_preserves_code_and_metadata(tmp_path):
    row = task()
    passed, _ = curate([row], tmp_path / "curate")
    assert passed == [row]
    assert passed[0]["reference_solution"] == "def a(x):\n    return x+1"


def test_release_does_not_leak_restricted_text():
    original = dict(task("lcd/a"), source="newfacade/LeetCodeDataset")
    record = {
        "task_id": original["task_id"],
        "messages": [{"content": "COPYRIGHTED"}],
        "raw_text": "COPYRIGHTED",
        "test_code": "COPYRIGHTED",
        "parsed": {"point_s": 1},
    }
    released = redact_record(record, {original["task_id"]: original})
    assert "COPYRIGHTED" not in json.dumps(released)
    assert released["restricted_statement"]
    assert record["raw_text"] == "COPYRIGHTED"
