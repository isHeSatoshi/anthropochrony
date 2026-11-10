import copy
import json

import pytest

from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
from anthropochrony.eval.hypotheses import confirmatory
from anthropochrony.eval.preservation import collect_preservation
from anthropochrony.eval.prompts import build_evaluation_prompts
from anthropochrony.eval.reporting import report
from anthropochrony.eval.targets import common_base_targets, primary_rows
from anthropochrony.gates import GateError


def estimate(task, estimator="M1-s0", error=0.1, target_source="own"):
    return {
        "estimator": estimator,
        "task_id": task,
        "split": "test",
        "family": "FUNC",
        "card_id": "REF",
        "frame": "self",
        "S": 96,
        "P": 26000,
        "seed": 0,
        "T_med_s": 1,
        "target_source": target_source,
        "parsed": {"point_s": 10**error, "has_human_units": False},
    }


def test_confirmatory_results_ignore_common_base_robustness_rows():
    rows = []
    for i in range(24):
        for split in ("val", "test"):
            for estimator, error in (("E0", 2), ("E1", 1.5), ("E2", 1), ("E3", 0.8)):
                row = estimate(str(i), estimator, error, "base")
                row.update(split=split, frame="generic" if estimator == "E3" else "self", seed=None)
                rows.append(row)
        for seed in range(3):
            own = estimate(str(i), f"M1-s{seed}", 0.1)
            own["seed"] = seed
            rows.extend([own, own | {"target_source": "base", "parsed": {"point_s": 1e8}}])
    result = confirmatory(rows, resamples=20)
    assert result["H2"]["mean_seed_mdae"] == pytest.approx(0.1)
    assert result["H2"]["passed"]
    invalid = estimate("a")
    del invalid["target_source"]
    with pytest.raises(ValueError, match="target source"):
        primary_rows([invalid])


def test_common_base_comparisons_retarget_every_model_without_new_generation():
    labels = [
        {
            "task_id": "a",
            "card_id": "REF",
            "S": 96,
            "P": 26000,
            "T_med_s": 2,
            "label_source": "base",
        }
    ]
    rows = [estimate("a", name) for name in ("E1", "M1-s0", "M2-s0", "M3-s0")]
    original = copy.deepcopy(rows)
    converted = common_base_targets(rows, labels)
    assert len(converted) == 4
    assert all(row["actual_s"] == row["T_med_s"] == 2 for row in converted)
    assert all(row["target_source"] == row["label_source"] == "base" for row in converted)
    assert [row["parsed"] for row in converted] == [row["parsed"] for row in rows]
    assert rows == original
    assert common_base_targets(converted + converted, labels) == converted


def test_report_does_not_pool_own_and_common_base_targets(tmp_path):
    own = estimate("a")
    robust = own | {"target_source": "base", "T_med_s": 100}
    tables = report(tmp_path, [own, robust], resamples=20)
    assert {row["target_source"] for row in tables} == {"own", "base"}
    assert len(tables) == 2
    assert tables[0]["mdae_log10"] != tables[1]["mdae_log10"]


def test_actual_auditkit_keeps_identical_messages_separate_by_estimate_id(tmp_path):
    from anthropochrony.eval.audit import evaluate

    prompts = [
        {
            "est_id": name,
            "messages": [{"role": "user", "content": "Identical prompt."}],
            "T_med_s": 2,
        }
        for name in ("first", "second")
    ]
    actual = {"first": "2 s.", "second": "3 s."}
    result, _ = evaluate(
        prompts,
        lambda inputs: [actual[json.loads(value)["est_id"]] for value in inputs],
        tmp_path,
        model_sha="fixture",
        code_sha="fixture",
    )
    assert {row.sample_id: row.raw_output for row in result.predictions} == actual


def test_human_prompts_are_matched_and_raw_preservation_is_training_only(tmp_path):
    tasks = [
        {"task_id": split, "task_turn": f"Implement {split}.", "family": "FUNC", "split": split}
        for split in ("train", "val", "test", "raw_extra")
    ]
    labels = [
        {
            "task_id": task["task_id"],
            "card_id": "REF",
            "S": 96,
            "P": 26000,
            "T_med_s": 1,
            "label_source": "base",
        }
        for task in tasks
    ]
    base = build_evaluation_prompts(tasks, labels, split="test")
    tuned = build_evaluation_prompts(tasks, labels, split="test", fine_tuned="m1-s0")
    assert (
        next(r for r in base if r["frame"] == "human")["messages"]
        == next(r for r in tuned if r["frame"] == "human")["messages"]
    )
    raw = build_evaluation_prompts(tasks, labels, split="raw_extra")
    assert len(raw) == 1
    assert raw[0]["frame"] == "human"
    assert raw[0]["max_tokens"] == 1024
    with pytest.raises(ValueError, match="training-only"):
        build_evaluation_prompts(tasks, labels, split="raw_extra", fine_tuned="m3-s0")
    write_jsonl(tmp_path / "data/tasks/tasks.jsonl", tasks)
    write_json(tmp_path / "configs/pins.yaml", {"subject": {"revision": "fixture"}})
    answers = []
    for split in ("train", "val", "raw_extra"):
        prompt = next(
            row
            for row in build_evaluation_prompts(tasks, labels, split=split)
            if row["frame"] == "human"
        )
        answers.append(
            prompt
            | {"raw_text": "About two hours.", "finish_reason": "stop", "model_sha": "fixture"}
        )
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_jsonl(first, answers[:1])
    collect_preservation(tmp_path, [first])
    write_jsonl(second, [answers[1], answers[2] | {"finish_reason": "length"}])
    with pytest.raises(GateError, match="Truncated"):
        collect_preservation(tmp_path, [second])
    assert len(read_jsonl(tmp_path / "data/eval/human-train-val.jsonl")) == 1
    write_jsonl(second, answers[1:])
    collect_preservation(tmp_path, [second])
    assert len(read_jsonl(tmp_path / "data/eval/human-train-val.jsonl")) == 2
    assert len(read_jsonl(tmp_path / "data/eval/human-raw-extra.jsonl")) == 1
    write_jsonl(second, [answers[1] | {"raw_text": "A conflicting answer."}])
    with pytest.raises(ValueError, match="Conflicting"):
        collect_preservation(tmp_path, [second])
    assert json.loads(
        (tmp_path / "data/eval/preservation-blockers.jsonl.manifest.json").read_text()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"raw_text": ""},
        {"raw_text": "   "},
        {"finish_reason": "context_exhausted"},
        {"model_called": False},
    ],
)
def test_incomplete_human_answers_never_become_training_targets(tmp_path, change):
    task = {"task_id": "a", "family": "FUNC", "split": "train", "task_turn": "Implement f()."}
    label = {"task_id": "a", "card_id": "REF", "S": 96, "P": 26000, "T_med_s": 1}
    row = build_evaluation_prompts([task], [label], split="train")[0] | {
        "raw_text": "Two hours.",
        "finish_reason": "stop",
        "model_sha": "fixture",
    }
    write_jsonl(tmp_path / "data/tasks/tasks.jsonl", [task])
    write_json(tmp_path / "configs/pins.yaml", {"subject": {"revision": "fixture"}})
    path = tmp_path / "answers.jsonl"
    write_jsonl(path, [row | change])
    with pytest.raises(GateError, match="incomplete"):
        collect_preservation(tmp_path, [path])
    assert not (tmp_path / "data/eval/human-train-val.jsonl").exists()
