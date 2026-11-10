import copy
import json

import pytest

from anthropochrony.artifacts import file_hash, write_json
from anthropochrony.curate import sft
from anthropochrony.curate.gates import sft_gate
from anthropochrony.gates import GateError
from anthropochrony.prompts import estimate_messages
from anthropochrony.timemodel import SpeedCard
from anthropochrony.train.sft import TARGETS


class ShortTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(50))


def matched_inputs():
    tasks, labels, human = [], [], {}
    for name, split in (("train-task", "train"), ("val-task", "val"), ("raw-task", "raw_extra")):
        task = {
            "task_id": name,
            "split": split,
            "family": "FUNC",
            "source": "fixture",
            "task_turn": f"Implement the unique function {name}.",
            "uncached_task_tokens": 30,
        }
        tasks.append(task)
        for index in range(6):
            labels.append(
                {
                    "task_id": name,
                    "card_id": f"train-{index}",
                    "S": 10 * (index + 1),
                    "P": 100 * (index + 1),
                    "T_med_s": 10 + index,
                    "T_q10_s": 5 + index,
                    "T_q90_s": 20 + index,
                    "TOK_med": 100,
                    "ATT_med": 1,
                    "PFP": 0.5,
                }
            )
        human[name] = {
            "template_id": "h01",
            "phrasing_id": "sc01",
            "finish_reason": "stop",
            "raw_text": "About two hours for a human.",
            "messages": estimate_messages(
                task, SpeedCard("REF", 96, 26000), "h01", "sc01", "agent"
            ),
        }
    teachers = {
        task["task_id"]: {
            "attempts_median": 2,
            "tokens_median": 900,
            "tokens_p10": 600,
            "tokens_p90": 1200,
            "test_seconds": 0.5,
            "p_first_pass": 0.2,
        }
        for task in tasks
    }
    return tasks, labels, human, teachers


def test_actual_sft_build_and_curatorkit_export_reach_g5_only_with_dry_run_evidence(tmp_path):
    tasks, labels, human, teachers = matched_inputs()
    variants, drops, report = sft.build_matched_sft(
        tasks, labels, human, ShortTokenizer(), teachers=teachers, feedback_by_family={"FUNC": 20}
    )
    assert len(variants["m1"]) == len(variants["m2"]) == 14
    assert len(variants["m3"]) == 21
    assert not any(drops.values())
    assert sft.check_matched_variants(variants["m1"], variants["m2"])
    for name, rows in variants.items():
        assert sft.export_sft(rows, tmp_path / "data/sft" / name) == rows
    write_json(tmp_path / "data/sft/matched-summary.json", report)
    write_json(tmp_path / "results/gates/G4.json", {"passed": True})
    summary = {
        "dry_run": True,
        "training": {"steps": 1},
        "train_data_sha256": file_hash(tmp_path / "data/sft/m1/train/data.jsonl"),
        "assistant_mask": {"rendering_identical": True, "masked_tokens": 20},
        "actual_collator": {
            "assistant_only": True,
            "padding_masked": True,
            "inspected_sequences": 7,
        },
        "lora": {"targets": sorted(TARGETS), "trainable_parameters": 16},
        "config": {"max_seq_length": 2048},
    }
    path = tmp_path / "unit-fixture-dry-run/training-summary.json"
    write_json(path, summary)
    write_json(path.parent / "hardware.json", {"cuda": True, "gpu": "NVIDIA L40S"})
    sft_gate(tmp_path, path)
    assert json.loads((tmp_path / "results/gates/G5.json").read_text())["passed"]
    summary["train_data_sha256"] = "stale-dataset"
    write_json(path, summary)
    with pytest.raises(GateError, match="dry_run_uses_current_data"):
        sft_gate(tmp_path, path)


@pytest.mark.parametrize("excluded_each,expected_limit", [(1, 2048), (2, 3072)])
def test_joint_length_losses_trigger_one_global_limit(monkeypatch, excluded_each, expected_limit):
    tasks = [{"task_id": str(i), "split": "train"} for i in range(100)]
    calls = []

    def build(tasks, labels, human, tokenizer, *, variant, max_length, **kwargs):
        calls.append((variant, max_length))
        excluded = (
            (
                set(range(excluded_each))
                if variant == "m1"
                else set(range(excluded_each, excluded_each * 2))
                if variant == "m2"
                else set()
            )
            if max_length == 2048
            else set()
        )
        rows, drops = [], []
        for index in range(100):
            row = {
                "example_id": str(index),
                "task_id": str(index),
                "family": "FUNC",
                "frame": "self",
                "messages": [
                    {"role": "user", "content": str(index)},
                    {"role": "assistant", "content": variant},
                ],
            }
            if index in excluded:
                drops.append(
                    {
                        "example_id": str(index),
                        "task_id": str(index),
                        "family": "FUNC",
                        "reason": "sequence_length",
                    }
                )
            else:
                rows.append(row)
        return rows, drops

    monkeypatch.setattr(sft, "build_sft", build)
    variants, exclusions, report = sft.build_matched_sft(
        tasks, [], {}, None, teachers={}, feedback_by_family={}
    )
    assert report["max_seq_length"] == expected_limit
    assert report["initial_drop_fractions"]["m1"]["FUNC"] == excluded_each * 2 / 100
    assert report["initial_drop_fractions"]["m2"]["FUNC"] == excluded_each * 2 / 100
    assert sft.check_matched_variants(variants["m1"], variants["m2"])
    assert {row["example_id"] for row in variants["m3"]} == {
        row["example_id"] for row in variants["m1"]
    }
    if expected_limit == 3072:
        assert len(calls) == 6
        assert not any(exclusions.values())
    else:
        assert len(calls) == 3


def test_export_counts_are_after_actual_curation(tmp_path, monkeypatch):
    rows = [
        {"split": split, "example_id": f"{split}-{i}"}
        for split in ("train", "val")
        for i in range(2)
    ]
    monkeypatch.setattr(
        sft, "curate", lambda subset, *args, **kwargs: (copy.deepcopy(subset[1:]), [])
    )
    exported = sft.export_sft(rows, tmp_path)
    assert len(exported) == 2
    assert json.loads((tmp_path / "counts.json").read_text()) == {"train": 1, "val": 1}
