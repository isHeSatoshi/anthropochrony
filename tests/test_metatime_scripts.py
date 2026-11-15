"""Pure helpers of the head-to-head scripts (no GPU, no benchmark download)."""

import json
import math
import sys
from pathlib import Path

import common
import pytest
import report
import score

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/metatime.json"


def test_answer_text_round_trips_through_the_parser():
    text = common.answer_text(n_in=383, n_out=517, S=34.0, P=5600.0)
    parts = common.parse_answer(text)
    assert parts["prompt_tokens"] == 380
    assert parts["response_tokens"] == 520
    assert parts["seconds"] == pytest.approx(380 / 5600 + 520 / 34, rel=5e-3)
    assert common.parse_answer("45 seconds") == {
        "prompt_tokens": None,
        "response_tokens": None,
        "seconds": None,
    }


def test_fewshot_examples_come_from_training_prompts_across_the_length_range():
    pool = [
        {
            "id": f"p{index}",
            "split": "val" if index % 10 == 0 else "train",
            "description": f"task {index}",
            "n_in": 400,
            "n_out": 10 * (index + 1),
        }
        for index in range(101)
    ]
    examples = common.fewshot_examples(pool, S=20.0, P=2000.0)
    assert len(examples) == 5
    lengths = [common.parse_answer(answer)["response_tokens"] for _, answer in examples]
    assert lengths[1] == min(lengths) and lengths[2] == max(lengths)
    val_tasks = {f"task {index}\n" for index in range(0, 101, 10)}
    assert not any(any(task in question for task in val_tasks) for question, _ in examples)
    assert all(question.startswith("I'm going to ask you") for question, _ in examples)


def test_answers_that_quote_the_task_are_detected():
    task = {
        "description": "Write a function that reverses a linked list in place",
        "prompt": "Implement reverse(head) for a singly linked list and explain the complexity.",
    }
    assert score.repeats_task("Sure: write a function that reverses a linked list quickly", task)
    assert not score.repeats_task("Prompt: about 380 tokens.\nTime: 12 seconds", task)
    assert not score.repeats_task(None, task)


def test_sampled_reading_takes_the_median_and_the_range():
    rows = [
        {"kind": "samples", "variant": "v", "task": "a", "answers": ["10", "20", "30", "x"]},
        {"kind": "samples", "variant": "v", "task": "b", "answers": ["x", "x"]},
    ]

    def read(text):
        return float(text) if text.isdigit() else None

    reading = score.sampled_reading(rows, "v", ["a", "b"], read)
    assert reading["median"] == [20.0, None]
    assert reading["low"][0] == pytest.approx(12.0) and reading["high"][0] == pytest.approx(28.0)
    assert score.sampled_reading(rows, "v", ["a", "c"], read) is None
    assert score.range_coverage({"low": [12.0, None], "high": [28.0, None]}, [15.0, 5.0]) == 0.5


def test_change_is_worded_as_a_factor():
    assert report.change(20.0, "slower", "faster") == "20x slower"
    assert report.change(0.158, "slower", "faster") == "6.33x faster"
    assert report.change(1.004, "longer", "shorter") == "unchanged"
    assert report.change(None, "longer", "shorter") == "n/a"


def test_category_rows_score_each_category_on_its_own_tasks():
    rows = score.by_category([10.0, 40.0, 5.0], [10.0, 10.0, 10.0], ["a", "b", "a"])
    assert rows["a"]["n"] == 2 and rows["b"]["n"] == 1
    assert rows["b"]["median_ratio"] == pytest.approx(4.0)
    assert rows["a"]["median_abs_log_error"] == pytest.approx(math.log(2) / 2)


@pytest.mark.skipif(not RESULT.exists(), reason="the head-to-head is not scored yet")
def test_report_prints_the_declared_rows_from_the_results_file(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["report.py", "--results", str(RESULT)])
    report.main()
    printed = capsys.readouterr().out
    result = json.loads(RESULT.read_text())
    primary = result["systems"]["g1-s0/card_description"]["stated"]
    assert f"| **Tiny Aya + G1 (primary)** | {primary['median_ratio']:.2f}x |" in printed
    assert "| GPT-5.5 (published) |" in printed and "| Claude Opus 4.8 (published) |" in printed
    assert "| **Tiny Aya + G1 (primary)** vs Claude Opus 4.8 |" in printed
