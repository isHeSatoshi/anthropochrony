"""Pure helpers of the head-to-head scripts (no GPU, no benchmark download)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/metatime"))

import common  # noqa: E402
import score  # noqa: E402


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
