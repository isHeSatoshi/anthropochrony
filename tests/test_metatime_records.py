"""The committed head-to-head records reproduce the numbers in `results/metatime.json`."""

import csv
import json
import statistics
from pathlib import Path

import pytest

from anthropochrony.eval import metatime

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/metatime.json"
RECORDS = ROOT / "results/metatime"
CHECKED = ("median_ratio", "median_abs_log_error", "pearson_log", "within_2x", "mae_s")

pytestmark = pytest.mark.skipif(not RESULT.exists(), reason="the head-to-head is not scored yet")


def _rows(name):
    return [json.loads(line) for line in (RECORDS / name).read_text().splitlines() if line]


def test_published_rows_recompute_from_the_committed_reference():
    result = json.loads(RESULT.read_text())
    with (RECORDS / "reference-published.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert set(result["published"]) == {row["model"] for row in rows}
    for model, expected in result["published"].items():
        mine = [row for row in rows if row["model"] == model]
        assert len(mine) == 68
        got = metatime.metrics(
            [float(row["estimate_before_seconds"]) for row in mine],
            [float(row["actual_seconds"]) for row in mine],
        )
        for name in CHECKED:
            assert got[name] == pytest.approx(expected[name]), (model, name)


def test_our_rows_recompute_from_the_committed_records():
    result = json.loads(RESULT.read_text())
    runs = {}
    for row in _rows("runs.jsonl"):
        runs.setdefault((row["executor"], row["task"]), []).append(row["wall_s"])
    estimates = {}
    for row in _rows("estimates.jsonl"):
        estimates.setdefault(f"{row['estimator']}/{row['variant']}", []).append(row)
    measured = [key for key in estimates if not result["systems"][key]["replayed_at_stated_speed"]]
    assert measured
    for key in measured:
        system, rows = result["systems"][key], estimates[key]
        assert len(rows) == 68
        actual = [statistics.median(runs[(system["scored_against"], row["task"])]) for row in rows]
        got = metatime.metrics([row["stated_seconds"] for row in rows], actual)
        for name in CHECKED:
            assert got[name] == pytest.approx(system["stated"][name]), (key, name)


def test_records_carry_no_benchmark_task_text():
    for row in _rows("estimates.jsonl"):
        assert set(row) <= {
            "estimator",
            "variant",
            "task",
            "answer",
            "answer_withheld",
            "stated_seconds",
            "token_reading_seconds",
            "response_tokens",
            "S",
            "P",
            "scored_against",
            "sampled_median_seconds",
            "sampled_p10_seconds",
            "sampled_p90_seconds",
        }
    for row in _rows("runs.jsonl"):
        assert "text" not in row and "prompt" not in row
