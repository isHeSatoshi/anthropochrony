import json

from anthropochrony.eval import metatime, plots


def _system(estimate, actual):
    return {**metatime.metrics(estimate, actual), "ci95": metatime.intervals(estimate, actual)}


def test_head_to_head_figure_is_drawn_from_committed_records(tmp_path):
    records = tmp_path / "results/metatime"
    records.mkdir(parents=True)
    figures = tmp_path / "figures"
    figures.mkdir()
    tasks = [f"t{i}" for i in range(6)]
    actual = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    ours = [2.5, 3.0, 9.0, 20.0, 30.0, 50.0]
    frontier = [value * 3 for value in actual]
    (records / "runs.jsonl").write_text(
        "".join(
            json.dumps({"executor": "g1-s0", "task": task, "run": run, "wall_s": seconds}) + "\n"
            for task, seconds in zip(tasks, actual, strict=True)
            for run in range(2)
        )
    )
    (records / "estimates.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "estimator": "g1-s0",
                    "variant": "card_description",
                    "task": task,
                    "stated_seconds": seconds,
                }
            )
            + "\n"
            for task, seconds in zip(tasks, ours, strict=True)
        )
    )
    lines = ["task,category,model,actual_seconds,estimate_before_seconds"]
    for model in ("gpt-5.5", "claude-opus-4-8"):
        lines += [
            f"{task},qa,{model},{seconds},{guess}"
            for task, seconds, guess in zip(tasks, actual, frontier, strict=True)
        ]
    (records / "reference-published.csv").write_text("\n".join(lines) + "\n")
    result = {
        "published": {model: _system(frontier, actual) for model in ("gpt-5.5", "claude-opus-4-8")},
        "executors": {"g1-s0": {"retest_ceiling_abs_log_error": 0.05}},
        "systems": {
            "g1-s0/card_description": {"scored_against": "g1-s0", "stated": _system(ours, actual)},
            # An estimator whose answers mostly cannot be read still gets a row.
            "untrained/protocol": {
                "scored_against": "base",
                "stated": _system([None, None, None, None, 5.0, 6.0], actual),
            },
        },
    }
    (tmp_path / "results/metatime.json").write_text(json.dumps(result))

    plots.head_to_head(tmp_path, figures)

    assert (figures / "F9-metatime-head-to-head.png").stat().st_size > 10_000


def test_head_to_head_figure_is_skipped_without_results(tmp_path):
    plots.head_to_head(tmp_path, tmp_path)
    assert not list(tmp_path.glob("*.png"))
