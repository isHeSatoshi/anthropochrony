import json
import math

import pytest

from anthropochrony.artifacts import write_jsonl
from anthropochrony.cli import main
from anthropochrony.eval.capability import capability_pairs
from anthropochrony.eval.hypotheses import confirmatory
from anthropochrony.eval.reporting import report


def row(task, estimator, seed, point, actual, card="REF", frame="self", template="q21"):
    return {
        "task_id": task,
        "family": "FUNC",
        "split": "test",
        "estimator": estimator,
        "seed": seed,
        "card_id": card,
        "frame": frame,
        "template_id": template,
        "phrasing_id": "sc07",
        "T_med_s": actual,
        "target_source": "base" if frame == "human" else "own",
        "parsed": {"point_s": point, "has_human_units": False},
        "messages": [{"role": "user", "content": f"{frame} {task} {template}"}],
    }


def speed_rows(errors=(0, 0, 1.2)):
    return [
        row(str(task), f"M1-s{seed}", seed, actual * 10**error, actual, card)
        for task in range(4)
        for seed, error in enumerate(errors)
        for card, actual in (("REF", 1), ("heldout-190", 2), ("heldout-230", 4))
    ]


def human_rows():
    rows = []
    for task in range(4):
        for template in ("h09", "h10"):
            rows.append(row(str(task), "E1", None, 10, 1, frame="human", template=template))
            for seed, error in enumerate((0, 0, 0.6)):
                rows.append(
                    row(
                        str(task),
                        f"M1-s{seed}",
                        seed,
                        10 * 10**error,
                        1,
                        frame="human",
                        template=template,
                    )
                )
    return rows


def test_h4_does_not_hide_one_bad_seed_behind_a_pooled_median():
    result = confirmatory(speed_rows(), resamples=20)["H4"]
    assert result["mdae_log10"] == pytest.approx(0.4)
    assert result["elasticity"] == pytest.approx(1)
    assert result["error_by_seed"]["per_seed"]["2"]["median"] == pytest.approx(1.2)
    assert result["passed"] is False
    assert confirmatory(speed_rows((0, 0, 0)), resamples=20)["H4"]["passed"]


def test_h4_gives_each_card_one_regression_point_regardless_of_template_count():
    rows = speed_rows((0, 0, 0))
    for r in rows:
        if r["card_id"] == "REF":
            r["parsed"]["point_s"] = 16
    extras = [
        r | {"template_id": f"extra-{i}"} for r in rows if r["card_id"] == "REF" for i in range(20)
    ]
    first = confirmatory(rows, resamples=20)["H4"]["elasticity"]
    repeated = confirmatory(rows + extras, resamples=20)["H4"]["elasticity"]
    # The equal-card points are log2(x)=(0,1,2), log2(y)=(4,1,2), slope -1.
    assert first == repeated == pytest.approx(-1)
    with pytest.raises(ValueError, match="complete evaluation-card"):
        confirmatory(rows[:-1], resamples=20)


def test_h6b_matches_templates_and_uses_all_three_seeds_without_robustness_duplicates():
    rows = human_rows()
    result = confirmatory(rows + rows, resamples=20)["H6b"]
    assert result["mean_seed_median_abs_log10_ratio"] == pytest.approx(0.2)
    assert result["seed_metrics"]["per_seed"]["2"]["median"] == pytest.approx(0.6)
    assert result["passed"] is False
    assert confirmatory([r for r in rows if r["seed"] != 2])["H6b"]["status"] == "not_measured"
    with pytest.raises(ValueError, match="same human prompts"):
        confirmatory(rows[:-1], resamples=20)
    changed = [dict(r) for r in rows]
    changed[-1]["messages"] = [{"role": "user", "content": "A different human question"}]
    with pytest.raises(ValueError, match="identical human"):
        confirmatory(changed, resamples=20)


def test_h6b_does_not_drop_unparseable_base_reference_answers():
    rows = human_rows()
    for r in rows:
        if r["estimator"] == "E1":
            r["parsed"]["point_s"] = None
    result = confirmatory(rows, resamples=20)["H6b"]
    assert math.isinf(result["mean_seed_median_abs_log10_ratio"])
    assert result["seed_metrics"]["sd"] is None
    assert result["passed"] is False


def test_h6a_uses_complete_k8_seed_zero_first_attempts_only():
    tasks = [{"task_id": str(i), "family": "FUNC", "split": "test"} for i in range(4)]

    def traces(sha, count):
        return [
            {
                "task_id": t["task_id"],
                "k": k,
                "model_sha": sha,
                "card": {"id": "REF", "S": 96, "P": 26000},
                "first_pass": k < count,
                "attempts": [{"exec": {"status": "pass" if k < count else "fail"}}],
            }
            for t in tasks
            for k in range(8)
        ]

    base, tuned = traces("base", 4), traces("fine-tuned", 3)
    pairs = capability_pairs(tasks, base, tuned, "base", "fine-tuned")
    assert all(r["base_pass1"] == 0.5 and r["ft_pass1"] == 0.375 for r in pairs)
    result = confirmatory([], capability=pairs, resamples=20)["H6a"]
    assert result["paired_pass1_difference_ci95"] == [-0.125, -0.125]
    assert result["passed"] is False
    with pytest.raises(ValueError, match="seed 0"):
        confirmatory([], capability=pairs + [pairs[0] | {"seed": 1}], resamples=20)
    with pytest.raises(ValueError, match="complete K"):
        capability_pairs(tasks, base, tuned[:-1], "base", "fine-tuned")
    with pytest.raises(ValueError, match="pinned own REF"):
        capability_pairs(tasks, base, tuned, "base", "wrong-model")


def test_report_writes_hypotheses_and_cli_blocks_unfrozen_finetuned_test_results(tmp_path):
    observation = row("a", "M1-s0", 0, 2, 2)
    report(tmp_path, [observation], resamples=20)
    hypotheses = json.loads((tmp_path / "results/hypotheses.json").read_text())
    assert all(r["status"] == "not_measured" for r in hypotheses.values())
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [observation])
    with pytest.raises(SystemExit, match="Freeze"):
        main(["--root", str(tmp_path), "eval", "--estimates", str(path)])
