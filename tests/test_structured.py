import json
import math

import pytest

from anthropochrony.eval.metrics import interval_metrics
from anthropochrony.eval.structured import (
    WorkloadPriors,
    apply_conformal_rows,
    build_structured_rows,
    fit_split_conformal,
    hybrid_ensemble_analysis,
    structured_duration,
    x10_benchmark_metrics,
    x10_gpu_deployment_struct,
    x10_hardware_tracking,
)
from anthropochrony.timemodel import PER_CALL_LATENCY_S, SpeedCard


def _priors():
    return WorkloadPriors(
        feedback_by_family={"FUNC": 60.0},
        feedback_default=60.0,
        exec_per_attempt_by_family={"FUNC": 0.1},
        exec_per_attempt_default=0.1,
    )


def test_structured_duration_exact_physics_and_slow_speed_scaling():
    priors = _priors()
    slow = SpeedCard("extrap-low", 1.5, 10.0)
    fast = SpeedCard("seen-dc", 400.0, 40000.0)
    # 2 attempts: u_total = 100 + 60 = 160; exec = 0.2; o_tokens = 300
    t_slow = structured_duration(300, 2, 100, "FUNC", slow, priors)
    expected_slow = 160 / 10.0 + 300 / 1.5 + 0.2 + 2 * PER_CALL_LATENCY_S
    assert t_slow == pytest.approx(expected_slow)

    t_local = structured_duration(300, 2, 100, "FUNC", slow, priors, latency_s=0.0)
    assert t_local == pytest.approx(160 / 10.0 + 300 / 1.5 + 0.2)

    t_fast = structured_duration(300, 2, 100, "FUNC", fast, priors)
    assert t_fast < t_slow / 50.0
    assert structured_duration(None, 1, 100, "FUNC", slow, priors) is None
    assert structured_duration(0, 1, 100, "FUNC", slow, priors) is None


def test_build_structured_rows_preserves_targets_and_clears_human_units():
    priors = _priors()
    tasks = {"t1": {"task_id": "t1", "family": "FUNC", "uncached_task_tokens": 100}}
    rows = [
        {
            "estimator": "M1-S0",
            "task_id": "t1",
            "split": "test",
            "family": "FUNC",
            "card_id": "extrap-low",
            "frame": "self",
            "S": 1.5,
            "P": 10.0,
            "tokens": 150.0,
            "attempts": 1.0,
            "point_s": 12.0,  # Under-extrapolated verbal estimate
            "actual_s": 111.0,
            "T_med_s": 111.0,
            "target_source": "own",
            "has_human_units": True,
        }
    ]
    built = build_structured_rows(rows, tasks, priors)
    assert len(built) == 1
    assert built[0]["estimator"] == "M1-Struct"
    assert built[0]["verbal_point_s"] == 12.0
    assert built[0]["point_s"] == pytest.approx(100 / 10.0 + 150 / 1.5 + 0.1 + PER_CALL_LATENCY_S)
    assert not built[0]["has_human_units"]
    assert built[0]["target_source"] == "own"


def test_split_conformal_guarantees_finite_sample_coverage_on_exchangeable_data():
    priors = _priors()
    tasks = {
        f"t{i}": {"task_id": f"t{i}", "family": "FUNC", "uncached_task_tokens": 100}
        for i in range(40)
    }
    m1_rows = []
    for i in range(40):
        split = "val" if i < 20 else "test"
        # True actual alternates multiplicatively wider than raw [lo_s, hi_s]
        factor = 10 ** (0.30 if i % 2 == 0 else -0.25)
        actual = 10.0 * factor
        m1_rows.append(
            {
                "estimator": "M1-S0",
                "task_id": f"t{i}",
                "split": split,
                "family": "FUNC",
                "card_id": "REF",
                "frame": "self",
                "S": 96.0,
                "P": 26000.0,
                "tokens": 860.0,
                "attempts": 1.0,
                "point_s": 10.0,
                "lo_s": 9.0,
                "hi_s": 11.0,
                "actual_s": actual,
                "T_med_s": actual,
                "target_source": "base" if split == "val" else "own",
            }
        )
    struct_rows = build_structured_rows(m1_rows, tasks, priors)
    # Decoy non-M1 rows on val must never contaminate conformal calibration
    decoy = [dict(m1_rows[0], estimator="E0", lo_s=1e5, hi_s=2e5, point_s=1.5e5)]
    cal = fit_split_conformal(m1_rows + decoy, struct_rows, alpha=0.20)
    assert cal.cqr_factor > 1.5
    assert cal.struct_factor > 1.5
    m1b, st_cp = apply_conformal_rows(m1_rows, struct_rows, cal)

    raw_test = [r for r in m1_rows if r["split"] == "test"]
    m1b_test = [r for r in m1b if r["split"] == "test"]
    st_test = [r for r in st_cp if r["split"] == "test"]

    raw_cov = interval_metrics(
        [r["actual_s"] for r in raw_test],
        [r["lo_s"] for r in raw_test],
        [r["hi_s"] for r in raw_test],
    )
    cqr_cov = interval_metrics(
        [r["actual_s"] for r in m1b_test],
        [r["lo_s"] for r in m1b_test],
        [r["hi_s"] for r in m1b_test],
    )
    st_cov = interval_metrics(
        [r["actual_s"] for r in st_test],
        [r["lo_s"] for r in st_test],
        [r["hi_s"] for r in st_test],
    )
    assert raw_cov["coverage80"] == 0.0
    assert cqr_cov["coverage80"] >= 0.80
    assert st_cov["coverage80"] >= 0.80
    assert st_cov["winkler"] < raw_cov["winkler"]


def test_hybrid_ensemble_tunes_strictly_on_val():
    struct_rows, b2_rows = [], []
    for split, offset in (("val", 0), ("test", 10)):
        for i in range(10):
            tid = f"t{offset + i}"
            actual = 10.0
            # M1-Struct overpredicts by +0.2 dex, B2 underpredicts by -0.2 dex -> 0.5 blend is exact!
            struct_rows.append(
                {
                    "estimator": "M1-Struct",
                    "task_id": tid,
                    "split": split,
                    "family": "FUNC",
                    "card_id": "REF",
                    "frame": "self",
                    "point_s": actual * (10**0.2),
                    "actual_s": actual,
                }
            )
            b2_rows.append(
                {
                    "estimator": "B2",
                    "task_id": tid,
                    "split": split,
                    "family": "FUNC",
                    "card_id": "REF",
                    "frame": "self",
                    "point_s": actual * (10**-0.2),
                    "actual_s": actual,
                }
            )
    res = hybrid_ensemble_analysis(struct_rows, b2_rows, resamples=20)
    assert res["val_selected_w_m1_struct"] == pytest.approx(0.5)
    assert res["splits"]["test"]["val_selected_blend"]["mdae_log10"] < 1e-9
    assert math.isclose(res["splits"]["test"]["val_selected_blend"]["within2x"], 1.0)


def _write_x10(root, name, card, rows):
    folder = root / "data/x10"
    folder.mkdir(parents=True, exist_ok=True)
    stem = "cpu" if name == "cpu-deployment" else name
    (folder / f"{stem}-speed-card.json").write_text(json.dumps(card.as_dict()))
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_x10_hardware_tracking_pairs_tasks_and_scores_predicted_speedup(tmp_path):
    priors = _priors()
    tasks = [
        {"task_id": "t1", "family": "FUNC", "uncached_task_tokens": 100},
        {"task_id": "t2", "family": "FUNC", "uncached_task_tokens": 100},
    ]
    (tmp_path / "data/tasks").mkdir(parents=True)
    (tmp_path / "data/tasks/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    tuned = "About 20 s. Breakdown: ~300 output tokens across 1 attempt(s)."
    cpu_card, gpu_card = SpeedCard("demo-llamacpp", 20, 200), SpeedCard("demo-llamacpp", 100, 7400)

    def row(task_id, wall_s, base_raw="About 7 minutes."):
        return {
            "task_id": task_id,
            "family": "FUNC",
            "base_raw": base_raw,
            "tuned_raw": tuned,
            "wall_s": wall_s,
        }

    _write_x10(tmp_path, "cpu-deployment", cpu_card, [row("t1", 40.0), row("t2", 30.0)])
    _write_x10(tmp_path, "gpu-deployment", gpu_card, [row("t1", 8.0, base_raw="no idea")])

    gpu = x10_gpu_deployment_struct(tmp_path, priors)
    assert gpu["speed_card"]["S"] == 100
    assert gpu["m1_struct_local"]["n_tasks"] == 1

    out = x10_hardware_tracking(tmp_path, priors)
    assert out["n_paired_tasks"] == 1
    assert out["actual_speedup_gm"] == pytest.approx(5.0)
    expected = structured_duration(300, 1, 100, "FUNC", cpu_card, priors, latency_s=0.0) / (
        structured_duration(300, 1, 100, "FUNC", gpu_card, priors, latency_s=0.0)
    )
    struct = out["m1_struct_local"]
    assert struct["predicted_speedup_gm"] == pytest.approx(expected)
    assert struct["speedup_mdae_log10"] == pytest.approx(abs(math.log10(expected / 5.0)))
    # The verbal answer does not change with the card, so it predicts no speedup.
    assert out["m1_verbal"]["predicted_speedup_gm"] == pytest.approx(1.0)
    # An unparseable answer on either machine counts as a miss.
    assert out["base_verbal"]["n_parsed_pairs"] == 0
    assert out["base_verbal"]["speedup_within2x"] == 0.0


def test_x10_hardware_tracking_needs_both_runs(tmp_path):
    assert x10_hardware_tracking(tmp_path, _priors()) is None


def test_x10_benchmark_metrics_use_natural_log_and_count_unparsed_as_misses(tmp_path):
    (tmp_path / "data/tasks").mkdir(parents=True)
    tasks = [{"task_id": f"t{i}", "family": "FUNC", "uncached_task_tokens": 100} for i in range(3)]
    (tmp_path / "data/tasks/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    rows = [
        {
            "task_id": f"t{i}",
            "family": "FUNC",
            "base_raw": "no idea" if i == 0 else "About 80 seconds.",
            "tuned_raw": f"About {seconds} s.",
            "wall_s": wall_s,
        }
        for i, (seconds, wall_s) in enumerate([(20, 10.0), (20, 20.0), (20, 40.0)])
    ]
    _write_x10(tmp_path, "cpu-deployment", SpeedCard("demo-llamacpp", 20, 200), rows)

    out = x10_benchmark_metrics(tmp_path, _priors())
    assert set(out) == {"cpu"}
    tuned = out["cpu"]["m1_verbal"]
    assert tuned["median_ratio"] == pytest.approx(1.0)
    assert tuned["median_abs_log_error"] == pytest.approx(math.log(2))
    assert tuned["within_2x"] == pytest.approx(1.0)
    base = out["cpu"]["base_verbal"]
    assert base["parsed"] == 2
    assert base["within_2x"] == pytest.approx(1 / 3)
    assert base["median_abs_log_error"] == pytest.approx(math.log(4))
