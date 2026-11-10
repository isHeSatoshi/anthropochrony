"""Confirmatory thresholds. Missing experiments remain unmeasured, never 'passed'."""

from collections import defaultdict

import numpy as np

from anthropochrony.eval.metrics import summary, task_records
from anthropochrony.eval.seeded import human_preservation, speed_conditioning
from anthropochrony.eval.stats import holm, paired_wilcoxon, stratified_bootstrap
from anthropochrony.eval.targets import primary_rows


def confirmatory(rows, *, capability=None, seeds=(0, 1, 2), resamples=10000, evaluation_cards=None):
    from anthropochrony.eval.reporting import normalized_rows

    all_rows = normalized_rows(rows)
    rows = primary_rows(all_rows)
    result = {
        key: {"status": "not_measured"} for key in ("H1", "H2", "H3", "H4", "H5", "H6a", "H6b")
    }
    tests = {}

    def select(prefix, split="test", frame="self", card="REF"):
        return [
            r
            for r in rows
            if r["estimator"].lower().split("-s")[0] == prefix.lower()
            and r["split"] == split
            and r.get("frame", "self") == frame
            and (card is None or r["card_id"] == card)
        ]

    e0, e1 = select("E0"), select("E1")
    if e0 and e1:
        a, b = summary(task_records(e0)), summary(task_records(e1))
        result["H1"] = {
            "status": "measured",
            "passed": a["gm_ratio"] is not None
            and b["gm_ratio"] is not None
            and a["gm_ratio"] >= 10
            and b["gm_ratio"] >= 10
            and a["aur"] >= 0.5,
            "E0_gm_ratio": a["gm_ratio"],
            "E1_gm_ratio": b["gm_ratio"],
            "E0_aur": a["aur"],
        }
    validation = {}

    def base_rows(estimator, split):
        return select(estimator, split=split, frame="generic" if estimator == "E3" else "self")

    for estimator in ("E0", "E1", "E2", "E3"):
        candidates = base_rows(estimator, "val")
        if candidates:
            validation[estimator] = summary(task_records(candidates))["mdae_log10"]
    best = min(validation, key=validation.get) if len(validation) == 4 else None

    def seed_averages(prefix, split):
        selected = select(prefix, split)
        observed_seeds = {r.get("seed") for r in selected}
        if not set(seeds).issubset(observed_seeds):
            return None
        # Templates are averaged within tasks and seeds, then seeds within tasks.
        by_seed = {}
        for seed in seeds:
            by_seed[seed] = task_records([r for r in selected if r.get("seed") == seed])
        grouped = defaultdict(list)
        for records in by_seed.values():
            for record in records:
                grouped[record["task_id"]].append(record)
        if any(len(values) != len(seeds) for values in grouped.values()):
            raise ValueError("Training seeds do not share the same evaluation tasks")
        errors = {
            task_id: float(np.mean([r["abs_error"] for r in values]))
            for task_id, values in grouped.items()
        }
        per_seed = [summary(records)["mdae_log10"] for records in by_seed.values()]
        return errors, per_seed

    for hypothesis, split, threshold in (("H2", "test", 0.30), ("H3", "test_ood", 0.50)):
        measured = seed_averages("M1", split)
        if measured and best:
            errors, per_seed = measured
            baseline = {r["task_id"]: r["abs_error"] for r in task_records(base_rows(best, split))}
            if set(errors) != set(baseline):
                raise ValueError("Paired baseline task IDs do not match")
            ids = sorted(errors)
            tests[hypothesis] = paired_wilcoxon(
                [errors[i] for i in ids], [baseline[i] for i in ids]
            )
            result[hypothesis] = {
                "status": "measured",
                "threshold_passed": float(np.mean(per_seed)) <= threshold,
                "mean_seed_mdae": float(np.mean(per_seed)),
                "sd_seed_mdae": float(np.std(per_seed, ddof=1)),
                "selected_base_prompt_on_val": best,
            }
    m1, m2 = seed_averages("M1", "test"), seed_averages("M2", "test")
    if m1 and m2:
        first, _ = m1
        second, _ = m2
        if set(first) != set(second):
            raise ValueError("H5 needs identical task IDs")
        ids = sorted(first)
        tests["H5"] = paired_wilcoxon([first[i] for i in ids], [second[i] for i in ids])
        result["H5"] = {"status": "measured", "threshold_passed": True}
    # Holm's family includes all three confirmatory tests, even if some are unmeasured.
    adjusted = holm({key: tests.get(key, 1.0) for key in ("H2", "H3", "H5")})
    for hypothesis in tests:
        result[hypothesis].update(
            p_raw=tests[hypothesis],
            p_holm=adjusted[hypothesis],
            passed=result[hypothesis]["threshold_passed"] and adjusted[hypothesis] < 0.05,
        )
    speed = speed_conditioning(select("M1", card=None), seeds, resamples, evaluation_cards)
    if speed:
        result["H4"] = speed
    human = [
        row
        for row in all_rows
        if row["split"] == "test" and row.get("frame") == "human" and row["card_id"] == "REF"
    ]
    preservation = human_preservation(
        [r for r in human if r["estimator"] == "E1"],
        [r for r in human if r["estimator"].lower().split("-s")[0] == "m1"],
        seeds,
        resamples,
    )
    if preservation:
        result["H6b"] = preservation
    if capability:
        if any(
            row.get("seed") != 0
            or row.get("split") != "test"
            or row.get("estimator", "").lower() != "m1-s0"
            or any(not 0 <= row[field] <= 1 for field in ("base_pass1", "ft_pass1"))
            for row in capability
        ) or len({row["task_id"] for row in capability}) != len(capability):
            raise ValueError("H6a requires unique paired test tasks for M1 seed 0 only")
        lower, upper = stratified_bootstrap(
            capability,
            lambda records: float(np.mean([r["ft_pass1"] - r["base_pass1"] for r in records])),
            resamples=resamples,
        )
        result["H6a"] = {
            "status": "measured",
            "paired_pass1_difference_ci95": [lower, upper],
            "passed": lower > -0.03,
        }
    return result
