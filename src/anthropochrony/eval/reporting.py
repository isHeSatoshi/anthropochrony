import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anthropochrony.artifacts import manifest, read_jsonl, write_json, write_jsonl
from anthropochrony.eval.metrics import interval_metrics, pairwise_accuracy, summary, task_records
from anthropochrony.eval.stats import json_safe, stratified_bootstrap
from anthropochrony.parse import parse


def normalized_rows(estimates):
    rows = []
    for record in estimates:
        parsed = record.get("parsed") or parse(record.get("raw_text", "")).as_dict()
        actual = record.get("actual_s", record.get("T_med_s"))
        if actual is None:
            raise ValueError("Every estimate needs an independently measured target")
        rows.append(record | parsed | {"actual_s": actual})
    return rows


def report(root, estimates, resamples=10000, capability=None):
    from anthropochrony.cli import config
    from anthropochrony.eval.hypotheses import confirmatory

    root = Path(root)
    if not estimates:
        raise ValueError("No measured estimates. Reports never fabricate research results")
    rows = normalized_rows(estimates)
    if capability is None and (root / "data/eval/capability-m1-s0.jsonl").exists():
        capability = read_jsonl(root / "data/eval/capability-m1-s0.jsonl")
    evaluation_cards = (
        ["REF"] + [row["id"] for row in config(root, "speeds.yaml")["evaluation"]]
        if (root / "configs/speeds.yaml").exists()
        else None
    )
    hypotheses = confirmatory(
        estimates, capability=capability, resamples=resamples, evaluation_cards=evaluation_cards
    )
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["estimator"],
            row["split"],
            row["card_id"],
            row.get("frame", "self"),
            row.get("target_source", "base"),
        )
        grouped[key].append(row)
    tables = []
    for key, group in sorted(grouped.items()):
        records = task_records(group)
        metrics = summary(records)
        ci = {}
        for name, statistic in {
            "mdae_log10": lambda rs: float(np.median([r["abs_error"] for r in rs])),
            "within2x": lambda rs: float(np.mean([r["abs_error"] <= np.log10(2) for r in rs])),
            "parse_rate": lambda rs: float(np.mean([r["parse_rate"] for r in rs])),
            "aur": lambda rs: float(np.mean([r["aur"] for r in rs])),
        }.items():
            ci[name] = stratified_bootstrap(records, statistic, resamples=resamples)
        intervals = [
            r
            for r in group
            if r.get("lo_s") is not None
            and r.get("hi_s") is not None
            and 0 < r["lo_s"] <= r["hi_s"]
        ]
        if intervals:
            metrics.update(
                interval_metrics(
                    [r["actual_s"] for r in intervals],
                    [r["lo_s"] for r in intervals],
                    [r["hi_s"] for r in intervals],
                )
            )
        tables.append(
            {
                "estimator": key[0],
                "split": key[1],
                "card_id": key[2],
                "frame": key[3],
                "target_source": key[4],
                **metrics,
                "ci95": ci,
                "pairwise": pairwise_accuracy(records),
            }
        )
    result_dir, figure_dir = root / "results", root / "paper/figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        result_dir / "hypotheses.json", json_safe(hypotheses), config={"resamples": resamples}
    )
    write_json(result_dir / "metrics.json", json_safe(tables), config={"resamples": resamples})
    write_jsonl(result_dir / "analysis_rows.jsonl", json_safe(rows))
    flat = [
        {k: v for k, v in row.items() if k not in {"ci95", "pairwise"}} for row in json_safe(tables)
    ]
    with (result_dir / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in flat for k in row}))
        writer.writeheader()
        writer.writerows(flat)
    manifest(result_dir / "metrics.csv", config={"resamples": resamples})
    primary = [
        r
        for r in rows
        if r["split"] == "test"
        and r["card_id"] == "REF"
        and r.get("frame", "self") == "self"
        and r.get("point_s") is not None
        and not (
            r["estimator"].lower().split("-s")[0] in {"m1", "m2"}
            and r.get("target_source") != "own"
        )
    ]
    if primary:
        estimators = sorted({r["estimator"] for r in primary})
        fig, axes = plt.subplots(
            1, len(estimators), figsize=(4 * len(estimators), 4), squeeze=False
        )
        for estimator, axis in zip(estimators, axes.flat, strict=True):
            selected = [r for r in primary if r["estimator"] == estimator]
            x, y = [r["actual_s"] for r in selected], [r["point_s"] for r in selected]
            axis.scatter(x, y, s=15, alpha=0.55)
            lower, upper = min(x + y) * 0.7, max(x + y) * 1.4
            line = np.geomspace(lower, upper, 100)
            axis.plot(line, line, color="black", linewidth=1)
            axis.fill_between(line, line / 2, line * 2, color="#37a899", alpha=0.15)
            axis.set(
                xscale="log",
                yscale="log",
                xlabel="Measured own completion time (s)",
                ylabel="Estimated completion time (s)",
                title=estimator,
            )
        fig.suptitle("Anthropochrony · separate-conversation estimates")
        fig.tight_layout()
        fig.savefig(figure_dir / "F1-estimated-vs-measured.png", dpi=180)
        plt.close(fig)
        manifest(figure_dir / "F1-estimated-vs-measured.png")
    write_json(
        result_dir / "report-status.json",
        {
            "synthetic": False,
            "n_estimates": len(rows),
            "bootstrap_resamples": resamples,
            "note": "Only supplied measured observations are reported. Missing experiments are not inferred.",
            "available_estimators": sorted({row["estimator"] for row in rows}),
        },
    )
    return tables


def roi(
    *,
    volume,
    input_tokens,
    output_tokens,
    tokens_per_second,
    gpu_usd_per_hour,
    api_input_usd_per_million,
    api_output_usd_per_million,
    price_source,
    price_date,
):
    if min(volume, tokens_per_second) <= 0:
        raise ValueError("Positive volume and measured throughput required")
    hours = volume * output_tokens / tokens_per_second / 3600
    return {
        "monthly_estimates": volume,
        "gpu_hours": hours,
        "self_hosted_usd": hours * gpu_usd_per_hour,
        "api_usd": volume
        * (input_tokens * api_input_usd_per_million + output_tokens * api_output_usd_per_million)
        / 1e6,
        "price_source": price_source,
        "price_date": price_date,
        "caveat": "Generation-only utilization estimate; excludes idle capacity, prefill and hosting overhead.",
    }
