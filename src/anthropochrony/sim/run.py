"""Exploratory scheduling and timeout simulations on measured estimates and trajectories (X6)."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.eval.metrics import task_records
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.sim import scheduling, timeout_tradeoffs
from anthropochrony.timemodel import SpeedCard, trajectory_time


def trace_path(root, estimator, target_source):
    run = estimator.lower() if target_source == "own" else "base"
    return Path(root) / f"data/measure/{run}/traces.jsonl"


def task_table(rows):
    """Task-level point estimate, upper estimate and measured time (templates averaged in log space)."""
    records = {r["task_id"]: r for r in task_records(rows)}
    uppers = defaultdict(list)
    for row in rows:
        if row.get("hi_s") and row["hi_s"] > 0:
            uppers[row["task_id"]].append(math.log10(row["hi_s"]))
    table = {}
    for task_id, record in records.items():
        if record["predicted"] is None:
            continue
        upper = 10 ** np.mean(uppers[task_id]) if uppers[task_id] else record["predicted"]
        table[task_id] = {
            "predicted": record["predicted"],
            "upper": max(upper, record["predicted"]),
            "actual": record["actual"],
        }
    return table, len(records)


def reference_trajectories(path, card):
    out = defaultdict(list)
    for trace in read_jsonl(path):
        out[trace["task_id"]].append((trajectory_time(trace, card), bool(trace["passed"])))
    return out


def simulate(root, estimates_path=None, card=None):
    root = Path(root)
    card = card or SpeedCard("REF", 96.0, 27000.0)
    rows = normalized_rows(read_jsonl(estimates_path or root / "data/eval/estimates.jsonl"))
    groups = defaultdict(list)
    for row in rows:
        if (
            row["split"] == "test"
            and row["card_id"] == "REF"
            and row.get("frame", "self") == "self"
            and row["family"] != "CLASS"
        ):
            groups[(row["estimator"], row.get("target_source"))].append(row)
    results = {}
    for (estimator, target_source), group in sorted(groups.items(), key=str):
        table, total = task_table(group)
        if len(table) < 50:
            continue
        ids = sorted(table)
        # Tasks without a parsed estimate are excluded; n_with_estimate vs n_tasks reports how many.
        entry = {
            "n_tasks": total,
            "n_with_estimate": len(table),
            "scheduling": scheduling(
                [table[t]["actual"] for t in ids], [table[t]["predicted"] for t in ids]
            ),
        }
        path = trace_path(root, estimator, target_source)
        if path.exists():
            trajectories = reference_trajectories(path, card)
            actual, upper, passed = [], [], []
            for task_id in ids:
                for seconds, ok in trajectories.get(task_id, ()):
                    actual.append(seconds)
                    upper.append(table[task_id]["upper"])
                    passed.append(ok)
            entry["tripwire"] = timeout_tradeoffs(actual, upper, passed)
            entry["tripwire_n_runs"] = len(actual)
            entry["tripwire_n_failed_runs"] = int(len(passed) - sum(passed))
        results[f"{estimator}|{target_source or 'base'}"] = entry
    write_json(root / "results/simulations.json", results)
    return results
