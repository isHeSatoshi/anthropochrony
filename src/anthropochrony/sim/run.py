"""Exploratory scheduling and timeout simulations on measured estimates and trajectories (X6)."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.eval.metrics import task_records
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.eval.structured import (
    apply_conformal_rows,
    build_structured_rows,
    fit_split_conformal,
    fit_workload_priors,
)
from anthropochrony.sim import scheduling, sla_tier_routing, timeout_tradeoffs
from anthropochrony.timemodel import SpeedCard, trajectory_time


def trace_path(root, estimator, target_source):
    if target_source == "own":
        base_name = estimator.split("+")[0]
        if base_name in {"M1-Struct", "M1b"}:
            run = "m1-s0"
        elif base_name == "M2-Struct":
            run = "m2-s0"
        elif base_name == "M3-Struct":
            run = "m3-s0"
        else:
            run = estimator.lower()
    else:
        run = "base"
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


def _enrich_posthoc_rows(root, rows):
    tasks_path = root / "data/tasks/tasks.jsonl"
    if not tasks_path.exists():
        return []
    try:
        priors = fit_workload_priors(root)
    except FileNotFoundError:
        return []
    tasks = {t["task_id"]: t for t in read_jsonl(tasks_path)}
    m1_struct = build_structured_rows(
        rows, tasks, priors, source_estimator="M1-S0", estimator_name="M1-Struct"
    )
    if not any(r.get("split") == "val" for r in m1_struct):
        return m1_struct
    val_traces = defaultdict(list)
    base_traces = root / "data/measure/base/traces.jsonl"
    if base_traces.exists():
        for tr in read_jsonl(base_traces):
            if tasks.get(tr["task_id"], {}).get("split") == "val":
                val_traces[tr["task_id"]].append(tr)
    cal = fit_split_conformal(
        rows,
        m1_struct,
        val_traces_by_task=val_traces or None,
        card=SpeedCard("REF", 96.0, 26000.0),
        alpha=0.20,
    )
    m1b, struct_cp = apply_conformal_rows(rows, m1_struct, cal)
    return m1_struct + m1b + struct_cp


def _ood_and_routing_summary(root, all_rows):
    edge_card = SpeedCard("seen-edge", 60.0, 3000.0)
    dc_card = SpeedCard("seen-dc", 400.0, 40000.0)
    ref_card = SpeedCard("REF", 96.0, 26000.0)

    # 1) OOD CLASS SJF + tripwires (queue_size=30 on 93 unseen CLASS tasks)
    ood_groups = defaultdict(list)
    for row in all_rows:
        if (
            row["split"] == "test_ood"
            and row["card_id"] == "REF"
            and row.get("frame", "self") == "self"
        ):
            ood_groups[(row["estimator"], row.get("target_source"))].append(row)
    ood_out = {}
    for (est, tsrc), group in sorted(ood_groups.items(), key=str):
        table, total = task_table(group)
        if len(table) < 30:
            continue
        ids = sorted(table)
        entry = {
            "n_tasks": total,
            "n_with_estimate": len(table),
            "scheduling_q30": scheduling(
                [table[t]["actual"] for t in ids],
                [table[t]["predicted"] for t in ids],
                queue_size=30,
            ),
        }
        p = trace_path(root, est, tsrc)
        if p.exists():
            trajs = reference_trajectories(p, ref_card)
            act, up, ok_list = [], [], []
            for tid in ids:
                for secs, ok in trajs.get(tid, ()):
                    act.append(secs)
                    up.append(table[tid]["upper"])
                    ok_list.append(ok)
            if act:
                entry["tripwire"] = timeout_tradeoffs(act, up, ok_list)
        ood_out[f"{est}|{tsrc or 'base'}"] = entry

    # 2) Multi-tier hardware SLA routing on held-out test trajectories (SLA = 15s)
    routing_out = {}
    base_trace_file = root / "data/measure/base/traces.jsonl"
    if base_trace_file.exists():
        traces_by_run = {}
        for run in ("base", "m1-s0", "m2-s0"):
            tp = root / f"data/measure/{run}/traces.jsonl"
            if tp.exists():
                by_t = defaultdict(list)
                for tr in read_jsonl(tp):
                    by_t[tr["task_id"]].append(
                        (trajectory_time(tr, edge_card), trajectory_time(tr, dc_card))
                    )
                traces_by_run[run] = by_t

        edge_groups = defaultdict(list)
        for row in all_rows:
            if (
                row["split"] == "test"
                and row["card_id"] == "seen-edge"
                and row.get("frame", "self") == "self"
            ):
                edge_groups[(row["estimator"], row.get("target_source"))].append(row)

        # Baselines: Always-Edge and Always-DC on M1-S0 trajectories
        if "m1-s0" in traces_by_run:
            test_ids = sorted(
                {
                    r["task_id"]
                    for r in all_rows
                    if r["split"] == "test" and r["estimator"] == "M1-S0"
                }
            )
            e_all, d_all = [], []
            for tid in test_ids:
                for et, dt in traces_by_run["m1-s0"].get(tid, ()):
                    e_all.append(et)
                    d_all.append(dt)
            if e_all:
                routing_out["Always-Edge"] = sla_tier_routing(
                    e_all, d_all, np.zeros(len(e_all)), sla_s=15.0
                )
                routing_out["Always-DC"] = sla_tier_routing(
                    e_all, d_all, np.full(len(e_all), 1e9), sla_s=15.0
                )

        for est in ("E1", "E2", "B2", "M2-S0", "M1-S0", "M1-Struct", "M1b", "M1-Struct+CP"):
            matches = [(k, v) for k, v in edge_groups.items() if k[0] == est]
            if not matches:
                continue
            (_, tsrc), group = matches[0]
            run_key = (
                "m1-s0"
                if tsrc == "own" and est.startswith("M1")
                else ("m2-s0" if tsrc == "own" and est.startswith("M2") else "base")
            )
            if run_key not in traces_by_run:
                continue
            table, _ = task_table(group)
            e_runs, d_runs, hi_runs = [], [], []
            for tid in sorted(table):
                for et, dt in traces_by_run[run_key].get(tid, ()):
                    e_runs.append(et)
                    d_runs.append(dt)
                    hi_runs.append(table[tid]["upper"])
            if e_runs:
                routing_out[est] = sla_tier_routing(e_runs, d_runs, hi_runs, sla_s=15.0)

    return {"ood_class_simulation": ood_out, "sla_tier_routing_15s": routing_out}


def simulate(root, estimates_path=None, card=None):
    root = Path(root)
    card = card or SpeedCard("REF", 96.0, 26000.0)
    base_rows = normalized_rows(read_jsonl(estimates_path or root / "data/eval/estimates.jsonl"))
    all_rows = base_rows + (_enrich_posthoc_rows(root, base_rows) if estimates_path is None else [])
    groups = defaultdict(list)
    for row in all_rows:
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
    if estimates_path is None:
        results["_systems_summary"] = _ood_and_routing_summary(root, all_rows)
    write_json(root / "results/simulations.json", results)
    return results
