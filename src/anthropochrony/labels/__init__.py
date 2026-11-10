"""Build measured labels and canonical targets without seeing evaluation outcomes."""

from collections import defaultdict

import numpy as np

from anthropochrony.prompts import fmt_speed, load
from anthropochrony.timemodel import trajectory_time


def fmt_duration(seconds):
    if seconds < 0 or not np.isfinite(seconds):
        raise ValueError("Duration must be finite and non-negative")
    if seconds < 10:
        return f"{seconds:.2g} s"
    if seconds < 100:
        return f"{seconds:.0f} s"
    if seconds < 6000:
        return f"{seconds / 60:.2g} min"
    return f"{seconds / 3600:.2g} h"


def canonical_answer(label):
    total = label["T_med_s"]
    tokens = label["TOK_med"]
    generation = tokens / label["S"]
    overhead = total - generation
    low, high = fmt_duration(label["T_q10_s"]), fmt_duration(label["T_q90_s"])
    if low.split()[-1] == high.split()[-1]:
        low = low.rsplit(" ", 1)[0]
    attempts = int(np.clip(np.floor(label["ATT_med"] + 0.5), 1, 3))
    attempt_text = {
        1: "1 attempt expected",
        2: "about 2 attempts expected",
        3: "all 3 attempts expected",
    }
    pfp = int(np.floor(label["PFP"] * 10 + 0.5)) * 10
    chance = "under 10%" if pfp <= 10 else "over 90%" if pfp >= 90 else f"about {pfp}%"
    values = {
        "T": fmt_duration(total),
        "LO": low,
        "HI": high,
        "TOK": fmt_speed(tokens),
        "S": fmt_speed(label["S"]),
        "GEN": fmt_duration(generation),
        "ATT": attempt_text[attempts],
        "PFP": chance,
        "OVH": f", plus ~{fmt_duration(overhead)} reading the task and running tests"
        if overhead >= 0.05 * total
        else "",
    }
    return load()["target"].format(**values)


def training_priors(tasks, traces):
    index = {task["task_id"]: task for task in tasks}
    rates = defaultdict(list)
    for trace in traces:
        task = index[trace["task_id"]]
        if task["split"] == "train" and not trace.get("harness_error"):
            rates[task["family"]].append(float(trace["first_pass"]))
    if not rates:
        raise ValueError("PFP shrinkage requires measured train trajectories")
    return {family: float(np.mean(values)) for family, values in rates.items()} | {
        "__pooled__": float(np.mean([value for values in rates.values() for value in values]))
    }


def build_labels(
    tasks, traces, cards, *, k=8, max_harness_errors=2, label_source="base", priors=None
):
    task_map = {t["task_id"]: t for t in tasks}
    grouped = defaultdict(list)
    for trace in traces:
        grouped[trace["task_id"]].append(trace)
    family_rates = priors if priors is not None else training_priors(tasks, traces)
    labels, dropped = [], []
    for task_id, runs in grouped.items():
        if len({r["run_id"] for r in runs}) != len(runs):
            raise ValueError(f"Duplicate runs for {task_id}")
        if len(runs) != k:
            raise ValueError(f"{task_id} has {len(runs)} runs; expected K={k}")
        errors = sum(bool(r.get("harness_error")) for r in runs)
        if errors > max_harness_errors:
            dropped.append({"task_id": task_id, "reason": "harness_errors", "count": errors})
            continue
        valid = [r for r in runs if not r.get("harness_error")]
        task = task_map[task_id]
        if task["family"] not in family_rates:
            # An unseen family has no legal train-only prior. Use the pooled training prior.
            prior = family_rates["__pooled__"]
        else:
            prior = family_rates[task["family"]]
        for card in cards.get(task_id, []) if isinstance(cards, dict) else cards:
            times = [trajectory_time(run, card) for run in valid]
            median = float(np.median(times))
            if median <= 0:
                raise ValueError("A measured label must have positive completion time")
            q10, q90 = np.quantile(times, [0.1, 0.9])
            tok = float(np.median([sum(a["o_tokens"] for a in r["attempts"]) for r in valid]))
            labels.append(
                {
                    "task_id": task_id,
                    "family": task["family"],
                    "split": task["split"],
                    "card_id": card.id,
                    "S": card.S,
                    "P": card.P,
                    "T_med_s": median,
                    "T_q10_s": min(float(q10), median / 1.25),
                    "T_q90_s": max(float(q90), median * 1.25),
                    "TOK_med": tok,
                    "ATT_med": float(np.median([r["n_attempts"] for r in valid])),
                    "PFP": (sum(r["first_pass"] for r in valid) + 2 * prior) / (len(valid) + 2),
                    "overhead_med_s": median - tok / card.S,
                    "n_runs": len(valid),
                    "label_source": label_source,
                    "label_version": "v2-latency",
                }
            )
    return labels, dropped


def build_labels_for_training_cards(tasks, traces, task_cards, *, k=8, label_source="base"):
    rows, _ = build_labels(tasks, traces, task_cards, k=k, label_source=label_source)
    return rows
