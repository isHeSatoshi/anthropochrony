"""Single source of metric definitions. Parse failures cost infinity, never zero."""

import math
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from anthropochrony.eval.errors import log_error as log_error


def task_records(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    records = []
    for task_id, group in sorted(grouped.items()):
        errors = [log_error(r.get("point_s"), r["actual_s"]) for r in group]
        parsed = [r for r, error in zip(group, errors, strict=True) if math.isfinite(error)]
        valid_errors = [e for e in errors if math.isfinite(e)]
        records.append(
            {
                "task_id": task_id,
                "family": group[0]["family"],
                "abs_error": float(np.mean(np.abs(errors))),
                "signed_error": float(np.mean(valid_errors)) if valid_errors else None,
                "parsed_abs_error": float(np.mean(np.abs(valid_errors))) if valid_errors else None,
                "predicted": float(10 ** np.mean([math.log10(r["point_s"]) for r in parsed]))
                if parsed
                else None,
                "actual": float(10 ** np.mean([math.log10(r["actual_s"]) for r in group])),
                "parse_rate": len(parsed) / len(group),
                "aur": float(np.mean([r.get("has_human_units", False) for r in group])),
            }
        )
    return records


def _correlation(function, x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(function(x, y).statistic)


def summary(records):
    if not records:
        raise ValueError("No task records")
    absolute = np.array([r["abs_error"] for r in records])
    signed = [r["signed_error"] for r in records if r["signed_error"] is not None]
    parsed_absolute = [r["parsed_abs_error"] for r in records if r["parsed_abs_error"] is not None]
    parsed = [r for r in records if r["predicted"] is not None]
    x = np.array([r["predicted"] for r in parsed])
    y = np.array([r["actual"] for r in parsed])
    return {
        "n_tasks": len(records),
        "mdae_log10": float(np.median(absolute)),
        "median_abs_ln_ratio": float(np.median(absolute)) * math.log(10),
        "mean_abs_log10": float(np.mean(parsed_absolute)) if parsed_absolute else None,
        "signed_median_bias": float(np.median(signed)) if signed else None,
        "gm_ratio": float(10 ** np.mean(signed)) if signed else None,
        "gm_deviation_factor": float(10 ** np.mean(parsed_absolute)) if parsed_absolute else None,
        "within2x": float(np.mean(absolute <= math.log10(2))),
        "within10x": float(np.mean(absolute <= 1)),
        "spearman": _correlation(spearmanr, x, y),
        "pearson_log": _correlation(pearsonr, np.log10(x), np.log10(y)) if len(x) else None,
        "aur": float(np.mean([r["aur"] for r in records])),
        "parse_rate": float(np.mean([r["parse_rate"] for r in records])),
    }


def interval_metrics(actuals, lows, highs, alpha=0.2):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    y, lo, hi = map(lambda a: np.asarray(a, dtype=float), (actuals, lows, highs))
    if np.any(lo > hi) or np.any(lo <= 0) or np.any(y <= 0):
        raise ValueError("Intervals must be ordered and durations positive")
    score = hi - lo + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)
    return {
        "coverage80": float(np.mean((lo <= y) & (y <= hi))),
        "winkler": float(np.mean(score)),
        "mean_log_width": float(np.mean(np.log10(hi / lo))),
    }


def pairwise_accuracy(records, human_predictions=None, count=2000, seed=20261109):
    rng = np.random.default_rng(seed)
    candidates = []
    for i, first in enumerate(records):
        for j in range(i + 1, len(records)):
            second = records[j]
            if (
                max(first["actual"], second["actual"]) / min(first["actual"], second["actual"])
                < 1.5
            ):
                continue
            if human_predictions is not None:
                a, b = (
                    human_predictions.get(first["task_id"]),
                    human_predictions.get(second["task_id"]),
                )
                if a is None or b is None or (a - b) * (first["actual"] - second["actual"]) >= 0:
                    continue
            candidates.append((i, j))
    if not candidates:
        return {"accuracy": None, "n_pairs": 0}
    choices = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    correct = 0
    for index in choices:
        a, b = (records[i] for i in candidates[index])
        if a["predicted"] is not None and b["predicted"] is not None:
            correct += (a["predicted"] - b["predicted"]) * (a["actual"] - b["actual"]) > 0
    return {"accuracy": float(correct / len(choices)), "n_pairs": len(choices)}


def speed_elasticity(predicted, actual):
    x, y = np.log(np.asarray(actual)), np.log(np.asarray(predicted))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def success_metrics(probabilities, outcomes, bins=10):
    from sklearn.metrics import roc_auc_score

    p, y = np.asarray(probabilities), np.asarray(outcomes)
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("Invalid probabilities")
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece,
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
    }


def arithmetic_consistency(stated_s, tokens, u_tokens, exec_s, card):
    from anthropochrony.timemodel import completion_time

    # The prompt never states a per-call latency, so a model's own stated
    # arithmetic is judged against the bare token/execution formula.
    implied = completion_time(u_tokens, tokens, exec_s, card, latency_s=0.0)
    return abs(log_error(stated_s, implied))


def noise_ceiling(traces, card):
    from anthropochrony.timemodel import trajectory_time

    by_task = defaultdict(list)
    for trace in traces:
        if not trace.get("harness_error"):
            by_task[trace["task_id"]].append(trajectory_time(trace, card))
    result = {}
    for task_id, times in by_task.items():
        if len(times) < 2:
            raise ValueError("Noise ceiling requires at least two runs")
        errors = [
            abs(log_error(float(np.median(times[:i] + times[i + 1 :])), value))
            for i, value in enumerate(times)
        ]
        result[task_id] = float(np.mean(errors))
    return result
