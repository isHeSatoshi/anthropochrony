"""The independent, validation-only speed-card check required before full measurement."""

from collections import defaultdict

import numpy as np

from anthropochrony.artifacts import digest


def invariance_metrics(tasks, reference, slow, fast):
    ids = {task["task_id"] for task in tasks}
    if len(tasks) != 100 or len(ids) != 100 or any(task["split"] != "val" for task in tasks):
        raise ValueError("Invariance requires exactly 100 distinct validation tasks")
    model_ids = {row["model_sha"] for group in (reference, slow, fast) for row in group}
    if len(model_ids) != 1:
        raise ValueError("Invariance conditions must use the same model")

    def quantities(rows, k, speeds=None):
        grouped = defaultdict(list)
        for row in rows:
            if not row["done"] or row["task_id"] not in ids:
                raise ValueError("Invariance contains unfinished or unexpected trajectories")
            if speeds is not None and (row["card"]["S"], row["card"]["P"]) != speeds:
                raise ValueError("Incorrect invariance speed card")
            grouped[row["task_id"]].append(row)
        if set(grouped) != ids or any(
            len(group) != k or {row["k"] for row in group} != set(range(k))
            for group in grouped.values()
        ):
            raise ValueError(f"Invariance requires K={k} unique trajectories for every task")
        totals = {
            task_id: float(
                np.median([sum(a["o_tokens"] for a in row["attempts"]) for row in group])
            )
            for task_id, group in grouped.items()
        }
        if any(value <= 0 for value in totals.values()):
            raise ValueError("Invariance requires positive measured token counts")
        return totals

    baseline = quantities(reference, 8)
    comparisons = {}
    for name, rows, speeds in (("slow", slow, (2, 20)), ("fast", fast, (2500, 50000))):
        observed = quantities(rows, 2, speeds)
        differences = [
            abs(float(np.log10(observed[task] / baseline[task]))) for task in sorted(ids)
        ]
        median = float(np.median(differences))
        comparisons[name] = {
            "median_abs_log10_token_ratio": median,
            "passed": median <= 0.05,
            "per_task": dict(zip(sorted(ids), differences, strict=True)),
        }
    return {
        "passed": all(row["passed"] for row in comparisons.values()),
        "samples_tasks": 100,
        "reference_k": 8,
        "condition_k": 2,
        "model_sha": next(iter(model_ids)),
        "tasks_sha256": digest(tasks),
        "trace_sha256": {
            name: digest(rows)
            for name, rows in (("REF", reference), ("slow", slow), ("fast", fast))
        },
        "comparisons": comparisons,
    }


def invariance_passed(record):
    comparisons = record.get("comparisons", {})
    return (
        record.get("passed") is True
        and record.get("samples_tasks") == 100
        and record.get("reference_k") == 8
        and record.get("condition_k") == 2
        and set(comparisons) == {"slow", "fast"}
        and all(row["median_abs_log10_token_ratio"] <= 0.05 for row in comparisons.values())
    )
