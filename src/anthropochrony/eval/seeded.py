"""Keep cards/templates within tasks and training seeds separate."""

import math
from collections import defaultdict

import numpy as np

from anthropochrony.eval.metrics import log_error, speed_elasticity, task_records
from anthropochrony.eval.stats import stratified_bootstrap


def seed_report(records, key, seeds, resamples):
    def median_for(rows, index):
        values = [row[key][index] for row in rows if row[key][index] is not None]
        return float(np.median(values)) if values else None

    medians = [median_for(records, i) for i in range(len(seeds))]
    if any(value is None for value in medians):
        return None
    return {
        "mean": float(np.mean(medians)),
        "sd": (float(np.std(medians, ddof=1)) if len(seeds) > 1 else 0)
        if all(math.isfinite(value) for value in medians)
        else None,
        "ci95": stratified_bootstrap(
            records,
            lambda rows: (
                float(np.mean([median_for(rows, i) for i in range(len(seeds))]))
                if all(median_for(rows, i) is not None for i in range(len(seeds)))
                else None
            ),
            resamples=resamples,
        ),
        "per_seed": {
            str(seed): {
                "median": medians[index],
                "ci95": stratified_bootstrap(
                    records, lambda rows, i=index: median_for(rows, i), resamples=resamples
                ),
            }
            for index, seed in enumerate(seeds)
        },
    }


def speed_conditioning(rows, seeds, resamples, expected_cards=None):
    if not rows or not set(seeds) <= {row.get("seed") for row in rows}:
        return None
    cards = set(expected_cards or {row["card_id"] for row in rows})
    heldout = {card for card in cards if card.startswith("heldout-")}
    if not heldout:
        return None
    grouped = defaultdict(list)
    for row in rows:
        if row.get("seed") in seeds:
            grouped[(row["task_id"], row["seed"], row["card_id"])].append(row)
    task_ids = sorted({key[0] for key in grouped})
    if set(grouped) != {
        (task, seed, card) for task in task_ids for seed in seeds for card in cards
    }:
        raise ValueError("H4 needs the same complete evaluation-card set for every task and seed")
    records = []
    for task in task_ids:
        errors, elasticities = [], []
        for seed in seeds:
            card_records = {
                card: task_records(grouped[(task, seed, card)])[0] for card in sorted(cards)
            }
            errors.append(float(np.mean([card_records[card]["abs_error"] for card in heldout])))
            valid = [record for record in card_records.values() if record["predicted"] is not None]
            elasticity = (
                speed_elasticity(
                    [record["predicted"] for record in valid],
                    [record["actual"] for record in valid],
                )
                if len(valid) > 1
                else None
            )
            elasticities.append(elasticity)
        records.append(
            {
                "task_id": task,
                "family": grouped[(task, seeds[0], sorted(cards)[0])][0]["family"],
                "errors": errors,
                "elasticities": elasticities,
            }
        )
    error = seed_report(records, "errors", seeds, resamples)
    elasticity = seed_report(records, "elasticities", seeds, resamples)
    return {
        "status": "measured",
        "n_tasks": len(records),
        "cards": sorted(cards),
        "seed_aggregation": "mean_of_per_seed_task_medians",
        "mdae_log10": error["mean"],
        "error_by_seed": error,
        "elasticity": elasticity["mean"] if elasticity else None,
        "elasticity_by_seed": elasticity,
        "passed": bool(elasticity) and error["mean"] <= 0.35 and 0.8 <= elasticity["mean"] <= 1.2,
    }


def human_preservation(base_rows, tuned_rows, seeds, resamples):
    if not base_rows or not set(seeds) <= {row.get("seed") for row in tuned_rows}:
        return None

    def key(row):
        return (row["task_id"], row.get("template_id"), row.get("phrasing_id"), row["card_id"])

    def unique(rows):
        result = {}
        for row in rows:
            item = key(row)
            if item in result and any(
                row.get(field) != result[item].get(field)
                for field in ("point_s", "raw_text", "messages", "model_sha")
            ):
                raise ValueError("Conflicting human answers for the same matched prompt")
            result[item] = row
        return result

    base = unique(base_rows)
    tuned = {seed: unique([r for r in tuned_rows if r.get("seed") == seed]) for seed in seeds}
    if any(set(rows) != set(base) for rows in tuned.values()):
        raise ValueError("H6b needs the same human prompts and tasks for every seed and base")
    if any(
        "messages" in base[item] and row.get("messages") != base[item]["messages"]
        for rows in tuned.values()
        for item, row in rows.items()
    ):
        raise ValueError("H6b requires identical human-frame prompt messages")
    records = []
    for task in sorted({item[0] for item in base}):
        keys = [item for item in base if item[0] == task]
        errors = []
        for seed in seeds:
            ratios = []
            for item in keys:
                reference = base[item].get("point_s")
                ratios.append(
                    abs(log_error(tuned[seed][item].get("point_s"), reference))
                    if reference is not None and math.isfinite(reference) and reference > 0
                    else math.inf
                )
            errors.append(float(np.mean(ratios)))
        records.append({"task_id": task, "family": base[keys[0]]["family"], "errors": errors})
    metric = seed_report(records, "errors", seeds, resamples)
    return {
        "status": "measured",
        "n_tasks": len(records),
        "seed_aggregation": "mean_of_per_seed_task_medians",
        "mean_seed_median_abs_log10_ratio": metric["mean"],
        "seed_metrics": metric,
        "passed": metric["mean"] <= 0.15,
    }
