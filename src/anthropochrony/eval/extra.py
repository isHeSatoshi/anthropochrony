"""Secondary tables: common-target robustness, speed cards, families, frames, behavior shift."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.eval.metrics import summary, task_records
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.eval.stats import json_safe, stratified_bootstrap

MODELS = ("E1", "E2", "B2", "M1-S0", "M2-S0", "M3-S0")


def _mdae(records):
    return float(np.median([r["abs_error"] for r in records]))


def _line(rows, resamples):
    records = task_records(rows)
    stats = summary(records)
    return {
        "n_tasks": stats["n_tasks"],
        "mdae_log10": stats["mdae_log10"],
        "ci95": stratified_bootstrap(records, _mdae, resamples=resamples),
        "within2x": stats["within2x"],
        "gm_ratio": stats["gm_ratio"],
        "aur": stats["aur"],
        "parse_rate": stats["parse_rate"],
    }


def _group(rows, **conditions):
    return [
        r
        for r in rows
        if all(
            (r.get(key) in value if isinstance(value, set) else r.get(key) == value)
            for key, value in conditions.items()
        )
    ]


def behavior_shift(root):
    """Per-task change in output tokens and attempts versus the base trajectories (K=8 means)."""
    root = Path(root)
    tasks = {t["task_id"]: t for t in read_jsonl(root / "data/tasks/tasks.jsonl")}

    def per_task(run):
        rows = defaultdict(list)
        for trace in read_jsonl(root / f"data/measure/{run}/traces.jsonl"):
            rows[trace["task_id"]].append(trace)
        return {
            task_id: {
                "tokens": float(np.mean([t["total_o"] for t in group])),
                "attempts": float(np.mean([t["n_attempts"] for t in group])),
                "first_pass": float(np.mean([t["first_pass"] for t in group])),
            }
            for task_id, group in rows.items()
        }

    base = per_task("base")
    result = {}
    for run in ("m1-s0", "m2-s0", "m3-s0"):
        tuned = per_task(run)
        ids = sorted(set(tuned) & set(base))
        splits = defaultdict(list)
        for task_id in ids:
            splits[tasks[task_id]["split"]].append(task_id)
        result[run] = {}
        for split, split_ids in splits.items():
            ratio = [np.log10(tuned[i]["tokens"] / base[i]["tokens"]) for i in split_ids]
            result[run][split] = {
                "n_tasks": len(split_ids),
                "median_abs_log10_token_ratio": float(np.median(np.abs(ratio))),
                "mean_signed_log10_token_ratio": float(np.mean(ratio)),
                "mean_attempts_base": float(np.mean([base[i]["attempts"] for i in split_ids])),
                "mean_attempts_tuned": float(np.mean([tuned[i]["attempts"] for i in split_ids])),
                "pass1_base": float(np.mean([base[i]["first_pass"] for i in split_ids])),
                "pass1_tuned": float(np.mean([tuned[i]["first_pass"] for i in split_ids])),
            }
    return result


def extra_report(root, resamples=2000):
    root = Path(root)
    rows = normalized_rows(read_jsonl(root / "data/eval/estimates.jsonl"))
    common = normalized_rows(read_jsonl(root / "data/eval/common-base-estimates.jsonl"))
    own = rows
    out = {"resamples": resamples}

    out["common_base_targets"] = {
        f"{estimator}|{split}": _line(
            _group(common, estimator=estimator, split=split, card_id="REF", frame="self"), resamples
        )
        for estimator in ("M1-S0", "M2-S0", "M3-S0")
        for split in ("test", "test_ood")
    }
    for split in ("test", "test_ood"):
        for estimator in ("E2", "B2"):
            out["common_base_targets"][f"{estimator}|{split}"] = _line(
                _group(rows, estimator=estimator, split=split, card_id="REF", frame="self"),
                resamples,
            )

    cards = sorted(
        {r["card_id"] for r in rows if r["split"] == "test" and r["estimator"] == "M1-S0"}
    )
    out["by_speed_card_test"] = {
        card: {
            estimator: summary(
                task_records(
                    _group(own, estimator=estimator, split="test", card_id=card, frame="self")
                )
            )["mdae_log10"]
            for estimator in MODELS
            if _group(own, estimator=estimator, split="test", card_id=card, frame="self")
        }
        for card in cards
    }

    families = sorted({r["family"] for r in rows if r["split"] == "test"})
    out["by_family_test"] = {
        family: {
            estimator: summary(
                task_records(
                    _group(
                        own,
                        estimator=estimator,
                        split="test",
                        card_id="REF",
                        frame="self",
                        family=family,
                    )
                )
            )["mdae_log10"]
            for estimator in MODELS
            if _group(
                own, estimator=estimator, split="test", card_id="REF", frame="self", family=family
            )
        }
        for family in families
    }

    out["frames_test_ref"] = {}
    for frame in ("self", "human", "plan", "generic"):
        for estimator in sorted({r["estimator"] for r in rows}):
            selected = _group(rows, estimator=estimator, split="test", card_id="REF", frame=frame)
            if estimator in {"E0", "E1", "E3", "M1-S0", "M2-S0", "M3-S0"} and selected:
                stats = summary(task_records(selected))
                out["frames_test_ref"][f"{estimator}|{frame}"] = {
                    "n_tasks": stats["n_tasks"],
                    "mdae_log10_vs_own_time": stats["mdae_log10"],
                    "median_estimate_s": float(
                        np.median(
                            [r["predicted"] for r in task_records(selected) if r["predicted"]]
                        )
                    ),
                    "aur": stats["aur"],
                    "parse_rate": stats["parse_rate"],
                    "gm_ratio": stats["gm_ratio"],
                }
    out["behavior_shift"] = behavior_shift(root)
    write_json(root / "results/extra-analyses.json", json_safe(out))
    return out
