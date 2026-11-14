"""Score the MetaTime head-to-head and write the committed records and `results/metatime.json`.

Runs on CPU in the pinned environment. Reads the local run, estimate and pool files, the
public artifact's per-task numbers for the two frontier models, and writes per-task records
without any benchmark task text.
"""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Optional

import numpy as np
from common import fetch_suite, load_tasks, parse_answer, read_jsonl, task_view
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV

from anthropochrony.eval import metatime

PUBLISHED_TABLE_22 = {
    "source": "MetaTime Appendix C, Table 22 (Qwen3-8B, 68 held-out single-turn tasks, before generating)",
    "rows": {
        "No training": {"mae_s": 125.77, "pearson": 0.31},
        "SFT, speed stated": {"mae_s": 30.45, "pearson": 0.39},
        "SFT, plain prompt": {"mae_s": 34.32, "pearson": 0.30},
        "Regression, linear head": {"mae_s": 24.70, "pearson": 0.21},
        "RAFT, asymmetric": {"mae_s": 24.04, "pearson": 0.44},
        "RAFT, symmetric": {"mae_s": 26.30, "pearson": 0.41},
        "RL, structured": {"mae_s": 61.84, "pearson": 0.09},
        "RL, with reasoning": {"mae_s": 43.77, "pearson": 0.28},
    },
    "serving_speed_tok_s": 16.63,
}


def benchmark_parser(suite):
    """The artifact's own duration parser, loaded from the downloaded copy."""
    source = (Path(suite) / "experiments" / "models.py").read_text()
    start = source.index("def parse_duration_from_text")
    namespace = {"re": re, "Optional": Optional}
    exec(source[start : source.index("# ====", start)], namespace)  # noqa: S102
    return namespace["parse_duration_from_text"]


def load_runs(path):
    rows = read_jsonl(path)
    cards = [row for row in rows if row["kind"] == "card"]
    runs = {}
    for row in rows:
        if row["kind"] == "run":
            runs.setdefault(row["task"], []).append(row)
    return runs, cards


def system_row(estimate, actual, reference):
    row = metatime.metrics(estimate, actual)
    row["ci95"] = metatime.intervals(estimate, actual)
    row["versus"] = {
        name: metatime.paired_log_error(estimate, actual, ref["estimate"], ref["actual"])
        for name, ref in reference.items()
    }
    return row


def static_baselines(pool, tasks, card):
    """Pool-only predictors of response length that never run the model on a benchmark task."""
    train = [row for row in pool if row["split"] == "train"]
    target = np.log([max(1.0, row["n_out"]) for row in train])
    prompt_tokens = statistics.median(row["n_in"] for row in train)
    out = {
        "pool_median_constant": [
            prompt_tokens / card["P"] + float(np.exp(np.median(target))) / card["S"]
        ]
        * len(tasks)
    }
    for view in ("description", "full"):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        features = vectorizer.fit_transform([task_view(row, view) for row in train])
        model = RidgeCV(alphas=np.logspace(-2, 2, 9)).fit(features, target)
        predicted = model.predict(vectorizer.transform([task_view(task, view) for task in tasks]))
        out[f"text_regressor_{view}"] = [
            prompt_tokens / card["P"] + float(np.exp(value)) / card["S"] for value in predicted
        ]
    return out


def pool_validation(data, pool):
    """Checks on held-out pool prompts, made before the fine-tuned model saw a benchmark task."""
    val = [row for row in pool if row["split"] == "val"]
    by_id = {row["id"]: row for row in val}
    out = {
        "prompts": len(val),
        "retest_ceiling_abs_log_error": metatime.retest_ceiling([row["tokens"] for row in val]),
        "estimators": {},
        "length_drift": {},
    }
    for path in sorted(data.glob("val-est-*.jsonl")):
        rows = read_jsonl(path)
        card = next(row for row in rows if row["kind"] == "card")
        actual = [row["n_in"] / card["P"] + row["n_out"] / card["S"] for row in val]
        scored = {}
        for variant in sorted({row["variant"] for row in rows if row["kind"] == "estimate"}):
            answers = {
                row["task"]: parse_answer(row["answer"])
                for row in rows
                if row["kind"] == "estimate" and row["variant"] == variant
            }
            if all(row["id"] in answers for row in val):
                scored[variant] = metatime.metrics(
                    [answers[row["id"]]["seconds"] for row in val], actual
                )
        for name, estimate in static_baselines(pool, val, card).items():
            scored[f"static/{name}"] = metatime.metrics(estimate, actual)
        out["estimators"][path.stem.removeprefix("val-est-")] = scored
    # Does the adapter change how long the model's ordinary responses are?
    for path in sorted(data.glob("val-lengths-*.jsonl")):
        ratios = [
            math.log(max(1, row["n_out"]) / max(1.0, by_id[row["id"]]["n_out"]))
            for row in read_jsonl(path)
            if row["kind"] == "label" and row["id"] in by_id
        ]
        out["length_drift"][path.stem.removeprefix("val-lengths-")] = {
            "prompts": len(ratios),
            "median_log_ratio_to_base": statistics.median(ratios),
            "median_abs_log_ratio_to_base": statistics.median(abs(value) for value in ratios),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--data", default="data/metatime")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    data, results = Path(args.data), Path(args.results)
    records = results / "metatime"
    records.mkdir(parents=True, exist_ok=True)
    suite = fetch_suite(args.suite)
    tasks = load_tasks(suite)
    order = [task["id"] for task in tasks]
    parse = benchmark_parser(suite)

    published = list(csv.DictReader((suite / "results" / "single_turn.csv").open()))
    reference = {}
    for model in sorted({row["model"] for row in published}):
        by_task = {row["task"]: row for row in published if row["model"] == model}
        reference[model] = {
            "actual": [float(by_task[task]["actual_seconds"]) for task in order],
            "estimate": [float(by_task[task]["estimate_before_seconds"]) for task in order],
        }
    with (records / "reference-published.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "category", "model", "actual_seconds", "estimate_before_seconds"])
        for row in published:
            writer.writerow(
                [
                    row["task"],
                    row["category"],
                    row["model"],
                    row["actual_seconds"],
                    row["estimate_before_seconds"],
                ]
            )

    def published_metrics(name, column):
        rows = list(csv.DictReader((suite / "results" / name).open()))
        out = {}
        for model in sorted({row["model"] for row in rows}):
            scored = [row for row in rows if row["model"] == model and row[column]]
            out[model] = metatime.metrics(
                [float(row[column]) for row in scored],
                [float(row["actual_seconds"]) for row in scored],
            )
        return out

    # Other published settings, for context only; none is the declared comparison.
    published_context = {
        "single_turn_after_completing": published_metrics(
            "single_turn.csv", "estimate_after_seconds"
        ),
        "short_agentic_before": published_metrics("short_agentic.csv", "estimate_before_seconds"),
        "long_horizon_before": published_metrics("long_horizon.csv", "estimate_before_seconds"),
    }

    executors, run_rows = {}, []
    for path in sorted(data.glob("time-*.jsonl")):
        name = path.stem.removeprefix("time-")
        runs, cards = load_runs(path)
        if not all(task in runs for task in order):
            continue
        executors[name] = {
            "actual": [statistics.median(run["wall_s"] for run in runs[task]) for task in order],
            "runs": [[run["wall_s"] for run in runs[task]] for task in order],
            "tokens": [statistics.median(run["n_out"] for run in runs[task]) for task in order],
            "prompt_tokens": [runs[task][0]["n_in"] for task in order],
            "cards": cards,
            "capped_runs": sum(run["finish"] == "length" for task in order for run in runs[task]),
            "n_runs": sorted({len(runs[task]) for task in order}),
        }
        for task in order:
            for run in runs[task]:
                run_rows.append(
                    {
                        "executor": name,
                        **{
                            k: run[k]
                            for k in (
                                "task",
                                "run",
                                "n_in",
                                "n_out",
                                "wall_s",
                                "prefill_s",
                                "decode_s",
                                "finish",
                            )
                        },
                    }
                )
    (records / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in run_rows))

    out = {
        "benchmark": {
            "name": "MetaTime single-turn suite (68 tasks)",
            "artifact": "https://anonymous.4open.science/r/Time-Estimation-B286",
            "question": "the artifact's prospective question, with the task description",
        },
        "published": {
            model: {
                **metatime.metrics(ref["estimate"], ref["actual"]),
                "ci95": metatime.intervals(ref["estimate"], ref["actual"]),
            }
            for model, ref in reference.items()
        },
        "published_context": published_context,
        "published_trained": PUBLISHED_TABLE_22,
        "executors": {},
        "systems": {},
    }
    for name, executor in executors.items():
        card = executor["cards"][0]
        formula = [
            n_in / card["P"] + tokens / card["S"]
            for n_in, tokens in zip(executor["prompt_tokens"], executor["tokens"], strict=True)
        ]
        out["executors"][name] = {
            "runs_per_task": executor["n_runs"],
            "speed_cards": [
                {k: card_row[k] for k in ("when", "S", "P")} for card_row in executor["cards"]
            ],
            "median_seconds": statistics.median(executor["actual"]),
            "min_seconds": min(executor["actual"]),
            "max_seconds": max(executor["actual"]),
            "median_response_tokens": statistics.median(executor["tokens"]),
            "capped_runs": executor["capped_runs"],
            "retest_ceiling_abs_log_error": metatime.retest_ceiling(executor["runs"]),
            "measured_length_through_formula": metatime.metrics(formula, executor["actual"]),
        }

    estimate_rows = []
    for path in sorted(data.glob("est-*.jsonl")):
        name = path.stem.removeprefix("est-")
        rows = read_jsonl(path)
        card = next(row for row in rows if row["kind"] == "card")
        target = "base" if card["executor"] == "base" else name.split("@")[0]
        if target not in executors:
            continue
        actual = executors[target]["actual"]
        for variant in sorted({row["variant"] for row in rows if row["kind"] == "estimate"}):
            answers = {
                row["task"]: row["answer"]
                for row in rows
                if row["kind"] == "estimate" and row["variant"] == variant
            }
            if not all(task in answers for task in order):
                continue
            stated, tokens = [], []
            for task in order:
                answer = answers[task]
                parts = parse_answer(answer)
                stated.append(parse(answer))
                tokens.append(
                    parts["prompt_tokens"] / card["P"] + parts["response_tokens"] / card["S"]
                    if parts["prompt_tokens"] and parts["response_tokens"]
                    else None
                )
                estimate_rows.append(
                    {
                        "estimator": name,
                        "variant": variant,
                        "task": task,
                        "answer": answer,
                        "stated_seconds": stated[-1],
                        "token_reading_seconds": tokens[-1],
                        "response_tokens": parts["response_tokens"],
                        "S": card["S"],
                        "P": card["P"],
                        "scored_against": target,
                    }
                )
            key = f"{name}/{variant}"
            out["systems"][key] = {
                "scored_against": target,
                "stated_speed": {"S": card["S"], "P": card["P"]},
                "stated": system_row(stated, actual, reference),
            }
            if any(value is not None for value in tokens):
                out["systems"][key]["token_reading"] = system_row(tokens, actual, reference)
    (records / "estimates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in estimate_rows)
    )

    probe = data / "probe.json"
    if probe.exists() and "base" in executors:
        base = executors["base"]
        card = base["cards"][0]
        for context, row in json.loads(probe.read_text())["contexts"].items():
            estimate = [
                n_in / card["P"] + row["predicted_tokens"][task] / card["S"]
                for task, n_in in zip(order, base["prompt_tokens"], strict=True)
            ]
            out["systems"][f"probe/{context}"] = {
                "scored_against": "base",
                "layer": row["layer"],
                "pool_validation": {
                    "median_abs_log_error": row["val_median_abs_log_error"],
                    "pearson_log": row["val_pearson_log"],
                },
                "stated": system_row(estimate, base["actual"], reference),
            }

    labels = data / "pool-labels.jsonl"
    if labels.exists() and "base" in executors:
        from train import load_pool

        pool = load_pool(data / "pool.jsonl", labels)
        card = executors["base"]["cards"][0]
        for name, estimate in static_baselines(pool, tasks, card).items():
            out["systems"][f"static/{name}"] = {
                "scored_against": "base",
                "stated": system_row(estimate, executors["base"]["actual"], reference),
            }
        (records / "pool-labels.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": row["id"],
                        "source": row["source"],
                        "split": row["split"],
                        "n_in": row["n_in"],
                        "response_tokens": row["tokens"],
                    }
                )
                + "\n"
                for row in pool
            )
        )
        out["pool"] = {
            "prompts": len(pool),
            "train": sum(row["split"] == "train" for row in pool),
            "runs_per_prompt": sorted({len(row["tokens"]) for row in pool}),
            "median_response_tokens": statistics.median(row["n_out"] for row in pool),
            "manifest": json.loads((data / "pool.manifest.json").read_text())["stats"],
        }
        out["pool_validation"] = pool_validation(data, pool)
    (results / "metatime.json").write_text(json.dumps(out, indent=2) + "\n")
    for key, system in out["systems"].items():
        for reading in ("stated", "token_reading"):
            if reading in system:
                row = system[reading]
                print(
                    f"{key:46s} {reading:13s} ratio {row['median_ratio']:.2f} "
                    f"logerr {row['median_abs_log_error']:.3f} r {row['pearson_log']} "
                    f"within2x {row['within_2x']:.2f} parsed {row['parsed']}"
                )


if __name__ == "__main__":
    main()
