"""AuditKIT stage: evaluate the untrained model and G1-kit on the 68 tasks, and capability.

Both are asked the benchmark's prospective question with the task description and the
stated speed of this GPU (`card_description`), greedy, through AuditKIT's evaluation
harness with three scorers of ours: the absolute natural-log error of the time read by the
benchmark's own parser against the untrained model's median timed run, whether that time
is within 2x, and whether a time was read at all. An unread answer scores the error of a
1,000x miss, so means stay finite; how many there are is reported. AuditKIT's paired
bootstrap compares the two, and every greedy answer is checked against the one
`estimate.py` wrote. Its lm-eval engine then measures ARC-Easy and ARC-Challenge zero-shot
with the adapter off and on. The token counts and wall time of the greedy answers are kept
for the cost comparison.

Writes task ids and numbers only, never answer or task text.
"""

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

from common import PRE_ESTIMATE, load_tasks, read_jsonl, system_prompt, task_view
from score import benchmark_parser, load_runs

VARIANT = "card_description"
UNREAD_LOG_ERROR = math.log(1000)
ARC = ["arc_easy", "arc_challenge"]


def estimate_file(path):
    rows = read_jsonl(path)
    card = next(row for row in rows if row["kind"] == "card")
    answers = {
        row["task"]: row["answer"]
        for row in rows
        if row["kind"] == "estimate" and row["variant"] == VARIANT
    }
    return card, answers


def fresh_cache():
    """AuditKIT replays a cached result by run fingerprint; every run here starts empty."""
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="auditkit-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--runs", default="data/metatime/time-base.jsonl")
    parser.add_argument("--base-estimates", default="data/metatime/est-untrained.jsonl")
    parser.add_argument("--estimates", required=True, help="estimate.py output for the adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lmeval-limit", type=int, help="dry run")
    parser.add_argument("--lmeval-batch", type=int, default=16)
    parser.add_argument("--skip-lmeval", action="store_true")
    args = parser.parse_args()

    import auditkit as ak
    import torch
    from auditkit import Direction, RunConfig, Sample
    from engine import Engine

    tasks = load_tasks(args.suite)
    parse = benchmark_parser(args.suite)
    runs, _ = load_runs(args.runs)
    actual = {
        task["id"]: statistics.median(r["wall_s"] for r in runs[task["id"]]) for task in tasks
    }

    def read(output):
        value = parse(output or "")
        return value if value and value > 0 else None

    @ak.scorer(direction=Direction.MINIMIZE, name="abs_log_error")
    def abs_log_error(sample, output):
        value = read(output)
        target = sample.metadata["actual_seconds"]
        return abs(math.log(value / target)) if value else UNREAD_LOG_ERROR

    @ak.scorer(direction=Direction.MAXIMIZE, name="within_2x")
    def within_2x(sample, output):
        value = read(output)
        target = sample.metadata["actual_seconds"]
        return float(bool(value) and abs(math.log(value / target)) <= math.log(2))

    @ak.scorer(direction=Direction.MAXIMIZE, name="parsed")
    def parsed(sample, output):
        return float(read(output) is not None)

    scorers = [abs_log_error, within_2x, parsed]
    engine = Engine(args.model, adapter=args.adapter)
    arms = {
        "untrained": (False, *estimate_file(args.base_estimates)),
        "g1-kit": (True, *estimate_file(args.estimates)),
    }
    runs_by_arm, report = {}, {"tool": "AuditKIT", "auditkit_version": ak.__version__}
    for arm, (adapter_on, card, written) in arms.items():
        system = system_prompt(card["S"], card["P"])
        samples = [
            Sample(
                input=PRE_ESTIMATE.format(description=task_view(task, "description")),
                id=task["id"],
                task="metatime_single_turn",
                metadata={"actual_seconds": actual[task["id"]], "category": task["category"]},
            )
            for task in tasks
        ]

        costs = []

        def ask(prompts, system=system, costs=costs):
            answers = []
            for prompt in prompts:
                out = engine.generate_timed(prompt, system=system, max_new_tokens=64, temperature=0)
                costs.append({k: out[k] for k in ("n_in", "n_out", "wall_s")})
                answers.append(out["text"])
            return answers

        ask.__name__ = f"tiny-aya-global-{arm}-S{card['S']:g}-P{card['P']:g}"
        engine.use_adapter(adapter_on)
        fresh_cache()
        result = ak.evaluate(
            samples,
            model=ask,
            scorers=scorers,
            config=RunConfig(seed=0, temperature=0.0, max_tokens=64, batch_size=1),
            experiment_name=None,
        )
        outputs = {p.sample_id: p.raw_output for p in result.predictions}
        same = sum(outputs[task_id] == answer for task_id, answer in written.items())
        runs_by_arm[arm] = result
        report[arm] = {
            "adapter": adapter_on,
            "stated_speed": {"S": card["S"], "P": card["P"]},
            "fingerprint": result.fingerprint,
            "headline": result.headline,
            "unread_answers": sum(read(text) is None for text in outputs.values()),
            "answers_equal_to_estimate_py": f"{same}/{len(written)}",
            "cost": {
                "answers": len(costs),
                "prompt_tokens": sum(c["n_in"] for c in costs),
                "answer_tokens": sum(c["n_out"] for c in costs),
                "wall_s": round(sum(c["wall_s"] for c in costs), 2),
                "gpu": torch.cuda.get_device_name(0),
            },
            "per_task": [
                {
                    "task": p.sample_id,
                    **{
                        s["name"]: s["value"]
                        for s in p.metadata.get("scores", [])
                        if s.get("name") in {"abs_log_error", "within_2x", "parsed"}
                    },
                }
                for p in result.predictions
            ],
        }
        print(arm, json.dumps(report[arm]["headline"]), report[arm]["answers_equal_to_estimate_py"])
    comparison = ak.compare(runs_by_arm["untrained"], runs_by_arm["g1-kit"])
    report["paired_bootstrap"] = {
        name: comparison.significance(name) for name in ("abs_log_error", "within_2x", "parsed")
    }
    print(json.dumps(report["paired_bootstrap"]))

    if not args.skip_lmeval:
        del engine
        torch.cuda.empty_cache()
        lm = {}
        for arm, extra in (("untrained", {}), ("g1-kit", {"peft": str(Path(args.adapter))})):
            fresh_cache()
            lm[arm] = ak.run_lmeval(
                ARC,
                model=f"hf:{args.model}",
                num_fewshot=0,
                limit=args.lmeval_limit,
                dtype="bfloat16",
                batch_size=args.lmeval_batch,
                **extra,
            )
            print(arm, json.dumps(lm[arm].headline))
        report["lmeval"] = {
            "tasks": ARC,
            "num_fewshot": 0,
            "limit": args.lmeval_limit,
            **{
                arm: {"fingerprint": r.fingerprint, "headline": r.headline} for arm, r in lm.items()
            },
            "paired_bootstrap": ak.compare(lm["untrained"], lm["g1-kit"]).significance(),
        }
        print(json.dumps(report["lmeval"]["paired_bootstrap"]))
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
