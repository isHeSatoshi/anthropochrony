"""Time the model's own single-turn runs on the MetaTime suite, one request at a time.

Follows the benchmark's protocol for local models: the task prompt alone, a 2,048-token
cap, several runs per task, wall time from request to last token. Sampling follows the
Tiny Aya model card (temperature 0.1, top-p 0.95).
"""

import argparse
import time

from common import append_jsonl, load_tasks, read_jsonl, stable_seed
from engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.suite)
    engine = Engine(args.model, adapter=args.adapter, load_in_4bit=args.load_in_4bit)
    if args.no_thinking:
        engine.template_kwargs = {"enable_thinking": False}
    done = {(row["task"], row["run"]) for row in read_jsonl(args.out) if row.get("kind") == "run"}
    engine.generate_timed(tasks[0]["prompt"], max_new_tokens=64)  # warm-up, not recorded
    append_jsonl(
        args.out, {"kind": "card", "when": "start", "time": time.time(), **engine.speed_card()}
    )
    for run in range(args.runs):
        for task in tasks:
            if (task["id"], run) in done:
                continue
            result = engine.generate_timed(task["prompt"], seed=stable_seed(task["id"], run))
            append_jsonl(args.out, {"kind": "run", "task": task["id"], "run": run, **result})
        print(f"run {run} finished", flush=True)
    append_jsonl(
        args.out, {"kind": "card", "when": "end", "time": time.time(), **engine.speed_card()}
    )


if __name__ == "__main__":
    main()
