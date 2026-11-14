"""Ask a model for prospective estimates of its own run time on the MetaTime tasks.

Every estimate is a separate call that never sees the response, as in the benchmark. The
stated speed is measured on this deployment just before the questions are asked.
"""

import argparse

from common import (
    PAPER_ESTIMATE,
    PRE_ESTIMATE,
    append_jsonl,
    load_pool,
    load_tasks,
    read_jsonl,
    sig,
    stable_seed,
    system_prompt,
    task_view,
)
from engine import Engine

# name -> (uses the stated speed, question, what the estimator sees of the task)
VARIANTS = {
    "protocol": (False, PRE_ESTIMATE, "description"),
    "card_description": (True, PRE_ESTIMATE, "description"),
    "card_full_prompt": (True, PRE_ESTIMATE, "full"),
    "card_paper_wording": (True, PAPER_ESTIMATE, "description"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument(
        "--executor",
        choices=["base", "adapter"],
        default="base",
        help="whose speed is stated: the plain model or the model with its adapter on",
    )
    parser.add_argument("--suite")
    parser.add_argument("--pool")
    parser.add_argument("--labels")
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--samples", type=int, default=0, help="also draw this many answers at temperature 1"
    )
    args = parser.parse_args()

    engine = Engine(args.model, adapter=args.adapter, load_in_4bit=args.load_in_4bit)
    if args.no_thinking:
        engine.template_kwargs = {"enable_thinking": False}
    card = next((row for row in read_jsonl(args.out) if row["kind"] == "card"), None)
    if card is None:  # a resumed file keeps the speed it already stated
        engine.use_adapter(args.executor == "adapter")
        measured = engine.speed_card()
        engine.use_adapter(True)
        # Stated to two significant digits, as every stated speed in training was.
        card = {key: sig(value, 2) for key, value in measured.items()}
        append_jsonl(
            args.out, {"kind": "card", "executor": args.executor, "measured": measured, **card}
        )
    if args.suite:
        tasks = load_tasks(args.suite)
    else:  # held-out pool prompts, for a check that never touches the benchmark
        tasks = [row for row in load_pool(args.pool, args.labels) if row["split"] == "val"]
    done = {
        (row["task"], row["variant"]) for row in read_jsonl(args.out) if row["kind"] == "estimate"
    }
    sampled = {
        (row["task"], row["variant"]) for row in read_jsonl(args.out) if row["kind"] == "samples"
    }
    for name in args.variants:
        stated, question, view = VARIANTS[name]
        system = system_prompt(card["S"], card["P"]) if stated else None
        for task in tasks:
            if (task["id"], name) in done:
                continue
            result = engine.generate_timed(
                question.format(description=task_view(task, view)),
                system=system,
                max_new_tokens=args.max_new_tokens,
                temperature=0,
            )
            append_jsonl(
                args.out,
                {
                    "kind": "estimate",
                    "task": task["id"],
                    "variant": name,
                    "answer": result["text"],
                    "S": card["S"],
                    "P": card["P"],
                    "executor": args.executor,
                },
            )
        for task in tasks if args.samples else []:
            if (task["id"], name) in sampled:
                continue
            encoded = engine.encode(question.format(description=task_view(task, view)), system)
            answers = engine.generate_batch(
                [encoded] * args.samples,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                seed=stable_seed("estimate", task["id"], name),
            )
            append_jsonl(
                args.out,
                {
                    "kind": "samples",
                    "task": task["id"],
                    "variant": name,
                    "answers": [answer["text"] for answer in answers],
                },
            )
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
