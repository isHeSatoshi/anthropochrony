"""Measure the model's own response lengths on the training pool (never on benchmark tasks).

Each pool prompt is answered several times under the same sampling settings as the timed
benchmark runs; the label is the median output-token count. A first step asks the model for
a one-line description of each prompt, because the benchmark shows the estimator a short
description rather than the full prompt.
"""

import argparse
import time

import torch
from common import append_jsonl, read_jsonl, stable_seed
from engine import MAX_NEW_TOKENS, Engine

DESCRIBE = (
    "Here is a task someone wants done:\n\n<task>\n{task}\n</task>\n\n"
    "Name this task in one short phrase of at most 12 words, the way a to-do list would. "
    "Do not do the task. Reply with the phrase only."
)
TOKEN_ROW_BUDGET = 110_000


def batches(rows, lengths, max_rows, new_tokens):
    """Group rows in the given order, bounding rows x (prompt + new tokens) for GPU memory."""
    group, widest = [], 0
    for row in rows:
        width = max(widest, lengths[row["id"]])
        if group and (
            len(group) >= max_rows or (len(group) + 1) * (width + new_tokens) > TOKEN_ROW_BUDGET
        ):
            yield group
            group, width = [], lengths[row["id"]]
        group.append(row)
        widest = width
    if group:
        yield group


def run_batch(engine, encoded, **kwargs):
    try:
        return engine.generate_batch(encoded, **kwargs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(encoded) == 1:
            raise
        half = len(encoded) // 2
        return run_batch(engine, encoded[:half], **kwargs) + run_batch(
            engine, encoded[half:], **kwargs
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=48)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter", help="measure the fine-tuned model instead (drift check)")
    parser.add_argument("--split", help="restrict to one pool split")
    parser.add_argument("--skip-descriptions", action="store_true")
    args = parser.parse_args()

    pool = [row for row in read_jsonl(args.pool) if args.split in (None, row["split"])]
    pool = pool[: args.limit]
    engine = Engine(args.model, adapter=args.adapter, load_in_4bit=args.load_in_4bit)
    if args.no_thinking:
        engine.template_kwargs = {"enable_thinking": False}
    sampling = {} if args.temperature is None else {"temperature": args.temperature}
    done = read_jsonl(args.out)

    described = {row["id"] for row in done if row["kind"] == "description"}
    todo = [] if args.skip_descriptions else [row for row in pool if row["id"] not in described]
    encoded = {row["id"]: engine.encode(DESCRIBE.format(task=row["prompt"][:2000])) for row in todo}
    lengths = {key: len(engine._ids(value)) for key, value in encoded.items()}
    todo.sort(key=lambda row: lengths[row["id"]])
    for group in batches(todo, lengths, 64, 40):
        results = run_batch(
            engine, [encoded[row["id"]] for row in group], max_new_tokens=40, temperature=0
        )
        for row, result in zip(group, results, strict=True):
            phrase = (
                result["text"].strip().splitlines()[0].strip(" \"'.*")
                if result["text"].strip()
                else ""
            )
            append_jsonl(args.out, {"kind": "description", "id": row["id"], "description": phrase})
    print("descriptions done", flush=True)

    encoded = {row["id"]: engine.encode(row["prompt"]) for row in pool}
    lengths = {key: len(engine._ids(value)) for key, value in encoded.items()}
    for index in range(args.passes):
        done = read_jsonl(args.out)
        finished = {row["id"] for row in done if row["kind"] == "label" and row["pass"] == index}
        previous = {
            row["id"]: row["n_out"]
            for row in done
            if row["kind"] == "label" and row["pass"] == index - 1
        }
        todo = [row for row in pool if row["id"] not in finished]
        if previous:  # rows of similar length finish together, so batches stay full
            todo.sort(key=lambda row: (-previous.get(row["id"], 0), lengths[row["id"]]))
        else:
            todo.sort(key=lambda row: (row["source"], row["category"], lengths[row["id"]]))
        started, tokens = time.time(), 0
        for number, group in enumerate(batches(todo, lengths, args.max_rows, MAX_NEW_TOKENS)):
            results = run_batch(
                engine,
                [encoded[row["id"]] for row in group],
                seed=stable_seed("pool", index, number),
                keep_text=index == 0,  # the first pass also supplies the replay targets
                **sampling,
            )
            for row, result in zip(group, results, strict=True):
                append_jsonl(args.out, {"kind": "label", "id": row["id"], "pass": index, **result})
            tokens += sum(result["n_out"] for result in results)
            print(
                f"pass {index} batch {number} rows {len(group)} tokens {tokens} "
                f"rate {tokens / (time.time() - started):.0f} tok/s "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                flush=True,
            )


if __name__ == "__main__":
    main()
