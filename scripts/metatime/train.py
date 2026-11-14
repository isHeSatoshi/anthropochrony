"""LoRA fine-tuning of the single-turn self-time adapter on pool measurements.

The project's M1 recipe (rank 16, alpha 32, all attention and MLP projections, cosine
schedule, loss on the answer only) with the stated speed varied across examples. Labels
are the model's own median response lengths on pool prompts; the time in each answer is
arithmetic from that length and the stated speeds.
"""

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from common import (
    QUESTIONS,
    TRAIN_PHRASINGS,
    answer_text,
    load_pool,
    sample_card,
    stable_seed,
    system_prompt,
    task_view,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_examples(rows, cards_per_task, seed):
    examples = []
    for row in rows:
        rng = random.Random(stable_seed("sft", seed, row["id"]))
        for _ in range(cards_per_task):
            speed, prefill = sample_card(rng)
            question = rng.choices([q for q, _ in QUESTIONS], [w for _, w in QUESTIONS])[0]
            view = rng.choice(["description", "full"])
            examples.append(
                {
                    "id": row["id"],
                    "view": view,
                    "system": system_prompt(speed, prefill, rng.choice(TRAIN_PHRASINGS)),
                    "user": question.format(description=task_view(row, view)),
                    "answer": answer_text(row["n_in"], row["n_out"], speed, prefill),
                }
            )
    return examples


def build_replay(rows):
    """The plain prompt with the base model's own response as target.

    Trained alongside the estimates so the adapter leaves ordinary task behaviour, and with
    it the response lengths the labels were measured on, unchanged.
    """
    return [
        {
            "id": row["id"],
            "view": "replay",
            "system": None,
            "user": row["prompt"],
            "answer": row["response"],
        }
        for row in rows
        if row["response"]
    ]


def tokenize(tokenizer, example, template_kwargs):
    messages = [{"role": "system", "content": example["system"]}] if example["system"] else []
    messages.append({"role": "user", "content": example["user"]})

    def ids(extra, **kwargs):
        encoded = tokenizer.apply_chat_template(
            messages + extra, tokenize=True, **kwargs, **template_kwargs
        )
        return encoded["input_ids"] if hasattr(encoded, "keys") else encoded

    prompt = ids([], add_generation_prompt=True)
    full = ids([{"role": "assistant", "content": example["answer"]}])
    if full[: len(prompt)] != prompt:
        raise ValueError("The chat template does not extend the prompt with the answer")
    return prompt, full[len(prompt) :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--cards-per-task", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-tokens", type=int, default=128, help="0 turns replay off")
    parser.add_argument("--replay-batch", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=1000, help="rows x tokens per pass")
    parser.add_argument("--worst-case", action="store_true", help="run the largest batches only")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    rows = [row for row in load_pool(args.pool, args.labels) if row["split"] == "train"]
    examples = build_examples(rows, args.cards_per_task, args.seed)
    data = [tokenize(tokenizer, example, template_kwargs) for example in examples]
    n_estimates = len(data)
    if args.replay_tokens:
        for example in build_replay(rows):
            prompt, completion = tokenize(tokenizer, example, template_kwargs)
            # A response cut at the replay length loses its end-of-turn tokens, so the
            # adapter is never taught to stop early; short responses keep theirs.
            data.append((prompt, completion[: args.replay_tokens]))
    print(
        f"{len(rows)} prompts, {n_estimates} estimate examples, {len(data) - n_estimates} "
        f"replay examples, median length {statistics.median(len(p) + len(c) for p, c in data)}, "
        f"longest {max(len(p) + len(c) for p, c in data)}",
        flush=True,
    )
    print(
        "example:\n"
        + examples[0]["system"]
        + "\n---\n"
        + examples[0]["user"][:400]
        + "\n---\n"
        + examples[0]["answer"],
        flush=True,
    )

    kwargs = {"device_map": {"": 0}}
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"
        ),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # Buckets of similar length keep padding small; bucket order is shuffled every epoch.
    def length(i):
        return len(data[i][0]) + len(data[i][1])

    # Micro-batches of similar length, bounded in rows x tokens so activations fit in GPU
    # memory. Replay examples carry long targets, so they go in their own, smaller batches.
    def epoch_batches(epoch):
        groups = []
        for kind, (first, last, size) in enumerate(
            [(0, n_estimates, args.micro_batch), (n_estimates, len(data), args.replay_batch)]
        ):
            order = list(range(first, last))
            random.Random(args.seed * 1000 + epoch * 10 + kind).shuffle(order)
            chunk = size * 32
            for start in range(0, len(order), chunk):
                group = []
                for i in sorted(order[start : start + chunk], key=length):
                    if group and (
                        len(group) >= size or (len(group) + 1) * length(i) > args.token_budget
                    ):
                        groups.append(group)
                        group = []
                    group.append(i)
                groups.append(group)
        random.Random(args.seed * 1000 + epoch * 10 + 9).shuffle(groups)
        return groups

    # One optimizer step per `accum` micro-batches of estimates, whatever their row count.
    def epoch_steps(epoch):
        steps, window, rows = [], [], 0
        for group in epoch_batches(epoch):
            window.append(group)
            rows += len(group)
            if rows >= args.micro_batch * args.accum:
                steps.append(window)
                window, rows = [], 0
        if window:
            steps.append(window)
        return steps

    def positions(group):
        return len(group) * max(length(i) for i in group)

    if args.worst_case:  # memory check: the largest micro-batches of each kind
        heavy = sorted(epoch_batches(0), key=positions, reverse=True)
        fitting = [group for group in heavy if positions(group) <= args.token_budget]
        replay = [group for group in fitting if group[0] >= n_estimates]
        schedule = [[[group] for group in heavy[:2] + fitting[:2] + replay[:2]]]
    else:
        schedule = [epoch_steps(epoch) for epoch in range(args.epochs)]
    total = sum(len(steps) for steps in schedule)
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, round(0.05 * total))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    state = {"checkpointing": False}

    # Activations of a pass over the budget do not fit, so those few long examples trade
    # speed for memory; everything else runs without checkpointing, about twice as fast.
    def checkpointing(enabled):
        if enabled != state["checkpointing"]:
            if enabled:
                model.gradient_checkpointing_enable({"use_reentrant": False})
            else:
                model.gradient_checkpointing_disable()
            state["checkpointing"] = enabled

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    log, step, started = [], 0, time.monotonic()
    model.train()
    for epoch, steps in enumerate(schedule):
        for window in steps:
            if step >= total:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale(step)
            tokens = sum(len(data[i][1]) for group in window for i in group)
            loss_sum = 0.0
            for group in window:
                width = max(len(data[i][0]) + len(data[i][1]) for i in group)
                answer = max(len(data[i][1]) for i in group)
                ids = torch.full((len(group), width), pad, dtype=torch.long, device="cuda")
                mask = torch.zeros_like(ids)
                target = torch.full((len(group), answer), -100, dtype=torch.long, device="cuda")
                for row, i in enumerate(group):
                    prompt, completion = data[i]
                    sequence = prompt + completion
                    ids[row, width - len(sequence) :] = torch.tensor(sequence, device="cuda")
                    mask[row, width - len(sequence) :] = 1
                    target[row, answer - len(completion) :] = torch.tensor(
                        completion, device="cuda"
                    )
                keep = torch.arange(width - answer - 1, width - 1, device="cuda")
                checkpointing(len(group) * width > args.token_budget)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.float().flatten(0, 1),
                    target.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                (loss / tokens).backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            entry = {
                "step": step,
                "epoch": epoch,
                "loss": loss_sum / tokens,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - started,
                "rows": sum(len(group) for group in window),
                "positions": sum(positions(group) for group in window),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
            torch.cuda.reset_peak_memory_stats()
            log.append(entry)
            if step % 10 == 0 or step <= 5:
                print(json.dumps(entry), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    (output / "training-log.json").write_text(json.dumps(log))
    (output / "training-summary.json").write_text(
        json.dumps(
            {
                "prompts": len(rows),
                "examples": n_estimates,
                "replay_examples": len(data) - n_estimates,
                "replay_tokens": args.replay_tokens,
                "steps": step,
                "epochs": args.epochs,
                "lr": args.lr,
                "micro_batch": args.micro_batch,
                "accum": args.accum,
                "seed": args.seed,
                "cards_per_task": args.cards_per_task,
                "gpu": torch.cuda.get_device_name(0),
                "elapsed_s": time.monotonic() - started,
                "first_loss": log[0]["loss"],
                "last_loss": log[-1]["loss"],
            },
            indent=2,
        )
        + "\n"
    )
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
