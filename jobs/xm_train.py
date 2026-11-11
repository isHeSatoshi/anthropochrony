# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.13.0", "transformers==5.21.1", "peft==0.21.2",
#   "huggingface-hub==2.4.0", "numpy",
# ]
# ///
"""LoRA SFT for the cross-model arm, on token IDs rendered and checked locally.

The M1 recipe (r16, alpha32, all attention and MLP projections, effective batch 32, cosine
schedule, assistant-only loss) without AlignTune: the pinned AlignTune/TRL path needs a chat
template with generation markers that the Aya Expanse template lacks. Loss covers only the
completion tokens, including the closing `<|END_OF_TURN_TOKEN|>`.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def localize(source, destination="/tmp/aya-expanse-8b"):
    """Sequential copy off the FUSE bucket mount; random-access mmap loads there are very slow."""
    import shutil

    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This research training job requires the approved HF L40S GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    started = time.monotonic()
    base = localize(args.base)
    print(f"weights copied locally after {time.monotonic() - started:.0f}s", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    per_epoch = math.ceil(len(rows) / args.accum)
    total = per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, round(0.05 * total))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    log, step = [], 0
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.accum):
            if step >= total:
                break
            group = [rows[i] for i in order[start : start + args.accum]]
            tokens = sum(len(r["completion_ids"]) for r in group)
            for group_row in optimizer.param_groups:
                group_row["lr"] = args.lr * scale(step)
            loss_sum = 0.0
            for row in group:
                ids = row["prompt_ids"] + row["completion_ids"]
                prompt_len = len(row["prompt_ids"])
                input_ids = torch.tensor([ids], device="cuda")
                keep = torch.arange(prompt_len - 1, len(ids) - 1, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, logits_to_keep=keep).logits[0]
                target = input_ids[0, prompt_len:]
                loss = torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum")
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
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # The adapter is published from the local machine: the card and config PEFT writes here name
    # the job's temporary weight copy, which the Hub rejects as a base model.
    model.save_pretrained(output)
    (output / "training-log.json").write_text(json.dumps(log))
    (output / "training-summary.json").write_text(
        json.dumps(
            {
                "examples": len(rows),
                "steps": step,
                "epochs": args.epochs,
                "lr": args.lr,
                "accum": args.accum,
                "seed": args.seed,
                "gpu": torch.cuda.get_device_name(0),
                "job_id": os.environ.get("HF_JOB_ID"),
                "elapsed_s": time.monotonic() - started,
                "first_loss": log[0]["loss"] if log else None,
                "last_loss": log[-1]["loss"] if log else None,
            }
        )
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
