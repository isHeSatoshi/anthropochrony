"""AlignTune stage: LoRA fine-tuning of the self-time adapter on the CuratorKIT-passed pool.

The examples are the ones `train.py` builds for `G1` (same questions, stated speeds, task
views, phrasings and answer format, drawn with the same seeds per prompt), restricted to
the prompts that passed `curate.py`. The replay examples, which keep the adapter's ordinary
answers unchanged, use the model's complete measured response, ending with its own
end-of-turn tokens; a response longer than the sequence limit is cut there and so carries
no end-of-turn target. Training runs through AlignTune's SFT trainer (TRL backend, PEFT
LoRA) with the loss on the answer only.
"""

import argparse
import json
import random
import time
from pathlib import Path

from common import load_pool
from train import TARGETS, build_examples, build_replay


def check_answer_only_labels(trainer, count=16):
    """The labels the trainer really uses cover the answer and its end tokens, nothing else."""
    rows = [trainer.train_dataset[i] for i in range(min(count, len(trainer.train_dataset)))]
    batch = trainer.data_collator([dict(row) for row in rows])
    for row, ids, labels in zip(
        rows, batch["input_ids"].tolist(), batch["labels"].tolist(), strict=True
    ):
        start = next(i for i, token in enumerate(ids) if token == row["input_ids"][0])
        # The collator cuts a sequence longer than the limit, so compare the part it kept.
        n = min(len(row["input_ids"]), len(ids) - start)
        ids, labels = ids[start : start + n], labels[start : start + n]
        expected = [
            t if m else -100
            for t, m in zip(row["input_ids"][:n], row["assistant_masks"][:n], strict=True)
        ]
        if ids != row["input_ids"][:n] or labels != expected or not any(row["assistant_masks"]):
            raise ValueError("Training labels are not the answer-only mask")
    return {"inspected_sequences": len(rows), "answer_only": True}


def to_record(example):
    prompt = [{"role": "system", "content": example["system"]}] if example["system"] else []
    prompt.append({"role": "user", "content": example["user"]})
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": example["answer"]}],
        "kind": example["view"],
    }


def build_dataset(pool, labels, curated, seed, cards_per_task):
    keep = set(json.loads(Path(curated).read_text()))
    rows = [row for row in load_pool(pool, labels) if row["split"] == "train" and row["id"] in keep]
    if len(rows) != len(keep):
        raise ValueError(f"{len(keep)} curated ids but {len(rows)} labelled training prompts")
    examples = build_examples(rows, cards_per_task, seed) + build_replay(rows)
    records = [to_record(example) for example in examples]
    random.Random(seed).shuffle(records)
    return rows, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--curated", required=True, help="passed.json from curate.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cards-per-task", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, help="dry run")
    parser.add_argument("--worst-case", action="store_true", help="dry run on the longest rows")
    args = parser.parse_args()

    import torch
    from aligntune.core.backend_factory import create_sft_trainer
    from datasets import Dataset
    from transformers import AutoTokenizer

    output = Path(args.output)
    data_dir = output / "data"
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    rows, records = build_dataset(
        args.pool, args.labels, args.curated, args.seed, args.cards_per_task
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The pinned tokenizer stores its templates as {"default": ...}; TRL's assistant mask needs
    # the default selected as a string. The rendered text does not change.
    tokenizer.chat_template = tokenizer.get_chat_template()
    tokenizer_dir = output / "normalized-tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    if args.worst_case:

        def length(record):
            return len(
                tokenizer.apply_chat_template(
                    record["prompt"] + record["completion"], tokenize=True
                )
            )

        records = sorted(records, key=length, reverse=True)[: args.batch * args.accum * 2]
    with (data_dir / "train/data.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {
        "prompts": len(rows),
        "estimate_examples": sum(r["kind"] != "replay" for r in records),
        "replay_examples": sum(r["kind"] == "replay" for r in records),
    }
    print(json.dumps(counts), flush=True)

    kwargs = {
        "backend": "trl",
        "num_epochs": args.epochs,
        "batch_size": args.batch,
        "gradient_accumulation_steps": args.accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_seq_length": args.max_seq_length,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": TARGETS,
        "assistant_only_loss": True,
        "packing": False,
        "gradient_checkpointing": True,
        "precision": "bf16",
        "seed": args.seed,
        "data_seed": args.seed,
        "group_by_length": True,
        "load_best_model_at_end": False,
        "report_to": "none",
        "curator_clean": False,
        "curator_dedup": "none",
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "no",
    }
    if args.max_steps:
        kwargs["max_steps"] = args.max_steps
    trainer = create_sft_trainer(
        model_name=args.model,
        tokenizer_name_or_path=str(tokenizer_dir),
        dataset_name=str(data_dir),
        output_dir=str(output),
        **kwargs,
    )
    trainer.setup_model()
    # AlignTune's data manager folds a system turn into the user turn, which would train a
    # different prompt format from the one asked at test time, so the conversations go to
    # its trainer as they are.
    trainer.train_dataset = Dataset.from_list(
        [{"messages": record["prompt"] + record["completion"]} for record in records]
    )
    trainer.dataset = trainer.train_dataset
    trainer.eval_dataset = None
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    trainer.trainer.model.save_pretrained(output)
    mask = check_answer_only_labels(trainer.trainer)
    history = trainer.trainer.state.log_history
    (output / "training-log.json").write_text(json.dumps(history))
    summary = {
        **counts,
        "trainer": "AlignTune SFT (TRL backend)",
        "steps": result.get("steps"),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "accum": args.accum,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "cards_per_task": args.cards_per_task,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "elapsed_s": elapsed,
        "final_loss": result.get("final_loss"),
        "label_check": mask,
        "losses": [entry["loss"] for entry in history if "loss" in entry],
    }
    (output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "losses"}), flush=True)
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
