import json
from copy import deepcopy
from pathlib import Path

from anthropochrony.artifacts import file_hash, manifest, write_json, write_jsonl

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def verify_training_collator(trainer):
    examples = [trainer.train_dataset[index] for index in range(min(8, len(trainer.train_dataset)))]
    if not examples:
        raise ValueError("The actual trainer has no processed training examples")
    batch = trainer.data_collator(deepcopy(examples))
    inputs, labels, attention = (
        batch[key].tolist() for key in ("input_ids", "labels", "attention_mask")
    )
    for example, tokens, targets, active in zip(examples, inputs, labels, attention, strict=True):
        compact_tokens = [token for token, present in zip(tokens, active, strict=True) if present]
        compact_targets = [label for label, present in zip(targets, active, strict=True) if present]
        expected = [
            token if assistant else -100
            for token, assistant in zip(
                example["input_ids"], example["assistant_masks"], strict=True
            )
        ]
        if compact_tokens != example["input_ids"] or compact_targets != expected:
            raise ValueError("Actual training labels differ from the assistant-only mask")
        if any(
            label != -100 for label, present in zip(targets, active, strict=True) if not present
        ):
            raise ValueError("Training padding tokens are unmasked")
        selected = [
            token
            for token, assistant in zip(
                example["input_ids"], example["assistant_masks"], strict=True
            )
            if assistant
        ]
        if len(selected) < 3 or selected[-2:] != [261001, 6]:
            raise ValueError("The assistant answer must include END_RESPONSE and END_OF_TURN")
    return {"inspected_sequences": len(examples), "assistant_only": True, "padding_masked": True}


def verify_lora_targets(model):
    modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    actual = {name.rsplit(".", 1)[-1] for name in modules}
    if actual != set(TARGETS):
        raise ValueError(f"Unexpected LoRA targets: {sorted(actual)}")
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        not any(part in name for part in (".lora_A.", ".lora_B.")) for name, _ in trainable
    ):
        raise ValueError("Only the specified LoRA matrices may be trainable")
    return {
        "targets": sorted(actual),
        "adapted_modules": modules,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
    }


def verify_assistant_mask(tokenizer, messages, *, training_template=None):
    original = tokenizer.apply_chat_template(messages, tokenize=False)
    # The pinned tokenizer exposes {"default": ...}; TRL matches string templates.
    # Select the existing default without changing a single rendered token.
    tokenizer.chat_template = tokenizer.get_chat_template()
    if training_template is None:
        from trl.chat_template_utils import get_training_chat_template

        training_template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    selected = [
        token
        for token, active in zip(
            encoded["input_ids"],
            encoded["assistant_masks"],
            strict=True,
        )
        if active
    ]
    expected = messages[-1]["content"] + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    decoded = tokenizer.decode(selected, skip_special_tokens=False)
    if decoded != expected:
        raise ValueError(f"Assistant mask mismatch: {decoded!r}")
    if (
        tokenizer.apply_chat_template(messages, tokenize=False, chat_template=training_template)
        != original
    ):
        raise ValueError("Training template changes rendered conversation")
    return {"masked_tokens": len(selected), "rendering_identical": True}


def trainer_kwargs(config, seed):
    return {
        "backend": "trl",
        "num_epochs": config["epochs"],
        # Effective batch 32. The 262k-token vocabulary makes 8 x 3,072 logits exceed 44 GB.
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": config["learning_rate"],
        "warmup_ratio": 0.05,
        "max_seq_length": config["max_seq_length"],
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": TARGETS,
        "assistant_only_loss": True,
        "packing": False,
        "gradient_checkpointing": True,
        "precision": "bf16",
        "seed": seed,
        "data_seed": seed,
        "split_seed": seed,
        "group_by_length": False,
        "load_best_model_at_end": False,
        "report_to": "none",
        "curator_clean": False,
        "curator_dedup": "none",
        "lr_scheduler_type": "cosine",
    }


def train(base, dataset, output, config, seed, *, dry_run=False, push_repo=None):
    from aligntune.core.backend_factory import create_sft_trainer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    first = json.loads(next((Path(dataset) / "train/data.jsonl").open()))
    mask = verify_assistant_mask(tokenizer, first["messages"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "normalized-tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    write_json(output / "mask_check.json", mask)
    kwargs = trainer_kwargs(config, seed)
    if dry_run:
        kwargs["max_steps"] = 1
    trainer = create_sft_trainer(
        model_name=str(base),
        tokenizer_name_or_path=str(tokenizer_path),
        dataset_name=str(dataset),
        output_dir=str(output),
        **kwargs,
    )
    trainer.setup_model()
    lora_check = verify_lora_targets(trainer.model)
    write_json(output / "lora_check.json", lora_check)
    if dry_run:
        print(trainer.model, flush=True)
    result = trainer.train()
    if dry_run and result.get("steps") != 1:
        raise ValueError("The real dry run must complete exactly one optimizer step")
    trainer.trainer.save_state()
    write_jsonl(output / "training-log.jsonl", trainer.trainer.state.log_history)
    collator_check = verify_training_collator(trainer.trainer)
    write_json(
        output / "training-summary.json",
        {
            "seed": seed,
            "dry_run": dry_run,
            "config": config,
            "train_data_sha256": file_hash(Path(dataset) / "train/data.jsonl"),
            "assistant_mask": mask,
            "actual_collator": collator_check,
            "lora": lora_check,
            "training": result,
        },
    )
    for file in output.rglob("*"):
        if file.is_file() and not file.name.endswith(".manifest.json"):
            manifest(file, config=config)
    if push_repo:
        from anthropochrony.train.publish import publish_adapter

        # Never use AlignTune's default (public) or push merged weights.
        publish_adapter(trainer, output, push_repo, config["base"])
    return trainer


def merge(base, adapter, output, *, minimum_disk_gb=15):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from anthropochrony.preflight import ensure_disk

    ensure_disk(output, minimum_disk_gb)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    # Training used the pinned base tokenizer; the adapter repo stores no tokenizer files.
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(output)
    hashes = {p.name: file_hash(p) for p in Path(output).glob("*.safetensors")}
    write_json(
        Path(output) / "merge.json", {"base": str(base), "adapter": str(adapter), "hashes": hashes}
    )
