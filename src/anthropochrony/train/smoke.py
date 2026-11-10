"""Real-GPU smoke of in-job adapter merge and merged serving, before any trained adapter exists.

The adapter is a small random LoRA with the real Cohere2 shapes. It checks the job tooling
(PEFT install, merge, vLLM serving, HF+PEFT parity). It says nothing about research results.
"""

import json
from pathlib import Path

from anthropochrony.artifacts import digest, read_jsonl, write_json, write_jsonl
from anthropochrony.cli import config
from anthropochrony.gates import GateError
from anthropochrony.gpu_jobs import GPUJob
from anthropochrony.train.sft import TARGETS

RUN = "smoke-merged"


def lora_shapes(model_config):
    hidden, head = model_config["hidden_size"], model_config["head_dim"]
    query = model_config["num_attention_heads"] * head
    kv = model_config["num_key_value_heads"] * head
    inner = model_config["intermediate_size"]
    shapes = {
        "self_attn.q_proj": (hidden, query),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (query, hidden),
        "mlp.gate_proj": (hidden, inner),
        "mlp.up_proj": (hidden, inner),
        "mlp.down_proj": (inner, hidden),
    }
    if {name.split(".")[1] for name in shapes} != set(TARGETS):
        raise ValueError("Smoke adapter shapes no longer match the training targets")
    return shapes


def build_adapter(root, destination, rank=16):
    import numpy as np
    from safetensors.numpy import save_file

    root, destination = Path(root), Path(destination)
    model_config = json.loads((root / "pins/tiny-aya-global/config.json").read_text())
    subject = config(root, "pins.yaml")["subject"]
    generator = np.random.default_rng(20261110)
    tensors = {}
    for layer in range(model_config["num_hidden_layers"]):
        for name, (inputs, outputs) in lora_shapes(model_config).items():
            prefix = f"base_model.model.model.layers.{layer}.{name}"
            tensors[f"{prefix}.lora_A.weight"] = generator.normal(
                0, 1 / rank**0.5, (rank, inputs)
            ).astype("float32")
            # Small but nonzero, so merged and unmerged weights genuinely differ.
            tensors[f"{prefix}.lora_B.weight"] = generator.normal(0, 0.002, (outputs, rank)).astype(
                "float32"
            )
    destination.mkdir(parents=True, exist_ok=True)
    save_file(tensors, destination / "adapter_model.safetensors")
    write_json(
        destination / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": subject["repo"],
            "revision": subject["revision"],
            "r": rank,
            "lora_alpha": 2 * rank,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": TARGETS,
            "inference_mode": True,
        },
    )
    return destination


def smoke_prompts(root, count=50):
    from anthropochrony.prompts import estimate_messages
    from anthropochrony.timemodel import SpeedCard

    tasks = sorted(
        (t for t in read_jsonl(Path(root) / "data/tasks/tasks.jsonl") if t["split"] == "val"),
        key=lambda t: t["task_id"],
    )[:count]
    card = SpeedCard("REF", 96, 26000)
    return [
        {
            "est_id": f"smoke-{index:03d}",
            "messages": estimate_messages(task, card, "q21", "sc07", "E1"),
            "max_tokens": 96,
            "T_med_s": 1.0,
            "split": "val",
            "card_id": "REF",
        }
        for index, task in enumerate(tasks)
    ]


def upload_adapter(root, adapter):
    from huggingface_hub import sync_bucket

    from anthropochrony.measure.remote import BUCKET_ID
    from anthropochrony.remote import authorization_guard

    with authorization_guard():
        sync_bucket(str(adapter), f"hf://buckets/{BUCKET_ID}/adapters/{RUN}", quiet=True)


def smoke_merged_serving(root, *, timeout_s=1800):
    """Launch both the G6-parity job and the multi-run round job against the random adapter."""
    from transformers import AutoTokenizer

    from anthropochrony.gen.engine import prepare_estimate_requests

    root = Path(root)
    adapter = build_adapter(root, root / "data/smoke/merged/adapter")
    upload_adapter(root, adapter)
    prompts = smoke_prompts(root)
    if len(prompts) != 50:
        raise GateError("The merged-serving smoke needs 50 validation prompts")
    directory = root / "data/smoke/merged"
    prompt_path = directory / "prompts.jsonl"
    write_jsonl(prompt_path, prompts)
    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    requests, _ = prepare_estimate_requests(tokenizer, prompts, 8192)
    request_path = directory / "requests.jsonl"
    write_jsonl(
        request_path,
        [row | {"request_id": f"{RUN}:{row['request_id']}"} for row in requests],
    )
    config_path = directory / "round-config.json"
    write_json(config_path, config(root, "measure.yaml"))
    context = {"adapter_sha": digest(sorted(p.name for p in adapter.iterdir()))}
    common = {"timeout_s": timeout_s, "required_gates": ("G0-security", "G1", "G2")}
    parity = GPUJob(
        root,
        directory / "g6",
        kind="smoke-g6",
        script="jobs/g6_check.py",
        files=[(prompt_path, "prompts.jsonl")],
        arguments=[
            "--prompts",
            "/code/prompts.jsonl",
            "--adapter",
            f"/out/adapters/{RUN}",
            "--config",
            "/code/configs/measure.yaml",
        ],
        outputs=["g6-equivalence.json", "hf-peft.jsonl"],
        context=context,
        **common,
    )
    serving = GPUJob(
        root,
        directory / "round",
        kind="smoke-ftround",
        script="jobs/ft_round.py",
        files=[(request_path, "requests.jsonl"), (config_path, "round-config.json")],
        arguments=["--requests", "/code/requests.jsonl", "--config", "/code/round-config.json"],
        outputs=[
            "generation.jsonl",
            "merge-evidence.json",
            "generation.performance.json",
            "generation.gpu.json",
        ],
        context=context,
        sources=("jobs/model_prep.py",),
        **common,
    )
    return parity, serving


def run_smoke(root):
    from concurrent.futures import ThreadPoolExecutor

    root = Path(root)
    parity, serving = smoke_merged_serving(root)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(parity.run), pool.submit(serving.run)]
        for future in futures:
            future.result()
    equivalence = json.loads((parity.directory / "g6-equivalence.json").read_text())
    rows = read_jsonl(serving.directory / "generation.jsonl")
    evidence = json.loads((serving.directory / "merge-evidence.json").read_text())
    result = {
        "g6_prompts": equivalence["prompts"],
        "numeric_mismatches": equivalence["numeric_mismatches"],
        "identical_texts": equivalence["identical_texts"],
        "round_rows": len(rows),
        "round_merge_hash": evidence[RUN]["adapter_weights_sha256"],
        "scope": "random-adapter tooling smoke; not a research result",
    }
    write_json(root / "results/smoke-merged.json", result)
    if equivalence["numeric_mismatches"] > 2 or len(rows) != 50:
        raise GateError(f"Merged-serving smoke failed: {result}")
    return result
