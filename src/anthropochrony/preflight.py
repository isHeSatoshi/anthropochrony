import importlib.metadata
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from anthropochrony.artifacts import file_hash, write_json
from anthropochrony.gates import record_gate
from anthropochrony.gen.engine import render, system_token_count
from anthropochrony.prompts import system_prompt
from anthropochrony.timemodel import SpeedCard


def ensure_disk(path, minimum_gb):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 1e9
    if free < minimum_gb:
        raise RuntimeError(f"Only {free:.1f} GB free at {path}; need {minimum_gb} GB")


def pinned_tokenizer(root, pins):
    root = Path(root)
    subject = pins["subject"]
    directory = root / "pins/tiny-aya-global"
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        hf_hub_download(
            subject["repo"], filename, revision=subject["revision"], local_dir=directory
        )
    if file_hash(directory / "tokenizer.json") != subject["tokenizer_sha256"]:
        raise ValueError("Pinned official tokenizer hash changed")
    return AutoTokenizer.from_pretrained(directory, local_files_only=True)


def _gate_passed(root, name):
    path = Path(root) / "results/gates" / f"{name}.json"
    return path.exists() and bool(json.loads(path.read_text()).get("passed"))


def assemble_g0(root, results):
    """Combine the individually recorded G0 evidence into the overall gate (PLAN S0).

    Remote candidate execution is not available on HF Jobs (G0-hf-security stays unpassed by
    design, docs/EXECUTION_BLOCKER.md), so the overall gate requires the approved alternative:
    candidate code runs only in the local sandbox and G2 validated that topology.
    """
    ledger = json.loads((Path(root) / "artifacts/budget.json").read_text())["entries"]
    gpu_smoke = any(
        e["name"].startswith("s0-gpu-smoke") and e.get("state") == "completed" for e in ledger
    )
    hf_gate = Path(root) / "results/gates/G0-hf-security.json"
    remote_blocked = hf_gate.exists() and not json.loads(hf_gate.read_text()).get("passed")
    checks = {
        "identity_and_special_tokens": bool(results["identity_verified"]),
        "sandbox_security_suite": _gate_passed(root, "G0-security"),
        "chat_template_and_assistant_mask": _gate_passed(root, "G0-template"),
        "hf_jobs_gpu_smoke_completed": gpu_smoke,
        "candidate_execution_local_only": remote_blocked and _gate_passed(root, "G2"),
        "work_disk_at_least_15gb": results["disk_free_gb"]["work"] >= 15,
        "home_disk_at_least_10gb": results["disk_free_gb"]["home"] >= 10,
    }
    record_gate(root, "G0", checks, evidence={"disk_free_gb": results["disk_free_gb"]})
    return []


def preflight(root, pins):
    root = Path(root)
    who = HfApi().whoami()
    if who["name"] != "isHeSatoshi":
        raise ValueError("HF login differs from the account authorized by the plan")
    tokenizer = pinned_tokenizer(root, pins)
    expected = {
        "<BOS_TOKEN>": 2,
        "<EOS_TOKEN>": 3,
        "<|START_OF_TURN_TOKEN|>": 5,
        "<|END_OF_TURN_TOKEN|>": 6,
        "<|START_RESPONSE|>": 261000,
        "<|END_RESPONSE|>": 261001,
        "<PAD>": 0,
    }
    for token, token_id in expected.items():
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            raise ValueError(f"Token ID mismatch: {token}")
    model_config = json.loads((root / "pins/tiny-aya-global/config.json").read_text())
    if model_config["max_position_embeddings"] != 8192:
        raise ValueError("Unexpected model context")
    hf_hub_download("CohereLabs/aya-expanse-8b", "config.json")
    system = system_prompt(SpeedCard("sc01-check", 35, 5000), estimator="agent")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "print(1)"}]
    results = {
        "identity_verified": True,
        "token_ids": expected,
        "context": 8192,
        "tokenizer_sha256": file_hash(root / "pins/tiny-aya-global/tokenizer.json"),
        "transformers_version": importlib.metadata.version("transformers"),
        "system_turn_tokens_sc01": system_token_count(tokenizer, system),
        "no_double_bos": render(tokenizer, messages)[:2] != [2, 2],
        "aya_expanse_license_download": True,
        "disk_free_gb": {
            "work": shutil.disk_usage(root).free / 1e9,
            "home": shutil.disk_usage(Path.home()).free / 1e9,
        },
    }
    results["remaining_G0"] = assemble_g0(root, results)
    write_json(root / "results/preflight.json", results, config=pins)
    return results
