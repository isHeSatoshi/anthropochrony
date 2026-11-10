import importlib.metadata
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from anthropochrony.artifacts import file_hash, write_json
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
        "remaining_G0": [
            "generation smoke",
            "HF Jobs volume and GPU smoke",
            "sandbox security tests",
            "stack/gen tokenizer parity",
            "execution dependency coverage",
        ],
    }
    write_json(root / "results/preflight.json", results, config=pins)
    return results
