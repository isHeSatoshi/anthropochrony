"""Merge a private adapter into the pinned base on CPU, then convert to GGUF with pinned llama.cpp.

Run with the llama.cpp venv (CPU torch + peft). GGUF files are demo and deployment artifacts only:
llama.cpp tokenizes some digit runs differently from the HF tokenizer, so no reported number may
come from them.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

LLAMA = Path("/home/adi/.local/share/anthropochrony-envs/llamacpp")
BUCKET = "isHeSatoshi/anthropochrony-runs"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--work", default="/home/adi/.cache/anthropochrony-tmp/gguf")
    parser.add_argument("--quantizations", nargs="+", default=["Q4_K_M", "Q8_0"])
    args = parser.parse_args()
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(args.work) / args.run
    work.mkdir(parents=True, exist_ok=True)
    pins = yaml.safe_load((ROOT / "configs/pins.yaml").read_text())["subject"]
    adapter = work / "adapter"
    adapter.mkdir(exist_ok=True)
    HfApi().download_bucket_files(
        BUCKET,
        [
            (f"adapters/{args.run}/{name}", adapter / name)
            for name in ("adapter_model.safetensors", "adapter_config.json")
        ],
        raise_on_missing_files=True,
    )
    base = snapshot_download(pins["repo"], revision=pins["revision"], cache_dir=work / "hf")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged = work / "merged"
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    del model
    python = LLAMA / "venv/bin/python"
    source = LLAMA / "src"
    bf16 = work / f"{args.run}-bf16.gguf"
    subprocess.run(
        [
            str(python),
            str(source / "convert_hf_to_gguf.py"),
            str(merged),
            "--outtype",
            "bf16",
            "--outfile",
            str(bf16),
        ],
        check=True,
    )
    record = {
        "run": args.run,
        "base_revision": pins["revision"],
        "llama_cpp_tag": "b12110",
        "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
        "files": {bf16.name: sha256(bf16)},
    }
    for quantization in args.quantizations:
        target = work / f"{args.run}-{quantization.lower()}.gguf"
        subprocess.run(
            [str(LLAMA / "build/bin/llama-quantize"), str(bf16), str(target), quantization],
            check=True,
        )
        record["files"][target.name] = sha256(target)
    (work / "gguf-manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    sys.exit(main())
