"""In-job merged-weight preparation. vLLM never loads a LoRA adapter (vLLM PR #65532)."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

PEFT = "peft==0.21.2"


def ensure_peft():
    if importlib.util.find_spec("peft") is None:
        # The pinned generation image ships vLLM but not PEFT; install only the pure-Python package.
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", PEFT], check=True)


def prepare_merged(base, adapter, destination):
    """Merge a private adapter into the pinned base bf16 weights; tokenizer comes from the base."""
    ensure_peft()
    import json

    from anthropochrony.artifacts import file_hash
    from anthropochrony.train.sft import merge

    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    merge(base, adapter, destination, minimum_disk_gb=10)
    record = json.loads((destination / "merge.json").read_text())
    record["adapter_weights_sha256"] = file_hash(Path(adapter) / "adapter_model.safetensors")
    return destination, record
