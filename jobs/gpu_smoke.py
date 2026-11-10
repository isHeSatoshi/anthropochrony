"""S0 smoke: pinned image, model, bucket writes, tokenizer parity, batched generation."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import yaml
    from calibrate_l4 import calibrate

    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.gen.engine import VLLMEngine, render

    output = Path("/out/smoke")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/measure.yaml").read_text())
    engine = VLLMEngine(
        "/out/base", config["engine"] | {"disable_log_stats": False, "enable_prefix_caching": False}
    )
    parity = read_jsonl(ROOT / "parity.jsonl")
    comparisons = [render(engine.tokenizer, row["messages"]) == row["token_ids"] for row in parity]
    requests = read_jsonl(ROOT / "requests.jsonl")
    generated = engine.generate(requests)
    write_jsonl(
        output / "generation.jsonl", generated, inputs=[ROOT / "requests.jsonl"], config=config
    )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = {
        "gpu": gpu,
        "uid": os.getuid(),
        "bucket_read_write": True,
        "model_staged": Path("/out/base/config.json").exists(),
        "transformers_version": importlib.metadata.version("transformers"),
        "vllm_version": importlib.metadata.version("vllm"),
        "tokenizer_parity_20_conversations": len(comparisons) == 20 and all(comparisons),
        "bwrap_available": bool(shutil.which("bwrap")),
        "generation_count": len(generated),
        "performance": engine.last_performance,
        "stop_conventions": [
            {
                "stop_reason": row["stop_reason"],
                "last_token": row["token_ids"][-1],
                "output_count": row["o_tokens"],
                "returned_count": len(row["token_ids"]),
            }
            for row in generated
        ],
    }
    write_json(output / "report.json", result, config=config)
    if not result["tokenizer_parity_20_conversations"] or len(generated) != 10:
        raise RuntimeError("S0 GPU smoke failed")
    write_json("/out/calibration/reference.json", calibrate(engine.llm), config=config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
