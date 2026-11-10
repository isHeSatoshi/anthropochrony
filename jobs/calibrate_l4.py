"""REF calibration on l40sx1. Name retained from the plan; no L4 jobs are launched."""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def calibrate(llm, *, reset_prefix_cache=False):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()
    filler = tok.encode("A deterministic calibration sequence. " * 1000, add_special_tokens=False)
    records, decode_speeds, prefill_speeds = [], [], []
    # Warm up CUDA graphs before timing. ignore_eos is calibration-only, never agent sampling.
    llm.generate(
        [TokensPrompt(prompt_token_ids=filler[:256])],
        SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
    )
    for length in (256, 512, 1024):
        for rep in range(5):
            start = time.monotonic()
            result = llm.generate(
                [TokensPrompt(prompt_token_ids=filler[:256])],
                SamplingParams(temperature=0, max_tokens=length, ignore_eos=True),
            )[0]
            metrics = result.metrics
            decode_time = metrics.last_token_ts - metrics.first_token_ts
            if decode_time <= 0:
                raise RuntimeError("vLLM decode timestamps are unavailable")
            speed = (len(result.outputs[0].token_ids) - 1) / decode_time
            decode_speeds.append(speed)
            records.append(
                {
                    "kind": "decode",
                    "length": length,
                    "rep": rep,
                    "speed": speed,
                    "wall_s": time.monotonic() - start,
                }
            )
    for length in (256, 512, 1024, 2048):
        for rep in range(5):
            if reset_prefix_cache:
                if not llm.reset_prefix_cache():
                    raise RuntimeError("Unable to clear prefix cache for uncached prefill")
            result = llm.generate(
                [TokensPrompt(prompt_token_ids=filler[:length])],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            )[0]
            metrics = result.metrics
            elapsed = metrics.first_token_ts - metrics.scheduled_ts
            if elapsed <= 0:
                raise RuntimeError("vLLM prefill timestamps are unavailable")
            speed = length / elapsed
            prefill_speeds.append(speed)
            records.append({"kind": "prefill", "length": length, "rep": rep, "speed": speed})
    clocks = (
        __import__("subprocess")
        .run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
    )
    return {
        "reference": {
            "id": "REF",
            "S": float(f"{statistics.median(decode_speeds):.2g}"),
            "P": float(f"{statistics.median(prefill_speeds):.2g}"),
        },
        "hardware": "HF l40sx1",
        "gpu_clocks": clocks,
        "records": records,
        "validation_gate_passed": False,
    }


def main():
    from vllm import LLM

    from anthropochrony.artifacts import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        disable_log_stats=False,
    )
    write_json(args.output, calibrate(llm))


if __name__ == "__main__":
    main()
