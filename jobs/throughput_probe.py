"""Benchmark one complete production-shaped batch in a fresh GPU process."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import yaml

    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.gen.engine import VLLMEngine
    from anthropochrony.gen.performance import GPUMonitor, preemptions

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/measure.yaml").read_text())
    engine_config = config["engine"] | {
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "disable_log_stats": False,
    }
    started = time.monotonic()
    output = Path(args.output)
    with GPUMonitor() as monitor:
        engine = VLLMEngine("/out/base", engine_config)
        before = preemptions(engine.llm)
        if before is None:
            raise RuntimeError("The required vllm:num_preemptions counter is unavailable")
        requests = read_jsonl(ROOT / "throughput-requests.jsonl")
        generate_start = time.monotonic()
        rows = engine.generate(requests)
        generate_end = time.monotonic()
        # Preserve costly outputs before optional statistics can fail.
        write_jsonl(output.with_suffix(".generation.jsonl"), rows, config=config)
        write_json(output.with_suffix(".gpu.json"), monitor.samples)
        after = preemptions(engine.llm)
        performance = engine.last_performance | {
            "engine": engine_config,
            "requests": len(requests),
            "process_elapsed_s": generate_end - started,
            "preemptions": after - before if after is not None else None,
            "gpu": monitor.summary(generate_start, generate_end),
            "monitor_errors": monitor.errors,
        }
        write_json(output, performance, config=engine_config)
        print(f"THROUGHPUT_RESULT {output}", flush=True)


if __name__ == "__main__":
    main()
