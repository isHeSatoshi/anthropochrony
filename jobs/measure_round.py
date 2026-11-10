"""GPU-only round. All requests enter one vLLM call; no generated code executes here."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
from anthropochrony.gen.engine import VLLMEngine
from anthropochrony.gen.performance import measured_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    engine = VLLMEngine(args.model, config["engine"])
    _, performance, samples = measured_generate(
        engine,
        read_jsonl(args.requests),
        output_sink=lambda rows: write_jsonl(output, rows, inputs=[args.requests], config=config),
    )
    performance["process_elapsed_s"] = time.monotonic() - started
    performance["engine"] = config["engine"]
    write_json(output.with_suffix(".gpu.json"), samples)
    write_json(output.with_suffix(".performance.json"), performance)


if __name__ == "__main__":
    main()
