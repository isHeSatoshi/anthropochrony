"""Single GPU engine over a VM-local socket; candidate execution is LOCAL only."""

import argparse
import json
import socketserver
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import yaml
    from calibrate_l4 import calibrate

    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.gen.engine import VLLMEngine, render
    from anthropochrony.gen.performance import GPUMonitor
    from anthropochrony.prompts import system_prompt
    from anthropochrony.timemodel import SpeedCard

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    engine_config = json.loads(Path(args.engine_config).read_text())["engine"]
    engine = VLLMEngine("/out/base", engine_config)
    config = yaml.safe_load((ROOT / "configs/measure.yaml").read_text())
    count = 0

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal count
            try:
                body = self.rfile.readline(1024 * 1024 + 1)
                if len(body) > 1024 * 1024:
                    raise ValueError("Oversized request")
                message = json.loads(body)
                operation = message["op"]
                if operation == "ready":
                    response = {"ready": True}
                elif operation == "calibrate":
                    response = calibrate(engine.llm, reset_prefix_cache=True)
                    write_json(output / "reference.json", response, config=engine_config)
                elif operation == "warmup":
                    card = SpeedCard(**message["card"])
                    ids = render(
                        engine.tokenizer,
                        [
                            {"role": "system", "content": system_prompt(card, estimator="agent")},
                            {"role": "user", "content": "print(1)"},
                        ],
                    )
                    engine.generate(
                        [
                            {
                                "request_id": "warmup",
                                "prompt_token_ids": ids,
                                "seed": 0,
                                "max_tokens": 1,
                            }
                        ]
                    )
                    response = {"warmed_up": True}
                elif operation == "generate":
                    requests = message["requests"]
                    if len(requests) != 1:
                        raise ValueError("S2 requires exactly one request at a time")
                    count += 1
                    rows = engine.generate(requests)
                    write_jsonl(output / f"call-{count:04d}.jsonl", rows, config=config)
                    response = {"outputs": rows, "performance": engine.last_performance}
                elif operation == "batch":
                    requests = read_jsonl(ROOT / "throughput-requests.jsonl")
                    start = time.monotonic()
                    rows = engine.generate(requests)
                    end = time.monotonic()
                    write_jsonl(output / "batch-generation.jsonl", rows, config=config)
                    response = engine.last_performance | {"gpu": monitor.summary(start, end)}
                else:
                    raise ValueError("Unsupported trusted generation operation")
            except Exception as error:
                response = {"error": type(error).__name__ + ": " + str(error)[:1000]}
            self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")

    socket = Path("/tmp/anthropochrony-generation.sock")
    if socket.exists():
        raise RuntimeError("A generation session already owns the socket")
    with GPUMonitor() as monitor, socketserver.UnixStreamServer(str(socket), Handler) as server:
        print("GENERATION_SESSION_READY", flush=True)
        try:
            server.serve_forever()
        finally:
            socket.unlink(missing_ok=True)
            write_json(output / "gpu-telemetry.json", monitor.samples)


if __name__ == "__main__":
    main()
