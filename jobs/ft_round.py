"""GPU-only round for merged fine-tuned models. No generated code executes here.

Request IDs are namespaced `{run}:{id}`. Each run's adapter is merged into the pinned base
(bf16, never vLLM LoRA) and served by a fresh vLLM process, then all outputs are recombined.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def serve(args):
    import yaml
    from model_prep import prepare_merged

    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.gen.engine import VLLMEngine
    from anthropochrony.gen.performance import measured_generate

    config = yaml.safe_load(Path(args.config).read_text())
    started = time.monotonic()
    merged, record = prepare_merged(args.base, args.adapter, "/tmp/merged")
    record["merge_s"] = time.monotonic() - started
    engine = VLLMEngine(merged, config["engine"])
    requests = read_jsonl(args.requests)
    rows, performance, samples = measured_generate(engine, requests)
    performance["engine"] = config["engine"]
    performance["process_elapsed_s"] = time.monotonic() - started
    out = Path(args.output_dir)
    write_jsonl(out / "rows.jsonl", rows, inputs=[args.requests], config=config)
    write_json(out / "merge.json", record)
    write_json(out / "performance.json", performance)
    write_json(out / "gpu.json", samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--base", default="/out/base")
    parser.add_argument("--adapter-root", default="/out/adapters")
    parser.add_argument("--adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    if args.single:
        serve(args)
        return
    import shutil

    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl

    requests = read_jsonl(args.requests)
    runs = sorted({row["request_id"].split(":", 1)[0] for row in requests})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, merges, performance, samples = [], {}, {}, {}
    scratch = Path("/tmp/ft-round")
    for run in runs:
        subset = [dict(r) for r in requests if r["request_id"].split(":", 1)[0] == run]
        directory = scratch / run
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "requests.jsonl", subset)
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--single",
                "--requests",
                str(directory / "requests.jsonl"),
                "--adapter",
                f"{args.adapter_root}/{run}",
                "--base",
                args.base,
                "--config",
                args.config,
                "--output-dir",
                str(directory / "out"),
            ],
            check=True,
        )
        rows += read_jsonl(directory / "out/rows.jsonl")
        merges[run] = json.loads((directory / "out/merge.json").read_text())
        performance[run] = json.loads((directory / "out/performance.json").read_text())
        samples[run] = json.loads((directory / "out/gpu.json").read_text())
        shutil.rmtree("/tmp/merged", ignore_errors=True)
    write_jsonl(output / "generation.jsonl", rows, inputs=[args.requests])
    write_json(output / "merge-evidence.json", merges)
    write_json(output / "generation.performance.json", performance)
    write_json(output / "generation.gpu.json", samples)


if __name__ == "__main__":
    main()
