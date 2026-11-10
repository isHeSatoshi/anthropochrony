"""AuditKIT callable using one pre-generated batched completion set, preserving cache hashes."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "jobs"))


def main():
    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.eval.audit import evaluate
    from anthropochrony.gen.engine import VLLMEngine, prepare_estimate_requests
    from anthropochrony.gen.performance import measured_generate
    from anthropochrony.parse import parse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    args.output = str(Path(args.output_dir) / "estimates.jsonl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    import yaml
    from transformers import AutoTokenizer

    prompts = read_jsonl(args.prompts)
    config = yaml.safe_load(Path(args.config).read_text())
    started = time.monotonic()
    merge_record, model_label = None, args.model
    if args.adapter:
        model_label = args.adapter
        # Merged bf16 weights only; vLLM never loads the adapter.
        from model_prep import prepare_merged

        args.model, merge_record = prepare_merged(args.model, args.adapter, "/tmp/merged")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    requests, exhausted = prepare_estimate_requests(
        tokenizer, prompts, config["engine"]["max_model_len"]
    )
    if requests:
        engine = VLLMEngine(args.model, config["engine"])
        generated, performance, samples = measured_generate(engine, requests, greedy=True)
    else:
        generated, samples = [], []
        performance = {"requests": 0, "gpu": {"samples": 0}, "reason": "no_in_context_prompts"}
    by_id = {row["request_id"]: row for row in generated}
    if len(by_id) != len(generated) or set(by_id) != {row["request_id"] for row in requests}:
        raise ValueError("Estimate generation outputs do not match the requested IDs")
    by_id.update(exhausted)
    request_limits = {row["request_id"]: row["max_tokens"] for row in requests}

    def generate(inputs):
        return [by_id[json.loads(value)["est_id"]]["text"] for value in inputs]

    result, fingerprint = evaluate(
        prompts,
        generate,
        Path(args.output).parent / "auditkit",
        model_sha=args.model_sha,
        code_sha=args.code_sha,
    )
    rows = []
    for prompt in prompts:
        generation = by_id[prompt["est_id"]]
        rows.append(
            prompt
            | {
                "model_id": model_label,
                "model_sha": args.model_sha,
                "raw_text": generation["text"],
                "finish_reason": generation["finish_reason"],
                "model_called": prompt["est_id"] not in exhausted,
                "prompt_tokens_total": generation["prompt_tokens_total"],
                "effective_max_tokens": request_limits.get(prompt["est_id"], 0),
                "parsed": parse(generation["text"]).as_dict(),
                "auditkit_fingerprint": fingerprint,
            }
        )
    write_jsonl(args.output, rows, inputs=[args.prompts], config=config)
    output = Path(args.output)
    performance["process_elapsed_s"] = time.monotonic() - started
    performance["engine"] = config["engine"]
    performance["context_exhausted_prompts"] = len(exhausted)
    if merge_record:
        write_json(output.with_name("merge.json"), merge_record)
    write_json(output.with_suffix(".gpu.json"), samples)
    write_json(output.with_suffix(".performance.json"), performance)


if __name__ == "__main__":
    main()
