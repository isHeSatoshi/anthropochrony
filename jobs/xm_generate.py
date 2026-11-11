"""GPU-only generation for the cross-model arm. No generated code executes here.

Requests carry pre-rendered prompt token IDs. Aya Expanse ends an assistant turn with
`<|END_OF_TURN_TOKEN|>` (255001) alone, unlike the tiny-aya pair that the pinned engine hardcodes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EOT = 255001


def localize(source, destination="/tmp/aya-expanse-8b"):
    """Copy weights from the FUSE bucket mount to local disk with sequential reads.

    vLLM and transformers mmap the safetensors shards; on the FUSE mount that random access
    took 16 minutes per 4 GB shard in the first attempt, while a sequential copy runs near 1 GB/s.
    """
    import shutil

    if str(source).startswith("/out/"):
        free = shutil.disk_usage("/tmp").free
        print(f"/tmp free: {free / 2**30:.0f} GiB", flush=True)
        if free < 36 * 2**30:
            raise RuntimeError(f"Only {free / 2**30:.0f} GiB free in /tmp; need 36")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return destination
    return source


def make_engine():
    from anthropochrony.gen.engine import VLLMEngine
    from anthropochrony.timemodel import output_token_count

    class AyaEngine(VLLMEngine):
        def generate(self, requests, *, greedy=False, max_tokens=None):
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            params = [
                SamplingParams(
                    temperature=0.0 if greedy else 0.1,
                    top_p=1.0 if greedy else 0.95,
                    seed=r["seed"],
                    max_tokens=max_tokens or r["max_tokens"],
                    stop_token_ids=[EOT],
                    n=1,
                )
                for r in requests
            ]
            started = time.monotonic()
            outputs = self.llm.generate(
                [TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in requests], params
            )
            elapsed = time.monotonic() - started
            results = []
            for request, output in zip(requests, outputs, strict=True):
                generated = output.outputs[0]
                results.append(
                    {
                        "request_id": request["request_id"],
                        "text": generated.text,
                        "token_ids": list(generated.token_ids),
                        "o_tokens": output_token_count(
                            generated.token_ids, generated.stop_reason, stop_ids=(EOT,)
                        ),
                        "prompt_tokens_total": len(output.prompt_token_ids),
                        "finish_reason": generated.finish_reason,
                        "stop_reason": generated.stop_reason,
                    }
                )
            total = sum(row["o_tokens"] for row in results)
            self.last_performance = {
                "elapsed_s": elapsed,
                "output_tokens": total,
                "tokens_per_second": total / elapsed if elapsed else None,
                "usd_per_million_output_tokens": elapsed * 1.8 / 3600 / total * 1e6
                if total
                else None,
            }
            return results

    return AyaEngine


def main():
    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.gen.performance import measured_generate

    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--base", default="/out/aya-expanse-8b")
    parser.add_argument("--adapter")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    args.base = localize(args.base)
    model, merge_record = args.base, None
    if args.adapter:
        from model_prep import prepare_merged

        model, merge_record = prepare_merged(args.base, args.adapter, "/tmp/merged")
        merge_record["merge_s"] = time.monotonic() - started
        if args.base.startswith("/tmp/"):
            import shutil

            shutil.rmtree(args.base)
    engine = make_engine()(model, config["engine"])
    requests = read_jsonl(args.requests)
    rows, performance, samples = measured_generate(engine, requests, greedy=args.greedy)
    performance["engine"] = config["engine"]
    performance["process_elapsed_s"] = time.monotonic() - started
    write_jsonl(output / "generation.jsonl", rows, inputs=[args.requests], config=config)
    write_json(output / "generation.performance.json", performance)
    write_json(output / "merge-evidence.json", merge_record or {"adapter": None})


if __name__ == "__main__":
    main()
