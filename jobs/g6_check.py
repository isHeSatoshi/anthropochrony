"""G6 equivalence: unmerged HF+PEFT greedy outputs vs merged-weight vLLM greedy outputs."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
STOP = [6, 261001]


def hf_peft(args):
    import torch
    from model_prep import ensure_peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_peft()
    from peft import PeftModel

    from anthropochrony.artifacts import read_jsonl, write_jsonl
    from anthropochrony.gen.engine import prepare_estimate_requests

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    requests, _ = prepare_estimate_requests(tokenizer, read_jsonl(args.prompts), 8192)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    rows = []
    for request in requests:
        ids = torch.tensor([request["prompt_token_ids"]], device="cuda")
        with torch.no_grad():
            output = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=request["max_tokens"],
                eos_token_id=STOP,
            )[0, ids.shape[1] :].tolist()
        while output and output[-1] in STOP:
            output.pop()
        rows.append(
            {
                "request_id": request["request_id"],
                "text": tokenizer.decode(output, skip_special_tokens=True),
            }
        )
    write_jsonl(Path(args.output_dir) / "hf-peft.jsonl", rows, inputs=[args.prompts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--base", default="/out/base")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf", action="store_true")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.hf:
        hf_peft(args)
        return
    import shutil

    import yaml
    from model_prep import prepare_merged
    from transformers import AutoTokenizer

    from anthropochrony.artifacts import read_jsonl, write_json
    from anthropochrony.gen.engine import VLLMEngine, prepare_estimate_requests
    from anthropochrony.gen.performance import measured_generate
    from anthropochrony.parse import parse

    started = time.monotonic()
    # A separate process guarantees the HF model's GPU memory is released before vLLM starts.
    subprocess.run([sys.executable, __file__, "--hf", *sys.argv[1:]], check=True)
    hf_seconds = time.monotonic() - started
    config = yaml.safe_load(Path(args.config).read_text())
    merged, record = prepare_merged(args.base, args.adapter, "/tmp/merged")
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    requests, _ = prepare_estimate_requests(tokenizer, read_jsonl(args.prompts), 8192)
    engine = VLLMEngine(merged, config["engine"])
    generated, performance, _ = measured_generate(engine, requests, greedy=True)
    shutil.rmtree(merged, ignore_errors=True)
    hf_rows = read_jsonl(Path(args.output_dir) / "hf-peft.jsonl")
    hf = {row["request_id"]: row["text"] for row in hf_rows}
    comparisons = []
    for row in generated:
        a, b = parse(hf[row["request_id"]]).point_s, parse(row["text"]).point_s
        same = (a is None and b is None) or (
            a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a))
        )
        comparisons.append(
            {
                "request_id": row["request_id"],
                "hf_peft_point_s": a,
                "merged_vllm_point_s": b,
                "numeric_match": same,
                "text_identical": hf[row["request_id"]] == row["text"],
            }
        )
    write_json(
        Path(args.output_dir) / "g6-equivalence.json",
        {
            "prompts": len(comparisons),
            "numeric_mismatches": sum(not row["numeric_match"] for row in comparisons),
            "identical_texts": sum(row["text_identical"] for row in comparisons),
            "merge": record,
            "hf_peft_seconds": hf_seconds,
            "performance": performance,
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    main()
