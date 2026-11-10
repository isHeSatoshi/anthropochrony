"""CPU check that merging the LoRA delta preserves the model, on the G6 validation prompts.

The G6 GPU job compares greedy text from vLLM-on-merged-weights with HF+PEFT. Greedy text flips
whenever two digit tokens are nearly tied, so this script measures the logits directly: for each
prompt it teacher-forces the HF+PEFT answer and compares next-token logits from (a) unmerged
PEFT in float32, (b) the float32 merge, and (c) the bf16-rounded merge that vLLM serves.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def forward(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="m1-s0")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--work", default="/home/adi/.cache/anthropochrony-tmp/gguf")
    args = parser.parse_args()
    pins = yaml.safe_load((ROOT / "configs/pins.yaml").read_text())["subject"]
    base = snapshot_download(
        pins["repo"], revision=pins["revision"], cache_dir=Path(args.work) / "hf"
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    directory = ROOT / f"data/eval/jobs/{args.run}-g6"
    prompts = {
        json.loads(line)["est_id"]: json.loads(line)
        for line in (directory / "prompts.jsonl").read_text().splitlines()
    }
    hf = {
        json.loads(line)["request_id"]: json.loads(line)["text"]
        for line in (directory / "hf-peft.jsonl").read_text().splitlines()
    }
    flags = {
        c["request_id"]: c
        for c in json.loads((directory / "g6-equivalence.json").read_text())["comparisons"]
    }
    wrong = [k for k, c in flags.items() if not c["numeric_match"]]
    right = [k for k, c in flags.items() if c["numeric_match"]]
    chosen = (wrong + right)[: args.limit]
    cases = []
    for key in chosen:
        messages = prompts[key]["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt = (
            prompt["input_ids"] if isinstance(prompt, dict) or hasattr(prompt, "keys") else prompt
        )
        answer = tokenizer.encode(hf[key], add_special_tokens=False)
        cases.append((key, list(prompt), answer))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    adapter = Path(args.work) / f"adapter-{args.run}"
    adapter.mkdir(exist_ok=True)
    HfApi().download_bucket_files(
        "isHeSatoshi/anthropochrony-runs",
        [
            (f"adapters/{args.run}/{name}", adapter / name)
            for name in ("adapter_model.safetensors", "adapter_config.json")
        ],
        raise_on_missing_files=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    reference = {}
    for key, prompt, answer in cases:
        reference[key] = forward(model, prompt + answer)[len(prompt) - 1 : -1].clone()
    model = model.merge_and_unload()
    merged32 = {
        key: forward(model, prompt + answer)[len(prompt) - 1 : -1].clone()
        for key, prompt, answer in cases
    }
    model = model.to(torch.bfloat16)
    merged16 = {
        key: forward(model, prompt + answer)[len(prompt) - 1 : -1].clone()
        for key, prompt, answer in cases
    }
    rows = []
    for key, _prompt, answer in cases:
        ref = reference[key]
        top2 = ref.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        digit = torch.tensor([tokenizer.decode([t]).strip().isdigit() for t in answer])
        rows.append(
            {
                "request_id": key,
                "numeric_mismatch_in_gpu_job": key in wrong,
                "fp32_merge_max_abs_logit_diff": float((merged32[key] - ref).abs().max()),
                "bf16_merge_max_abs_logit_diff": float((merged16[key] - ref).abs().max()),
                "bf16_merge_argmax_agreement": float(
                    (merged16[key].argmax(-1) == ref.argmax(-1)).float().mean()
                ),
                "digit_positions": int(digit.sum()),
                "min_top2_margin_at_digits": float(margin[digit].min()) if digit.any() else None,
                "digit_margin_below_0_5": int((margin[digit] < 0.5).sum()) if digit.any() else 0,
            }
        )
    out = ROOT / f"results/g6-cpu-parity-{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "rows": rows}, indent=2))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
