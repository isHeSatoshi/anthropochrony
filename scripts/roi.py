"""T7: closed-API token billing versus self-hosting the fine-tuned model, from measured throughput."""

import json
from pathlib import Path

from anthropochrony.artifacts import write_json
from anthropochrony.eval.reporting import roi

ROOT = Path(__file__).resolve().parents[1]
GPU_USD_PER_HOUR = 1.80
# OpenAI GPT-5 mini list price, read from developers.openai.com/api/docs/models/gpt-5-mini on 2026-11-10.
API = {
    "model": "gpt-5-mini",
    "input": 0.25,
    "output": 2.00,
    "source": "https://developers.openai.com/api/docs/models/gpt-5-mini",
    "date": "2026-11-10",
}


def main():
    performance = json.loads(
        (ROOT / "data/eval/jobs/m1-s0-final/estimates.performance.json").read_text()
    )
    requests = performance["requests"]
    input_tokens = performance["prompt_tokens"] / requests
    output_tokens = performance["output_tokens"] / requests
    per_second = performance["output_tokens"] / performance["elapsed_s"]
    rows = []
    for volume in (10_000, 100_000, 1_000_000, 10_000_000):
        entry = roi(
            volume=volume,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=per_second,
            gpu_usd_per_hour=GPU_USD_PER_HOUR,
            api_input_usd_per_million=API["input"],
            api_output_usd_per_million=API["output"],
            price_source=API["source"],
            price_date=API["date"],
        )
        rows.append(entry)
    api_per_estimate = rows[0]["api_usd"] / rows[0]["monthly_estimates"]
    month_hours = 730
    result = {
        "measured": {
            "requests": requests,
            "mean_input_tokens": input_tokens,
            "mean_output_tokens": output_tokens,
            "generation_elapsed_s": performance["elapsed_s"],
            "output_tokens_per_second": per_second,
            "requests_per_second": requests / performance["elapsed_s"],
            "mean_gpu_utilization_pct": performance["gpu"]["mean_gpu_utilization_pct"],
            "gpu": "HF Jobs l40sx1",
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        },
        "api": API,
        "volumes": rows,
        "api_usd_per_estimate": api_per_estimate,
        "self_hosted_usd_per_estimate_saturated": rows[0]["self_hosted_usd"]
        / rows[0]["monthly_estimates"],
        "always_on_gpu_usd_per_month": GPU_USD_PER_HOUR * month_hours,
        "break_even_estimates_per_month_always_on": GPU_USD_PER_HOUR * month_hours / api_per_estimate,
        "caveats": [
            "GPU hours assume a saturated batch of 6,216 requests (99% mean GPU utilization); interactive traffic uses the GPU less well.",
            "The API price excludes any hidden reasoning tokens, so it is a lower bound for a reasoning model.",
            "The API model's accuracy at this task was not measured; this table compares cost only.",
            "Local CPU deployment (llama.cpp Q4_K_M) has no per-estimate bill but took about 15 s per estimate on an 8-thread desktop CPU (median 45 s for three estimates, X10).",
        ],
    }
    write_json(ROOT / "results/roi.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
