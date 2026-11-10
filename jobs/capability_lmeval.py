"""AuditKIT lm-eval capability check (IFEval, ARC-Challenge) on base or merged-adapter weights."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "jobs"))
LMEVAL = "lm-eval[ifeval]==0.4.13"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/out/base")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The pinned image ships vLLM but not the harness. IFEval also needs punkt for its checkers.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", LMEVAL, "accelerate"], check=True
    )
    import nltk

    for resource in ("punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)
    model, merge_record = args.base, None
    if args.adapter:
        from model_prep import prepare_merged

        model, merge_record = prepare_merged(args.base, args.adapter, "/tmp/merged")
    import auditkit as ak

    from anthropochrony.artifacts import write_json

    results = {}
    # ARC scores log-likelihoods, so it runs in float32 (AuditKIT issue #26: bf16 log_softmax).
    # IFEval generates up to 1,280 tokens per prompt with the HF backend, which is the slow
    # part (the full 541 prompts did not finish in 50 min at batch 2), so --limit applies to
    # IFEval only. ARC-Challenge always runs in full.
    for task, dtype, chat, limit in (
        ("ifeval", "bfloat16", True, args.limit),
        ("arc_challenge", "float32", False, None),
    ):
        result = ak.run_lmeval(
            task,
            model=f"hf:{model}",
            dtype=dtype,
            apply_chat_template=chat,
            config=ak.RunConfig(limit=limit, batch_size=16),
        )
        results[task] = {
            "headline": result.headline,
            "stats": {
                name: stat.__dict__ if hasattr(stat, "__dict__") else str(stat)
                for name, stat in result.stats.items()
            },
            "fingerprint": result.fingerprint,
        }
    record = {
        "adapter": args.adapter,
        "limit": args.limit,
        "results": results,
        "lm_eval": LMEVAL,
        "merge": merge_record,
    }
    write_json(out / "lmeval.json", json.loads(json.dumps(record, default=str)))


if __name__ == "__main__":
    main()
