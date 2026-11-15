"""CuratorKIT stage: quality-gate the measured training pool before AlignTune sees it.

Each training-split pool prompt becomes one sample: the prompt as the instruction and the
model's own first measured response as the output, with the three measured response
lengths attached. CuratorKIT then runs its schema gate, exact and MinHash near-duplicate
removal, and its secrets gate. One rule is ours and runs as CuratorKIT's preprocessing
hook: a prompt is dropped when any of its measured runs stopped at the generation cap,
because its label is then a lower bound rather than a measurement. Text is never rewritten
(`clean=False`), since the labels were measured on the exact prompt.

Writes the passed ids, every rejected id with its reason, and the per-stage counts.
"""

import argparse
import json
from pathlib import Path

from common import read_jsonl
from engine import MAX_NEW_TOKENS

MINHASH_THRESHOLD = 0.7
MINHASH_SEED = 20261115


def samples(pool_path, labels_path, split):
    runs, first_text = {}, {}
    for row in read_jsonl(labels_path):
        if row["kind"] != "label":
            continue
        runs.setdefault(row["id"], []).append(row)
        if row.get("text") is not None:
            first_text[row["id"]] = row["text"]
    out = []
    for row in read_jsonl(pool_path):
        if row["split"] != split or row["id"] not in runs:
            continue
        measured = runs[row["id"]]
        out.append(
            {
                "pool_id": row["id"],
                "source": row["source"],
                "instruction": row["prompt"],
                "output": first_text.get(row["id"]) or "",
                "n_out_runs": [run["n_out"] for run in measured],
                "capped_runs": sum(
                    run.get("finish") == "length" or run["n_out"] >= MAX_NEW_TOKENS
                    for run in measured
                ),
            }
        )
    return out


def _pool_id(rejected):
    """The pool id of a rejected sample, wherever this CuratorKIT version keeps the sample."""
    for name in ("sample", "original", "data_sample"):
        inner = getattr(rejected, name, None)
        if inner is not None and "pool_id" in (getattr(inner, "metadata", None) or {}):
            return inner.metadata["pool_id"]
    meta = getattr(rejected, "metadata", None) or {}
    return meta.get("pool_id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    from curatorkit.curator import Curator, CuratorConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = samples(args.pool, args.labels, args.split)
    source = out / "input.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    ours = {}

    def preprocess(row):
        if row.get("capped_runs"):
            ours[row["pool_id"]] = "label_cut_at_generation_cap"
            return None
        if not row.get("output", "").strip():
            ours[row["pool_id"]] = "empty_measured_response"
            return None
        return row

    config = CuratorConfig(
        dataset=str(source),
        format="alpaca",
        min_tokens=1,
        max_tokens=100_000,
        clean=False,
        dedup="minhash",
        minhash_threshold=MINHASH_THRESHOLD,
        minhash_seed=MINHASH_SEED,
        secrets_gate=True,
        preprocessing_fn=preprocess,
        output_dir=str(out / "curatorkit"),
        export_formats=["messages"],
        write_hf_readme=True,
        name="anthropochrony-metatime-pool",
    )
    result = Curator(config).run()
    passed = sorted(sample.metadata["pool_id"] for sample in result.passed)
    rejected = []
    for sample in result.rejected:
        rid = _pool_id(sample)
        rejected.append({"id": rid, "reason": ours.get(rid) or str(sample.rejection_reason)})
    seen = {row["id"] for row in rejected} | set(passed)
    rejected += [{"id": rid, "reason": why} for rid, why in ours.items() if rid not in seen]
    reasons = {}
    for row in rejected:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    report = {
        "tool": "CuratorKIT",
        "split": args.split,
        "input": len(rows),
        "passed": len(passed),
        "rejected": len(rejected),
        "rejected_by_reason": dict(sorted(reasons.items())),
        "stage_counts": result.stage_counts,
        "settings": {
            "dedup": "exact + minhash",
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_seed": MINHASH_SEED,
            "secrets_gate": True,
            "clean": False,
            "generation_cap": MAX_NEW_TOKENS,
        },
        "passed_by_source": {
            name: sum(row["source"] == name and row["pool_id"] in set(passed) for row in rows)
            for name in sorted({row["source"] for row in rows})
        },
    }
    (out / "passed.json").write_text(json.dumps(passed) + "\n")
    (out / "rejected.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rejected))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
