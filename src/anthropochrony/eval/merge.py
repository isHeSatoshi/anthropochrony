"""Combine measured model estimates and baseline predictions into one analysis table."""

from pathlib import Path

from anthropochrony.artifacts import read_jsonl, write_jsonl


def merge_estimates(root):
    root = Path(root)
    paths = sorted(root.glob("data/eval/*-estimates.jsonl"))
    paths = [p for p in paths if p.name != "common-base-estimates.jsonl" and "lr-" not in p.name]
    paths += [
        p
        for p in (root / "data/eval/statistical.jsonl", root / "data/eval/teacher.jsonl")
        if p.exists()
    ]
    rows, seen = [], set()
    for path in paths:
        for row in read_jsonl(path):
            key = row.get("est_id") or (
                row["task_id"],
                row["estimator"],
                row["card_id"],
                row.get("frame"),
            )
            if key in seen:
                raise ValueError(f"Duplicate estimate row {key} in {path.name}")
            seen.add(key)
            rows.append(row)
    write_jsonl(root / "data/eval/estimates.jsonl", rows, inputs=paths)
    return len(rows)
