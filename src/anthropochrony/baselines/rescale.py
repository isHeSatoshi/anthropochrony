"""E4: rescale the base model's reference-card estimate by the decode-speed ratio (MetaTime rule)."""

import math
from collections import defaultdict
from pathlib import Path

from anthropochrony.artifacts import digest, read_jsonl, write_jsonl

REFERENCE_DECODE_SPEED = 96.0


def derive_e4(root, splits=("val", "test", "test_ood")):
    root = Path(root)
    rows = []
    paths = []
    for split in splits:
        path = root / f"data/eval/base-{split}-estimates.jsonl"
        paths.append(path)
        rows += [
            r
            for r in read_jsonl(path)
            if r["estimator"] == "E1" and r["frame"] == "self" and r["split"] == split
        ]
    reference = defaultdict(list)
    for row in rows:
        point = row["parsed"]["point_s"]
        if row["card_id"] == "REF" and point:
            reference[row["task_id"]].append(point)
    derived = []
    for row in rows:
        points = reference.get(row["task_id"])
        if not points:
            # Without a parsed reference-card answer there is nothing to rescale.
            point = None
        else:
            point = 10 ** (sum(math.log10(p) for p in points) / len(points))
        scale = REFERENCE_DECODE_SPEED / row["S"]
        parsed = dict(row["parsed"])
        parsed |= {
            "point_s": point * scale if point else None,
            "lo_s": None,
            "hi_s": None,
            "parse_method": "derived",
        }
        keep = {
            k: v
            for k, v in row.items()
            if k not in {"messages", "raw_text", "auditkit_fingerprint"}
        }
        keep |= {
            "estimator": "E4",
            "prompt_estimator": "E1",
            "parsed": parsed,
            "model_called": False,
            "raw_text": "",
        }
        keep["est_id"] = digest({"e4": row["est_id"]})
        derived.append(keep)
    write_jsonl(root / "data/eval/e4-derived-estimates.jsonl", derived, inputs=paths)
    return len(derived)
