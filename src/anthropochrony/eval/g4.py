"""Gate G4: base E0/E1 parse rates, parser-versus-hand audit and the produced H1 table."""

import json
import random
from pathlib import Path

from anthropochrony.artifacts import digest, file_hash, read_jsonl, write_jsonl
from anthropochrony.gates import GateError, record_gate

AUDIT_SIZE = 60


def base_self_rows(root, estimators=("E0", "E1")):
    rows = []
    for split in ("val", "test"):
        path = Path(root) / f"data/eval/base-{split}-estimates.jsonl"
        if not path.exists():
            raise GateError(f"Missing base estimates for {split}")
        rows += [
            row
            for row in read_jsonl(path)
            if row["frame"] == "self" and row["card_id"] == "REF" and row["estimator"] in estimators
        ]
    return rows


def sample_parser_audit(root, seed=20261110):
    """All parse failures (up to a third) plus seeded random successes, for manual reading."""
    root = Path(root)
    rows = base_self_rows(root)
    failures = [r for r in rows if r["parsed"]["point_s"] is None]
    successes = [r for r in rows if r["parsed"]["point_s"] is not None]
    random.Random(seed).shuffle(failures)
    random.Random(seed + 1).shuffle(successes)
    chosen = failures[: AUDIT_SIZE // 3]
    chosen += successes[: AUDIT_SIZE - len(chosen)]
    sample = [
        {
            "est_id": r["est_id"],
            "estimator": r["estimator"],
            "split": r["split"],
            "raw_text": r["raw_text"],
            "parser_point_s": r["parsed"]["point_s"],
        }
        for r in sorted(chosen, key=lambda r: r["est_id"])
    ]
    write_jsonl(root / "data/eval/parser-audit-sample.jsonl", sample, inputs=[])
    return sample


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def g4_gate(root):
    root = Path(root)
    rows = base_self_rows(root)
    rates = {}
    for estimator in ("E0", "E1"):
        group = [r for r in rows if r["estimator"] == estimator]
        rates[estimator] = (
            sum(r["parsed"]["point_s"] is not None for r in group) / len(group) if group else 0.0
        )
    sample_path = root / "data/eval/parser-audit-sample.jsonl"
    hand_path = root / "data/eval/parser-audit.jsonl"
    if not sample_path.exists() or not hand_path.exists():
        raise GateError("Sample the parser audit and record manual readings first")
    sample = {r["est_id"]: r for r in read_jsonl(sample_path)}
    hand = {r["est_id"]: r for r in read_jsonl(hand_path)}
    if set(hand) != set(sample) or len(sample) < AUDIT_SIZE:
        raise GateError("Every sampled parser-audit row needs one manual reading")
    agreement = sum(
        _same(sample[i]["parser_point_s"], hand[i]["hand_point_s"]) for i in sample
    ) / len(sample)
    h1 = root / "results/hypotheses.json"
    h1_ok = False
    if h1.exists():
        report = json.loads(h1.read_text())
        h1_ok = report.get("H1", {}).get("status") == "measured"
    record_gate(
        root,
        "G4",
        {
            "e0_parse_rate_at_least_90pct": rates["E0"] >= 0.9,
            "e1_parse_rate_at_least_90pct": rates["E1"] >= 0.9,
            "parser_hand_audit_agreement_at_least_90pct": agreement >= 0.9,
            "h1_table_produced": h1_ok,
        },
        evidence={
            "parse_rates": rates,
            "audit_agreement": agreement,
            "audit_rows": len(sample),
            "audit_sha256": digest([hand[i] for i in sorted(hand)]),
            "audit_file_sha256": file_hash(hand_path),
            "auditor": "the research agent, reading raw outputs without parser output visible "
            "in the readings file",
        },
    )
    return rates, agreement
