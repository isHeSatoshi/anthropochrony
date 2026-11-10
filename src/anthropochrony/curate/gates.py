import json
from collections import Counter
from pathlib import Path

from anthropochrony.artifacts import file_hash, read_jsonl
from anthropochrony.curate.sft import check_matched_variants
from anthropochrony.gates import record_gate, require_gate
from anthropochrony.train.sft import TARGETS


def sft_gate(root, dry_run_summary):
    root, dry_run_summary = Path(root), Path(dry_run_summary)
    require_gate(root, "G4")
    report = json.loads((root / "data/sft/matched-summary.json").read_text())
    summary = json.loads(dry_run_summary.read_text())
    hardware = json.loads((dry_run_summary.parent / "hardware.json").read_text())
    variants, training_hashes = {}, set()
    counts_match = True
    for name in ("m1", "m2", "m3"):
        directory = root / "data/sft" / name
        rows = read_jsonl(directory / "train/data.jsonl") + read_jsonl(
            directory / "validation/data.jsonl"
        )
        variants[name] = rows
        training_hashes.add(file_hash(directory / "train/data.jsonl"))
        counts_match &= (
            len(rows) == len({r["example_id"] for r in rows})
            and sorted(r["example_id"] for r in rows) == report["expected_example_ids"][name]
            and dict(Counter(r["split"] for r in rows))
            == json.loads((directory / "counts.json").read_text())
        )
    try:
        check_matched_variants(variants["m1"], variants["m2"])
        matched = True
    except ValueError:
        matched = False
    second = {row["example_id"]: row for row in variants["m2"]}
    self_rows = [row for row in variants["m1"] if row["frame"] == "self"]
    changed = sum(
        row["example_id"] in second
        and row["messages"][-1]["content"] != second[row["example_id"]]["messages"][-1]["content"]
        for row in self_rows
    )
    collator, mask, lora = (
        summary.get(key, {}) for key in ("actual_collator", "assistant_mask", "lora")
    )
    record_gate(
        root,
        "G5",
        {
            "counts_and_retained_ids_match": bool(counts_match) and all(variants.values()),
            "m1_m2_prompts_and_human_targets_match": matched,
            # Trivial tasks can coincide exactly (two HumanEval/23 cards), so allow 0.1%.
            "self_assistant_targets_differ": bool(self_rows) and changed >= 0.999 * len(self_rows),
            "one_real_optimizer_step": summary.get("dry_run") is True
            and summary.get("training", {}).get("steps") == 1,
            "dry_run_uses_current_data": summary.get("train_data_sha256") in training_hashes,
            "assistant_only_mask": mask.get("rendering_identical") is True
            and mask.get("masked_tokens", 0) > 2,
            "actual_training_collator_mask": collator.get("assistant_only") is True
            and collator.get("padding_masked") is True
            and collator.get("inspected_sequences", 0) > 0,
            "approved_gpu": hardware.get("cuda") is True and "L40S" in hardware.get("gpu", ""),
            "lora_targets_verified": lora.get("targets") == sorted(TARGETS)
            and lora.get("trainable_parameters", 0) > 0,
            "shared_sequence_limit": summary.get("config", {}).get("max_seq_length")
            == report["max_seq_length"],
        },
        evidence={
            "dry_run_summary": str(dry_run_summary),
            "dry_run_sha256": file_hash(dry_run_summary),
            "self_examples": len(self_rows),
            "different_self_targets": changed,
            "identical_self_targets": sorted(
                row["example_id"]
                for row in self_rows
                if row["example_id"] in second
                and row["messages"][-1]["content"]
                == second[row["example_id"]]["messages"][-1]["content"]
            ),
            "counts": {name: len(rows) for name, rows in variants.items()},
        },
    )
