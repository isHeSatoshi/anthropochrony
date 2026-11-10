from anthropochrony.artifacts import (
    _atomic_text,
    digest,
    file_hash,
    manifest,
    read_jsonl,
    write_jsonl,
)
from anthropochrony.cli import config
from anthropochrony.gates import GateError
from anthropochrony.prompts import estimate_messages
from anthropochrony.timemodel import SpeedCard

MAX_TRUNCATED_SHARE = 0.02


def collect_preservation(root, paths):
    tasks = {task["task_id"]: task for task in read_jsonl(root / "data/tasks/tasks.jsonl")}
    model_sha = config(root, "pins.yaml")["subject"]["revision"]
    selected, blockers = {}, []
    paths = list(paths)
    existing = [
        root / f"data/eval/human-{suffix}.jsonl"
        for suffix in ("train-val", "raw-extra")
        if (root / f"data/eval/human-{suffix}.jsonl").exists()
    ]
    for path in existing + paths:
        for row in read_jsonl(path):
            if row["frame"] != "human" or row["split"] not in {"train", "val", "raw_extra"}:
                continue
            task = tasks[row["task_id"]]
            assignment = int(digest(task["task_id"])[:8], 16)
            expected = estimate_messages(
                task,
                SpeedCard("REF", row["S"], row["P"]),
                f"h{1 + assignment % 8:02d}",
                f"sc{1 + assignment % 6:02d}",
                "E1",
            )
            if (
                row["model_sha"] != model_sha
                or row["card_id"] != "REF"
                or row["split"] != task["split"]
                or row["messages"] != expected
                or row["estimator"] != "E1"
            ):
                raise ValueError(
                    "Preservation targets must be the base model's matching seeded prompts"
                )
            if row["finish_reason"] == "length":
                blockers.append({"task_id": row["task_id"], "reason": "truncated_human_target"})
                continue
            if (
                row["finish_reason"] != "stop"
                or row.get("model_called", True) is False
                or not row.get("raw_text", "").strip()
            ):
                blockers.append({"task_id": row["task_id"], "reason": "incomplete_human_target"})
                continue
            previous = selected.get(row["task_id"])
            if previous and (
                previous["raw_text"] != row["raw_text"] or previous["messages"] != row["messages"]
            ):
                raise ValueError("Conflicting cached human preservation targets")
            selected[row["task_id"]] = row
    write_jsonl(root / "data/eval/preservation-blockers.jsonl", blockers, inputs=paths)
    unusable = {row["task_id"] for row in blockers if row["task_id"] not in selected}
    truncated_only = all(
        row["reason"] == "truncated_human_target" for row in blockers if row["task_id"] in unusable
    )
    # A capped answer is never a target. Those tasks keep their self-frame examples and lose
    # the human-frame one; anything beyond a small share means the cap itself is wrong.
    if unusable and (not truncated_only or len(unusable) > MAX_TRUNCATED_SHARE * len(tasks)):
        raise GateError(
            "Truncated or incomplete human targets cannot enter SFT; collect complete answers first"
        )
    if not selected:
        raise GateError("No real human preservation answers supplied")
    snapshots = {}
    for cache in existing:
        snapshot = root / "data/eval/preservation-cache" / f"{file_hash(cache)}.jsonl"
        if not snapshot.exists():
            _atomic_text(snapshot, cache.read_text())
            manifest(snapshot, inputs=[cache])
        snapshots[cache] = snapshot
    inputs = list(
        dict.fromkeys([snapshots.get(path, path) for path in paths] + list(snapshots.values()))
    )
    for suffix, splits in (("train-val", {"train", "val"}), ("raw-extra", {"raw_extra"})):
        rows = sorted(
            (row for row in selected.values() if row["split"] in splits),
            key=lambda row: row["task_id"],
        )
        write_jsonl(root / f"data/eval/human-{suffix}.jsonl", rows, inputs=inputs)
    return selected
