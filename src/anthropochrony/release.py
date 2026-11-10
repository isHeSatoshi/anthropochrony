"""License-aware release staging. No publication side effect is implicit."""

import copy
from pathlib import Path

from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
from anthropochrony.gates import GateError

RESTRICTED_SOURCES = {"newfacade/LeetCodeDataset", "deepmind/code_contests", "BAAI/TACO"}


def redact_record(record, task_index):
    result = copy.deepcopy(record)
    task = task_index[result["task_id"]]
    if task["source"] not in RESTRICTED_SOURCES:
        return result
    # Release only IDs/provenance, not paraphrases, completions, prompt hashes of statements,
    # tests, reference code, human answers or embedded judge/provider transcripts.
    safe = {
        key: result[key]
        for key in (
            "task_id",
            "family",
            "source",
            "source_revision",
            "license",
            "split",
            "card_id",
            "variant",
            "estimator",
            "frame",
            "template_id",
            "label_version",
            "label_source",
            "label",
            "parsed",
            "T_med_s",
            "S",
            "P",
            "example_id",
        )
        if key in result
    }
    safe.update(
        task_id=task["task_id"],
        source=task["source"],
        source_revision=task["source_revision"],
        restricted_statement=True,
        rebuild={
            "task_id": task["task_id"],
            "source": task["source"],
            "revision": task["source_revision"],
        },
    )
    return safe


def prepare_release(root, approved=False):
    root = Path(root)
    if approved:
        raise GateError(
            "Public release must be performed explicitly after reviewing the staged bundle and licenses"
        )
    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    index = {t["task_id"]: t for t in tasks}
    destination = root / "artifacts/release"
    for filename, source in {
        "tasks.jsonl": root / "data/tasks/tasks.jsonl",
        "sft-m1-train.jsonl": root / "data/sft/m1/train/data.jsonl",
        "eval.jsonl": root / "data/eval/estimates.jsonl",
    }.items():
        if source.exists():
            write_jsonl(
                destination / filename,
                [redact_record(r, index) for r in read_jsonl(source)],
                inputs=[source],
            )
    write_json(
        destination / "release-status.json",
        {
            "private_staging_only": True,
            "publication_approved": False,
            "requires": [
                "G0-G9",
                "license confirmation U6",
                "public-release approval",
                "model/data/log cards",
                "account details removed from public docs",
            ],
        },
    )
    return destination


def rebuild_task(record, pins, local_root=None):
    from anthropochrony.tasks.loaders import load_pool

    if not record.get("restricted_statement"):
        return record
    tasks, _ = load_pool(
        {"datasets": {record["source"]: pins["datasets"][record["source"]]}}, local_root
    )
    for task in tasks:
        if task["task_id"] == record["task_id"]:
            return task
    raise KeyError(record["task_id"])
