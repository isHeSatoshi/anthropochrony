import json
from pathlib import Path

from curatorkit.curator import Curator, CuratorConfig

from anthropochrony.artifacts import write_json, write_jsonl


def curate(rows, directory, *, kind="tasks", rejected_ids=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rejected_ids = rejected_ids or {}
    source = []
    for row in rows:
        if kind == "sft":
            instruction = json.dumps(row["messages"][:-1], ensure_ascii=False)
            output = row["messages"][-1]["content"]
        else:
            instruction, output = row["task_turn"], row["reference_solution"]
        source.append(
            {"instruction": instruction, "output": output, "record": row, "task_id": row["task_id"]}
        )
    input_path = directory / "input.jsonl"
    write_jsonl(input_path, source)

    def preprocess(row):
        return None if row.get("task_id") in rejected_ids else row

    config = CuratorConfig(
        dataset=str(input_path),
        format="alpaca",
        min_tokens=1,
        max_tokens=100000,
        clean=False,
        dedup="exact" if kind == "sft" else "minhash",
        minhash_threshold=0.7,
        minhash_seed=20261109,
        preprocessing_fn=preprocess,
        output_dir=str(directory / "curatorkit"),
        export_formats=["messages"],
        write_hf_readme=True,
        name=f"anthropochrony-{kind}",
    )
    result = Curator(config).run()
    passed = [sample.metadata["record"] for sample in result.passed]
    rejected = [
        {
            "task_id": sample.metadata.get("task_id"),
            "reason": rejected_ids.get(sample.metadata.get("task_id"), sample.rejection_reason),
        }
        for sample in result.rejected
    ]
    write_jsonl(directory / "passed.jsonl", passed, inputs=[input_path])
    write_jsonl(directory / "rejected.jsonl", rejected, inputs=[input_path])
    write_json(
        directory / "report.json",
        {
            "tool": "CuratorKIT",
            "version": "0.1.23",
            "kind": kind,
            "stage_counts": result.stage_counts,
            "passed": len(passed),
            "rejected": len(rejected),
            "clean": False,
            "dedup": config.dedup,
        },
        inputs=[input_path],
    )
    return passed, rejected
