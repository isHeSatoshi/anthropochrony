"""Fill restricted task statements back into released rows.

Rows from LeetCodeDataset, CodeContests and TACO ship without statement text. Their
prompts carry the marker ``{{TASK_TURN}}`` where the statement belongs. This script reads the
pinned public sources (``configs/pins.yaml``), rebuilds each task with the same loader used for
the study, and writes rows with complete ``messages``.

Usage (from the repository root, with the pinned datasets downloadable):

    uv run python scripts/rebuild_messages.py INPUT.jsonl OUTPUT.jsonl [--local-data DIR]
"""

import argparse
import json
from pathlib import Path

from anthropochrony.release import TASK_MARKER, rebuild_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local-data", type=Path)
    args = parser.parse_args()
    pins = json.loads((ROOT / "configs/pins.yaml").read_text())
    cache = {}
    with args.input.open() as source, args.output.open("w") as target:
        for line in source:
            row = json.loads(line)
            if row.get("restricted_statement") and row.get("messages"):
                key = row["task_id"]
                if key not in cache:
                    cache[key] = rebuild_task(
                        {
                            "task_id": key,
                            "restricted_statement": True,
                            "source": row["source"],
                        },
                        pins,
                        args.local_data,
                    )
                turn = cache[key]["task_turn"]
                for message in row["messages"]:
                    message["content"] = message["content"].replace(TASK_MARKER, turn)
                row["restricted_statement"] = False
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
