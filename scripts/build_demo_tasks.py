"""Pick the demo's verified tasks: PROJ-8 plus test tasks the base model finishes quickly.

Only tasks with open licenses and no restricted statements are listed, so the bundle can ship
with the Space.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = {"newfacade/LeetCodeDataset", "deepmind/code_contests", "BAAI/TACO"}


def main():
    tasks = [
        json.loads(line) for line in (ROOT / "data/tasks/tasks.jsonl").read_text().splitlines()
    ]
    labels = pd.read_parquet(ROOT / "data/labels/base.parquet")
    ref = labels[(labels.card_id == "REF") & (labels.split == "test")]
    quick = ref[(ref.T_med_s.between(3, 40)) & (ref.PFP >= 0.1)].set_index("task_id")
    chosen = []
    for family in ("FUNC", "LIB", "ALGO_FN"):
        pool = [
            t
            for t in tasks
            if t["task_id"] in quick.index
            and t["family"] == family
            and t["source"] not in RESTRICTED
            and t["gt"].get("passes")
        ]
        pool.sort(key=lambda t: t["task_id"])
        chosen += pool[:7]
    showcase = [
        json.loads(line) for line in (ROOT / "data/tasks/proj8.jsonl").read_text().splitlines()
    ]
    out = ROOT / "data/demo/tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t) + "\n" for t in showcase + chosen))
    print(len(showcase), "showcase +", len(chosen), "test tasks ->", out)


if __name__ == "__main__":
    main()
