"""Materialize the public Hub release into the repository's local `data/` layout.

The research artifacts (tasks, estimates, traces) are published as public Hugging Face
datasets. This script downloads them at pinned revisions and places them where the
pipeline expects them, so that a fresh clone can rebuild every reported number, run
the Gradio workbench, and reproduce the paper without private credentials.

    uv run python scripts/fetch_release.py            # tasks + estimates + traces
    uv run python scripts/fetch_release.py --include-x10

Downloaded data is never committed: `data/` is gitignored by design.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
USER = "isHeSatoshi"

# Pinned revisions of the public release. Bumping these changes the numbers, so they
# are pinned here and asserted in CI.
REVISIONS = {
    "tasks": ("isHeSatoshi/anthropochrony-tasks", "73ac9b151c10b5ccc9b0530b2880c8bbcec964a2"),
    "eval": ("isHeSatoshi/anthropochrony-eval", "b9a6812506a225031364f288fa1556b839955ecd"),
    "traces": ("isHeSatoshi/anthropochrony-traces", "6f81d028005b15f1a61dff3d456acf62ce4df52a"),
}

# (dataset key, repo file) -> (local path under data/, required)
FILES = [
    ("tasks", "tasks.jsonl", "tasks/tasks.jsonl", True),
    ("tasks", "proj8.jsonl", "tasks/proj8.jsonl", False),
    ("eval", "estimates.jsonl", "eval/estimates.jsonl", True),
    ("traces", "base/traces.jsonl", "measure/base/traces.jsonl", True),
    ("traces", "m1-s0/traces.jsonl", "measure/m1-s0/traces.jsonl", True),
    ("traces", "feedback-medians.json", "measure/feedback-medians.json", True),
    ("traces", "labels-base.parquet", "labels/base.parquet", False),
    ("traces", "labels-m1-s0.parquet", "labels/m1-s0.parquet", False),
    ("eval", "x10/cpu-deployment.jsonl", "x10/cpu-deployment.jsonl", False),
]


def fetch(include_x10: bool, out: Path) -> None:
    for key, name, destination, required in FILES:
        if name.startswith("x10/") and not include_x10:
            continue
        repo, revision = REVISIONS[key]
        try:
            path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
        except Exception as error:  # noqa: BLE001 - surface the underlying reason
            if required:
                raise SystemExit(f"Failed to download {repo}/{name}: {error}") from error
            print(f"skipped optional {repo}/{name}: {type(error).__name__}")
            continue
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
        print(f"{repo}/{name} -> data/{destination}")


def build_demo_tasks(out: Path) -> None:
    """Rebuild `data/demo/tasks.jsonl` from the published artifacts.

    The demo bundle is the PROJ-8 showcase plus seven quick, openly-licensed `test`
    tasks per family. It is derived (not published) so that the Space and the
    workbench can offer verified, executable tasks without a private checkout.
    """
    import pandas as pd

    tasks_path = out / "tasks/tasks.jsonl"
    labels_path = out / "labels/base.parquet"
    proj8_path = out / "tasks/proj8.jsonl"
    if not (tasks_path.exists() and labels_path.exists() and proj8_path.exists()):
        print("skipped demo tasks: missing published tasks/labels/proj8 inputs")
        return

    restricted = {"newfacade/LeetCodeDataset", "deepmind/code_contests", "BAAI/TACO"}
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    labels = pd.read_parquet(labels_path)
    ref = labels[(labels.card_id == "REF") & (labels.split == "test")]
    quick = ref[(ref.T_med_s.between(3, 40)) & (ref.PFP >= 0.1)].set_index("task_id")

    chosen = []
    for family in ("FUNC", "LIB", "ALGO_FN"):
        pool = [
            task
            for task in tasks
            if task["task_id"] in quick.index
            and task["family"] == family
            and task["source"] not in restricted
            and task.get("gt", {}).get("passes")
        ]
        pool.sort(key=lambda task: task["task_id"])
        chosen += pool[:7]

    showcase = [json.loads(line) for line in proj8_path.read_text().splitlines()]
    target = out / "demo/tasks.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(t) + "\n" for t in showcase + chosen))
    print(f"{len(showcase)} showcase + {len(chosen)} test tasks -> data/demo/tasks.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data"))
    parser.add_argument("--include-x10", action="store_true", help="also fetch the X10 CPU rows")
    parser.add_argument("--print-revisions", action="store_true")
    args = parser.parse_args()

    if args.print_revisions:
        print(json.dumps({k: v[1] for k, v in REVISIONS.items()}, indent=2))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(args.include_x10, out)
    build_demo_tasks(out)
    print("release materialized under", out)


if __name__ == "__main__":
    main()
