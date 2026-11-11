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
    print("release materialized under", out)


if __name__ == "__main__":
    main()
