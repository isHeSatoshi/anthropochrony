"""Upload the staged release bundle to the Hub as PRIVATE repos and collections.

Making anything public is a separate, explicit step (``--make-public``) that needs the user's
approval. Run ``anthropochrony release`` and ``scripts/build_cards.py`` first.

    uv run python scripts/publish_hub.py collections
    uv run python scripts/publish_hub.py upload [--gguf-dir DIR]
    uv run python scripts/publish_hub.py visibility --make-public   # only after approval
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
USER = "isHeSatoshi"
RELEASE = ROOT / "artifacts/release"
STATE = RELEASE / "collections.json"
ADAPTERS = ["m1-s0", "m2-s0", "m3-s0", "lr-0.0001", "lr-0.0002"]
DATASETS = [
    "anthropochrony-tasks",
    "anthropochrony-traces",
    "anthropochrony-sft",
    "anthropochrony-eval",
    "anthropochrony-logs",
]
GGUF_REPO = "anthropochrony-tiny-aya-global-gguf"
GGUF_FILES = {
    "m1-s0/m1-s0-q4_k_m.gguf": "m1-s0-q4_k_m.gguf",
    "m1-s0/m1-s0-q8_0.gguf": "m1-s0-q8_0.gguf",
    "base/base-q4_k_m.gguf": "base-q4_k_m.gguf",
}


def model_ids():
    return [f"{USER}/anthropochrony-tiny-aya-global-lora-{a}" for a in ADAPTERS] + [
        f"{USER}/{GGUF_REPO}"
    ]


def collections(api):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for key, title, description in (
        (
            "models",
            "Anthropochrony models",
            "LoRA adapters and GGUF files for self-estimation of completion time (Tiny Aya Global).",
        ),
        (
            "data",
            "Anthropochrony data",
            "Tasks, measured runs, SFT sets, estimates, results and logs for Anthropochrony.",
        ),
    ):
        if key not in state:
            created = api.create_collection(
                title=title, namespace=USER, description=description, private=True
            )
            state[key] = created.slug
    STATE.write_text(json.dumps(state, indent=2))
    for slug, items, kind in (
        (state["models"], model_ids(), "model"),
        (state["data"], [f"{USER}/{d}" for d in DATASETS], "dataset"),
    ):
        for item in items:
            try:
                api.add_collection_item(slug, item_id=item, item_type=kind, exists_ok=True)
            except Exception as error:  # repo not uploaded yet; rerun after upload
                print("skipped", item, type(error).__name__)
    print(json.dumps(state, indent=2))


def upload(api, gguf_dir):
    for name in DATASETS:
        repo = f"{USER}/{name}"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=RELEASE / "datasets" / name,
            commit_message="Release bundle (private staging)",
        )
        print("uploaded", repo)
    for adapter in ADAPTERS:
        repo = f"{USER}/anthropochrony-tiny-aya-global-lora-{adapter}"
        card = RELEASE / "models" / repo.split("/")[1] / "README.md"
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="Model card",
        )
        print("card", repo)
    if gguf_dir:
        repo = f"{USER}/{GGUF_REPO}"
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        card = RELEASE / "models" / GGUF_REPO / "README.md"
        api.upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="model"
        )
        manifest = {}
        for source, target in GGUF_FILES.items():
            path = Path(gguf_dir) / source
            api.upload_file(
                path_or_fileobj=str(path), path_in_repo=target, repo_id=repo, repo_type="model"
            )
            print("uploaded", target)
        for run in ("base", "m1-s0"):
            data = json.loads((Path(gguf_dir) / run / "gguf-manifest.json").read_text())
            manifest[run] = {k: v for k, v in data.items() if k != "files"} | {
                "files": {
                    k: v for k, v in data.get("files", {}).items() if k in GGUF_FILES.values()
                }
            }
        api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2).encode(),
            path_in_repo="gguf-manifest.json",
            repo_id=repo,
            repo_type="model",
        )


def visibility(api, make_public):
    if not make_public:
        raise SystemExit("Pass --make-public only after the user approved public release")
    state = json.loads(STATE.read_text())
    for name in DATASETS:
        api.update_repo_settings(f"{USER}/{name}", repo_type="dataset", private=False)
    for repo in model_ids():
        api.update_repo_settings(repo, repo_type="model", private=False)
    for slug in state.values():
        api.update_collection_metadata(slug, private=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collections", "upload", "visibility"])
    parser.add_argument("--gguf-dir")
    parser.add_argument("--make-public", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    if args.action == "collections":
        collections(api)
    elif args.action == "upload":
        upload(api, args.gguf_dir)
    else:
        visibility(api, args.make_public)


if __name__ == "__main__":
    main()
