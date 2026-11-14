"""Build the training pool for the single-turn self-time adapter from public instruction data.

The pool is a fixed stratified sample of four public datasets plus this project's own
train-split coding tasks. No MetaTime task is in it: prompts that resemble any of the 68
benchmark tasks (TF-IDF cosine at or above the threshold) are removed before any run.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd
from common import load_tasks
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer

SOURCES = {
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"),
    "code_alpaca": ("sahil2801/CodeAlpaca-20k", "152bb5e9a29651266b018106053980070a0521a1"),
    "no_robots": ("HuggingFaceH4/no_robots", "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
}
MAX_CHARS = 6000
SIMILARITY_THRESHOLD = 0.25


def _fetch(name, filename):
    repo, revision = SOURCES[name]
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def _take(rows, count, rng):
    rows = sorted(rows, key=lambda row: row["key"])
    rng.shuffle(rows)
    return rows[:count]


def candidates(rng, own_tasks):
    out = []
    dolly = [
        json.loads(line) for line in Path(_fetch("dolly", "databricks-dolly-15k.jsonl")).open()
    ]
    for category in sorted({row["category"] for row in dolly}):
        rows = [
            {
                "key": f"{index}",
                "category": category,
                "instruction": row["instruction"].strip(),
                "context": row["context"].strip(),
            }
            for index, row in enumerate(dolly)
            if row["category"] == category
        ]
        out += [{"source": "dolly", **row} for row in _take(rows, 70, rng)]
    alpaca = json.loads(Path(_fetch("code_alpaca", "code_alpaca_20k.json")).read_text())
    rows = [
        {
            "key": f"{index}",
            "category": "code",
            "instruction": row["instruction"].strip(),
            "context": row["input"].strip(),
        }
        for index, row in enumerate(alpaca)
    ]
    out += [{"source": "code_alpaca", **row} for row in _take(rows, 340, rng)]
    robots = pd.read_parquet(_fetch("no_robots", "data/train-00000-of-00001.parquet"))
    for category in sorted(robots["category"].unique()):
        rows = []
        for index, row in robots[robots["category"] == category].iterrows():
            users = [m["content"] for m in row["messages"] if m["role"] == "user"]
            if len(users) == 1:
                rows.append(
                    {
                        "key": f"{index}",
                        "category": category,
                        "instruction": users[0].strip(),
                        "context": "",
                    }
                )
        out += [{"source": "no_robots", **row} for row in _take(rows, 50, rng)]
    gsm = pd.read_parquet(_fetch("gsm8k", "main/train-00000-of-00001.parquet"))
    rows = [
        {"key": f"{index}", "category": "math", "instruction": question.strip(), "context": ""}
        for index, question in enumerate(gsm["question"])
    ]
    out += [{"source": "gsm8k", **row} for row in _take(rows, 100, rng)]
    rows = [
        {
            "key": row["task_id"],
            "category": row["family"],
            "instruction": row["task_turn"].strip(),
            "context": "",
        }
        for row in own_tasks
        if row["split"] == "train"
    ]
    out += [{"source": "anthropochrony_train", **row} for row in _take(rows, 100, rng)]
    return out


def _normal(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--tasks", default="data/tasks/tasks.jsonl")
    parser.add_argument("--out", default="data/metatime/pool.jsonl")
    args = parser.parse_args()

    benchmark = load_tasks(args.suite)
    own = [json.loads(line) for line in Path(args.tasks).open()]
    pool = candidates(random.Random(0), own)
    stats = {"sampled": len(pool)}
    seen, kept = set(), []
    for row in pool:
        row["prompt"] = row["instruction"] + ("\n\n" + row["context"] if row["context"] else "")
        key = _normal(row["prompt"])
        if not row["instruction"] or len(row["prompt"]) > MAX_CHARS or key in seen:
            continue
        seen.add(key)
        kept.append(row)
    stats["after_length_and_duplicates"] = len(kept)

    reference = [task["prompt"] for task in benchmark] + [task["description"] for task in benchmark]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    matrix = vectorizer.fit_transform([row["instruction"] for row in kept] + reference)
    similarity = (matrix[: len(kept)] @ matrix[len(kept) :].T).toarray()
    nearest = similarity.max(axis=1)
    removed = [
        {
            "instruction": row["instruction"][:120],
            "similarity": round(float(score), 3),
            "benchmark_text": reference[int(similarity[index].argmax())][:120],
        }
        for index, (row, score) in enumerate(zip(kept, nearest, strict=True))
        if score >= SIMILARITY_THRESHOLD
    ]
    kept = [row for row, score in zip(kept, nearest, strict=True) if score < SIMILARITY_THRESHOLD]
    stats["removed_as_similar_to_benchmark"] = len(removed)
    stats["final"] = len(kept)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in kept:
            digest = hashlib.sha256(f"{row['source']}|{row['key']}".encode()).hexdigest()
            row["id"] = f"{row['source']}-{digest[:10]}"
            row["split"] = "val" if int(digest[10:14], 16) % 10 == 0 else "train"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["val"] = sum(row["split"] == "val" for row in kept)
    stats["by_source"] = {
        name: sum(row["source"] == name for row in kept)
        for name in sorted({row["source"] for row in kept})
    }
    manifest = {
        "sources": {
            name: {"repo": repo, "revision": revision} for name, (repo, revision) in SOURCES.items()
        },
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "stats": stats,
        "removed": removed,
        "max_remaining_similarity": round(float(nearest[nearest < SIMILARITY_THRESHOLD].max()), 3),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print("removed:", json.dumps(removed, indent=1)[:3000])


if __name__ == "__main__":
    main()
