"""Port of the audited loader sketch. Loading never executes benchmark code."""

import ast
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from anthropochrony.artifacts import digest
from anthropochrony.prompts import task_turn
from anthropochrony.tasks.schema import Task

EXCLUDED_LIBS = {
    "tensorflow",
    "keras",
    "librosa",
    "soundfile",
    "pytesseract",
    "selenium",
    "wikipedia",
    "mechanize",
    "geopandas",
    "shapely",
    "gensim",
    "folium",
    "sendgrid",
    "python_http_client",
    "nltk",
    "skimage",
    "wordcloud",
    "textblob",
    "turtle",
    "tkinter",
}
EXCLUDED_IDS = {f"BigCodeBench/{i}" for i in (176, 339, 990, 1005, 1027, 1040, 1137)}


def _native(value):
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def records(repo, pin, local_root=None):
    for filename in pin["files"]:
        local = Path(local_root) / repo.replace("/", "__") / filename if local_root else None
        path = (
            local
            if local and local.is_file()
            else Path(
                hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset",
                    revision=pin["revision"],
                )
            )
        )
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=True)
        )
        for row in frame.to_dict(orient="records"):
            yield filename, _native(row)


def load_pool(pins, local_root=None):
    tasks, dropped = [], []
    for repo, pin in pins["datasets"].items():
        if repo.endswith("-hard"):
            continue
        for filename, row in records(repo, pin, local_root):
            common = {"source": repo, "source_revision": pin["revision"], "license": pin["license"]}
            if repo == "evalplus/humanevalplus":
                task = Task(
                    row["task_id"],
                    "FUNC",
                    **common,
                    task_turn=task_turn(
                        f"```python\n{row['prompt']}\n```\nComplete this function."
                    ),
                    entry_point=row["entry_point"],
                    harness="evalplus_check",
                    reference_solution=row["prompt"] + row["canonical_solution"],
                    test_code=row["test"] + f"\ncheck({row['entry_point']})\n",
                )
            elif repo == "evalplus/mbppplus":
                entry = re.search(r"assertion\((\w+)\(\*inp\)", row["test"])
                task = Task(
                    f"Mbpp/{row['task_id']}",
                    "FUNC",
                    **common,
                    task_turn=task_turn(
                        row["prompt"]
                        + "\nYour code should pass tests like:\n"
                        + row["test_list"][0]
                    ),
                    entry_point=entry[1] if entry else None,
                    harness="evalplus_check",
                    reference_solution=row["code"],
                    test_code="\n".join(row["test_imports"]) + "\n" + row["test"],
                    public_examples={"assert": row["test_list"][0]},
                )
            elif repo == "FudanSELab/ClassEval":
                task = Task(
                    row["task_id"],
                    "CLASS",
                    **common,
                    task_turn=task_turn(
                        f"```python\n{row['skeleton']}\n```\nImplement every method."
                    ),
                    entry_point=row["class_name"],
                    harness="unittest",
                    reference_solution="\n".join(row["import_statement"])
                    + "\n"
                    + row["solution_code"],
                    test_code=row["test"],
                    time_limits={"per_test_s": 30, "per_task_s": 30},
                )
            elif repo == "bigcode/bigcodebench":
                libs = (
                    ast.literal_eval(row["libs"]) if isinstance(row["libs"], str) else row["libs"]
                )
                if set(libs) & EXCLUDED_LIBS or row["task_id"] in EXCLUDED_IDS:
                    dropped.append({"task_id": row["task_id"], "reason": "dependency_or_network"})
                    continue
                task = Task(
                    row["task_id"],
                    "LIB",
                    **common,
                    task_turn=task_turn(row["instruct_prompt"]),
                    entry_point=row["entry_point"],
                    harness="unittest",
                    reference_solution=row["code_prompt"] + row["canonical_solution"],
                    test_code=row["test"],
                    time_limits={"per_test_s": 120, "per_task_s": 120},
                )
            elif repo == "newfacade/LeetCodeDataset":
                split = "test" if "test" in filename else "train"
                task = Task(
                    f"lcd/{split}/{row['task_id']}",
                    "ALGO_FN",
                    **common,
                    task_turn=task_turn(
                        row["problem_description"]
                        + "\n\nStarter code:\n```python\n"
                        + row["starter_code"]
                        + "\n```"
                    ),
                    entry_point=row["entry_point"],
                    harness="lcd_check",
                    reference_solution=row["completion"],
                    test_code=row["prompt"]
                    + "\n# <SOLUTION>\n"
                    + row["test"]
                    + f"\ncheck({row['entry_point']})\n",
                    difficulty=f"{row['difficulty']}|{split}",
                    time_limits={"per_test_s": 10, "per_task_s": 10},
                    public_examples={"examples": row.get("input_output", [])[:2]},
                )
            elif repo == "deepmind/code_contests":
                refs = [
                    s
                    for lang, s in zip(
                        row["solutions"]["language"], row["solutions"]["solution"], strict=True
                    )
                    if lang == 3
                ]
                if not refs or row.get("is_description_translated"):
                    dropped.append({"task_id": "cc/" + row["name"], "reason": "language"})
                    continue
                pairs = list(
                    zip(row["public_tests"]["input"], row["public_tests"]["output"], strict=True)
                )
                pairs += list(
                    zip(row["private_tests"]["input"], row["private_tests"]["output"], strict=True)
                )
                generated = list(
                    zip(
                        row["generated_tests"]["input"],
                        row["generated_tests"]["output"],
                        strict=True,
                    )
                )
                rng = random.Random(int(digest(row["name"])[:16], 16))
                rng.shuffle(generated)
                if len(pairs) > 50:
                    dropped.append(
                        {"task_id": "cc/" + row["name"], "reason": "over_50_public_private_tests"}
                    )
                    continue
                pairs += generated[: 50 - len(pairs)]
                public = row["public_tests"]
                sample = (
                    (
                        "\nSample input:\n"
                        + public["input"][0]
                        + "\nSample output:\n"
                        + public["output"][0]
                    )
                    if public["input"]
                    else ""
                )
                limit = row.get("time_limit") or {}
                limit_s = limit.get("seconds", 0) + limit.get("nanos", 0) / 1e9
                task = Task(
                    "cc/" + row["name"],
                    "ALGO_IO",
                    **common,
                    task_turn=task_turn(
                        row["description"]
                        + "\nRead from standard input and write to standard output."
                        + sample
                    ),
                    harness="stdio",
                    reference_solution=sorted(refs, key=len)[len(refs) // 2],
                    io_examples={"input": [p[0] for p in pairs], "output": [p[1] for p in pairs]},
                    public_examples=public,
                    difficulty=str(row.get("cf_rating") or "unrated"),
                    time_limits={"per_test_s": max(1, 3 * limit_s), "per_task_s": 60},
                )
            elif repo == "BAAI/TACO":
                # Plan S3 fallback: only EASY/MEDIUM stdin problems, because the
                # base model passes almost no CodeContests problems.
                if row["difficulty"] not in {"EASY", "MEDIUM"}:
                    continue
                tests = json.loads(row["input_output"])
                if "fn_name" in tests or not tests.get("inputs"):
                    continue
                refs = []
                for solution in json.loads(row["solutions"] or "[]"):
                    try:
                        ast.parse(solution)
                    except SyntaxError:
                        continue  # Python 2 solutions
                    refs.append(solution)
                task_id = "taco/" + digest(row["question"])[:16]
                if not refs or len(tests["inputs"]) != len(tests["outputs"]):
                    dropped.append({"task_id": task_id, "reason": "taco_no_python3_reference"})
                    continue
                order = list(range(len(tests["inputs"])))
                random.Random(int(digest(task_id)[:16], 16)).shuffle(order)
                chosen = order[:50]
                task = Task(
                    task_id,
                    "ALGO_IO",
                    **common,
                    task_turn=task_turn(
                        row["question"] + "\nRead from standard input and write to standard output."
                    ),
                    harness="stdio",
                    reference_solution=sorted(refs, key=len)[len(refs) // 2],
                    io_examples={
                        "input": [tests["inputs"][i] for i in chosen],
                        "output": [tests["outputs"][i] for i in chosen],
                    },
                    difficulty=f"taco-{row['difficulty']}",
                    time_limits={"per_test_s": 4, "per_task_s": 60},
                )
            else:
                raise ValueError(f"Unknown source {repo}")
            tasks.append(task.as_dict())
    return tasks, dropped


def stratified_sample(tasks, count, *, key, seed=20261109):
    if count >= len(tasks):
        return sorted(tasks, key=lambda t: t["task_id"])
    groups = {}
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        groups.setdefault(key(task), []).append(task)
    rng = random.Random(seed)
    allocation = {group: int(count * len(pool) / len(tasks)) for group, pool in groups.items()}
    remainder = count - sum(allocation.values())
    order = sorted(groups, key=lambda g: (-(count * len(groups[g]) / len(tasks) % 1), str(g)))
    for group in order[:remainder]:
        allocation[group] += 1
    return sorted(
        [t for group, pool in groups.items() for t in rng.sample(pool, allocation[group])],
        key=lambda t: t["task_id"],
    )


def build_mix(tasks, hard_ids):
    family = {
        f: [t for t in tasks if t["family"] == f]
        for f in ("FUNC", "CLASS", "LIB", "ALGO_FN", "ALGO_IO")
    }

    def by_length(pool, n):
        edges = np.quantile([t["ref_tokens"] for t in pool], [0.2, 0.4, 0.6, 0.8])
        return stratified_sample(
            pool, n, key=lambda t: int(np.searchsorted(edges, t["ref_tokens"]))
        )

    func = family["FUNC"]
    result = [t for t in func if t["source"] == "evalplus/humanevalplus"]
    result += by_length([t for t in func if t["source"] == "evalplus/mbppplus"], 235)
    result += family["CLASS"]
    hard = [t for t in family["LIB"] if t["task_id"] in hard_ids]
    result += hard + by_length(
        [t for t in family["LIB"] if t["task_id"] not in hard_ids], 350 - len(hard)
    )
    lcd = family["ALGO_FN"]
    test = [t for t in lcd if t["difficulty"].endswith("|test")]
    result += test
    for difficulty, target in {"Easy": 110, "Medium": 140, "Hard": 100}.items():
        have = sum(t["difficulty"].startswith(difficulty) for t in test)
        pool = [t for t in lcd if t["difficulty"] == f"{difficulty}|train"]
        result += by_length(pool, target - have)
    result += algo_io_mix(family["ALGO_IO"])
    return result


def algo_io_mix(pool):
    """CodeContests with the hardest stratum capped, plus TACO EASY/MEDIUM.

    The REF pilot passed 0 of 13 CodeContests tasks, which triggers the plan's
    S3 fallback: cap the CF >= 2000 stratum and add TACO EASY/MEDIUM.
    """

    def rating(task):
        try:
            return float(task["difficulty"])
        except ValueError:
            return None

    contests = [t for t in pool if t["source"] == "deepmind/code_contests"]
    hard = [t for t in contests if (rating(t) or 0) >= 2000]
    easier = [t for t in contests if t not in hard]
    result = easier + stratified_sample(hard, 30, key=lambda t: rating(t) // 400)
    for level in ("taco-EASY", "taco-MEDIUM"):
        result += stratified_sample(
            [t for t in pool if t["difficulty"] == level], 60, key=lambda t: t["source"]
        )
    return result
