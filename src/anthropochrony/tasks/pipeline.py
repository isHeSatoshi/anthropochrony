import ast
import itertools
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from anthropochrony.artifacts import digest, file_hash, read_jsonl, write_json, write_jsonl
from anthropochrony.curate.pipeline import curate
from anthropochrony.gates import record_gate
from anthropochrony.gen.engine import render, system_token_count
from anthropochrony.prompts import system_prompt
from anthropochrony.sandbox import Sandbox
from anthropochrony.tasks.loaders import build_mix, load_pool, records
from anthropochrony.tasks.splits import assign_splits, duplicate_groups, verify_splits


def vacuous_solution(task):
    if task["harness"] == "stdio":
        return ""
    try:
        tree = ast.parse(task["reference_solution"])
    except SyntaxError:
        return "pass"

    class Stub(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.body = [ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    return ast.unparse(ast.fix_missing_locations(Stub().visit(tree)))


def verify_reference(task, sandbox):
    outcomes = [sandbox.run(task["reference_solution"], task) for _ in range(3)]
    times = [result["wall_s"] for result in outcomes]
    if any(result["status"] != "pass" for result in outcomes):
        return dict(
            task, gt={"passes": False, "times_s": times, "outcomes": outcomes}
        ), "reference_failed"
    if max(times) > 1 and np.std(times) / np.mean(times) > 0.5:
        return dict(task, gt={"passes": True, "times_s": times}), "flaky"
    vacuous = sandbox.run(vacuous_solution(task), task)
    if vacuous["status"] == "pass":
        return dict(task, gt={"passes": True, "times_s": times}), "vacuous_tests"
    limits = task["time_limits"]
    if task["family"] != "ALGO_IO":
        limits = limits | {"per_test_s": max(4 if task["family"] == "FUNC" else 1, 4 * max(times))}
    return dict(task, gt={"passes": True, "times_s": times}, time_limits=limits), None


def build_tasks(root, pins, tokenizer, sandbox, card, *, local_root=None, workers=8):
    root = Path(root)
    directory = root / "data/tasks"
    directory.mkdir(parents=True, exist_ok=True)
    pool, dropped = load_pool(pins, local_root)
    for task in pool:
        task["ref_tokens"] = len(
            tokenizer.encode(task["reference_solution"], add_special_tokens=False)
        )
        task["prompt_tokens"] = len(
            render(
                tokenizer,
                [
                    {"role": "system", "content": system_prompt(card, estimator="agent")},
                    {"role": "user", "content": task["task_turn"]},
                ],
            )
        )
        task["tests_ref"] = digest({"test": task["test_code"], "io": task["io_examples"]})
        task["uncached_task_tokens"] = task["prompt_tokens"] - system_token_count(
            tokenizer, system_prompt(card, estimator="agent")
        )
    hard_ids = {
        r["task_id"]
        for _, r in records(
            "bigcode/bigcodebench-hard", pins["datasets"]["bigcode/bigcodebench-hard"], local_root
        )
    }
    mix = build_mix(pool, hard_ids)
    legal = []
    for task in mix:
        if task["prompt_tokens"] > 2500:
            dropped.append({"task_id": task["task_id"], "reason": "prompt_length"})
        else:
            legal.append(task)
    groups = duplicate_groups(legal)
    # TACO re-uses CodeContests/APPS problems: drop any TACO task that shares a
    # near-duplicate group with a task from another source.
    group_sources = {}
    for task in legal:
        group_sources.setdefault(groups[task["task_id"]], set()).add(task["source"])
    shared = [
        t
        for t in legal
        if t["source"] == "BAAI/TACO" and len(group_sources[groups[t["task_id"]]]) > 1
    ]
    dropped += [{"task_id": t["task_id"], "reason": "taco_duplicates_other_source"} for t in shared]
    legal = [t for t in legal if t not in shared]
    cache_path = directory / "gt-checkpoint.jsonl"
    cache = {r["task_id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    package = Path(__file__).parents[1]
    execution_lock = {
        name: file_hash(package / name)
        for name in ("execenv/requirements.lock", "sandbox/runner.py", "sandbox/worker.py")
    }

    def check_hash(task):
        inputs = {
            key: task.get(key)
            for key in (
                "reference_solution",
                "test_code",
                "io_examples",
                "harness",
                "entry_point",
                "time_limits",
            )
        }
        return digest({"task": inputs, "execution_lock": execution_lock, "workers": workers})

    pending = [t for t in legal if check_hash(t) != cache.get(t["task_id"], {}).get("input_sha")]
    local = threading.local()
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    worker_index = itertools.count()

    def initialize():
        local.sandbox = Sandbox(
            python=sandbox.python,
            memory_mb=sandbox.memory_mb,
            tasks_max=sandbox.tasks_max,
            cpu=cpus[next(worker_index) % len(cpus)],
        )

    def check(task):
        verified, reason = verify_reference(task, local.sandbox)
        return {
            "task_id": task["task_id"],
            "input_sha": check_hash(task),
            "task": verified,
            "reason": reason,
        }

    with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool_executor:
        for checked in pool_executor.map(check, pending):
            cache[checked["task_id"]] = checked
            write_jsonl(cache_path, list(cache.values()))
    good, raw = [], []
    for task in legal:
        entry = cache[task["task_id"]]
        (raw if entry["reason"] else good).append(entry["task"])
        if entry["reason"]:
            dropped.append({"task_id": task["task_id"], "reason": entry["reason"]})
    split_tasks, overlap_drops = assign_splits(good, groups)
    dropped += overlap_drops
    curated = []
    for split in ("train", "val", "test", "test_ood"):
        subset = [t for t in split_tasks if t["split"] == split]
        passed, rejects = curate(subset, directory / "curation" / split)
        curated += passed
        dropped += rejects
        if split == "train":
            rejected_ids = {r["task_id"] for r in rejects}
            raw += [t for t in subset if t["task_id"] in rejected_ids]
    evaluation_groups = {
        t["near_dup_group"]
        for t in split_tasks
        if t["split"] in {"val", "test", "test_ood", "showcase"}
    } | {groups[t["task_id"]] for t in legal if t["family"] in {"CLASS", "PROJ-8"}}
    raw_extra = []
    for task in raw:
        if task["family"] in {"CLASS", "PROJ-8"}:
            continue
        group = groups[task["task_id"]]
        if group not in evaluation_groups:
            raw_extra.append(dict(task, split="raw_extra", near_dup_group=group))
    verify_splits(curated + raw_extra)
    # raw_extra can share training groups but never evaluation or holdout-family groups.
    assert all(t["near_dup_group"] not in evaluation_groups for t in raw_extra)
    write_jsonl(directory / "tasks.jsonl", curated + raw_extra, config=pins)
    write_jsonl(directory / "tasks_dropped.jsonl", dropped, config=pins)
    counts = Counter(t["family"] for t in curated)
    splits = Counter(t["split"] for t in curated)
    write_json(
        directory / "counts.json",
        {
            "families": dict(counts),
            "splits": dict(splits),
            "raw_extra": len(raw_extra),
            "drops": dict(Counter(r["reason"] for r in dropped)),
        },
    )
    targets = {"FUNC": 399, "CLASS": 100, "LIB": 350, "ALGO_FN": 350, "ALGO_IO": 217}
    record_gate(
        root,
        "G1",
        {
            "kept_at_least_1250": len(curated) >= 1250,
            "all_families_80pct": all(counts[f] >= 0.8 * n for f, n in targets.items()),
            "test_at_least_240": splits["test"] >= 240,
            "val_at_least_90": splits["val"] >= 90,
            "ood_at_least_90": splits["test_ood"] >= 90,
            "all_references_3_of_3": all(
                t["gt"]["passes"] and len(t["gt"]["times_s"]) == 3 for t in curated
            ),
            "no_group_leakage": True,
        },
        evidence={"families": dict(counts), "splits": dict(splits)},
    )
    return curated + raw_extra
