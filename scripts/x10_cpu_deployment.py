"""X10: real CPU deployment. Measure the machine's own speed card, ask the base and fine-tuned
models for estimates in separate calls, run each task for real, and compare with the wall clock.

Quantized llama.cpp, so this is a deployment check and not a replacement for the bf16 results.
Resumable: finished tasks are skipped on restart.
"""

import argparse
import json
import random
import subprocess
import time
import urllib.request
from pathlib import Path

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.demo import Demo, LlamaBackend
from anthropochrony.eval.errors import log_error
from anthropochrony.parse import parse
from anthropochrony.sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[1]
LLAMA = Path.home() / ".local/share/anthropochrony-envs/llamacpp/build/bin"
GGUF = Path.home() / ".cache/anthropochrony-tmp/gguf"
RESTRICTED = {"newfacade/LeetCodeDataset", "deepmind/code_contests", "BAAI/TACO"}
THREADS = 8


def start(model, port):
    process = subprocess.Popen(
        [
            str(LLAMA / "llama-server"),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-t",
            str(THREADS),
            "-c",
            "8192",
            "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return process
        except OSError:
            time.sleep(2)
    process.kill()
    raise RuntimeError(f"llama-server on {port} did not start")


def sample(n, seed):
    tasks = [t for t in read_jsonl(ROOT / "data/tasks/tasks.jsonl")]
    pool = [
        t
        for t in tasks
        if t["split"] == "test"
        and t["family"] in {"FUNC", "LIB", "ALGO_FN"}
        and t["source"] not in RESTRICTED
        and t["gt"].get("passes")
    ]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--output", type=Path, default=ROOT / "data/x10/cpu-deployment.jsonl")
    parser.add_argument(
        "--proj8", action="store_true", help="run the eight PROJ-8 showcase tasks instead"
    )
    parser.add_argument("--tasks", type=Path, help="run tasks from this JSONL instead")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in read_jsonl(args.output)} if args.output.exists() else set()
    if args.tasks:
        tasks = read_jsonl(args.tasks)
    else:
        tasks = (
            read_jsonl(ROOT / "data/tasks/proj8.jsonl") if args.proj8 else sample(args.n, args.seed)
        )
    servers = [
        start(GGUF / "base/base-q4_k_m.gguf", 8081),
        start(GGUF / "m1-s0/m1-s0-q4_k_m.gguf", 8082),
    ]
    try:
        backend = LlamaBackend(
            "http://127.0.0.1:8081",
            "http://127.0.0.1:8082",
            GGUF / "m1-s0/m1-s0-q4_k_m.gguf",
            LLAMA / "llama-bench",
            threads=THREADS,
        )
        demo = Demo(tasks, backend=backend, sandbox=Sandbox())
        card = backend.calibrate()
        # One measurement at the start; per-call recalibration would add minutes per task.
        backend.calibrate = lambda: card
        demo.last_card = card
        card_name = (
            "cpu-speed-card.json"
            if args.output.stem == "cpu-deployment"
            else f"{args.output.stem}-speed-card.json"
        )
        write_json(args.output.with_name(card_name), card.as_dict())
        with args.output.open("a") as out:
            for task in tasks:
                if task["task_id"] in done:
                    continue
                started = time.monotonic()
                estimates = demo.estimate(task["task_id"])
                estimate_s = time.monotonic() - started
                run = demo.run(task["task_id"], card=card)
                row = {
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "card": card.as_dict(),
                    "estimate_card": estimates["card"],
                    "base_raw": estimates["base"],
                    "tuned_raw": estimates["anthropochrony"],
                    "human_raw": estimates["human"],
                    "wall_s": run["wall_s"],
                    "passed": run["passed"],
                    "attempts": len(run["attempts"]),
                    "estimate_call_s": estimate_s,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
        summarize(args.output)
    finally:
        for server in servers:
            server.terminate()


def summarize(path):
    rows = read_jsonl(path)
    result = {"n_tasks": len(rows), "estimators": {}}
    for name in ("base", "tuned"):
        errors = []
        for row in rows:
            point = parse(row[f"{name}_raw"]).as_dict()["point_s"]
            errors.append(log_error(point, row["wall_s"]))
        finite = sorted(abs(e) for e in errors if e == e and abs(e) != float("inf"))
        result["estimators"][name] = {
            "parse_rate": len(finite) / len(rows),
            "median_abs_log10_error": finite[len(finite) // 2] if finite else None,
            "within2x": sum(e <= 0.30103 for e in finite) / len(rows),
        }
    result["pass_rate"] = sum(r["passed"] for r in rows) / len(rows)
    result["median_wall_s"] = sorted(r["wall_s"] for r in rows)[len(rows) // 2]
    write_json(path.with_name(f"{path.stem}-summary.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
