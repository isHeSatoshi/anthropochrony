"""Local orchestrator: tuned batched probe, sequential validation, immediate GPU exit."""

import argparse
import itertools
import json
import os
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def replay_pool(replays, tasks, sandbox, workers):
    import numpy as np

    from anthropochrony.sandbox import Sandbox

    index = {t["task_id"]: t for t in tasks}
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    counter, local = itertools.count(), threading.local()

    def initialize():
        local.sandbox = Sandbox(
            python=sandbox.python,
            memory_mb=sandbox.memory_mb,
            tasks_max=sandbox.tasks_max,
            cpu=cpus[next(counter) % len(cpus)],
        )

    def run(row):
        result = local.sandbox.run(row["code"], index[row["task_id"]])
        return {"task_id": row["task_id"], "inline_wall_s": row["inline_wall_s"], "pooled": result}

    with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        results = list(pool.map(run, replays))
    ratio = float(
        np.median(
            [r["pooled"]["wall_s"] / r["inline_wall_s"] for r in results if r["inline_wall_s"] > 0]
        )
    )
    return {"workers": workers, "median_pooled_to_inline_ratio": ratio, "results": results}


def main():
    import yaml
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl
    from anthropochrony.budget import Budget
    from anthropochrony.cli import config, reference_card
    from anthropochrony.gates import GateError, record_gate, require_gate
    from anthropochrony.gen.performance import select_batch_config
    from anthropochrony.gen.session import launch_session
    from anthropochrony.jobs import payload
    from anthropochrony.measure.rounds import initial_states, prepare_round
    from anthropochrony.measure.validation import (
        HFGenerationClient,
        fit_latency,
        observe_sequential,
        validation_metrics,
    )
    from anthropochrony.sandbox import Sandbox
    from anthropochrony.tasks.loaders import stratified_sample
    from anthropochrony.timemodel import SpeedCard

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args()
    require_gate(ROOT, "G1")
    require_gate(ROOT, "G0-security")
    measure = config(ROOT, "measure.yaml")
    tasks = read_jsonl(ROOT / "data/tasks/tasks.jsonl")
    tasks = [t for t in tasks if t["split"] != "raw_extra"]
    validation_tasks = stratified_sample(
        [t for t in tasks if t["split"] == "val"], 100, key=lambda t: t["family"]
    )
    if len(validation_tasks) != 100:
        raise GateError("S2 requires 100 stratified validation tasks")
    benchmark_tasks = stratified_sample(tasks, 100, key=lambda t: t["family"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "pins/tiny-aya-global", local_files_only=True)
    states = initial_states(benchmark_tasks, reference_card(ROOT), measure)
    requests = prepare_round(states, benchmark_tasks, tokenizer, 1)
    directory = ROOT / f"data/calibration/{args.name}"
    directory.mkdir(parents=True, exist_ok=False)
    write_jsonl(directory / "throughput-requests.jsonl", requests, config=measure)
    write_jsonl(
        directory / "validation-task-ids.jsonl",
        [{"task_id": t["task_id"], "family": t["family"]} for t in validation_tasks],
    )
    code = payload(
        ROOT,
        ROOT / f"artifacts/payloads/{args.name}",
        files=[(directory / "throughput-requests.jsonl", "throughput-requests.jsonl")],
    )
    job, session = launch_session(
        ROOT,
        args.name,
        config(ROOT, "pins.yaml")["generation_image"],
        code,
        timeout_s=args.timeout_s,
    )
    budget = Budget(ROOT)
    remote = f"/out/sessions/{args.name}"
    candidates = []
    try:
        for sequences, batched_tokens in ((512, 16384), (256, 32768), (128, 16384)):
            target = f"{remote}/probe-{sequences}-{batched_tokens}.json"
            print(
                f"Throughput probe: {sequences} sequences / {batched_tokens} batched tokens",
                flush=True,
            )
            result = session.run(
                [
                    "python3",
                    "/code/jobs/throughput_probe.py",
                    "--max-num-seqs",
                    str(sequences),
                    "--max-num-batched-tokens",
                    str(batched_tokens),
                    "--output",
                    target,
                ],
                timeout=1200,
                check=False,
                on_stdout=lambda s: print(s, end="", flush=True),
            )
            if result.exit_code != 0:
                write_json(
                    directory / f"probe-{sequences}-failure.json",
                    {"exit_code": result.exit_code, "stderr": result.stderr[-5000:]},
                )
                if not any(
                    marker in result.stderr.lower()
                    for marker in ("out of memory", "no available memory for the cache blocks")
                ):
                    raise RuntimeError(
                        "Throughput probe failed outside memory pressure; stop before another paid probe"
                    )
                continue
            result = json.loads(session.files.read(target))
            candidates.append(result | {"remote_result": target})
            write_json(directory / f"probe-{sequences}.json", result)
            if len(candidates) >= 2 and any(c["preemptions"] == 0 for c in candidates):
                break
        best = select_batch_config(candidates)
        write_json(directory / "selected-engine.json", best)
        session.files.write(f"{remote}/selected-engine.json", json.dumps(best))
        process = session.run(
            [
                "python3",
                "/code/jobs/inference_session.py",
                "--engine-config",
                f"{remote}/selected-engine.json",
                "--output",
                f"{remote}/validation",
            ],
            background=True,
        )
        write_json(directory / "remote-process.json", {"id": process.id, "pid": process.pid})
        client = HFGenerationClient(session)
        client.request({"op": "ready"}, wait_ready=480, timeout=510)
        calibration = client.request({"op": "calibrate"}, timeout=240)
        calibration["job_id"] = job.id
        write_json(directory / "reference.json", calibration)
        card = SpeedCard(**calibration["reference"])
        client.request({"op": "warmup", "card": card.as_dict()})
        records, replays = observe_sequential(
            validation_tasks, tokenizer, measure, card, client, Sandbox(), directory
        )
        write_json(ROOT / "data/calibration/reference.json", calibration)
        measure["engine"] = best["engine"]
        (ROOT / "configs/measure.yaml").write_text(yaml.safe_dump(measure, sort_keys=False))
        write_json(
            ROOT / "results/inference-tuning.json", {"selected": best, "candidates": candidates}
        )
    finally:
        # No GPU waits for CPU replays, statistics, labeling, or user interaction.
        cancelled = False
        try:
            HfApi().cancel_job(job_id=job.id, namespace=job.owner.name)
            cancelled = True
        finally:
            budget.update(
                args.name,
                state="finished_cancel_requested"
                if cancelled
                else "cancel_failed_hard_timeout_reserved",
            )
            session.close()
            with (ROOT / "RUNLOG.md").open("a") as log:
                log.write(
                    f"\nSession {job.id}: cancellation accepted={cancelled}; hard timeout remains reserved. Billing not yet reconciled.\n"
                )

    pools = []
    for workers in (8, 4, 2, 1):
        result = replay_pool(replays, validation_tasks, Sandbox(), workers)
        pools.append(result)
        write_json(directory / f"pool-{workers}.json", result)
        if result["median_pooled_to_inline_ratio"] <= 1.10:
            measure["workers"] = workers
            (ROOT / "configs/measure.yaml").write_text(yaml.safe_dump(measure, sort_keys=False))
            break
    raw = validation_metrics(records, card)
    latency = fit_latency(records, card)
    corrected = validation_metrics(records, card, latency_s=latency)
    write_json(
        directory / "validation-summary.json",
        {
            "raw": raw,
            "candidate_latency_fallback": corrected,
            "pool_ratios": [{k: v for k, v in p.items() if k != "results"} for p in pools],
            "fallback_adopted": False,
        },
    )
    record_gate(
        ROOT,
        "G2",
        {
            "100_real_sequential_val_loops": len(records) == 100,
            "median_relative_error_at_most_10pct": raw["median_absolute_relative_error"] <= 0.10,
            "log_time_r2_at_least_095": raw["log_time_r2"] >= 0.95,
            "pooled_execution_ratio_at_most_110": pools[-1]["median_pooled_to_inline_ratio"]
            <= 1.10,
        },
        evidence={
            "metrics": raw,
            "candidate_latency_fallback": corrected,
            "observations": str(directory.relative_to(ROOT)),
            "topology": "HF-generation/local-bwrap",
        },
    )
    print("G2 passed without latency correction", flush=True)


if __name__ == "__main__":

    def stop(signum, frame):
        raise KeyboardInterrupt("Stop requested; cancel the paid GPU session")

    signal.signal(signal.SIGTERM, stop)
    main()
