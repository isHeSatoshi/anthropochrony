"""S2 observations: a real sequential HF generation / local sandbox loop."""

import json
import time

import numpy as np
from sklearn.metrics import r2_score

from anthropochrony.artifacts import write_json, write_jsonl
from anthropochrony.measure.rounds import apply_round, initial_states, prepare_round
from anthropochrony.sandbox.runner import extract_code
from anthropochrony.timemodel import completion_time, trajectory_time


def validation_metrics(records, card, *, latency_s=0.0):
    if not records:
        raise ValueError("Validation requires observed agent loops")
    actual = np.asarray([r["actual_wall_s"] for r in records])
    predicted = np.asarray(
        [trajectory_time(r["trace"], card, latency_s=latency_s) for r in records]
    )
    if np.any(actual <= 0) or np.any(predicted <= 0):
        raise ValueError("Completion times must be positive")
    relative = (predicted - actual) / actual
    return {
        "observed_runs": len(records),
        "median_absolute_relative_error": float(np.median(np.abs(relative))),
        "log_time_r2": float(r2_score(np.log(actual), np.log(predicted)))
        if len(records) > 1
        else None,
        "median_relative_bias": float(np.median(relative)),
        "latency_per_call_s": latency_s,
    }


def fit_latency(records, card):
    residuals = []
    for row in records:
        for call, attempt in zip(row["calls"], row["trace"]["attempts"], strict=True):
            residuals.append(
                call["rpc_wall_s"]
                - completion_time(attempt["u_tokens"], attempt["o_tokens"], 0, card, latency_s=0.0)
            )
    if not residuals:
        raise ValueError("No measured calls for the latency fallback")
    return max(0.0, float(np.median(residuals)))


def adopt_latency_fallback(root, observations="data/calibration/s2-throughput-validation-v3"):
    """Re-evaluate G2 with the plan's per-call-latency fallback on saved observations.

    The unchanged thresholds apply. The failed raw attempt is preserved, and the
    adopted latency must equal the central `PER_CALL_LATENCY_S`.
    """
    from pathlib import Path

    from anthropochrony.artifacts import now, read_jsonl
    from anthropochrony.gates import record_gate
    from anthropochrony.timemodel import PER_CALL_LATENCY_S, SpeedCard

    root = Path(root)
    directory = root / observations
    gate_path = root / "results/gates/G2.json"
    raw_gate = json.loads(gate_path.read_text())
    preserved = root / "results/gates/G2-raw-attempt.json"
    if raw_gate.get("evidence", {}).get("fallback_adopted"):
        raise ValueError("The latency fallback is already adopted")
    if raw_gate["passed"]:
        raise ValueError("G2 already passed without the fallback")
    if not preserved.exists():
        preserved.write_text(gate_path.read_text())
    records = read_jsonl(directory / "observed-loops.jsonl")
    card = SpeedCard(**json.loads((directory / "reference.json").read_text())["reference"])
    fitted = fit_latency(records, card)
    if round(fitted, 3) != PER_CALL_LATENCY_S:
        raise ValueError(f"Fitted latency {fitted} does not match adopted {PER_CALL_LATENCY_S}")
    raw = validation_metrics(records, card, latency_s=0.0)
    adopted = validation_metrics(records, card, latency_s=PER_CALL_LATENCY_S)
    rng = np.random.default_rng(0)
    order = np.arange(len(records))
    halves = []
    for _ in range(200):
        rng.shuffle(order)
        fit_on, test_on = order[: len(order) // 2], order[len(order) // 2 :]
        latency = fit_latency([records[i] for i in fit_on], card)
        halves.append(
            validation_metrics([records[i] for i in test_on], card, latency_s=latency)
            | {"fitted_latency_s": latency}
        )
    families = {}
    for family in sorted({r["family"] for r in records}):
        held_out = [r for r in records if r["family"] == family]
        if len(held_out) > 1:
            latency = fit_latency([r for r in records if r["family"] != family], card)
            families[family] = validation_metrics(held_out, card, latency_s=latency)[
                "median_absolute_relative_error"
            ]
    cross_validation = {
        "random_half_splits": len(halves),
        "worst_heldout_median_absolute_relative_error": max(
            h["median_absolute_relative_error"] for h in halves
        ),
        "worst_heldout_log_time_r2": min(h["log_time_r2"] for h in halves),
        "fitted_latency_range_s": [
            min(h["fitted_latency_s"] for h in halves),
            max(h["fitted_latency_s"] for h in halves),
        ],
        "leave_one_family_out_median_absolute_relative_error": families,
    }
    pool = json.loads((directory / "pool-8.json").read_text())
    evidence = {
        "adopted_at": now(),
        "fallback_adopted": True,
        "per_call_latency_s": PER_CALL_LATENCY_S,
        "raw_metrics": raw,
        "metrics": adopted,
        "cross_validation": cross_validation,
        "observations": observations,
        "raw_attempt": str(preserved.relative_to(root)),
        "topology": "HF-generation/local-bwrap",
    }
    record_gate(
        root,
        "G2",
        {
            "100_real_sequential_val_loops": len(records) == 100,
            "raw_formula_failed_before_fallback": not raw_gate["passed"],
            "median_relative_error_at_most_10pct": adopted["median_absolute_relative_error"]
            <= 0.10,
            "log_time_r2_at_least_095": adopted["log_time_r2"] >= 0.95,
            "heldout_median_error_at_most_10pct": cross_validation[
                "worst_heldout_median_absolute_relative_error"
            ]
            <= 0.10,
            "pooled_execution_ratio_at_most_110": pool["median_pooled_to_inline_ratio"] <= 1.10,
        },
        evidence=evidence,
    )
    return evidence


class HFGenerationClient:
    """Only trusted model inference runs remotely. No candidate is executed there."""

    def __init__(self, session):
        self.session = session

    def request(self, message, *, wait_ready=0, timeout=240):
        command = ["python3", "/code/jobs/session_request.py"]
        if wait_ready:
            command += ["--wait-ready", str(wait_ready)]
        result = self.session.run(command, stdin=json.dumps(message), timeout=timeout, check=True)
        response = json.loads(result.stdout)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response


def observe_sequential(tasks, tokenizer, config, card, client, sandbox, directory):
    directory.mkdir(parents=True, exist_ok=True)
    records, replays = [], []
    for index, task in enumerate(tasks):
        states = initial_states([task], card, config | {"k": 1})
        calls, started, ended = [], None, None
        for round_number in range(1, 4):
            requests = prepare_round(states, [task], tokenizer, round_number)
            if not requests:
                break
            if started is None:
                started = time.monotonic()
            call_start = time.monotonic()
            response = client.request({"op": "generate", "requests": requests})
            call_end = time.monotonic()
            states, texts = apply_round(
                states, [task], requests, response["outputs"], sandbox, round_number, workers=1
            )
            ended = time.monotonic()
            attempt = states[0]["attempts"][-1]
            calls.append(
                {
                    "rpc_wall_s": call_end - call_start,
                    "generation_wall_s": response["performance"]["elapsed_s"],
                }
            )
            code = extract_code(texts[0]["text"])
            if code is not None:
                replays.append(
                    {
                        "task_id": task["task_id"],
                        "code": code,
                        "inline_wall_s": attempt["exec"]["wall_s"],
                    }
                )
            if states[0]["done"]:
                break
        if started is None or ended is None:
            raise RuntimeError("Validation task never entered the real agent loop")
        trace = {k: v for k, v in states[0].items() if k != "messages"}
        records.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "actual_wall_s": ended - started,
                "trace": trace,
                "calls": calls,
            }
        )
        write_jsonl(directory / "observed-loops.jsonl", records, config=config)
        write_jsonl(directory / "execution-replays.jsonl", replays, config=config)
        if (index + 1) % 10 == 0:
            print(f"S2 observed {index + 1}/{len(tasks)} real sequential loops", flush=True)
    write_json(directory / "raw-metrics.json", validation_metrics(records, card))
    return records, replays
