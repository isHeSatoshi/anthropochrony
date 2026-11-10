"""Estimate generation, adapter evidence and G6 equivalence on HF GPU jobs."""

import json
from pathlib import Path

from huggingface_hub import HfApi

from anthropochrony.artifacts import digest, file_hash, read_jsonl, write_json, write_jsonl
from anthropochrony.cli import config
from anthropochrony.gates import GateError, record_gate, require_gate, require_preregistration
from anthropochrony.gpu_jobs import GPUJob
from anthropochrony.measure.remote import BUCKET_ID
from anthropochrony.remote import authorization_guard

ADAPTER_FILES = (
    "identity.json",
    "training-summary.json",
    "training-log.jsonl",
    "trainer_state.json",
    "mask_check.json",
    "lora_check.json",
    "hardware.json",
    "adapter_config.json",
)
ESTIMATE_SOURCES = (
    "jobs/eval_estimates.py",
    "src/anthropochrony/eval/audit.py",
    "src/anthropochrony/eval/errors.py",
    "src/anthropochrony/parse.py",
)


def fetch_adapter_evidence(root, run):
    """Download identity/provenance only; the adapter weights stay in the private bucket."""
    root = Path(root)
    target = root / "adapters" / run
    target.mkdir(parents=True, exist_ok=True)
    with authorization_guard():
        HfApi().download_bucket_files(
            BUCKET_ID,
            [(f"adapters/{run}/{name}", target / name) for name in ADAPTER_FILES],
            raise_on_missing_files=True,
        )
    identity = json.loads((target / "identity.json").read_text())
    summary = target / "training-summary.json"
    if (
        identity["training_summary_sha256"] != file_hash(summary)
        or identity.get("private") is not True
    ):
        raise ValueError("Adapter evidence does not match its private identity")
    return identity


def _identity(root, run):
    if run == "base":
        return None
    path = Path(root) / "adapters" / run / "identity.json"
    if not path.exists():
        raise GateError(f"Download and check adapter evidence for {run} first")
    return json.loads(path.read_text())


def run_estimates(root, run, prompt_paths, *, label, timeout_s=2400, adapter=None, name=None):
    """One GPU job for every prompt file of a single model; returns rows per input split.

    `adapter` serves a different checkpoint than the prompt identity (LR-sweep runs use the
    m1-s0 prompts); `name` keeps those outputs from colliding with the main run.
    """
    root = Path(root)
    adapter, name = adapter or run, name or run
    fine_tuned = adapter != "base"
    prompts = []
    for path in prompt_paths:
        prompts += read_jsonl(path)
    if not prompts or len({row["est_id"] for row in prompts}) != len(prompts):
        raise ValueError("Estimate prompts must be non-empty and have unique IDs")
    require_preregistration(root, {row["split"] for row in prompts}, fine_tuned=fine_tuned)
    if fine_tuned:
        require_gate(root, "G6")
    identity = _identity(root, adapter)
    pins = config(root, "pins.yaml")["subject"]
    model_sha = identity["model_sha"] if fine_tuned else pins["revision"]
    code_hashes = {
        source: file_hash(root / source)
        for source in ESTIMATE_SOURCES + ("src/anthropochrony/gen/engine.py",)
    }
    code_sha = digest(code_hashes)
    directory = root / "data/eval/jobs" / f"{name}-{label}"
    directory.mkdir(parents=True, exist_ok=True)
    combined = directory / "prompts.jsonl"
    write_jsonl(combined, prompts, inputs=list(prompt_paths))
    fingerprint = digest(
        {"prompts": prompts, "model_sha": model_sha, "code_sha": code_sha, "callable_schema": 2}
    )
    outputs = [
        "estimates.jsonl",
        "estimates.gpu.json",
        "estimates.performance.json",
        f"auditkit/{fingerprint[:20]}/run.json",
    ] + (["merge.json"] if fine_tuned else [])
    job = GPUJob(
        root,
        directory,
        kind=f"est-{name}",
        script="jobs/eval_estimates.py",
        files=[(combined, "prompts.jsonl")],
        arguments=[
            "--prompts",
            "/code/prompts.jsonl",
            "--model",
            "/out/base",
            "--model-sha",
            model_sha,
            "--code-sha",
            code_sha,
            "--config",
            "/code/configs/measure.yaml",
            *(["--adapter", f"/out/adapters/{adapter}"] if fine_tuned else []),
        ],
        outputs=outputs,
        timeout_s=timeout_s,
        context={"run": name, "model_sha": model_sha, "code_sha": code_sha, "label": label},
        sources=ESTIMATE_SOURCES,
        include_auditkit=True,
        required_gates=("G0-security", "G1", "G2", "G3"),
    )
    job.run()
    rows = read_jsonl(directory / "estimates.jsonl")
    if {row["est_id"] for row in rows} != {row["est_id"] for row in prompts} or any(
        row["model_sha"] != model_sha or row["auditkit_fingerprint"] != fingerprint for row in rows
    ):
        raise ValueError("Estimate job returned rows that do not match the requested prompts")
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    for split, split_rows in by_split.items():
        write_jsonl(
            root / f"data/eval/{name}-{split}-estimates.jsonl",
            sorted(split_rows, key=lambda row: row["est_id"]),
            inputs=[directory / "estimates.jsonl"],
        )
    return by_split


def run_g6(root, run, *, timeout_s=1800):
    """G6: loss/validation evidence is read locally; this GPU job checks merged-vs-PEFT parity."""
    root = Path(root)
    require_gate(root, "G5")
    identity = _identity(root, run)
    prompts_path = root / f"data/eval/{run}-val-prompts.jsonl"
    prompts = [row for row in read_jsonl(prompts_path) if row["card_id"] == "REF"]
    chosen = sorted(prompts, key=lambda row: digest(row["est_id"]))[:50]
    if len(chosen) != 50:
        raise GateError("G6 needs 50 validation prompts")
    directory = root / "data/eval/jobs" / f"{run}-g6"
    directory.mkdir(parents=True, exist_ok=True)
    selected = directory / "prompts.jsonl"
    write_jsonl(selected, chosen, inputs=[prompts_path])
    job = GPUJob(
        root,
        directory,
        kind=f"g6-{run}",
        script="jobs/g6_check.py",
        files=[(selected, "prompts.jsonl")],
        arguments=[
            "--prompts",
            "/code/prompts.jsonl",
            "--adapter",
            f"/out/adapters/{run}",
            "--config",
            "/code/configs/measure.yaml",
        ],
        outputs=["g6-equivalence.json", "hf-peft.jsonl"],
        timeout_s=timeout_s,
        context={"run": run, "model_sha": identity["model_sha"]},
        required_gates=("G0-security", "G1", "G2", "G5"),
    )
    job.run()
    return json.loads((directory / "g6-equivalence.json").read_text())


def g6_gate(root, run):
    """Assemble G6 from real training logs, val metrics, parse rate and merged-serving parity."""
    root = Path(root)
    evidence = run_g6(root, run)
    summary = root / "adapters" / run / "training-summary.json"
    log = read_jsonl(root / "adapters" / run / "training-log.jsonl")
    losses = [row["loss"] for row in log if "loss" in row]
    estimates = read_jsonl(root / f"data/eval/{run}-val-estimates.jsonl")
    own = [row for row in estimates if row["frame"] == "self" and row["card_id"] == "REF"]
    parse_rate = sum(row["parsed"]["point_s"] is not None for row in own) / max(1, len(own))
    base = read_jsonl(root / "data/eval/base-val-estimates.jsonl")
    base_own = [row for row in base if row["frame"] == "self" and row["card_id"] == "REF"]
    checks = {
        "training_loss_finite": bool(losses) and all(map(_finite, losses)),
        "loss_decreases": len(losses) >= 4 and _tail_mean(losses) < _head_mean(losses),
        "val_parse_at_least_98pct": parse_rate >= 0.98,
        "val_median_abs_log_error_beats_base": median_abs_error(own) < median_abs_error(base_own),
        "merged_matches_hf_peft_within_2": evidence["numeric_mismatches"] <= 2,
    }
    record_gate(
        root,
        "G6",
        checks,
        evidence={
            "run": run,
            "val_parse_rate": parse_rate,
            "val_error": median_abs_error(own),
            "base_val_error": median_abs_error(base_own),
            "numeric_mismatches": evidence["numeric_mismatches"],
            "training_summary_sha256": file_hash(summary),
        },
    )
    write_json(root / "results/gates/G6-evidence.json", evidence)
    return checks


def median_abs_error(rows):
    from anthropochrony.eval.metrics import summary, task_records
    from anthropochrony.eval.reporting import normalized_rows

    return summary(task_records(normalized_rows(rows)))["mdae_log10"]


def _finite(value):
    return value == value and abs(value) != float("inf")


def _head_mean(values):
    quarter = len(values) // 4
    return sum(values[:quarter]) / quarter


def _tail_mean(values):
    quarter = len(values) // 4
    return sum(values[-quarter:]) / quarter
