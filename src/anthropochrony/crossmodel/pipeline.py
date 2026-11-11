"""Cross-model arm driver: stage weights, measure, label, train, estimate, score.

Paid steps run on HF Jobs (GPU-only; candidate code runs only in the local sandbox). Every launch
is budget-reserved and logged through the pinned launch helpers.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi

from anthropochrony.artifacts import (
    digest,
    file_hash,
    now,
    read_jsonl,
    write_json,
    write_jsonl,
)
from anthropochrony.budget import Budget, reconcile_terminal_jobs
from anthropochrony.cli import config, reference_card
from anthropochrony.crossmodel import (
    ADAPTER_RUN,
    BASE_PATH,
    EOT,
    EVAL_CARDS,
    MODEL_ID,
    MODEL_SHA,
    TRAIN_TASKS,
    K,
)
from anthropochrony.crossmodel.rounds import prepare_round, system_token_count
from anthropochrony.gates import require_gate
from anthropochrony.gen.engine import render
from anthropochrony.gpu_jobs import GPUJob
from anthropochrony.jobs import launch, payload
from anthropochrony.measure.remote import (
    BUCKET_ID,
    TERMINAL,
    QueueTimeout,
    settle_unstarted,
    wait_for_job,
)
from anthropochrony.prompts import system_prompt
from anthropochrony.remote import authorization_guard
from anthropochrony.timemodel import SpeedCard

TOKENIZER = "pins/aya-expanse-8b"
DIRECTORY = "data/crossmodel"


def tokenizer(root):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(root) / TOKENIZER, local_files_only=True)


def select_tasks(root):
    """Deterministic subset: stratified train tasks plus every test and test_ood task."""
    root = Path(root)
    pool = read_jsonl(root / "data/tasks/tasks.jsonl")
    chosen = []
    for family, count in TRAIN_TASKS.items():
        train = sorted(
            (t for t in pool if t["split"] == "train" and t["family"] == family),
            key=lambda t: digest([t["task_id"], "xm-subset"]),
        )
        chosen += train[:count]
    chosen += [t for t in pool if t["split"] in {"test", "test_ood"}]
    tok, card = tokenizer(root), reference_card(root)
    system = system_prompt(card, estimator="agent")
    system_tokens = system_token_count(tok, system)
    tasks = []
    for task in sorted(chosen, key=lambda t: t["task_id"]):
        total = len(
            render(
                tok,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": task["task_turn"]},
                ],
            )
        )
        tasks.append(task | {"prompt_tokens": total, "uncached_task_tokens": total - system_tokens})
    return tasks, tok, system_tokens


def _directory(root):
    path = Path(root) / DIRECTORY
    path.mkdir(parents=True, exist_ok=True)
    return path


def _wait(root, name, job_id, timeout_s):
    """Wait to a verified terminal state; cancel on any other exit so nothing keeps billing."""
    api, terminal = HfApi(), None
    try:
        with authorization_guard():
            terminal = wait_for_job(api, job_id, timeout_s)
            Budget(root).update(name, state=terminal.lower(), terminal_verified=True)
            if terminal != "COMPLETED":
                raise RuntimeError(f"{name} ended {terminal}; inspect it before retrying")
    except BaseException as error:
        if isinstance(error, QueueTimeout) and settle_unstarted(root, api, name, job_id):
            terminal = "CANCELED_UNSTARTED"
            raise
        if terminal not in TERMINAL:
            accepted = False
            try:
                api.cancel_job(job_id=job_id)
                accepted = True
            finally:
                Budget(root).update(
                    name, state="cancel_requested" if accepted else "cancel_failed_timeout_reserved"
                )
        raise
    finally:
        with (Path(root) / "RUNLOG.md").open("a") as log:
            log.write(f"\n{now()}: {name} {job_id} terminal={terminal}; cross-model arm.\n")


def run_plain_job(root, name, *, flavor, timeout_s, secret, files, volumes_extra, command, script):
    """Launch once (or resume the recorded job) and wait for it."""
    root = Path(root)
    metadata = root / f"artifacts/jobs/{name}.json"
    if metadata.exists():
        job = json.loads(metadata.read_text())
    else:
        code = payload(root, root / f"artifacts/payloads/{name}", files=files)
        launch(
            root,
            name,
            flavor=flavor,
            timeout_s=timeout_s,
            image=str(root / script),
            uv_script=True,
            secret=secret,
            volumes=[f"{code}:/code", f"hf://buckets/{BUCKET_ID}:/out", *volumes_extra],
            command=command,
        )
        job = json.loads(metadata.read_text())
    _wait(root, name, job["job_id"], timeout_s)
    return job


def stage(root):
    root = Path(root)
    reconcile_terminal_jobs(root, HfApi())
    require_gate(root, "G0-security")
    run_plain_job(
        root,
        "xm-stage-aya-expanse-8b",
        flavor="cpu-basic",
        timeout_s=1800,
        secret=True,
        files=[],
        volumes_extra=[],
        script="jobs/xm_stage.py",
        command=["--", "--output", BASE_PATH],
    )


ENGINE = {
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.92,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 16384,
    "disable_log_stats": False,
}


def generation_job(root, directory, requests, *, kind, timeout_s, adapter=None, greedy=False):
    root, directory = Path(root), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "requests.jsonl"
    if not request_path.exists():
        write_jsonl(request_path, requests)
    elif read_jsonl(request_path) != requests:
        raise ValueError("Saved requests differ from the current ones")
    config_path = directory / "config.json"
    cfg = {"engine": ENGINE, "model_id": MODEL_ID, "model_sha": MODEL_SHA, "greedy": greedy}
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    arguments = ["--requests", "/code/requests.jsonl", "--config", "/code/config.json"]
    if adapter:
        arguments += ["--adapter", f"/out/adapters/{adapter}"]
    if greedy:
        arguments += ["--greedy"]
    job = GPUJob(
        root,
        directory,
        kind=kind,
        script="jobs/xm_generate.py",
        files=[(request_path, "requests.jsonl"), (config_path, "config.json")],
        arguments=arguments,
        outputs=["generation.jsonl", "generation.performance.json", "merge-evidence.json"],
        timeout_s=timeout_s,
        context={"config": cfg, "adapter": adapter},
        sources=("jobs/xm_generate.py",),
    )
    done = job.run()
    return read_jsonl(done / "generation.jsonl")


def measure(root):
    """Round-synchronous K=4 trajectories on Aya Expanse 8B at the reference card."""
    from anthropochrony.measure.rounds import apply_round, initial_states
    from anthropochrony.sandbox import Sandbox

    root = Path(root)
    for gate in ("G0-security", "G1", "G2"):
        require_gate(root, gate)
    tasks, tok, system_tokens = select_tasks(root)
    card = reference_card(root)
    base = _directory(root) / "measure"
    base.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_id": MODEL_ID,
        "model_sha": MODEL_SHA,
        "k": K,
        "temperature": 0.1,
        "top_p": 0.95,
        "workers": 8,
    }
    checkpoint = base / "checkpoint.json"
    identity = digest({"tasks": [t["task_id"] for t in tasks], "config": cfg})
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        if record["identity"] != identity:
            raise ValueError("Refusing to resume with a changed subset or configuration")
        states = record["states"]
    else:
        states = initial_states(tasks, card, cfg)
    sandbox = Sandbox()
    timeouts = {1: 4200, 2: 2700, 3: 2100}
    for number in (1, 2, 3):
        requests = prepare_round(states, tasks, tok, number, system_tokens)
        if not requests:
            continue
        outputs = generation_job(
            root,
            base / f"r{number}",
            requests,
            kind=f"xmgen{number}",
            timeout_s=timeouts[number],
        )
        states, texts = apply_round(
            states, tasks, requests, outputs, sandbox, number, cfg["workers"]
        )
        write_json(checkpoint, {"identity": identity, "states": states})
        write_jsonl(
            base / "traces.jsonl", [{k: v for k, v in s.items() if k != "messages"} for s in states]
        )
        write_jsonl(base / f"outputs-r{number}.jsonl", texts)
    return states


def _cards(root, tasks):
    from anthropochrony.labels.speeds import sample_cards

    speeds = config(root, "speeds.yaml")
    shared = [reference_card(root)] + [SpeedCard(**c) for c in speeds["evaluation"]]
    cards = {}
    for task in tasks:
        cards[task["task_id"]] = list(shared)
        if task["split"] == "train":
            cards[task["task_id"]] += sample_cards(task["task_id"], speeds)
    return cards


def build_labels_and_priors(root):
    from anthropochrony.eval.structured import WorkloadPriors
    from anthropochrony.labels import build_labels, training_priors

    root = Path(root)
    tasks, _, _ = select_tasks(root)
    traces = read_jsonl(_directory(root) / "measure/traces.jsonl")
    by_task = {t["task_id"]: t for t in tasks}
    priors = training_priors(tasks, traces)
    labels, dropped = build_labels(
        tasks, traces, _cards(root, tasks), k=K, label_source="aya-expanse-8b", priors=priors
    )
    feedback, execution = defaultdict(list), defaultdict(list)
    for trace in traces:
        task = by_task[trace["task_id"]]
        if task["split"] != "train":
            continue
        for index, attempt in enumerate(trace["attempts"]):
            execution[task["family"]].append(float(attempt["exec"]["wall_s"]))
            if index:
                feedback[task["family"]].append(float(attempt["u_tokens"]))
    fb = {f: float(np.median(v)) for f, v in feedback.items()}
    ex = {f: float(np.median(v)) for f, v in execution.items()}
    work = WorkloadPriors(
        feedback_by_family=fb,
        feedback_default=float(np.median(list(fb.values()))),
        exec_per_attempt_by_family=ex,
        exec_per_attempt_default=float(np.median(list(ex.values()))),
    )
    directory = _directory(root)
    write_jsonl(directory / "labels.jsonl", labels)
    write_json(directory / "workload-priors.json", work.as_dict())
    write_json(directory / "label-dropped.json", dropped)
    return tasks, labels, work


def build_training_data(root):
    from anthropochrony.curate.sft import build_sft

    root = Path(root)
    tasks, tok, _ = select_tasks(root)
    labels = read_jsonl(_directory(root) / "labels.jsonl")
    train_tasks = [t for t in tasks if t["split"] == "train"]
    rows, drops = build_sft(train_tasks, labels, {}, tok, variant="m4", max_length=3072)
    examples = []
    for row in rows:
        prompt = row["messages"][:-1]
        answer = row["messages"][-1]["content"]
        prompt_ids = render(tok, prompt)
        completion_ids = tok.encode(answer, add_special_tokens=False) + [EOT]
        full = render(tok, row["messages"], generation_prompt=False)
        if full != prompt_ids + completion_ids:
            raise ValueError(f"Template mismatch for {row['example_id']}")
        examples.append(
            {
                "example_id": row["example_id"],
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
            }
        )
    path = _directory(root) / "train.jsonl"
    write_jsonl(path, examples)
    write_jsonl(_directory(root) / "sft-rows.jsonl", rows)
    summary = {
        "examples": len(examples),
        "dropped": len(drops),
        "tasks": len({r["task_id"] for r in rows}),
        "mean_tokens": float(
            np.mean([len(e["prompt_ids"]) + len(e["completion_ids"]) for e in examples])
        ),
        "max_tokens": max(len(e["prompt_ids"]) + len(e["completion_ids"]) for e in examples),
        "completion_tokens": sum(len(e["completion_ids"]) for e in examples),
    }
    write_json(_directory(root) / "train-summary.json", summary)
    return summary


def train(root, *, max_steps=None):
    root = Path(root)
    reconcile_terminal_jobs(root, HfApi())
    data = _directory(root) / "train.jsonl"
    stamp = digest(
        {
            "data": file_hash(data),
            "script": file_hash(root / "jobs/xm_train.py"),
            "max_steps": max_steps,
        }
    )[:12]
    name = f"xm-train-{stamp}"
    timeout_s = 900 if max_steps else 6000
    command = [
        "--",
        "--base",
        BASE_PATH,
        "--data",
        "/code/train.jsonl",
        "--output",
        f"/out/adapters/{ADAPTER_RUN}" + ("-dry" if max_steps else ""),
        "--lr",
        "0.0002",
        "--epochs",
        "2",
    ]
    if max_steps:
        command += ["--max-steps", str(max_steps)]
    run_plain_job(
        root,
        name,
        flavor="l40sx1",
        timeout_s=timeout_s,
        secret=False,
        files=[(data, "train.jsonl")],
        volumes_extra=[],
        script="jobs/xm_train.py",
        command=command,
    )


def _estimate_prompts(root):
    from anthropochrony.eval.prompts import build_evaluation_prompts

    tasks, tok, _ = select_tasks(root)
    labels = read_jsonl(_directory(root) / "labels.jsonl")
    wanted = set(EVAL_CARDS)
    labels = [r for r in labels if r["card_id"] in wanted]

    def keep_first(rows):
        seen, kept = set(), []
        for row in rows:
            if row["frame"] != "self" or row["card_id"] not in wanted:
                continue
            key = (row["task_id"], row["card_id"], row["estimator"])
            if key not in seen:
                seen.add(key)
                kept.append(row)
        return kept

    base, tuned = [], []
    for split in ("test", "test_ood"):
        base += keep_first(
            build_evaluation_prompts(tasks, labels, split=split, include_fewshot=True)
        )
        tuned += keep_first(build_evaluation_prompts(tasks, labels, split=split, fine_tuned="xm8b"))
    return base, tuned, tok


def _requests(prompts, tok):
    requests = []
    for row in prompts:
        ids = render(tok, row["messages"])
        if row.get("prefill"):
            ids = ids + tok.encode(row["prefill"], add_special_tokens=False)
        requests.append(
            {
                "request_id": row["est_id"],
                "prompt_token_ids": ids,
                "max_tokens": min(row["max_tokens"], 8192 - len(ids)),
                "seed": 0,
            }
        )
    return requests


def estimate(root, which):
    root = Path(root)
    reconcile_terminal_jobs(root, HfApi())
    base, tuned, tok = _estimate_prompts(root)
    prompts = base if which == "base" else tuned
    requests = [r for r in _requests(prompts, tok) if r["max_tokens"] > 0]
    outputs = generation_job(
        root,
        _directory(root) / f"estimates-{which}",
        requests,
        kind=f"xmest{which}",
        timeout_s=2400 if which == "base" else 3000,
        adapter=None if which == "base" else ADAPTER_RUN,
        greedy=True,
    )
    by_id = {row["request_id"]: row for row in outputs}
    rows = []
    for prompt in prompts:
        generation = by_id[prompt["est_id"]]
        rows.append(
            prompt
            | {
                "raw_text": prompt.get("prefill", "") + generation["text"],
                "finish_reason": generation["finish_reason"],
                "model_called": True,
            }
        )
    write_jsonl(_directory(root) / f"estimates-{which}.rows.jsonl", rows)
    return rows


def report(root):
    from anthropochrony.eval.reporting import normalized_rows
    from anthropochrony.eval.stats import json_safe
    from anthropochrony.eval.structured import (
        WorkloadPriors,
        _metric_line,
        build_structured_rows,
    )

    root = Path(root)
    directory = _directory(root)
    tasks, _, _ = select_tasks(root)
    by_task = {t["task_id"]: t for t in tasks}
    work = WorkloadPriors(**json.loads((directory / "workload-priors.json").read_text()))
    rows = normalized_rows(
        read_jsonl(directory / "estimates-base.rows.jsonl")
        + read_jsonl(directory / "estimates-tuned.rows.jsonl")
    )
    rows += build_structured_rows(
        rows, by_task, work, source_estimator="XM8B", estimator_name="XM8B-Struct"
    )
    rows += build_structured_rows(
        rows, by_task, work, source_estimator="E2", estimator_name="E2-Struct"
    )
    labels = read_jsonl(directory / "labels.jsonl")
    train_ref = defaultdict(list)
    for label in labels:
        if by_task[label["task_id"]]["split"] == "train":
            train_ref[label["card_id"]].append(label["T_med_s"])
    constant = {card: statistics.median(values) for card, values in train_ref.items()}
    for row in [r for r in rows if r["estimator"] == "E1"]:
        rows.append(row | {"estimator": "B1-constant", "point_s": constant[row["card_id"]]})
    out = {"generated_at": now(), "model": MODEL_ID, "model_sha": MODEL_SHA, "results": {}}
    estimators = sorted({r["estimator"] for r in rows})
    for split in ("test", "test_ood"):
        for scope, cards in (
            ("REF", {"REF"}),
            ("extrap-low", {"extrap-low"}),
            ("all-cards", set(EVAL_CARDS)),
        ):
            for estimator in estimators:
                subset = [
                    r
                    for r in rows
                    if r["split"] == split and r["estimator"] == estimator and r["card_id"] in cards
                ]
                if subset:
                    out["results"][f"{split}/{scope}/{estimator}"] = _metric_line(subset)
    token_errors = {}
    for estimator in ("E1", "E2", "XM8B"):
        errors = []
        for r in rows:
            if r["estimator"] != estimator or r["card_id"] != "REF" or r["split"] != "test":
                continue
            truth = next(
                lab["TOK_med"]
                for lab in labels
                if lab["task_id"] == r["task_id"] and lab["card_id"] == "REF"
            )
            if r.get("tokens") and r["tokens"] > 0:
                errors.append(abs(np.log10(r["tokens"] / truth)))
            else:
                errors.append(float("inf"))
        if errors:
            token_errors[estimator] = float(np.median(errors))
    out["token_mdae_log10_test_ref"] = token_errors
    write_json(root / "results/crossmodel.json", json_safe(out))
    return out
