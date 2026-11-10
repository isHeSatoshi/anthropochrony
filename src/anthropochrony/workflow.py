"""Pipeline entry points. Each stage validates prerequisites before spending money."""

import json
import shutil
import subprocess

import pandas as pd

from anthropochrony.artifacts import digest, file_hash, git_sha, read_jsonl, write_json, write_jsonl
from anthropochrony.cli import config, reference_card
from anthropochrony.gates import GateError, record_gate, require_gate, require_preregistration
from anthropochrony.jobs import launch, payload

BUCKET = "hf://buckets/isHeSatoshi/anthropochrony-runs:/out"


def security_gate(root):
    completed = subprocess.run(
        ["uv", "run", "pytest", "-m", "security", "-q"],
        cwd=root,
        check=False,
    )
    record_gate(root, "G0-security", {"sandbox_security_suite": completed.returncode == 0})


def smoke(root, remote=False):
    from anthropochrony.preflight import preflight
    from anthropochrony.prompts import compile_prompts

    compile_prompts(check=True)
    preflight(root, config(root, "pins.yaml"))
    security_gate(root)
    if remote:
        require_gate(root, "G0-security")
        print(
            launch(
                root,
                "s0-cpu-smoke",
                flavor="cpu-basic",
                timeout_s=120,
                image="python:3.12",
                command=["python", "-c", "print(1)"],
            )
        )


def measurement(root, args):
    from transformers import AutoTokenizer

    from anthropochrony.measure.rounds import Measurement, apply_round, prepare_round
    from anthropochrony.sandbox import Sandbox
    from anthropochrony.tasks.loaders import stratified_sample

    require_gate(root, "G1")
    require_gate(root, "G2")
    phase = getattr(args, "phase", "pilot")
    if phase == "full":
        require_gate(root, "G3")
    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    if args.run != "base":
        require_gate(root, "G6")
        tasks = [t for t in tasks if t["split"] in {"test", "test_ood", "showcase"}]
    if phase == "pilot":
        tasks = stratified_sample(
            [t for t in tasks if t["split"] != "raw_extra"], 100, key=lambda t: t["family"]
        )
        if len(tasks) != 100:
            raise GateError("The measurement pilot requires 100 stratified tasks")
    require_preregistration(root, {t["split"] for t in tasks}, fine_tuned=args.run != "base")
    measure_config = config(root, "measure.yaml")
    if args.run != "base":
        adapter_info = root / "adapters" / args.run / "identity.json"
        identity = json.loads(adapter_info.read_text())
        measure_config.update(model_id=identity["model_id"], model_sha=identity["model_sha"])
    measurement_id = f"{args.run}-pilot" if phase == "pilot" else args.run
    experiment = Measurement(
        root / "data/measure" / measurement_id, tasks, measure_config, reference_card(root)
    )
    states = experiment.states()
    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    if args.command == "execute-round":
        round_number = args.round
        request_path = experiment.directory / f"requests-r{round_number}.jsonl"
        outputs = read_jsonl(args.generation)
        requests = read_jsonl(request_path)
        states, texts = apply_round(
            states, tasks, requests, outputs, Sandbox(), round_number, measure_config["workers"]
        )
        write_jsonl(
            experiment.directory / f"outputs-r{round_number}.jsonl", texts, inputs=[args.generation]
        )
        experiment.save(states)
        if all(state["done"] for state in states):
            measurement_gate(root, states, tasks, phase=phase, run=args.run)
        return
    round_number = getattr(args, "round", None) or next(
        (
            r
            for r in range(1, 4)
            if any(not s["done"] and len(s["attempts"]) == r - 1 for s in states)
        ),
        0,
    )
    if round_number == 0:
        print("All trajectories are complete")
        return
    requests = prepare_round(states, tasks, tokenizer, round_number)
    experiment.save(states)
    request_path = experiment.directory / f"requests-r{round_number}.jsonl"
    write_jsonl(request_path, requests, config=measure_config)
    print(f"Prepared {len(requests)} requests in {request_path}")
    if args.command != "measure":
        return
    stamp = digest(requests)[:10]
    code = payload(
        root,
        root / "artifacts/payloads" / f"measure-{measurement_id}-r{round_number}-{stamp}",
        files=[(request_path, "requests.jsonl")],
    )
    model_path = "/out/base" if args.run == "base" else f"/out/merged/{args.run}"
    print(
        launch(
            root,
            f"measure-{measurement_id}-r{round_number}-{stamp}",
            flavor="l40sx1",
            timeout_s=3600,
            image=config(root, "pins.yaml")["generation_image"],
            secret=False,
            volumes=[f"{code}:/code", BUCKET],
            command=[
                "python3",
                "/code/jobs/measure_round.py",
                "--requests",
                "/code/requests.jsonl",
                "--model",
                model_path,
                "--output",
                f"/out/measure/{measurement_id}/r{round_number}.jsonl",
                "--config",
                "/code/configs/measure.yaml",
            ],
            dry_run=args.dry_run,
        )
    )
    print(
        "After completion: download the generation JSONL, then run execute-round. No GPU waits for tests."
    )


def measurement_gate(root, states, tasks, *, phase="pilot", run="base", require_invariance=True):
    from collections import Counter

    from anthropochrony.measure.invariance import invariance_passed

    families = {t["task_id"]: t["family"] for t in tasks}
    attempts = [a for state in states for a in state["attempts"]]
    by_family = {}
    for task_id, family in families.items():
        by_family.setdefault(family, []).extend(
            a for s in states if s["task_id"] == task_id for a in s["attempts"]
        )
    checks = {
        "harness_errors_below_1pct": sum(a["exec"]["status"] == "error" for a in attempts)
        / max(1, len(attempts))
        < 0.01,
        "runs_complete_99pct": sum(s["done"] for s in states) / len(states) >= 0.99,
        "length_below_5pct_each_family": all(
            sum(a["finish_reason"] == "length" for a in group) / len(group) < 0.05
            for group in by_family.values()
            if group
        ),
    }
    if phase == "pilot":
        checks["pilot_100_tasks_k8"] = (
            len(families) == 100
            and len(states) == 800
            and Counter(s["task_id"] for s in states)
            == Counter({task_id: 8 for task_id in families})
        )
    # Invariance needs independently generated slow/fast-card runs. Never silently waive it.
    invariance = root / "data/measure/invariance.json"
    if require_invariance:
        checks["speed_card_invariance"] = invariance.exists() and invariance_passed(
            json.loads(invariance.read_text())
        )
    pass_at_one = {
        family: sum(bool(s["first_pass"]) for s in states if families[s["task_id"]] == family)
        / sum(families[s["task_id"]] == family for s in states)
        for family in set(families.values())
    }
    record_gate(
        root,
        ("G3" if phase == "pilot" and run == "base" else f"G3-{run}-{phase}")
        if require_invariance
        else f"G3-{run}-{phase}-quality",
        checks,
        evidence={
            "phase": phase,
            "pass_at_1_by_family": pass_at_one,
            "tasks_sha256": digest(tasks),
        },
    )


def calibrate(root):
    require_gate(root, "G0-security")
    stamp = git_sha(root)[:10]
    code = payload(root, root / "artifacts/payloads" / f"calibrate-{stamp}")
    print(
        launch(
            root,
            f"calibrate-{stamp}",
            flavor="l40sx1",
            timeout_s=1800,
            image=config(root, "pins.yaml")["generation_image"],
            volumes=[f"{code}:/code", BUCKET],
            secret=True,
            command=[
                "python3",
                "/code/jobs/calibrate_l4.py",
                "--model",
                "/out/base",
                "--output",
                "/out/calibration/reference.json",
            ],
        )
    )


def baselines(root, teacher=False):
    from anthropochrony.baselines.features import embeddings
    from anthropochrony.baselines.regressors import ConstantBaseline, TunedInputRegressor
    from anthropochrony.eval.metrics import noise_ceiling
    from anthropochrony.eval.stats import json_safe
    from anthropochrony.timemodel import SpeedCard

    require_gate(root, "G3")
    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    traces = read_jsonl(root / "data/measure/base/traces.jsonl")
    labels = pd.read_parquet(root / "data/labels/base.parquet").to_dict("records")
    constant = ConstantBaseline().fit(tasks, labels)
    baseline_config = config(root, "baselines.yaml")
    features = embeddings(tasks, baseline_config["embedding"], root / "data/embeddings")
    tuning = {"alphas": baseline_config["ridge_alpha"], "blends": baseline_config["ridge_blend"]}
    regressor = TunedInputRegressor().fit(
        tasks, traces, embeddings=features, labels=labels, **tuning
    )
    privileged = TunedInputRegressor(privileged=True).fit(
        tasks, traces, embeddings=features, labels=labels, **tuning
    )
    write_json(
        root / "results/baseline-selection.json",
        {"B2": regressor.selection, "B2+": privileged.selection},
        config=baseline_config,
    )
    task_map = {t["task_id"]: t for t in tasks}
    rows = []
    for label in labels:
        if label["split"] not in {"val", "test", "test_ood"} or label["card_id"].startswith(
            "train-"
        ):
            continue
        task = task_map[label["task_id"]]
        card = SpeedCard(label["card_id"], label["S"], label["P"])
        for estimator, prediction in {
            "B0": constant.predict(task, card),
            "B1": constant.predict(task, card, family=True),
            "B2": regressor.predict(task, card),
            "B2+": privileged.predict(task, card),
        }.items():
            rows.append(
                {
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "split": task["split"],
                    "card_id": card.id,
                    "estimator": estimator,
                    "frame": "self",
                    "parsed": {"point_s": prediction},
                    "T_med_s": label["T_med_s"],
                }
            )
    write_jsonl(root / "data/eval/statistical.jsonl", rows)
    write_json(
        root / "results/noise-ceiling.json", json_safe(noise_ceiling(traces, reference_card(root)))
    )
    if teacher:
        collect_teacher(root)


def feedback_medians(root):
    """Median new tokens of a retry turn per family, from training-split runs only."""
    import statistics

    tasks = {t["task_id"]: t for t in read_jsonl(root / "data/tasks/tasks.jsonl")}
    values = {}
    for trace in read_jsonl(root / "data/measure/base/traces.jsonl"):
        task = tasks[trace["task_id"]]
        if task["split"] != "train":
            continue
        values.setdefault(task["family"], []).extend(
            a["u_tokens"] for a in trace["attempts"] if a["j"] >= 2
        )
    pooled = float(statistics.median([x for v in values.values() for x in v]))
    # The unseen CLASS family has no training tasks, so it uses the pooled training median.
    result = {
        family: float(statistics.median(values[family])) if values.get(family) else pooled
        for family in {t["family"] for t in tasks.values()}
    }
    if not values:
        raise GateError("Training runs contain no retry-turn feedback tokens")
    write_json(root / "data/measure/feedback-medians.json", result)
    return result


def collect_teacher(root):
    """B4 and the M3 teacher rows: one cached DeepSeek-V4-Pro call per task."""
    from anthropochrony.baselines.provider import CachedProvider
    from anthropochrony.curate.sft import teacher_label
    from anthropochrony.timemodel import SpeedCard

    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    labels = pd.read_parquet(root / "data/labels/base.parquet").to_dict("records")
    feedback = feedback_medians(root)
    provider = CachedProvider(
        root,
        model="deepseek-ai/DeepSeek-V4-Pro",
        provider="deepinfra",
        input_per_million=1.3,
        output_per_million=2.6,
    )
    outputs = {}
    for task in tasks:
        value = provider.teacher(task)
        if value is not None:
            outputs[task["task_id"]] = value
    write_json(root / "data/teacher/quantities.json", outputs)
    missing = [t["task_id"] for t in tasks if t["task_id"] not in outputs]
    task_map = {t["task_id"]: t for t in tasks}
    rows = []
    for label in labels:
        task = task_map[label["task_id"]]
        if task["split"] not in {"val", "test", "test_ood"} or task["task_id"] not in outputs:
            continue
        if label["card_id"].startswith("train-"):
            continue
        card = SpeedCard(label["card_id"], label["S"], label["P"])
        estimate = teacher_label(task, card, outputs[task["task_id"]], feedback[task["family"]])
        rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "split": task["split"],
                "card_id": card.id,
                "estimator": "B4",
                "frame": "self",
                "parsed": {
                    "point_s": estimate["T_med_s"],
                    "lo_s": estimate["T_q10_s"],
                    "hi_s": estimate["T_q90_s"],
                },
                "T_med_s": label["T_med_s"],
            }
        )
    write_jsonl(root / "data/eval/teacher.jsonl", rows)
    write_json(root / "data/teacher/missing.json", missing)
    return outputs


def sft_data(root, variant):
    from transformers import AutoTokenizer

    from anthropochrony.artifacts import now
    from anthropochrony.curate.sft import build_matched_sft, check_matched_variants, export_sft

    require_gate(root, "G4")
    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    exclusions_path = root / "data/labels/base.dropped.jsonl"
    excluded = read_jsonl(exclusions_path) if exclusions_path.exists() else []
    excluded_ids = {row["task_id"] for row in excluded}
    tasks = [task for task in tasks if task["task_id"] not in excluded_ids]
    labels = pd.read_parquet(root / "data/labels/base.parquet").to_dict("records")
    human = read_jsonl(root / "data/eval/human-train-val.jsonl")
    raw_human = root / "data/eval/human-raw-extra.jsonl"
    if raw_human.exists():
        human += read_jsonl(raw_human)
    targets = {r["task_id"]: r for r in human}
    if len(targets) != len(human):
        raise GateError("Human preservation targets must identify one seeded answer per task")
    blockers_path = root / "data/eval/preservation-blockers.jsonl"
    truncated = (
        {
            row["task_id"]
            for row in read_jsonl(blockers_path)
            if row["reason"] == "truncated_human_target" and row["task_id"] not in targets
        }
        if blockers_path.exists()
        else set()
    )
    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    teachers = json.loads((root / "data/teacher/quantities.json").read_text())
    feedback = json.loads((root / "data/measure/feedback-medians.json").read_text())
    configs = {name: config(root, f"sft/{name}.yaml") for name in ("m1", "m2", "m3")}
    limits = {cfg["max_seq_length"] for cfg in configs.values()}
    if len(limits) != 1:
        raise GateError("All SFT variants must share one sequence limit before data preparation")
    variants, drops, report = build_matched_sft(
        tasks,
        labels,
        targets,
        tokenizer,
        teachers=teachers,
        feedback_by_family=feedback,
        max_length=limits.pop(),
        truncated_ids=frozenset(truncated),
    )
    if report["expanded_to_3072"]:
        with (root / "docs/PREREGISTRATION.md").open("a") as prereg:
            prereg.write(
                f"\n- {now()[:10]}, before training: the S5 family-wise sequence-loss rule "
                "triggered on matched M1/M2/M3 data at 2048 tokens. Rebuild every variant "
                "with max_seq_length=3072, retain matched common example IDs, and record "
                "both initial and final per-family drop shares in data/sft/matched-summary.json.\n"
            )
        for name, cfg in configs.items():
            cfg["max_seq_length"] = 3072
            write_json(root / f"configs/sft/{name}.yaml", cfg)
    exported = {}
    for name, rows in variants.items():
        write_jsonl(root / f"data/sft/{name}/dropped.jsonl", drops[name])
        exported[name] = export_sft(rows, root / "data/sft" / name)
    report["label_exclusions"] = excluded
    report["exported_counts"] = {name: len(rows) for name, rows in exported.items()}
    write_json(root / "data/sft/matched-summary.json", report)
    check_matched_variants(exported["m1"], exported["m2"])
    print(
        f"Exported all matched P0 variants ({len(exported[variant])} {variant} examples). "
        "G5 still requires the real one-step assistant-mask check."
    )


def train_job(root, args):
    stage = getattr(args, "stage", "main")
    require_gate(root, "G4" if stage == "dry-run" else "G5")
    cfg = config(root, f"sft/{args.variant}.yaml")
    if stage == "main" and not cfg["learning_rate_selected_on_val"]:
        raise GateError("Run the two-point, one-epoch val-only LR sweep before main training")
    if stage == "main":
        from anthropochrony.train.selection import require_selected_learning_rate

        require_selected_learning_rate(root, cfg)
        if args.variant == "m3" and args.seed != 0:
            raise GateError("The preregistered P0 M3 run uses seed 0 only")
    if stage == "lr-sweep":
        if args.variant != "m1" or args.seed != 0 or args.learning_rate not in {1e-4, 2e-4}:
            raise GateError("The preregistered sweep uses M1 seed 0 and lr 1e-4 or 2e-4")
        cfg.update(epochs=1, learning_rate=args.learning_rate)
        run = f"lr-{args.learning_rate:g}"
    else:
        run = f"{args.variant}-{'dryrun-' if stage == 'dry-run' else ''}s{args.seed}"
    cfg["base"] = config(root, "pins.yaml")["subject"]
    source = root / "data/sft" / args.variant
    # AlignTune's directory loader fails on extra files that map to a split, so ship only the
    # two trainer files, not the CuratorKIT curation output.
    dataset = (
        root
        / "artifacts/datasets"
        / f"{args.variant}-{file_hash(source / 'train/data.jsonl')[:12]}"
    )
    for split in ("train", "validation"):
        target = dataset / split / "data.jsonl"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / split / "data.jsonl", target)
    stamp = digest(
        {
            "config": cfg,
            "stage": stage,
            "data": file_hash(dataset / "train/data.jsonl"),
            "validation": file_hash(dataset / "validation/data.jsonl"),
            "trainer": file_hash(root / "src/anthropochrony/train/sft.py"),
            "job": file_hash(root / "jobs/train_sft.py"),
        }
    )[:12]
    name = getattr(args, "launch_name", None) or f"train-{run}-{stamp}"
    training_config = root / f"artifacts/training-configs/{name}.json"
    write_json(training_config, cfg)
    code = payload(
        root,
        root / "artifacts/payloads" / name,
        files=[(training_config, "training-config.json")],
    )
    print(
        launch(
            root,
            name,
            flavor="l40sx1",
            timeout_s=1200 if stage == "dry-run" else 3600 if stage == "lr-sweep" else 6300,
            secret=stage != "dry-run",
            uv_script=True,
            image=str(root / "jobs/train_sft.py"),
            volumes=[f"{code}:/code", BUCKET, f"{dataset}:/dataset"],
            command=[
                "--",
                "--config",
                "/code/training-config.json",
                "--base",
                "/out/base",
                "--dataset",
                "/dataset",
                "--output",
                f"/out/adapters/{run}",
                "--seed",
                str(args.seed),
                *(
                    ["--dry-run"]
                    if stage == "dry-run"
                    else [
                        "--push-repo",
                        f"isHeSatoshi/anthropochrony-tiny-aya-global-lora-{run}",
                    ]
                ),
            ],
            dry_run=args.dry_run,
        )
    )
