import argparse
import json
import math
import subprocess
from pathlib import Path

import yaml

from anthropochrony.artifacts import read_jsonl, write_json, write_jsonl, write_parquet
from anthropochrony.families import FAMILIES
from anthropochrony.gates import GateError, require_gate
from anthropochrony.timemodel import SpeedCard

ROOT = Path(__file__).resolve().parents[2]


def config(root, name):
    return yaml.safe_load((root / "configs" / name).read_text())


def reference_card(root):
    data = config(root, "speeds.yaml")
    if data["reference"] is None:
        calibration = root / "data/calibration/reference.json"
        if not calibration.exists():
            raise GateError("REF has not been measured. Run calibration on HF l40sx1 first")
        return SpeedCard(**json.loads(calibration.read_text())["reference"])
    return SpeedCard(**data["reference"])


def labels_command(root, args):
    from anthropochrony.labels import build_labels, training_priors
    from anthropochrony.labels.speeds import sample_cards

    tasks, traces = read_jsonl(args.tasks), read_jsonl(args.traces)
    speeds = config(root, "speeds.yaml")
    cards = [reference_card(root)] + [SpeedCard(**c) for c in speeds["evaluation"]]
    if args.source == "base":
        priors = training_priors(tasks, traces)
        write_json(root / "data/labels/training-priors.json", priors)
    else:
        priors = json.loads((root / "data/labels/training-priors.json").read_text())
    rows, drops = build_labels(
        tasks, traces, cards, k=args.k, label_source=args.source, priors=priors
    )
    training_cards = {}
    for task in tasks:
        if task["split"] in {"train", "val", "raw_extra"}:
            training_cards[task["task_id"]] = sample_cards(task["task_id"], speeds)
    # Family priors are computed once from all train runs, never from isolated task slices.
    if training_cards and args.source == "base":
        # Build only each task's six cards, while retaining the train-only prior.
        from anthropochrony.labels import build_labels_for_training_cards

        rows += build_labels_for_training_cards(
            tasks, traces, training_cards, k=args.k, label_source=args.source
        )
    write_parquet(args.output, rows, inputs=[args.tasks, args.traces], config=speeds)
    write_jsonl(Path(args.output).with_suffix(".dropped.jsonl"), drops)
    print(f"Wrote {len(rows)} labels")


def speed_pair(text):
    try:
        decode, prefill = (float(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected S,P such as 20,200") from error
    if not (decode > 0 and prefill > 0 and math.isfinite(decode) and math.isfinite(prefill)):
        raise argparse.ArgumentTypeError("S and P must be positive tokens per second")
    return decode, prefill


def bench_arguments(command, required):
    command.add_argument("--llama-bench", required=required, help="Path to llama-bench")
    command.add_argument("--gguf", required=required, help="GGUF file to benchmark")
    command.add_argument("--threads", type=int, default=8)
    command.add_argument("--gpu-layers", type=int, default=0, help="Layers to offload, as -ngl")


def selftime_command(args):
    from urllib.error import URLError

    from anthropochrony.demo import ContextExhausted
    from anthropochrony.selftime import SelfTimeEstimator

    if args.speed is None and not (args.llama_bench and args.gguf):
        raise ValueError("Give --speed S,P, or --llama-bench PATH and --gguf PATH to measure it")
    task = args.task.read_text(encoding="utf-8") if args.task else args.text
    estimator = SelfTimeEstimator.from_llama_server(
        args.server,
        speed=args.speed,
        llama_bench=args.llama_bench,
        gguf=args.gguf,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
    )
    try:
        estimate = estimator.estimate(task, family=args.family)
        human = estimator.human_estimate(task) if args.human else None
    except (URLError, ConnectionError, TimeoutError) as error:
        raise SystemExit(f"Could not reach llama-server at {args.server}: {error}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"llama-bench failed: {error.stderr or error}") from error
    except ContextExhausted as error:
        raise SystemExit(f"The task is too long for the model context: {error}") from error
    if args.json:
        payload = {"self": estimate.as_dict(), "human": human.as_dict() if human else None}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(estimate.summary())
        if human:
            print(human.summary())


def parser():
    result = argparse.ArgumentParser(description="Anthropochrony research pipeline")
    result.add_argument("--root", type=Path, default=ROOT)
    commands = result.add_subparsers(dest="command", required=True)
    prompts = commands.add_parser("prompts")
    prompts.add_argument("--check", action="store_true")
    commands.add_parser("preflight")
    commands.add_parser("status")
    commands.add_parser("security")
    tasks = commands.add_parser("tasks")
    tasks.add_argument("--local-data", type=Path)
    tasks.add_argument("--workers", type=int, default=8)
    labels = commands.add_parser("labels")
    labels.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks.jsonl")
    labels.add_argument("--traces", type=Path, default=ROOT / "data/measure/base/traces.jsonl")
    labels.add_argument("--output", type=Path, default=ROOT / "data/labels/base.parquet")
    labels.add_argument("--k", type=int, default=8)
    labels.add_argument("--source", default="base")
    for name in ("prepare-round", "execute-round"):
        round_parser = commands.add_parser(name)
        round_parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
        round_parser.add_argument("--run", default="base")
        round_parser.add_argument("--phase", choices=["pilot", "full"], default="pilot")
        round_parser.add_argument("--generation", type=Path)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--remote", action="store_true")
    commands.add_parser("calibrate")
    measure = commands.add_parser("measure")
    measure.add_argument("--run", default="base")
    measure.add_argument("--dry-run", action="store_true")
    measure.add_argument("--auto", action="store_true")
    measure.add_argument("--phase", choices=["pilot", "full"], default="pilot")
    invariance = commands.add_parser("invariance")
    for name in ("reference", "slow", "fast"):
        invariance.add_argument(f"--{name}", type=Path, required=True)
    baseline = commands.add_parser("baselines")
    baseline.add_argument("--teacher", action="store_true")
    commands.add_parser("teacher")
    commands.add_parser("proj8")
    showcase = commands.add_parser("showcase")
    showcase.add_argument("--runs", default="base")
    commands.add_parser("smoke-merged")
    sft = commands.add_parser("sft-data")
    sft.add_argument("--variant", choices=["m1", "m2", "m3"], default="m1")
    sft_gate = commands.add_parser("sft-gate")
    sft_gate.add_argument("--dry-run-summary", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--variant", choices=["m1", "m2", "m3"], default="m1")
    train.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--stage", choices=["dry-run", "lr-sweep", "main"], default="main")
    train.add_argument("--learning-rate", type=float, choices=[1e-4, 2e-4])
    train.add_argument("--launch-name")
    selection = commands.add_parser("select-lr")
    for name in (
        "low-estimates",
        "low-training-summary",
        "high-estimates",
        "high-training-summary",
    ):
        selection.add_argument(f"--{name}", type=Path, required=True)
    estimates = commands.add_parser("estimate-prompts")
    estimates.add_argument(
        "--split", choices=["train", "val", "test", "test_ood", "raw_extra"], default="val"
    )
    estimates.add_argument("--run", default="base")
    estimate_run = commands.add_parser("estimate-run")
    estimate_run.add_argument("--run", default="base")
    estimate_run.add_argument("--splits", nargs="+", required=True)
    estimate_run.add_argument("--label", required=True)
    estimate_run.add_argument("--adapter")
    estimate_run.add_argument("--name")
    estimate_run.add_argument("--timeout", type=int, default=2400)
    adapter = commands.add_parser("fetch-adapter")
    adapter.add_argument("--run", required=True)
    audit = commands.add_parser("parser-audit")
    audit.add_argument("--gate", action="store_true")
    commands.add_parser("merge-estimates")
    lmeval = commands.add_parser("lmeval")
    lmeval.add_argument("--run", default="base")
    lmeval.add_argument("--limit", type=int)
    g6 = commands.add_parser("g6")
    g6.add_argument("--run", default="m1-s0")
    preservation = commands.add_parser("collect-preservation")
    preservation.add_argument("--estimates", type=Path, nargs="+", required=True)
    robustness = commands.add_parser("robustness-targets")
    robustness.add_argument("--estimates", type=Path, nargs="+", required=True)
    robustness.add_argument(
        "--output", type=Path, default=ROOT / "data/eval/common-base-estimates.jsonl"
    )
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--estimates", type=Path, default=ROOT / "data/eval/estimates.jsonl")
    figures = commands.add_parser("figures")
    figures.add_argument("--estimates", type=Path, default=ROOT / "data/eval/estimates.jsonl")
    figures.add_argument("--resamples", type=int, default=10000)
    for command in (evaluation, figures):
        command.add_argument("--capability", type=Path)
    commands.add_parser("simulate")
    commands.add_parser("extra-analyses")
    commands.add_parser("plots")
    commands.add_parser("derive-e4")
    capability = commands.add_parser("capability")
    capability.add_argument("--base-traces", type=Path, required=True)
    capability.add_argument("--fine-tuned-traces", type=Path, required=True)
    capability.add_argument("--output", type=Path)
    release = commands.add_parser("release")
    release.add_argument("--approve-public", action="store_true")
    merge = commands.add_parser("merge")
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    self_estimate = commands.add_parser(
        "estimate", help="Estimate how long Tiny Aya takes to solve a coding task on this machine"
    )
    source = self_estimate.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", type=Path, help="File with the task description")
    source.add_argument("--text", help="Task description")
    self_estimate.add_argument(
        "--server",
        default="http://127.0.0.1:8082",
        help="Local llama-server serving the tuned GGUF (default: %(default)s)",
    )
    self_estimate.add_argument(
        "--speed", type=speed_pair, metavar="S,P", help="Decode and prefill tokens/s"
    )
    bench_arguments(self_estimate, required=False)
    self_estimate.add_argument("--family", choices=FAMILIES, help="Default: inferred from text")
    self_estimate.add_argument("--human", action="store_true", help="Also ask for human time")
    self_estimate.add_argument("--json", action="store_true")
    speedcard = commands.add_parser(
        "speedcard", help="Measure decode (S) and prefill (P) tokens/s with llama-bench"
    )
    bench_arguments(speedcard, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "prompts":
            from anthropochrony.prompts import compile_prompts

            print(f"{compile_prompts(args.check)} prompt templates synchronized")
        elif args.command == "preflight":
            from anthropochrony.preflight import preflight

            print(json.dumps(preflight(root, config(root, "pins.yaml")), indent=2))
        elif args.command == "status":
            from anthropochrony.status import status

            print(json.dumps(status(root), indent=2))
        elif args.command == "security":
            from anthropochrony.workflow import security_gate

            security_gate(root)
        elif args.command == "smoke":
            from anthropochrony.workflow import smoke

            smoke(root, remote=args.remote)
        elif args.command == "tasks":
            from anthropochrony.preflight import pinned_tokenizer
            from anthropochrony.sandbox import Sandbox
            from anthropochrony.tasks.pipeline import build_tasks

            require_gate(root, "G0-security")
            pins = config(root, "pins.yaml")
            build_tasks(
                root,
                pins,
                pinned_tokenizer(root, pins),
                Sandbox(),
                reference_card(root)
                if (root / "data/calibration/reference.json").exists()
                else SpeedCard("token-budget-only", 35, 5000),
                local_root=args.local_data,
                workers=args.workers,
            )
        elif args.command == "showcase":
            from anthropochrony.measure.showcase import showcase_measurement

            showcase_measurement(root, args.runs.split(","))
        elif args.command == "proj8":
            from anthropochrony.baselines.provider import CachedProvider
            from anthropochrony.preflight import pinned_tokenizer
            from anthropochrony.sandbox import Sandbox
            from anthropochrony.tasks.proj8 import build_proj8

            require_gate(root, "G0-security")
            kept, report = build_proj8(
                root,
                CachedProvider(
                    root,
                    model="deepseek-ai/DeepSeek-V4-Pro",
                    provider="deepinfra",
                    input_per_million=1.3,
                    output_per_million=2.6,
                ),
                Sandbox(),
                pinned_tokenizer(root, config(root, "pins.yaml")),
                reference_card(root),
            )
            print(json.dumps(report, indent=2))
        elif args.command == "labels":
            labels_command(root, args)
        elif args.command in {"prepare-round", "execute-round", "measure"}:
            from anthropochrony.workflow import measurement

            if args.command == "measure" and args.auto:
                from anthropochrony.measure.automatic import automatic_measurement

                automatic_measurement(root, args)
            else:
                measurement(root, args)
        elif args.command == "calibrate":
            from anthropochrony.workflow import calibrate

            calibrate(root)
        elif args.command == "invariance":
            from anthropochrony.measure.invariance import invariance_metrics
            from anthropochrony.tasks.loaders import stratified_sample

            require_gate(root, "G2")
            tasks = stratified_sample(
                [t for t in read_jsonl(root / "data/tasks/tasks.jsonl") if t["split"] == "val"],
                100,
                key=lambda t: t["family"],
            )
            result = invariance_metrics(
                tasks, read_jsonl(args.reference), read_jsonl(args.slow), read_jsonl(args.fast)
            )
            write_json(
                root / "data/measure/invariance.json",
                result,
                inputs=[args.reference, args.slow, args.fast],
            )
            if not result["passed"]:
                raise GateError(
                    "Speed-card invariance failed; stop and ask the user before full measurement"
                )
        elif args.command == "baselines":
            from anthropochrony.workflow import baselines

            baselines(root, teacher=args.teacher)
        elif args.command == "smoke-merged":
            from anthropochrony.train.smoke import run_smoke

            print(json.dumps(run_smoke(root), indent=2))
        elif args.command == "teacher":
            from anthropochrony.workflow import collect_teacher

            require_gate(root, "G3")
            print(f"Collected teacher quantities for {len(collect_teacher(root))} tasks")
        elif args.command == "sft-data":
            from anthropochrony.workflow import sft_data

            sft_data(root, args.variant)
        elif args.command == "sft-gate":
            from anthropochrony.curate.gates import sft_gate

            sft_gate(root, args.dry_run_summary)
        elif args.command == "train":
            from anthropochrony.workflow import train_job

            train_job(root, args)
        elif args.command == "select-lr":
            from anthropochrony.train.selection import select_learning_rate

            select_learning_rate(
                root,
                {
                    1e-4: {
                        "estimates": args.low_estimates,
                        "training_summary": args.low_training_summary,
                    },
                    2e-4: {
                        "estimates": args.high_estimates,
                        "training_summary": args.high_training_summary,
                    },
                },
            )
        elif args.command == "estimate-prompts":
            import pandas as pd

            from anthropochrony.eval.prompts import build_evaluation_prompts
            from anthropochrony.gates import require_preregistration

            tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
            require_preregistration(root, [args.split], fine_tuned=args.run != "base")
            fine_tuned = args.run != "base"
            if fine_tuned:
                import re

                if not re.fullmatch(r"m[123]-s[012]", args.run):
                    raise ValueError(
                        "Fine-tuned runs must identify their variant and seed, e.g. m1-s0"
                    )
            has_own_labels = (root / f"data/labels/{args.run}.parquet").exists()
            own_targets = (
                fine_tuned
                and args.split in {"test", "test_ood"}
                and (not args.run.startswith("m3-") or has_own_labels)
            )
            label_name = args.run if own_targets else "base"
            labels = pd.read_parquet(root / f"data/labels/{label_name}.parquet").to_dict("records")
            if own_targets and any(
                row.get("label_source") != args.run for row in labels if row["split"] == args.split
            ):
                raise GateError(
                    "Primary fine-tuned evaluation requires the run's own re-measured labels"
                )
            rows = build_evaluation_prompts(
                tasks,
                labels,
                split=args.split,
                include_fewshot=args.run == "base",
                fine_tuned=args.run if fine_tuned else None,
            )
            for row in rows:
                row["target_source"] = "own" if own_targets else "base"
                if fine_tuned:
                    row["seed"] = int(args.run[-1])
            write_jsonl(root / f"data/eval/{args.run}-{args.split}-prompts.jsonl", rows)
        elif args.command == "estimate-run":
            from anthropochrony.eval.remote import run_estimates

            paths = [root / f"data/eval/{args.run}-{split}-prompts.jsonl" for split in args.splits]
            result = run_estimates(
                root,
                args.run,
                paths,
                label=args.label,
                timeout_s=args.timeout,
                adapter=args.adapter,
                name=args.name,
            )
            print({split: len(rows) for split, rows in result.items()})
        elif args.command == "fetch-adapter":
            from anthropochrony.eval.remote import fetch_adapter_evidence

            print(json.dumps(fetch_adapter_evidence(root, args.run), indent=2))
        elif args.command == "parser-audit":
            from anthropochrony.eval.g4 import g4_gate, sample_parser_audit

            print(g4_gate(root) if args.gate else f"{len(sample_parser_audit(root))} rows")
        elif args.command == "merge-estimates":
            from anthropochrony.eval.merge import merge_estimates as merge_all

            print(merge_all(root))
        elif args.command == "lmeval":
            from anthropochrony.eval.remote import run_lmeval

            print(json.dumps(run_lmeval(root, args.run, limit=args.limit), indent=2))
        elif args.command == "g6":
            from anthropochrony.eval.remote import g6_gate

            print(json.dumps(g6_gate(root, args.run), indent=2))
        elif args.command == "collect-preservation":
            from anthropochrony.eval.preservation import collect_preservation

            collect_preservation(root, args.estimates)
        elif args.command == "robustness-targets":
            import pandas as pd

            from anthropochrony.eval.targets import common_base_targets

            rows = common_base_targets(
                [row for path in args.estimates for row in read_jsonl(path)],
                pd.read_parquet(root / "data/labels/base.parquet").to_dict("records"),
            )
            write_jsonl(args.output, rows, inputs=args.estimates)
        elif args.command in {"eval", "figures"}:
            from anthropochrony.eval.reporting import report
            from anthropochrony.gates import require_preregistration

            rows = read_jsonl(args.estimates)
            tuned_splits = {
                row["split"] for row in rows if row["estimator"].lower().startswith("m")
            }
            require_preregistration(root, tuned_splits, fine_tuned=bool(tuned_splits))
            report(
                root,
                rows,
                resamples=getattr(args, "resamples", 10000),
                capability=read_jsonl(args.capability) if args.capability else None,
            )
        elif args.command == "derive-e4":
            from anthropochrony.baselines.rescale import derive_e4

            print(derive_e4(root))
        elif args.command == "plots":
            from anthropochrony.eval.plots import make_figures

            make_figures(root)
        elif args.command == "extra-analyses":
            from anthropochrony.eval.extra import extra_report

            extra_report(root)
        elif args.command == "simulate":
            from anthropochrony.sim.run import simulate

            print(json.dumps(simulate(root), indent=2))
        elif args.command == "capability":
            from anthropochrony.eval.capability import capability_pairs
            from anthropochrony.gates import require_preregistration

            require_gate(root, "G6")
            require_preregistration(root, ["test"], fine_tuned=True)
            identity = json.loads((root / "adapters/m1-s0/identity.json").read_text())
            pairs = capability_pairs(
                read_jsonl(root / "data/tasks/tasks.jsonl"),
                read_jsonl(args.base_traces),
                read_jsonl(args.fine_tuned_traces),
                config(root, "pins.yaml")["subject"]["revision"],
                identity["model_sha"],
            )
            write_jsonl(
                args.output or root / "data/eval/capability-m1-s0.jsonl",
                pairs,
                inputs=[args.base_traces, args.fine_tuned_traces],
            )
        elif args.command == "merge":
            from anthropochrony.train.sft import merge

            merge(args.base, args.adapter, args.output)
        elif args.command == "release":
            from anthropochrony.release import prepare_release

            prepare_release(root, approved=args.approve_public)
        elif args.command == "estimate":
            selftime_command(args)
        elif args.command == "speedcard":
            from anthropochrony.selftime import measure_speed_card

            try:
                card = measure_speed_card(
                    args.llama_bench, args.gguf, threads=args.threads, gpu_layers=args.gpu_layers
                )
            except subprocess.CalledProcessError as error:
                raise SystemExit(f"llama-bench failed: {error.stderr or error}") from error
            print(json.dumps(card.as_dict(), indent=2))
    except (GateError, FileNotFoundError, ValueError, PermissionError) as error:
        raise SystemExit(f"Blocked: {error}") from error


if __name__ == "__main__":
    main()
