"""Resumable round-synchronous generation, with CPU-only sandbox execution."""

import json
from pathlib import Path

from anthropochrony.artifacts import digest, file_hash, read_jsonl, write_jsonl
from anthropochrony.measure.rounds import apply_round, prepare_round


def run_experiments(experiments, tokenizer, sandbox, batch_directory, generator_factory):
    states = {name: experiment.states() for name, experiment in experiments.items()}
    for number in range(1, 4):
        requests, combined = {}, []
        for name, experiment in experiments.items():
            prepared = prepare_round(states[name], experiment.tasks, tokenizer, number)
            experiment.save(states[name])
            if not prepared:
                continue
            request_path = experiment.directory / f"requests-r{number}.jsonl"
            output_path = experiment.directory / f"generation-r{number}.jsonl"
            if request_path.exists() and read_jsonl(request_path) != prepared:
                raise ValueError("Saved automatic-round requests changed")
            if not request_path.exists():
                write_jsonl(request_path, prepared, config=experiment.config)
            requests[name] = prepared
            if not output_path.exists():
                combined += [
                    row | {"request_id": f"{name}:{row['request_id']}"} for row in prepared
                ]
        if combined:
            directory = batch_directory / f"r{number}"
            directory.mkdir(parents=True, exist_ok=True)
            batch_path = directory / "requests.jsonl"
            if batch_path.exists():
                previous = {row["request_id"]: row for row in read_jsonl(batch_path)}
                if any(previous.get(row["request_id"]) != row for row in combined):
                    raise ValueError(
                        "Existing GPU batch cannot satisfy the current resumed requests"
                    )
            else:
                write_jsonl(batch_path, combined)
            # One full vLLM batch, not a sequence of small transport-limited calls.
            outputs = generator_factory(directory).generate_file(batch_path)
            by_id = {row["request_id"]: row for row in outputs}
            if len(by_id) != len(outputs) or set(by_id) != {
                row["request_id"] for row in read_jsonl(batch_path)
            }:
                raise ValueError("Automatic GPU batch returned incorrect or duplicate IDs")
            for name, prepared in requests.items():
                experiment = experiments[name]
                output_path = experiment.directory / f"generation-r{number}.jsonl"
                if not output_path.exists():
                    selected = [
                        by_id[f"{name}:{row['request_id']}"] | {"request_id": row["request_id"]}
                        for row in prepared
                    ]
                    write_jsonl(
                        output_path,
                        selected,
                        inputs=[experiment.directory / f"requests-r{number}.jsonl"],
                        config=experiment.config,
                    )
        # Every paid GPU process is terminal before these candidate executions.
        for name, prepared in requests.items():
            experiment = experiments[name]
            output_path = experiment.directory / f"generation-r{number}.jsonl"
            metadata = json.loads(Path(f"{output_path}.manifest.json").read_text())
            request_path = experiment.directory / f"requests-r{number}.jsonl"
            if (
                metadata["artifact_sha256"] != file_hash(output_path)
                or set(metadata["input_hashes"].values()) != {file_hash(request_path)}
                or metadata["config_hash"] != digest(experiment.config)
            ):
                raise ValueError(
                    "Cached generation failed its request/configuration content hashes"
                )
            states[name], texts = apply_round(
                states[name],
                experiment.tasks,
                prepared,
                read_jsonl(output_path),
                sandbox,
                number,
                experiment.config["workers"],
            )
            write_jsonl(
                experiment.directory / f"outputs-r{number}.jsonl", texts, inputs=[output_path]
            )
            experiment.save(states[name])
    return states


def run_speed_invariance(root, tokenizer, sandbox):
    from anthropochrony.artifacts import write_json
    from anthropochrony.cli import config, reference_card
    from anthropochrony.gates import GateError, require_gate
    from anthropochrony.measure.invariance import invariance_metrics
    from anthropochrony.measure.remote import HFRemoteRound
    from anthropochrony.measure.rounds import Measurement
    from anthropochrony.tasks.loaders import stratified_sample
    from anthropochrony.timemodel import SpeedCard

    require_gate(root, "G0-security")
    require_gate(root, "G1")
    require_gate(root, "G2")
    result_path = root / "data/measure/invariance.json"
    if result_path.exists() and not json.loads(result_path.read_text()).get("passed"):
        raise GateError("Speed-card invariance failed; stop and ask the user before rerunning")
    tasks = stratified_sample(
        [t for t in read_jsonl(root / "data/tasks/tasks.jsonl") if t["split"] == "val"],
        100,
        key=lambda t: t["family"],
    )
    if len(tasks) != 100:
        raise GateError("Invariance requires 100 validation tasks")
    cfg = config(root, "measure.yaml")
    directory = root / "data/measure/invariance"
    conditions = {
        "ref": (8, reference_card(root)),
        "slow": (2, SpeedCard("invariance-slow", 2, 20)),
        "fast": (2, SpeedCard("invariance-fast", 2500, 50000)),
    }
    experiments = {
        name: Measurement(directory / name, tasks, cfg | {"k": k}, card)
        for name, (k, card) in conditions.items()
    }
    batch_config = cfg | {
        "conditions": {
            name: {"k": k, "card": card.as_dict()} for name, (k, card) in conditions.items()
        }
    }
    traces = run_experiments(
        experiments,
        tokenizer,
        sandbox,
        directory / "batches",
        lambda path: HFRemoteRound(root, path, batch_config, timeout_s=1200),
    )
    result = invariance_metrics(tasks, traces["ref"], traces["slow"], traces["fast"])
    paths = [experiment.directory / "traces.jsonl" for experiment in experiments.values()]
    write_json(result_path, result, inputs=paths)
    if not result["passed"]:
        raise GateError(
            "Speed-card invariance failed; stop and ask the user before full measurement"
        )
    return result


def automatic_measurement(root, args):
    from transformers import AutoTokenizer

    from anthropochrony.cli import config, reference_card
    from anthropochrony.gates import GateError, require_gate
    from anthropochrony.measure.remote import HFRemoteRound
    from anthropochrony.measure.rounds import Measurement
    from anthropochrony.sandbox import Sandbox
    from anthropochrony.tasks.loaders import stratified_sample
    from anthropochrony.workflow import measurement_gate

    require_gate(root, "G0-security")
    require_gate(root, "G1")
    require_gate(root, "G2")
    if args.run != "base":
        import re

        from anthropochrony.measure.finetuned import FT_RUN, finetuned_measurement

        runs = args.run.split(",")
        if args.phase != "full" or not all(re.fullmatch(FT_RUN, run) for run in runs):
            raise GateError("Fine-tuned measurement takes full-phase runs like m1-s0,m2-s0")
        if args.dry_run:
            print(f"Preview only: three merged-weight GPU rounds for {runs}.")
            return
        return finetuned_measurement(root, runs)
    if args.phase == "full":
        require_gate(root, "G3")
    if args.dry_run:
        print(
            "Preview only: up to three GPU-only round jobs, with local sandbox tests between rounds."
        )
        print("Pilot quality must pass before the independent validation-only invariance jobs.")
        return
    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    if args.phase == "pilot":
        tasks = stratified_sample(
            [t for t in tasks if t["split"] != "raw_extra"], 100, key=lambda t: t["family"]
        )
        if len(tasks) != 100:
            raise GateError("The pilot requires 100 stratified tasks")
    cfg = config(root, "measure.yaml")
    if cfg["k"] != 8:
        raise GateError("Measurement requires K=8")
    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    sandbox = Sandbox()
    directory = root / "data/measure" / ("base-pilot" if args.phase == "pilot" else "base")
    experiment = Measurement(directory, tasks, cfg, reference_card(root))
    states = run_experiments(
        {"base": experiment},
        tokenizer,
        sandbox,
        directory / "batches",
        lambda path: HFRemoteRound(
            root, path, cfg, timeout_s=1200 if args.phase == "pilot" else 5400
        ),
    )["base"]
    if args.phase == "pilot":
        # This preliminary gate cannot mark G3 passed without invariance.
        measurement_gate(root, states, tasks, require_invariance=False)
        run_speed_invariance(root, tokenizer, sandbox)
    measurement_gate(root, states, tasks, phase=args.phase)
    return states
