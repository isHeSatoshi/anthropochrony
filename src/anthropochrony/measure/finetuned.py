"""Merged-weight GPU rounds for fine-tuned runs, then the same local sandbox protocol."""

import json
from pathlib import Path

from anthropochrony.artifacts import read_jsonl
from anthropochrony.gates import GateError, require_gate, require_preregistration
from anthropochrony.gpu_jobs import GPUJob

FT_RUN = r"(m[12]-s[012]|m3-s0)"
ROUND_SOURCES = ("jobs/ft_round.py", "jobs/model_prep.py", "src/anthropochrony/gen/engine.py")


def adapter_identity(root, run):
    path = Path(root) / "adapters" / run / "identity.json"
    if not path.exists():
        raise GateError(f"Adapter identity for {run} has not been downloaded and checked")
    return json.loads(path.read_text())


class MergedRound:
    """GPU-only job serving every requested fine-tuned run from its own merged weights."""

    def __init__(self, root, directory, measure_config, runs, *, timeout_s=None):
        self.root, self.directory, self.config = Path(root), Path(directory), measure_config
        self.runs = sorted(runs)
        self.timeout_s = timeout_s or 900 * len(self.runs) + 600

    def generate_file(self, requests):
        requests = Path(requests)
        rows = read_jsonl(requests)
        names = {row["request_id"].split(":", 1)[0] for row in rows}
        if names - set(self.runs):
            raise GateError("Requests name a fine-tuned run that has no adapter identity")
        identities = {run: adapter_identity(self.root, run) for run in sorted(names)}
        config_path = self.directory / "round-config.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(self.config, indent=2, sort_keys=True))
        job = GPUJob(
            self.root,
            self.directory,
            kind="ftround",
            script="jobs/ft_round.py",
            files=[(requests, "requests.jsonl"), (config_path, "round-config.json")],
            arguments=[
                "--requests",
                "/code/requests.jsonl",
                "--config",
                "/code/round-config.json",
            ],
            outputs=[
                "generation.jsonl",
                "merge-evidence.json",
                "generation.performance.json",
                "generation.gpu.json",
            ],
            timeout_s=self.timeout_s,
            context={"identities": identities, "config": self.config},
            sources=ROUND_SOURCES,
            required_gates=("G0-security", "G1", "G2", "G3", "G6"),
        )
        directory = job.run()
        evidence = json.loads((directory / "merge-evidence.json").read_text())
        for run, identity in identities.items():
            if evidence[run]["adapter_weights_sha256"] != identity["adapter_weights_sha256"]:
                raise ValueError(f"{run}: merged weights came from an unexpected adapter")
        return read_jsonl(directory / "generation.jsonl")


def finetuned_measurement(root, runs):
    from transformers import AutoTokenizer

    from anthropochrony.cli import config, reference_card
    from anthropochrony.measure.automatic import run_experiments
    from anthropochrony.measure.rounds import Measurement
    from anthropochrony.sandbox import Sandbox
    from anthropochrony.workflow import measurement_gate

    root = Path(root)
    for gate in ("G0-security", "G1", "G2", "G3", "G6"):
        require_gate(root, gate)
    tasks = [
        t for t in read_jsonl(root / "data/tasks/tasks.jsonl") if t["split"] in {"test", "test_ood"}
    ]
    require_preregistration(root, {"test", "test_ood"}, fine_tuned=True)
    cfg = config(root, "measure.yaml")
    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    experiments = {}
    for run in runs:
        identity = adapter_identity(root, run)
        experiments[run] = Measurement(
            root / "data/measure" / run,
            tasks,
            cfg | {"model_id": identity["model_id"], "model_sha": identity["model_sha"]},
            reference_card(root),
        )
    batch_config = cfg | {"runs": {run: adapter_identity(root, run) for run in runs}}
    states = run_experiments(
        experiments,
        tokenizer,
        Sandbox(),
        root / "data/measure/ft-batches" / "-".join(sorted(runs)),
        lambda path: MergedRound(root, path, batch_config, runs),
    )
    for run in runs:
        measurement_gate(root, states[run], tasks, phase="full", run=run, require_invariance=False)
    return states
