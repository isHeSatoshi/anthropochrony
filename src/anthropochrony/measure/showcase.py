"""PROJ-8 showcase runs: the same K=8 protocol on the eight verified project tasks.

They are never in train, val or test, so these runs feed the demo and the qualitative figure
only and carry no hypothesis test.
"""

from pathlib import Path

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.gates import require_gate


def showcase_measurement(root, runs):
    from transformers import AutoTokenizer

    from anthropochrony.cli import config, reference_card
    from anthropochrony.measure.automatic import run_experiments
    from anthropochrony.measure.finetuned import MergedRound, adapter_identity
    from anthropochrony.measure.remote import HFRemoteRound
    from anthropochrony.measure.rounds import Measurement
    from anthropochrony.sandbox import Sandbox

    root = Path(root)
    for gate in ("G0-security", "G1", "G2", "G3"):
        require_gate(root, gate)
    tasks = read_jsonl(root / "data/tasks/proj8.jsonl")
    cfg = config(root, "measure.yaml")
    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    sandbox, card = Sandbox(), reference_card(root)
    summary = {}
    if "base" in runs:
        experiment = Measurement(root / "data/measure/showcase-base", tasks, cfg, card)
        summary["base"] = run_experiments(
            {"base": experiment},
            tokenizer,
            sandbox,
            experiment.directory / "batches",
            lambda path: HFRemoteRound(root, path, cfg, timeout_s=2400),
        )["base"]
    tuned = [run for run in runs if run != "base"]
    if tuned:
        require_gate(root, "G6")
        experiments = {}
        for run in tuned:
            identity = adapter_identity(root, run)
            experiments[run] = Measurement(
                root / "data/measure" / f"showcase-{run}",
                tasks,
                cfg | {"model_id": identity["model_id"], "model_sha": identity["model_sha"]},
                card,
            )
        batch_config = cfg | {"runs": {run: adapter_identity(root, run) for run in tuned}}
        summary |= run_experiments(
            experiments,
            tokenizer,
            sandbox,
            root / "data/measure/showcase-batches" / "-".join(sorted(tuned)),
            lambda path: MergedRound(root, path, batch_config, tuned, timeout_s=2400),
        )
    write_json(
        root / "data/measure/showcase-summary.json",
        {
            run: {
                "tasks": len(states),
                "first_pass": sum(bool(s["first_pass"]) for s in states) / len(states),
                "done": sum(bool(s["done"]) for s in states) / len(states),
            }
            for run, states in summary.items()
        },
    )
    return summary
