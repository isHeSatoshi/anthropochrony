import json
import math

import pandas as pd

from anthropochrony.artifacts import file_hash, read_jsonl, write_json
from anthropochrony.cli import config
from anthropochrony.eval.metrics import summary, task_records
from anthropochrony.eval.prompts import build_evaluation_prompts
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.gates import GateError, require_gate


def select_learning_rate(root, candidates):
    require_gate(root, "G5")
    if set(candidates) != {1e-4, 2e-4}:
        raise GateError("The sweep requires exactly the preregistered learning rates")
    tasks = read_jsonl(root / "data/tasks/tasks.jsonl")
    labels = pd.read_parquet(root / "data/labels/base.parquet").to_dict("records")
    expected = {
        (row["task_id"], row["template_id"]): row
        for row in build_evaluation_prompts(tasks, labels, split="val", fine_tuned="m1-s0")
        if row["frame"] == "self" and row["card_id"] == "REF"
    }
    if not expected:
        raise GateError("No measured validation targets for the learning-rate sweep")
    data_sha = file_hash(root / "data/sft/m1/train/data.jsonl")
    expected_steps = math.ceil(len(read_jsonl(root / "data/sft/m1/train/data.jsonl")) / 32)
    shared = {name: config(root, f"sft/{name}.yaml") for name in ("m1", "m2", "m3")}
    subject = config(root, "pins.yaml")["subject"]
    if len({cfg["max_seq_length"] for cfg in shared.values()}) != 1:
        raise GateError("The variants must share one sequence limit before LR selection")
    scores, identities, inputs = {}, set(), {}
    for rate, paths in sorted(candidates.items()):
        training_path = paths["training_summary"]
        training = json.loads(training_path.read_text())
        identity_path = training_path.parent / "identity.json"
        identity = json.loads(identity_path.read_text())
        hardware_path = training_path.parent / "hardware.json"
        hardware = json.loads(hardware_path.read_text())
        cfg = training["config"]
        if (
            training.get("dry_run") is not False
            or training.get("seed") != 0
            or cfg.get("variant") != "m1"
            or cfg.get("epochs") != 1
            or cfg.get("learning_rate") != rate
            or training.get("train_data_sha256") != data_sha
            or training.get("training", {}).get("steps") != expected_steps
            or cfg.get("max_seq_length") != shared["m1"]["max_seq_length"]
            or cfg.get("base") != subject
            or identity.get("private") is not True
            or identity.get("base") != subject
            or identity.get("training_summary_sha256") != file_hash(training_path)
            or hardware.get("cuda") is not True
            or "L40S" not in hardware.get("gpu", "")
        ):
            raise GateError("LR selection requires matched M1 seed-0 one-epoch GPU sweep runs")
        identities.add(identity["model_sha"])
        rows = read_jsonl(paths["estimates"])
        if any(row["split"] != "val" for row in rows):
            raise GateError("Learning-rate selection must never inspect test or test_ood results")
        selected = [row for row in rows if row["frame"] == "self" and row["card_id"] == "REF"]
        observed = {(row["task_id"], row["template_id"]): row for row in selected}
        if len(observed) != len(selected) or set(observed) != set(expected):
            raise GateError(
                "Every sweep candidate needs the identical complete self-REF validation set"
            )
        for key, row in observed.items():
            target = expected[key]
            if (
                row.get("seed") != 0
                or row.get("estimator") != target["estimator"]
                or row.get("target_source") != "base"
                or row.get("label_source") != "base"
                or row.get("model_sha") != identity["model_sha"]
                or row["messages"] != target["messages"]
                or row["T_med_s"] != target["T_med_s"]
                or (row["S"], row["P"]) != (target["S"], target["P"])
                or row.get("actual_s", row["T_med_s"]) != target["T_med_s"]
            ):
                raise GateError(
                    "Sweep outputs do not match their checkpoint, prompts, or measured val targets"
                )
        score = summary(task_records(normalized_rows(selected)))["mdae_log10"]
        scores[str(rate)] = score if math.isfinite(score) else "Infinity"
        inputs[str(rate)] = {
            "estimates_sha256": file_hash(paths["estimates"]),
            "training_summary_sha256": file_hash(training_path),
            "identity_sha256": file_hash(identity_path),
            "hardware_sha256": file_hash(hardware_path),
        }
    if len(identities) != 2:
        raise GateError("The learning-rate candidates must identify two distinct checkpoints")
    finite = {float(rate): value for rate, value in scores.items() if value != "Infinity"}
    if not finite:
        raise GateError("Neither sweep candidate has a finite validation MdAE")
    chosen = min(finite, key=lambda rate: (finite[rate], rate))
    result = {
        "learning_rate": chosen,
        "scores_val_mdae_log10": scores,
        "train_data_sha256": data_sha,
        "inputs": inputs,
        "tie_break": "lower_learning_rate",
        "selection_split": "val",
    }
    path = root / "results/lr-selection.json"
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise GateError(
                "The learning-rate selection is frozen; a change needs a dated deviation"
            )
        require_selected_learning_rate(root, shared["m1"])
        return result
    write_json(path, result)
    selection_sha = file_hash(path)
    for name in ("m1", "m2", "m3"):
        cfg = shared[name]
        cfg.update(
            learning_rate=chosen,
            learning_rate_selected_on_val=True,
            learning_rate_selection_sha256=selection_sha,
        )
        write_json(root / f"configs/sft/{name}.yaml", cfg)
    return result


def require_selected_learning_rate(root, cfg):
    path = root / "results/lr-selection.json"
    if not path.exists():
        raise GateError("Run and freeze the two-point validation-only learning-rate selection")
    result = json.loads(path.read_text())
    if result["train_data_sha256"] != file_hash(root / "data/sft/m1/train/data.jsonl"):
        raise GateError("The frozen learning-rate selection used a different training dataset")
    sha = file_hash(path)
    for name in ("m1", "m2", "m3"):
        variant = config(root, f"sft/{name}.yaml")
        if (
            not variant["learning_rate_selected_on_val"]
            or variant["learning_rate"] != result["learning_rate"]
            or variant.get("learning_rate_selection_sha256") != sha
            or variant["max_seq_length"] != cfg["max_seq_length"]
        ):
            raise GateError(
                "Every variant must reuse the same frozen validation-selected learning rate"
            )
