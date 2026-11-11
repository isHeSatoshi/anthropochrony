"""Score the fresh-task CPU run: base vs tuned (verbal) vs tuned + exact timing formula.

Fresh tasks come from outside the 1,465-task pool (see data/fresh/README.md). Wall times are real
llama.cpp CPU runs. Post-hoc and exploratory; nothing here is pre-registered.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anthropochrony.artifacts import read_jsonl, write_json  # noqa: E402
from anthropochrony.eval.errors import log_error  # noqa: E402
from anthropochrony.eval.stats import json_safe  # noqa: E402
from anthropochrony.eval.structured import fit_workload_priors, structured_duration  # noqa: E402
from anthropochrony.parse import parse  # noqa: E402
from anthropochrony.timemodel import SpeedCard  # noqa: E402


def pack(errors, predictions, actuals):
    errors = np.asarray(errors, dtype=float)
    finite = np.isfinite(predictions)
    rho = (
        float(spearmanr(np.asarray(predictions)[finite], np.asarray(actuals)[finite]).statistic)
        if finite.sum() > 2
        else None
    )
    return {
        "n": int(len(errors)),
        "mdae_log10": float(np.median(errors)),
        "median_factor_off": float(10 ** np.median(errors)),
        "within2x": float(np.mean(errors <= math.log10(2))),
        "within10x": float(np.mean(errors <= 1.0)),
        "spearman_vs_wall": rho,
    }


def main():
    rows = read_jsonl(ROOT / "data/x10/fresh-cpu.jsonl")
    tasks = {t["task_id"]: t for t in read_jsonl(ROOT / "data/fresh/tasks.jsonl")}
    info = json.loads((ROOT / "data/x10/fresh-cpu-speed-card.json").read_text())
    card = SpeedCard(info["id"], float(info["S"]), float(info["P"]))
    priors = fit_workload_priors(ROOT)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        task = tasks[row["task_id"]]
        actual = float(row["wall_s"])
        base = parse(row["base_raw"]).point_s
        tuned = parse(row["tuned_raw"])
        struct = structured_duration(
            tuned.tokens,
            tuned.attempts,
            float(task["uncached_task_tokens"]),
            "FRESH",
            card,
            priors,
            latency_s=0.0,
        )
        struct = struct if struct is not None else tuned.point_s
        for scope in ("all", task["flags"]["size"]):
            for name, value in (
                ("base", base),
                ("tuned_verbal", tuned.point_s),
                ("tuned_struct", struct),
            ):
                groups[scope][name].append((value, actual))
            groups[scope]["actual"].append(actual)
            groups[scope]["passed"].append(bool(row["passed"]))
    result = {"speed_card": info, "n_tasks": len(rows), "scopes": {}}
    for scope, data in groups.items():
        entry = {
            "n": len(data["actual"]),
            "pass_rate": float(np.mean(data["passed"])),
            "median_wall_s": float(np.median(data["actual"])),
        }
        for name in ("base", "tuned_verbal", "tuned_struct"):
            pairs = data[name]
            predictions = [p if p else float("nan") for p, _ in pairs]
            errors = [abs(log_error(p, a)) for p, a in pairs]
            entry[name] = pack(errors, predictions, [a for _, a in pairs])
        result["scopes"][scope] = entry
    write_json(ROOT / "results/fresh-tasks.json", json_safe(result))
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
