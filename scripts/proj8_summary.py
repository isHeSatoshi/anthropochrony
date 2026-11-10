"""Summarize the PROJ-8 showcase: base vs M1 trajectories on eight project-sized tasks, plus the
CPU deployment run (estimates vs real wall-clock). Writes results/proj8-showcase.json.

No hypothesis test: eight tasks, never in train, val or test.
"""

import json
import statistics
from pathlib import Path

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.eval.errors import log_error
from anthropochrony.parse import parse
from anthropochrony.timemodel import SpeedCard, trajectory_time

ROOT = Path(__file__).resolve().parents[1]


def trajectories(run):
    path = ROOT / "data/measure" / f"showcase-{run}" / "traces.jsonl"
    reference = json.loads((ROOT / "data/calibration/reference.json").read_text())
    card = SpeedCard("REF", reference["reference"]["S"], reference["reference"]["P"])
    rows = {}
    for trace in read_jsonl(path):
        rows.setdefault(trace["task_id"], []).append(trace)
    out = {}
    for task, traces in sorted(rows.items()):
        out[task] = {
            "runs": len(traces),
            "first_pass": sum(bool(t["first_pass"]) for t in traces) / len(traces),
            "pass_within_3": sum(bool(t["passed"]) for t in traces) / len(traces),
            "median_time_s": statistics.median(trajectory_time(t, card) for t in traces),
            "median_output_tokens": statistics.median(t["total_o"] for t in traces),
        }
    return out, card.as_dict()


def main():
    result = {}
    for run in ("base", "m1-s0"):
        if (ROOT / "data/measure" / f"showcase-{run}" / "traces.jsonl").exists():
            tasks, card = trajectories(run)
            result[run] = {
                "card": card,
                "tasks": tasks,
                "mean_first_pass": statistics.mean(v["first_pass"] for v in tasks.values()),
                "mean_pass_within_3": statistics.mean(v["pass_within_3"] for v in tasks.values()),
            }
    cpu = ROOT / "data/x10/proj8-cpu.jsonl"
    if cpu.exists():
        rows = read_jsonl(cpu)
        entries = []
        for row in rows:
            entry = {"task_id": row["task_id"], "wall_s": row["wall_s"], "passed": row["passed"]}
            entry["attempts"] = row["attempts"]
            for name in ("base", "tuned", "human"):
                point = parse(row[f"{name}_raw"]).as_dict()["point_s"]
                entry[f"{name}_estimate_s"] = point
                entry[f"{name}_log10_error"] = log_error(point, row["wall_s"])
            entries.append(entry)
        result["cpu_deployment"] = {"card": rows[0]["card"], "tasks": entries}
    write_json(ROOT / "results/proj8-showcase.json", result)
    print(json.dumps(result, indent=1)[:3000])


if __name__ == "__main__":
    main()
