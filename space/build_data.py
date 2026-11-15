"""Write the small data files the demo Space reads, from the committed records (no task text)."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data"
PRIMARY = "g1-kit/card_description"
ROWS = {
    "GPT-5.5 (published)": ("published", "gpt-5.5"),
    "Claude Opus 4.8 (published)": ("published", "claude-opus-4-8"),
    "Tiny Aya, untrained, told its speed": ("systems", "untrained/card_description"),
    "Tiny Aya + G1-kit (CuratorKIT -> AlignTune), told its speed": ("systems", PRIMARY),
}


def main():
    results = json.loads((ROOT / "results/metatime.json").read_text())
    summary = []
    for label, (section, key) in ROWS.items():
        entry = results[section].get(key)
        if entry is None:
            continue
        row = entry if section == "published" else entry["stated"]
        summary.append(
            {
                "system": label,
                "typical_factor_off": round(2.718281828 ** row["median_abs_log_error"], 2),
                "median_abs_log_error": round(row["median_abs_log_error"], 3),
                "ci95": [round(v, 3) for v in row["ci95"]["median_abs_log_error"]],
                "within_2x": round(row["within_2x"], 3),
                "median_estimate_over_actual": round(row["median_ratio"], 3),
                "pearson_log": round(row["pearson_log"], 3),
            }
        )
    records = ROOT / "results/metatime"
    published = list(csv.DictReader((records / "reference-published.csv").open()))
    tasks = {}
    for row in published:
        task = tasks.setdefault(row["task"], {"task": row["task"], "category": row["category"]})
        name = "gpt55" if row["model"] == "gpt-5.5" else "opus48"
        task[f"{name}_estimate_s"] = float(row["estimate_before_seconds"])
        task[f"{name}_actual_s"] = float(row["actual_seconds"])
    runs, executor = {}, results["systems"][PRIMARY]["scored_against"]
    for line in (records / "runs.jsonl").open():
        run = json.loads(line)
        if run["executor"] == executor:
            runs.setdefault(run["task"], []).append(run["wall_s"])
    for line in (records / "estimates.jsonl").open():
        row = json.loads(line)
        if f"{row['estimator']}/{row['variant']}" == PRIMARY:
            tasks[row["task"]]["g1_estimate_s"] = row["stated_seconds"]
    for task_id, task in tasks.items():
        task["tiny_aya_actual_s"] = round(statistics.median(runs[task_id]), 2)
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "task",
        "category",
        "gpt55_estimate_s",
        "gpt55_actual_s",
        "opus48_estimate_s",
        "opus48_actual_s",
        "g1_estimate_s",
        "tiny_aya_actual_s",
    ]
    with (OUT / "tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in sorted(tasks.values(), key=lambda t: (t["category"], t["task"])):
            writer.writerow({k: task.get(k) for k in fields})
    print(f"{len(summary)} summary rows, {len(tasks)} tasks")


if __name__ == "__main__":
    main()
