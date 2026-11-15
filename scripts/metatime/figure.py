"""Draw the head-to-head figure from the committed records.

Reads `results/metatime.json` and `results/metatime/`; needs no GPU and no benchmark download.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, read_jsonl

NAME = "F9-metatime-head-to-head.png"
PRIMARY = "g1-kit/card_description"
GREY, PINK, TEAL, DARK_TEAL = "#9e9e9e", "#c2185b", "#2a9d8f", "#0b5d56"
# (key in results/metatime.json, label, colour); "published:" rows come from the artifact.
ROWS = [
    ("published:gpt-5.5", "GPT-5.5 (published)", GREY),
    ("published:claude-opus-4-8", "Claude Opus 4.8 (published)", PINK),
    ("untrained/protocol", "Tiny Aya, untrained", "#757575"),
    ("untrained/card_description", "Untrained, speed stated", "#bdbdbd"),
    ("untrained/card_fewshot", "Untrained, speed + 5 examples", "#d9822b"),
    ("static/pool_median_constant", "One constant (no model)", "#8d6e63"),
    ("static/text_regressor_description", "Text regressor (no model)", "#37474f"),
    ("probe/question", "Probe on hidden state", "#7b5ea7"),
    ("g1-kit/protocol", "G1-kit, no speed stated", "#2f6db5"),
    (PRIMARY, "Tiny Aya + G1-kit, speed stated (ours)", DARK_TEAL),
]


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def head_to_head(root: Path, directory: Path):
    """Each system against its own measured time; nothing is drawn until the results exist."""
    path, records = root / "results/metatime.json", root / "results/metatime"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    systems = result["systems"]
    if PRIMARY not in systems:
        return None
    primary = systems[PRIMARY]
    runs = {}
    for row in read_jsonl(records / "runs.jsonl"):
        if row["executor"] == primary["scored_against"]:
            runs.setdefault(row["task"], []).append(row["wall_s"])
    ours = np.array(
        [
            (float(np.median(runs[row["task"]])), row["stated_seconds"])
            for row in read_jsonl(records / "estimates.jsonl")
            if f"{row['estimator']}/{row['variant']}" == PRIMARY and row["stated_seconds"]
        ]
    )
    with (records / "reference-published.csv").open(encoding="utf-8") as handle:
        published = list(csv.DictReader(handle))

    _style()
    fig, axes = plt.subplots(
        1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.5}
    )
    axis = axes[0]
    line = np.geomspace(0.3, 5000.0, 60)
    axis.fill_between(line, line / 2, line * 2, color=TEAL, alpha=0.15, zorder=1)
    axis.plot(line, line, color="#111111", linewidth=1.0, zorder=2)
    for model, label, color, marker in (
        ("gpt-5.5", "GPT-5.5", GREY, "o"),
        ("claude-opus-4-8", "Claude Opus 4.8", PINK, "s"),
    ):
        pts = np.array(
            [
                (float(row["actual_seconds"]), float(row["estimate_before_seconds"]))
                for row in published
                if row["model"] == model
            ]
        )
        axis.scatter(
            pts[:, 0],
            pts[:, 1],
            s=17,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            alpha=0.8,
            label=f"{label}: {result['published'][model]['within_2x']:.0%} within 2×",
            zorder=3,
        )
    axis.scatter(
        ours[:, 0],
        ours[:, 1],
        s=22,
        color=DARK_TEAL,
        alpha=0.85,
        label=f"Tiny Aya + G1-kit (ours): {primary['stated']['within_2x']:.0%} within 2×",
        zorder=4,
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(0.8, 400.0),
        ylim=(0.8, 3000.0),
        xlabel="Measured time of the system's own run (s)",
        ylabel="Its estimate before starting (s)",
        title="(a) Estimate vs own measured time, one point per task",
    )
    axis.legend(loc="upper left", fontsize=7.4, frameon=False)

    axis = axes[1]
    rows = []
    for key, label, color in ROWS:
        row = (
            result["published"].get(key.split(":")[1])
            if key.startswith("published:")
            else systems.get(key, {}).get("stated")
        )
        if row:
            rows.append((label, color, row))
    limit = 2.6
    for position, (_, color, row) in zip(range(len(rows) - 1, -1, -1), rows, strict=True):
        value = row["median_abs_log_error"]
        axis.barh(position, min(value, limit), color=color, height=0.62, zorder=3)
        edge = min(value, limit)
        if math.isfinite(value) and "median_abs_log_error" in row.get("ci95", {}):
            low, high = row["ci95"]["median_abs_log_error"]
            axis.plot([low, min(high, limit)], [position] * 2, color="#111111", lw=1.1, zorder=4)
            edge = max(edge, min(high, limit))
        correlation = row["pearson_log"]
        note = f"{value:.2f}" if math.isfinite(value) else "most answers unusable"
        if correlation is not None:
            note += f"   r = {correlation:.2f}"
        axis.text(edge + 0.04, position, note, va="center", fontsize=7.6, zorder=5)
    noise = result["executors"][primary["scored_against"]]["retest_ceiling_abs_log_error"]
    axis.axvline(math.log(2), color=TEAL, lw=1.0, ls="--", zorder=2)
    axis.text(math.log(2) + 0.02, len(rows) - 0.45, "factor of 2", fontsize=7, color="#1b7f79")
    axis.axvline(noise, color="#555555", lw=0.9, ls=":", zorder=2)
    axis.text(noise + 0.02, -0.75, "run-to-run noise", fontsize=7, color="#555555")
    axis.set(
        yticks=range(len(rows) - 1, -1, -1),
        yticklabels=[label for label, _, _ in rows],
        xlim=(0, limit + 0.2),
        ylim=(-0.95, len(rows) - 0.2),
        xlabel="Median absolute log error (0 is exact; lower is better), with 95% interval",
        title="(b) Size of the typical miss, and r on log durations",
    )
    axis.grid(axis="y", visible=False)
    fig.suptitle(
        "MetaTime single-turn benchmark (68 tasks): how well each system predicts its own time "
        "before starting",
        y=1.02,
        fontsize=10.5,
        fontweight="bold",
    )
    target = directory / NAME
    fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="directory that holds results/")
    parser.add_argument("--out", default=str(ROOT / "paper/figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(head_to_head(Path(args.root), out) or "results/metatime.json is missing; nothing drawn")


if __name__ == "__main__":
    main()
