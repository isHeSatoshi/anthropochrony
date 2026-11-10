"""Paper and article figures, drawn only from the released result files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anthropochrony.artifacts import manifest, read_jsonl
from anthropochrony.eval.metrics import summary, task_records
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.parse import parse

LABELS = {
    "E0": "Base, zero-shot",
    "E1": "Base, with speed card",
    "E2": "Base, 5 measured examples (few-shot)",
    "E4": "Base, rescaled by speed ratio",
    "M1-S0": "Anthropochrony (M1, measured labels)",
    "M2-S0": "Teacher-guessed labels (M2)",
    "M3-S0": "Raw measured labels, uncurated (M3)",
    "B2": "Regression on task features",
}
COLORS = {
    "E0": "#b8b8b8",
    "E1": "#8c8c8c",
    "E2": "#c28f3e",
    "E4": "#d8b878",
    "M1-S0": "#1b7f79",
    "M2-S0": "#7b5ea7",
    "M3-S0": "#2f6db5",
    "B2": "#444444",
}


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    manifest(path)


def _task_points(rows, estimator, split="test", card="REF", frame="self"):
    selected = [
        r
        for r in rows
        if r["estimator"] == estimator
        and r["split"] == split
        and r["card_id"] == card
        and r.get("frame", "self") == frame
    ]
    records = task_records(selected)
    kept = [r for r in records if r["predicted"]]
    return (
        np.array([r["actual"] for r in kept]),
        np.array([r["predicted"] for r in kept]),
        summary(records),
    )


def hero(rows, directory):
    names = ["E1", "E2", "M2-S0", "M1-S0"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
    for axis, name in zip(axes, names, strict=True):
        actual, predicted, stats = _task_points(rows, name)
        axis.scatter(actual, predicted, s=14, alpha=0.5, color=COLORS[name], linewidths=0)
        line = np.geomspace(0.2, 400, 50)
        axis.plot(line, line, color="black", linewidth=0.9)
        axis.fill_between(line, line / 2, line * 2, color="#37a899", alpha=0.13)
        axis.set(xscale="log", yscale="log", xlim=(0.5, 300), ylim=(0.5, 300))
        axis.set_title(
            f"{LABELS[name]}\nMdAE {stats['mdae_log10']:.2f} log10 · within 2×: {stats['within2x']:.0%}",
            fontsize=9,
        )
        axis.set_xlabel("measured own time (s)")
    axes[0].set_ylabel("the model's own estimate (s)")
    fig.suptitle("Tiny Aya Global estimating its own coding-task time, 255 held-out tasks", y=1.02)
    _save(fig, directory / "F1-estimated-vs-measured.png")


def accuracy_bars(metrics, directory):
    order = ["E0", "E1", "E2", "E4", "B0", "B2", "M2-S0", "M1-S0", "M3-S0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for axis, split in zip(axes, ("test", "test_ood"), strict=True):
        found = [
            m
            for name in order
            for m in metrics
            if m["estimator"] == name
            and m["split"] == split
            and m["card_id"] == "REF"
            and m["frame"] == "self"
        ]
        values = [m["mdae_log10"] for m in found]
        low = [m["mdae_log10"] - m["ci95"]["mdae_log10"][0] for m in found]
        high = [m["ci95"]["mdae_log10"][1] - m["mdae_log10"] for m in found]
        axis.bar(
            range(len(found)),
            values,
            yerr=[low, high],
            color=[COLORS.get(m["estimator"], "#999999") for m in found],
            capsize=3,
        )
        axis.set_xticks(range(len(found)))
        axis.set_xticklabels([m["estimator"].replace("-S0", "") for m in found], rotation=0)
        axis.set_yscale("log")
        axis.axhline(np.log10(2), color="black", linestyle=":", linewidth=0.8)
        axis.set_title("Held-out tasks" if split == "test" else "Unseen task family (class-level)")
    axes[0].set_ylabel("median |log10(estimate / measured)|  (lower is better)")
    fig.suptitle("Estimate error by method, reference speed card, 95% CIs", y=1.03)
    _save(fig, directory / "F2-accuracy-by-method.png")


def speed_cards(extra, directory):
    cards = extra["by_speed_card_test"]
    order = [
        "seen-edge",
        "seen-cpu",
        "extrap-low",
        "REF",
        "seen-dc",
        "heldout-190",
        "heldout-230",
        "extrap-high",
    ]
    order = [c for c in order if c in cards]
    fig, axis = plt.subplots(figsize=(9, 3.8))
    for name in ("E2", "B2", "M2-S0", "M1-S0", "M3-S0"):
        axis.plot(
            range(len(order)),
            [cards[c].get(name, np.nan) for c in order],
            marker="o",
            label=LABELS[name],
            color=COLORS[name],
        )
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=20)
    axis.set_ylabel("MdAE (log10)")
    axis.set_title(
        "Accuracy across serving speeds (stated in the prompt, never seen together at test)"
    )
    axis.legend(fontsize=7, ncol=2)
    _save(fig, directory / "F3-speed-cards.png")


def operations(simulations, directory):
    names = ["E1", "E2", "B2", "M2-S0", "M1-S0", "M3-S0"]
    present = [n for n in names if any(k.startswith(n + "|") for k in simulations)]
    entries = {n: next(v for k, v in simulations.items() if k.startswith(n + "|")) for n in present}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fifo = np.mean([e["scheduling"]["fifo_relative_to_oracle"] for e in entries.values()])
    axes[0].bar(
        range(len(present) + 1),
        [fifo] + [entries[n]["scheduling"]["sjf_relative_to_oracle"] for n in present],
        color=["#cccccc"] + [COLORS[n] for n in present],
    )
    axes[0].set_xticks(range(len(present) + 1))
    axes[0].set_xticklabels(["FIFO"] + [n.replace("-S0", "") for n in present])
    axes[0].axhline(1, color="black", linewidth=0.8)
    axes[0].set_ylabel("mean completion time / oracle shortest-job-first")
    axes[0].set_ylim(1, None)
    axes[0].set_title("Queue of 50 tasks ordered by each estimate")
    for name in present:
        trip = entries[name].get("tripwire")
        if trip:
            axes[1].plot(
                [t["false_stop_rate"] for t in trip],
                [t["failed_time_saved_share"] for t in trip],
                marker="o",
                color=COLORS[name],
                label=name.replace("-S0", ""),
            )
    axes[1].set_xlabel("false-stop rate on runs that would pass")
    axes[1].set_ylabel("share of failing-run time saved")
    axes[1].set_title("Stop a run at k × its own upper estimate, k = 3, 2, 1.5, 1")
    axes[1].legend(fontsize=7)
    _save(fig, directory / "F4-scheduling-and-tripwires.png")


def deployment(path, directory):
    rows = read_jsonl(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    for axis, key, title in zip(
        axes, ("base_raw", "tuned_raw"), ("Base model", "Anthropochrony M1"), strict=True
    ):
        points = [
            (r["wall_s"], parse(r[key]).as_dict()["point_s"])
            for r in rows
            if parse(r[key]).as_dict()["point_s"]
        ]
        actual, predicted = np.array(points).T
        error = np.median(np.abs(np.log10(predicted / actual)))
        axis.scatter(
            actual, predicted, s=16, alpha=0.6, color="#1b7f79" if "M1" in title else "#999"
        )
        line = np.geomspace(1, 600, 50)
        axis.plot(line, line, color="black", linewidth=0.9)
        axis.fill_between(line, line / 2, line * 2, color="#37a899", alpha=0.13)
        axis.set(
            xscale="log",
            yscale="log",
            ylim=(0.8, 3000),
            xlabel="stopwatch time on this machine (s)",
        )
        axis.set_title(f"{title}: MdAE {error:.2f}", fontsize=10)
    axes[0].set_ylabel("estimate (s)")
    fig.suptitle("Real CPU deployment (llama.cpp, Q4_K_M), 100 held-out tasks", y=1.02)
    _save(fig, directory / "F5-real-cpu-deployment.png")


def make_figures(root):
    root = Path(root)
    directory = root / "paper/figures"
    directory.mkdir(parents=True, exist_ok=True)
    rows = normalized_rows(read_jsonl(root / "data/eval/estimates.jsonl"))
    metrics = json.loads((root / "results/metrics.json").read_text())
    hero(rows, directory)
    accuracy_bars(metrics, directory)
    extra = root / "results/extra-analyses.json"
    if extra.exists():
        speed_cards(json.loads(extra.read_text()), directory)
    simulations = root / "results/simulations.json"
    if simulations.exists():
        operations(json.loads(simulations.read_text()), directory)
    x10 = root / "data/x10/cpu-deployment.jsonl"
    if x10.exists():
        deployment(x10, directory)
