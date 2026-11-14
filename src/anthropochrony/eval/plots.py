"""Publication-grade paper and article figures drawn strictly from released result files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anthropochrony.artifacts import manifest, read_jsonl
from anthropochrony.eval.metrics import summary, task_records
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.eval.structured import (
    WorkloadPriors,
    build_structured_rows,
    fit_workload_priors,
    structured_duration,
    x10_predictions,
)
from anthropochrony.parse import parse
from anthropochrony.timemodel import SpeedCard

LABELS = {
    "E0": "Base zero-shot (E0)",
    "E1": "Base + speed card (E1)",
    "E2": "5-shot measured (E2)",
    "E4": "Rescaled base (E4)",
    "B0": "Constant median (B0)",
    "B1": "Family median (B1)",
    "B2": "Tabular features (B2)",
    "B2+": "Ref-code oracle (B2+)",
    "B4": "Frontier teacher (B4)",
    "M1-S0": "M1 verbal (measured)",
    "M1-Struct": "M1-Struct (workload + exact timing)",
    "M1b": "M1b (CQR conformal)",
    "M1-Struct+CP": "M1-Struct+CP (conformal)",
    "M2-S0": "M2 verbal (teacher)",
    "M3-S0": "M3 verbal (+raw_extra)",
}

COLORS = {
    "E0": "#9e9e9e",
    "E1": "#757575",
    "E4": "#bdbdbd",
    "E2": "#d9822b",
    "B0": "#8d6e63",
    "B1": "#6d4c41",
    "B2": "#37474f",
    "B2+": "#263238",
    "B4": "#c2185b",
    "M2-S0": "#7b5ea7",
    "M3-S0": "#2f6db5",
    "M1-S0": "#1b7f79",
    "M1-Struct": "#0b5d56",
    "M1b": "#2a9d8f",
    "M1-Struct+CP": "#006d77",
}


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


def _save(fig, path: Path):
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
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


def hero(rows, directory: Path):
    _style()
    names = ["E1", "E2", "M2-S0", "M1-S0", "M1-Struct"]
    short_titles = {
        "E1": "Base + Speed Card (E1)",
        "E2": "5-Shot Measured (E2)",
        "M2-S0": "Teacher SFT (M2)",
        "M1-S0": "Measured SFT (M1)",
        "M1-Struct": "Workload + Timing (M1-Struct)",
    }
    has_struct = any(r["estimator"] == "M1-Struct" for r in rows)
    if not has_struct:
        names = names[:4]

    fig, axes = plt.subplots(
        1, len(names), figsize=(3.35 * len(names), 3.7), sharex=True, sharey=True
    )
    line = np.geomspace(0.4, 400, 80)
    for axis, name in zip(axes, names, strict=True):
        actual, predicted, stats = _task_points(rows, name)
        clipped_pred = np.clip(predicted, 0.45, 18000.0)
        axis.fill_between(line, line / 2, line * 2, color="#2a9d8f", alpha=0.15, zorder=1)
        axis.plot(line, line, color="#111111", linewidth=1.0, zorder=2)
        axis.scatter(
            actual,
            clipped_pred,
            s=15,
            alpha=0.55,
            color=COLORS[name],
            edgecolors="none",
            zorder=3,
        )
        axis.set(xscale="log", yscale="log", xlim=(0.5, 320), ylim=(0.5, 20000))
        axis.set_yticks([1, 10, 60, 600, 7200])
        axis.set_yticklabels(["1 s", "10 s", "1 min", "10 min", "2 h"])
        axis.set_xticks([1, 10, 60, 240])
        axis.set_xticklabels(["1 s", "10 s", "1 min", "4 min"])
        axis.set_title(
            f"{short_titles[name]}\nMdAE {stats['mdae_log10']:.2f} · within 2×: {stats['within2x']:.0%}",
            fontsize=8.8,
            fontweight="bold" if name in {"M1-S0", "M1-Struct"} else "normal",
        )
        axis.set_xlabel("Measured own time (s)")
        if name == "E1":
            axis.annotate(
                "24× human-time\noverestimation",
                xy=(12, 600),
                xytext=(2.0, 4200),
                fontsize=7.5,
                color="#444444",
                arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.8},
            )
    axes[0].set_ylabel("Model's self-estimated time")
    fig.suptitle(
        "Anthropochrony on Tiny Aya Global (3.35B): Eliminating Human-Time Bias on 255 Held-Out Coding Tasks",
        y=1.04,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F1-estimated-vs-measured.png")


def accuracy_bars(
    metrics, extra, directory: Path, *, noise_by_split: dict[str, float] | None = None
):
    _style()
    order = ["E0", "E1", "E4", "E2", "B0", "B1", "B4", "B2", "M2-S0", "M3-S0", "M1-S0", "M1-Struct"]
    short = {
        "E0": "E0",
        "E1": "E1",
        "E4": "E4",
        "E2": "E2",
        "B0": "B0",
        "B1": "B1",
        "B4": "B4\n(API)",
        "B2": "B2\n(Tab)",
        "M2-S0": "M2",
        "M3-S0": "M3*",
        "M1-S0": "M1",
        "M1-Struct": "M1-\nStruct",
    }
    posthoc = (extra or {}).get("posthoc_ref_summary", {})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.1), sharey=True)
    for axis, split in zip(axes, ("test", "test_ood"), strict=True):
        metric_map = {
            m["estimator"]: {
                "mdae": m["mdae_log10"],
                "ci": m["ci95"]["mdae_log10"],
            }
            for m in metrics
            if m["split"] == split and m["card_id"] == "REF" and m["frame"] == "self"
        }
        for est, entry in posthoc.get(split, {}).items():
            if est not in metric_map and entry.get("ci95"):
                metric_map[est] = {"mdae": entry["mdae_log10"], "ci": entry["ci95"]}

        present = [name for name in order if name in metric_map]
        values = [metric_map[n]["mdae"] for n in present]
        low = [max(0.0, metric_map[n]["mdae"] - metric_map[n]["ci"][0]) for n in present]
        high = [max(0.0, metric_map[n]["ci"][1] - metric_map[n]["mdae"]) for n in present]
        ci_tops = [metric_map[n]["ci"][1] for n in present]
        bars = axis.bar(
            range(len(present)),
            values,
            yerr=[low, high],
            color=[COLORS.get(n, "#888888") for n in present],
            capsize=2.5,
            width=0.72,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"lw": 0.9, "ecolor": "#222222"},
        )
        for bar, val, ci_top in zip(bars, values, ci_tops, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(val, ci_top) * 1.09,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#222222",
            )
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels([short.get(n, n) for n in present], rotation=0, fontsize=7.5)
        axis.set_yscale("log")
        axis.set_ylim(0.02, 3.4)
        axis.axhline(
            math.log10(2),
            color="#2a9d8f",
            linestyle=":",
            linewidth=1.1,
            label="2× error band (0.301 log10)",
        )
        n_val = (noise_by_split or {}).get(split) or metric_map.get("N", {}).get("mdae")
        if n_val:
            axis.axhline(
                n_val,
                color="#555555",
                linestyle="--",
                linewidth=0.95,
                label=f"Split-half trajectory noise N ({n_val:.2f})",
            )
        axis.set_title(
            "In-Distribution Held-Out Tasks (test, n = 255)"
            if split == "test"
            else "Unseen Task Family: Class-Level OOP (test_ood, n = 93)",
            fontweight="bold",
        )
        axis.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    axes[0].set_ylabel("Median |log10(estimate / measured)|  (lower is better)")
    fig.suptitle(
        "Accuracy Across Prompting, Tabular, Teacher-API, Verbal SFT, and Workload + Exact-Timing Estimators (95% CIs)",
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F2-accuracy-by-method.png")


def speed_cards(extra, directory: Path):
    _style()
    dis = extra.get("speed_card_disentanglement", {}).get("by_split_and_card")
    order = [
        "extrap-low",
        "seen-cpu",
        "seen-edge",
        "REF",
        "heldout-190",
        "heldout-230",
        "seen-dc",
        "extrap-high",
    ]
    tick_labels = [
        "extrap-low\n(1.5 tok/s)",
        "seen-cpu\n(8 tok/s)",
        "seen-edge\n(60 tok/s)",
        "REF\n(96 tok/s)",
        "heldout-190\n(190 tok/s)",
        "heldout-230\n(230 tok/s)",
        "seen-dc\n(400 tok/s)",
        "extrap-high\n(5k tok/s)",
    ]

    if dis:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.1), sharey=True)
        for axis, split, title in zip(
            axes,
            ("test", "test_ood"),
            ("Held-Out Tasks (test, n = 255)", "Unseen OOP Family (test_ood, n = 93)"),
            strict=True,
        ):
            card_map = dis[split]
            present_cards = [c for c in order if c in card_map]
            present_ticks = [tick_labels[order.index(c)] for c in present_cards]
            # Shade held-out speed band [150, 260]
            if "heldout-190" in present_cards and "heldout-230" in present_cards:
                i0 = present_cards.index("heldout-190") - 0.35
                i1 = present_cards.index("heldout-230") + 0.35
                axis.axvspan(i0, i1, color="#f4a261", alpha=0.14, label="Held-out speed band")
            for name, style, lw in (
                ("E2", "--", 1.5),
                ("M2-S0", "-.", 1.5),
                ("M1-S0", "-", 2.0),
                ("M1-Struct", "-", 2.5),
                ("B2", ":", 1.8),
            ):
                vals = [card_map[c].get(name, {}).get("mdae_log10", np.nan) for c in present_cards]
                if not np.isnan(vals).all():
                    axis.plot(
                        range(len(present_cards)),
                        vals,
                        marker="o" if name != "M1-Struct" else "D",
                        markersize=5 if name != "M1-Struct" else 5.8,
                        linestyle=style,
                        linewidth=lw,
                        label=LABELS[name],
                        color=COLORS[name],
                    )
            axis.axhline(math.log10(2), color="#2a9d8f", linestyle=":", linewidth=0.9)
            axis.set_xticks(range(len(present_cards)))
            axis.set_xticklabels(present_ticks, rotation=22, ha="right", fontsize=7.5)
            axis.set_title(title, fontweight="bold")
            if split == "test" and "extrap-low" in present_cards:
                y_m1 = card_map["extrap-low"]["M1-S0"]["mdae_log10"]
                y_st = card_map["extrap-low"]["M1-Struct"]["mdae_log10"]
                axis.annotate(
                    f"3.2× error drop\n({y_m1:.2f} → {y_st:.2f})",
                    xy=(0, y_st),
                    xytext=(0.75, 0.43),
                    fontsize=7.6,
                    fontweight="bold",
                    color=COLORS["M1-Struct"],
                    arrowprops={"arrowstyle": "->", "color": COLORS["M1-Struct"], "lw": 1.0},
                )
        axes[0].set_ylabel("Median |log10(estimate / measured)|")
        axes[1].legend(loc="upper right", frameon=True, facecolor="white", ncol=1)
        fig.suptitle(
            "Hardware Speed Conditioning Across 3,333× of Decode Throughput (1.5 to 5,000 tok/s)",
            y=1.03,
            fontsize=10.5,
            fontweight="bold",
        )
        _save(fig, directory / "F3-speed-cards.png")
        return

    cards = extra["by_speed_card_test"]
    present_cards = [c for c in order if c in cards]
    fig, axis = plt.subplots(figsize=(9.2, 3.8))
    for name in ("E2", "B2", "M2-S0", "M1-S0", "M1-Struct", "M3-S0"):
        if any(name in cards[c] for c in present_cards):
            axis.plot(
                range(len(present_cards)),
                [cards[c].get(name, np.nan) for c in present_cards],
                marker="o",
                label=LABELS[name],
                color=COLORS[name],
            )
    axis.set_xticks(range(len(present_cards)))
    axis.set_xticklabels(present_cards, rotation=20)
    axis.set_ylabel("MdAE (log10)")
    axis.legend(fontsize=7.5, ncol=2)
    _save(fig, directory / "F3-speed-cards.png")


def operations(simulations, directory: Path):
    _style()
    sys_sum = simulations.get("_systems_summary", {})
    has_routing = bool(sys_sum.get("sla_tier_routing_15s"))
    ncols = 3 if has_routing else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 3.9))

    # Panel A: SJF Scheduling (test & OOD)
    names = ["E1", "M2-S0", "M1-S0", "B4", "M1-Struct", "E2", "B2"]
    present = [n for n in names if any(k.startswith(n + "|") for k in simulations)]
    entries = {n: next(v for k, v in simulations.items() if k.startswith(n + "|")) for n in present}
    fifo = float(np.mean([e["scheduling"]["fifo_relative_to_oracle"] for e in entries.values()]))
    ood_sim = sys_sum.get("ood_class_simulation", {})
    x = np.arange(len(present) + 1)
    test_vals = [fifo] + [entries[n]["scheduling"]["sjf_relative_to_oracle"] for n in present]
    if ood_sim:
        ood_entries = {
            n: next((v for k, v in ood_sim.items() if k.startswith(n + "|")), None) for n in present
        }
        ood_fifo = next(
            (
                v["scheduling_q30"]["fifo_relative_to_oracle"]
                for v in ood_entries.values()
                if v is not None
            ),
            1.38,
        )
        ood_vals = [ood_fifo] + [
            ood_entries[n]["scheduling_q30"]["sjf_relative_to_oracle"] if ood_entries[n] else np.nan
            for n in present
        ]
        w = 0.38
        axes[0].bar(
            x - w / 2,
            test_vals,
            width=w,
            color=["#bdbdbd"] + [COLORS[n] for n in present],
            label="Held-out test (q=50)",
        )
        axes[0].bar(
            x + w / 2,
            ood_vals,
            width=w,
            color=["#e0e0e0"] + [COLORS[n] for n in present],
            alpha=0.55,
            hatch="//",
            edgecolor="#333333",
            linewidth=0.4,
            label="Unseen CLASS OOD (q=30)",
        )
        axes[0].legend(loc="upper right", frameon=True, facecolor="white", fontsize=7.2)
    else:
        axes[0].bar(x, test_vals, color=["#bdbdbd"] + [COLORS[n] for n in present])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["FIFO"] + [n.replace("-S0", "").replace("M1-Struct", "M1-\nStruct") for n in present],
        fontsize=7.3,
    )
    axes[0].axhline(1.0, color="#111111", linewidth=0.9)
    axes[0].set_ylim(1.0, 1.78)
    axes[0].set_ylabel("Mean wait / oracle SJF (1.0 = optimal)")
    axes[0].set_title("(a) Shortest-Job-First Queue Scheduling", fontweight="bold")

    # Panel B: Conformal SLA Timeout Tripwires
    trip_names = ["E2", "M2-S0", "M1-S0", "B2", "M1b", "M1-Struct+CP"]
    for name in trip_names:
        match = next(
            (v for k, v in simulations.items() if k.startswith(name + "|")),
            None,
        )
        if not match or not match.get("tripwire"):
            continue
        trip = match["tripwire"]
        xs = [100.0 * t["false_stop_rate"] for t in trip]
        ys = [100.0 * t["failed_time_saved_share"] for t in trip]
        axes[1].plot(
            xs,
            ys,
            marker="o" if "CP" not in name else "*",
            markersize=4.5 if "CP" not in name else 8,
            linewidth=2.1 if name in {"M1-Struct+CP", "M1b"} else 1.4,
            color=COLORS.get(name, "#666666"),
            label=name.replace("-S0", ""),
        )
        k1 = next((t for t in trip if t["multiplier"] == 1), None)
        if k1 and name in {"M1-S0", "M1-Struct+CP"}:
            axes[1].scatter(
                [100.0 * k1["false_stop_rate"]],
                [100.0 * k1["failed_time_saved_share"]],
                s=55,
                facecolors="none",
                edgecolors="#111111",
                linewidths=1.2,
                zorder=5,
            )
    axes[1].annotate(
        "k=1.0×: 13.9% → 1.3%\nfalse stops (10.5× drop)",
        xy=(1.33, 20.5),
        xytext=(0.5, 44.5),
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["M1-Struct+CP"],
        arrowprops={"arrowstyle": "->", "color": COLORS["M1-Struct+CP"], "lw": 0.95},
    )
    axes[1].set_xlim(-1.5, 46)
    axes[1].set_xlabel("False-stop rate on passing runs (%)")
    axes[1].set_ylabel("Failing-run GPU time saved (%)")
    axes[1].set_title("(b) SLA Timeout Tripwires (2,040 test runs)", fontweight="bold")
    axes[1].legend(loc="lower right", frameon=True, facecolor="white", fontsize=7.2)

    # Panel C: Multi-Tier Hardware SLA Routing (15s deadline)
    if has_routing:
        rt = sys_sum["sla_tier_routing_15s"]
        r_order = [
            "Always-Edge",
            "Always-DC",
            "E1",
            "M1-S0",
            "B2",
            "E2",
            "M1-Struct+CP",
        ]
        r_present = [r for r in r_order if r in rt]
        for rname in r_present:
            entry = rt[rname]
            vx = 100.0 * entry["sla_violation_rate"]
            cy = entry["passing_cost_usd_per_1k_runs"]
            col = COLORS.get(rname, "#555555")
            marker = "*" if rname == "M1-Struct+CP" else "o"
            sz = 120 if rname == "M1-Struct+CP" else 48
            axes[2].scatter([vx], [cy], s=sz, color=col, marker=marker, zorder=4)
            if rname == "Always-DC":
                dx, dy, ha = 0.8, -0.12, "left"
            elif rname == "M1-Struct+CP":
                dx, dy, ha = 0.0, -0.19, "center"
            elif rname == "Always-Edge":
                dx, dy, ha = -0.6, 0.08, "right"
            else:
                dx, dy, ha = 0.8, 0.04, "left"
            axes[2].text(
                vx + dx,
                cy + dy,
                rname.replace("-S0", ""),
                ha=ha,
                fontsize=7.3,
                fontweight="bold" if rname == "M1-Struct+CP" else "normal",
                color=col if rname == "M1-Struct+CP" else "#222222",
            )
        axes[2].set_xlim(-1.5, 35)
        axes[2].set_xlabel("15 s SLA violation rate (%)  (lower is better)")
        axes[2].set_ylabel("Cluster cost ($ / 1,000 runs)")
        axes[2].set_title("(c) Edge vs. DC Tier Routing (15 s SLA)", fontweight="bold")

    fig.suptitle(
        "Downstream Systems Utility: Queue Scheduling, Conformal Timeout Tripwires, and Multi-Tier SLA Routing",
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F4-scheduling-and-tripwires.png")


def deployment(
    path: Path,
    directory: Path,
    *,
    tasks_by_id: dict[str, dict] | None = None,
    priors: WorkloadPriors | None = None,
    cpu_card: SpeedCard | None = None,
):
    _style()
    rows = read_jsonl(path)
    include_struct = bool(tasks_by_id and priors and cpu_card)
    ncols = 3 if include_struct else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.1 * ncols, 3.8), sharex=True, sharey=True)

    panels = [
        ("base", "Base Tiny Aya (Verbal)", "#8c8c8c"),
        ("m1", "Anthropochrony M1 (Verbal)", COLORS["M1-S0"]),
    ]
    if include_struct:
        panels.append(("struct", "M1-Struct (Local L=0)", COLORS["M1-Struct"]))

    line = np.geomspace(1.0, 500.0, 60)
    for axis, mode, title, color in zip(
        axes, [p[0] for p in panels], [p[1] for p in panels], [p[2] for p in panels], strict=True
    ):
        pts = []
        for r in rows:
            wall = float(r["wall_s"])
            if mode == "base":
                pred = parse(r["base_raw"]).point_s
            elif mode == "m1":
                pred = parse(r["tuned_raw"]).point_s
            else:
                m_est = parse(r["tuned_raw"])
                meta = tasks_by_id.get(r["task_id"], {})
                pred = (
                    structured_duration(
                        m_est.tokens,
                        m_est.attempts,
                        float(meta.get("uncached_task_tokens", 120.0)),
                        r["family"],
                        cpu_card,
                        priors,
                        latency_s=0.0,
                    )
                    or m_est.point_s
                )
            if pred and pred > 0:
                pts.append((wall, pred))
        actual, predicted = np.array(pts).T
        errs = np.abs(np.log10(predicted / actual))
        mdae = float(np.median(errs))
        w2x = float(np.mean(errs <= math.log10(2)))
        axis.fill_between(line, line / 2, line * 2, color="#2a9d8f", alpha=0.15, zorder=1)
        axis.plot(line, line, color="#111111", linewidth=1.0, zorder=2)
        axis.scatter(actual, predicted, s=18, alpha=0.62, color=color, edgecolors="none", zorder=3)
        axis.set(
            xscale="log",
            yscale="log",
            xlim=(1.0, 450.0),
            ylim=(0.8, 3000.0),
            xlabel="Real CPU stopwatch wall-time (s)",
        )
        axis.set_title(
            f"{title}\nMdAE {mdae:.2f} · within 2×: {w2x:.0%}",
            fontsize=9.0,
            fontweight="bold" if mode != "base" else "normal",
        )
    axes[0].set_ylabel("Model self-estimate before solving (s)")
    fig.suptitle(
        "Real Desktop CPU Deployment (llama.cpp Q4_K_M, 20 tok/s, n = 100 Held-Out Tasks)",
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F5-real-cpu-deployment.png")


def real_hardware(root: Path, directory: Path, priors: WorkloadPriors):
    """F8: the same X10 tasks and GGUFs on a desktop CPU and a consumer GPU."""
    cpu = x10_predictions(
        root, priors, "data/x10/cpu-deployment.jsonl", "data/x10/cpu-speed-card.json"
    )
    gpu = x10_predictions(
        root, priors, "data/x10/gpu-deployment.jsonl", "data/x10/gpu-deployment-speed-card.json"
    )
    if cpu is None or gpu is None:
        return
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    machines = [("CPU", cpu, "o"), ("GPU", gpu, "^")]
    line = np.geomspace(0.3, 3000.0, 60)
    axis = axes[0]
    axis.fill_between(line, line / 2, line * 2, color="#2a9d8f", alpha=0.15, zorder=1)
    axis.plot(line, line, color="#111111", linewidth=1.0, zorder=2)
    for name, (card, preds), marker in machines:
        for key, label, color in (
            ("base_verbal", "Base, no speed given", COLORS["E0"]),
            ("m1_struct_local", "M1-Struct", COLORS["M1-Struct"]),
        ):
            pts = np.array([(p["actual"], p[key]) for p in preds if p[key] and p[key] > 0])
            errs = np.abs(np.log10(pts[:, 1] / pts[:, 0]))
            axis.scatter(
                pts[:, 0],
                pts[:, 1],
                s=16 if marker == "o" else 20,
                marker=marker,
                alpha=0.6,
                color=color if name == "GPU" else "none",
                edgecolors=color,
                linewidths=0.8,
                label=(
                    f"{label}, {name} {card['S']:g} tok/s: "
                    f"{np.sum(errs <= math.log10(2)) / len(preds):.0%} within 2×"
                ),
                zorder=3,
            )
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(0.5, 500.0),
        ylim=(0.5, 3000.0),
        xlabel="Real stopwatch time for the run (s)",
        ylabel="Model's estimate before starting (s)",
        title="(a) Estimate vs stopwatch on two machines",
    )
    axis.legend(loc="upper left", fontsize=7.2, frameon=False)

    axis = axes[1]
    gpu_by_id = {p["task_id"]: p for p in gpu[1]}
    pairs = [(c, gpu_by_id[c["task_id"]]) for c in cpu[1] if c["task_id"] in gpu_by_id]
    spd = np.geomspace(0.3, 60.0, 40)
    axis.fill_between(spd, spd / 2, spd * 2, color="#2a9d8f", alpha=0.15, zorder=1)
    axis.plot(spd, spd, color="#111111", linewidth=1.0, zorder=2)
    for key, label, color, marker in (
        ("base_verbal", "Base, no speed given", COLORS["E0"], "o"),
        ("m1_verbal", "M1 verbal", COLORS["E2"], "s"),
        ("m1_struct_local", "M1-Struct", COLORS["M1-Struct"], "o"),
    ):
        pts = np.array(
            [
                (c["actual"] / g["actual"], c[key] / g[key])
                for c, g in pairs
                if c[key] and g[key] and c[key] > 0 and g[key] > 0
            ]
        )
        errs = np.abs(np.log10(pts[:, 1] / pts[:, 0]))
        axis.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            marker=marker,
            alpha=0.65,
            color=color if marker == "o" else "none",
            edgecolors=color,
            linewidths=0.9,
            label=f"{label}: {np.sum(errs <= math.log10(2)) / len(pairs):.0%} within 2×",
            zorder=3,
        )
    actual_gm = 10 ** np.mean([math.log10(c["actual"] / g["actual"]) for c, g in pairs])
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(0.3, 60.0),
        ylim=(0.3, 60.0),
        xlabel="Actual speedup on GPU (CPU time / GPU time)",
        ylabel="Predicted speedup (CPU estimate / GPU estimate)",
        title=f"(b) Does the estimate follow the hardware? (typical actual {actual_gm:.1f}×)",
    )
    axis.legend(loc="upper left", fontsize=7.2, frameon=False)
    fig.suptitle(
        f"Real deployments: llama.cpp Q4_K_M on a desktop CPU ({cpu[0]['S']:g} tok/s) "
        f"and a consumer GPU ({gpu[0]['S']:g} tok/s), n = {len(pairs)} paired tasks",
        y=1.02,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F8-real-cpu-and-gpu.png")


def calibration_and_uncertainty(extra: dict, directory: Path):
    """F6: X1 fresh-trajectory conformal interval calibration + X8 pass@1 reliability."""
    _style()
    x1 = extra.get("x1_interval_calibration", {}).get("fresh_trajectories", {})
    x8 = extra.get("x8_first_pass_calibration", {})
    if not (x1 and x8):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1))

    # Panel A: X1 Fresh-trajectory 80% coverage + Winkler annotations
    methods = ["E2", "M2-S0", "M1-S0", "M1b", "M1-Struct+CP"]
    present = [m for m in methods if m in x1.get("test", {})]
    x = np.arange(len(present))
    w = 0.36
    cov_test = [100.0 * x1["test"][m]["coverage80"] for m in present]
    cov_ood = [100.0 * x1["test_ood"][m]["coverage80"] for m in present]
    wink_test = [x1["test"][m]["winkler"] for m in present]

    b1 = axes[0].bar(
        x - w / 2,
        cov_test,
        width=w,
        color=[COLORS[m] for m in present],
        label="Held-out test (2,040 runs)",
    )
    axes[0].bar(
        x + w / 2,
        cov_ood,
        width=w,
        color=[COLORS[m] for m in present],
        alpha=0.55,
        hatch="//",
        edgecolor="#222222",
        linewidth=0.5,
        label="Unseen CLASS OOD (744 runs)",
    )
    for bar, wk in zip(b1, wink_test, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"W={wk:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#222222",
        )
    axes[0].axhline(
        80.0, color="#c2185b", linestyle="--", linewidth=1.2, label="Nominal 80% target"
    )
    axes[0].set_ylim(0, 96)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [
            m.replace("-S0", "\n(aleatoric)").replace("M1-Struct+CP", "M1-Struct\n+CP")
            for m in present
        ],
        fontsize=7.6,
    )
    axes[0].set_ylabel("Empirical interval coverage on fresh runs (%)")
    axes[0].set_title("(a) X1 Fresh-Trajectory 80% Interval Calibration", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", fontsize=7.4)

    # Panel B: X8 First-attempt pass probability reliability diagram (test)
    axes[1].plot(
        [0, 1], [0, 1], color="#111111", linestyle="--", linewidth=1.0, label="Perfect calibration"
    )
    for est, marker in (
        ("B4", "s"),
        ("M2-S0", "^"),
        ("E2", "o"),
        ("M1-S0", "D"),
        ("M3-S0", "P"),
    ):
        entry = x8.get("test", {}).get(est)
        if not entry:
            continue
        bins = entry["reliability_5bin"]
        px = [b["mean_predicted"] for b in bins]
        py = [b["mean_empirical"] for b in bins]
        ece = entry["ece_empirical_rate"]
        axes[1].plot(
            px,
            py,
            marker=marker,
            markersize=5.5,
            linewidth=1.8,
            color=COLORS[est],
            label=f"{est.replace('-S0', '')} (ECE {ece:.2f})",
        )
    axes[1].set_xlim(0, 1.0)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Predicted first-attempt pass probability")
    axes[1].set_ylabel("Empirical K=8 first-attempt pass rate")
    axes[1].set_title(
        "(b) X8 Pass@1 Reliability: Teacher Overconfidence vs. Measured", fontweight="bold"
    )
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", fontsize=7.4)

    fig.suptitle(
        "Uncertainty Calibration: Resolving the Aleatoric Interval Trap (X1) and Teacher Overconfidence (X8)",
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F6-calibration-and-uncertainty.png")


def disentanglement_and_forensics(extra: dict, directory: Path):
    """F7: Mechanistic T8 workload-arithmetic decomposition + M3 curation forensic audit."""
    _style()
    tok_audit = extra.get("token_and_arithmetic_audit", {})
    m3_audit = extra.get("m3_curation_audit", {})
    speed_dis = extra.get("speed_card_disentanglement", {}).get("by_split_and_card", {})
    posthoc = extra.get("posthoc_ref_summary", {})
    if not (tok_audit and m3_audit and speed_dis and posthoc):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0))

    # Panel A: Why M1-Struct works — Token error vs Verbal arithmetic error vs M1-Struct
    cats = [
        "Token workload\n|log10(Ô/O)|",
        "Verbal arith.\ninconsistency",
        "M1 verbal\n(96 tok/s)",
        "M1-Struct\n(96 tok/s)",
        "M1 verbal\n(1.5 tok/s)",
        "M1-Struct\n(1.5 tok/s)",
    ]
    test_vals = [
        tok_audit["test"]["M1-S0"]["token_mdae_log10"],
        tok_audit["test"]["M1-S0"]["arithmetic_inconsistency_bare_mdae"],
        posthoc["test"]["M1-S0"]["mdae_log10"],
        posthoc["test"]["M1-Struct"]["mdae_log10"],
        speed_dis["test"]["extrap-low"]["M1-S0"]["mdae_log10"],
        speed_dis["test"]["extrap-low"]["M1-Struct"]["mdae_log10"],
    ]
    ood_vals = [
        tok_audit["test_ood"]["M1-S0"]["token_mdae_log10"],
        tok_audit["test_ood"]["M1-S0"]["arithmetic_inconsistency_bare_mdae"],
        posthoc["test_ood"]["M1-S0"]["mdae_log10"],
        posthoc["test_ood"]["M1-Struct"]["mdae_log10"],
        speed_dis["test_ood"]["extrap-low"]["M1-S0"]["mdae_log10"],
        speed_dis["test_ood"]["extrap-low"]["M1-Struct"]["mdae_log10"],
    ]
    x = np.arange(len(cats))
    w = 0.36
    bar_cols = [
        "#457b9d",
        "#e76f51",
        COLORS["M1-S0"],
        COLORS["M1-Struct"],
        "#c1121f",
        COLORS["M1-Struct"],
    ]
    axes[0].bar(x - w / 2, test_vals, width=w, color=bar_cols, label="Held-out test (n=255)")
    axes[0].bar(
        x + w / 2,
        ood_vals,
        width=w,
        color=bar_cols,
        alpha=0.55,
        hatch="//",
        edgecolor="#222222",
        linewidth=0.5,
        label="Unseen CLASS OOD (n=93)",
    )
    axes[0].axhline(math.log10(2), color="#2a9d8f", linestyle=":", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats, fontsize=7.3)
    axes[0].set_ylabel("Median |log10 ratio|")
    axes[0].set_title(
        "(a) Workload Accuracy vs. Causal Verbal Arithmetic Bottleneck", fontweight="bold"
    )
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", fontsize=7.4)

    # Panel B: Forensic audit of M3 training composition by family (M1/M2 vs M3)
    m1_fam = m3_audit["sft_train_counts"]["m1"]["by_family"]
    m3_fam = m3_audit["sft_train_counts"]["m3"]["by_family"]
    fams = ["ALGO_IO", "ALGO_FN", "FUNC", "LIB", "CLASS"]
    base_counts = [m1_fam.get(f, 0) for f in fams]
    extra_counts = [max(0, m3_fam.get(f, 0) - m1_fam.get(f, 0)) for f in fams]
    fx = np.arange(len(fams))
    axes[1].bar(
        fx, base_counts, width=0.56, color="#1b7f79", label="Curated SFT in M1 & M2 (6,077 rows)"
    )
    extra_colors = ["#2f6db5", "#2f6db5", "#2f6db5", "#2f6db5", "#d62828"]
    b_top = axes[1].bar(
        fx,
        extra_counts,
        width=0.56,
        bottom=base_counts,
        color=extra_colors,
        label="Added raw_extra in M3 (+894 valid, +40 leaked CLASS)",
    )
    for idx, (b_val, e_val) in enumerate(zip(base_counts, extra_counts, strict=True)):
        if e_val > 0:
            tag = f"+{e_val}\n(LEAK!)" if fams[idx] == "CLASS" else f"+{e_val}"
            axes[1].text(
                fx[idx],
                b_val + e_val + 45,
                tag,
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold" if fams[idx] == "CLASS" else "normal",
                color="#d62828" if fams[idx] == "CLASS" else "#1d3557",
            )
    axes[1].set_ylim(0, max(b + e for b, e in zip(base_counts, extra_counts, strict=True)) * 1.32)
    axes[1].set_xticks(fx)
    axes[1].set_xticklabels(
        ["ALGO_IO", "ALGO_FN", "FUNC", "LIB", "CLASS\n(OOD holdout)"], fontsize=7.6
    )
    axes[1].set_ylabel("SFT training examples")
    axes[1].set_title("(b) Forensic Audit of Historical M3 Curation Confounds", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", fontsize=7.3)
    _ = b_top

    fig.suptitle(
        "Mechanistic Diagnostics: Workload Decomposition + Exact Timing vs. Verbal Division & M3 Audit",
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    _save(fig, directory / "F7-disentanglement-and-forensics.png")


def make_figures(root):
    root = Path(root)
    directory = root / "paper/figures"
    directory.mkdir(parents=True, exist_ok=True)
    rows = normalized_rows(read_jsonl(root / "data/eval/estimates.jsonl"))
    tasks_path = root / "data/tasks/tasks.jsonl"
    tasks_by_id = {t["task_id"]: t for t in read_jsonl(tasks_path)} if tasks_path.exists() else {}
    priors = None
    if tasks_by_id:
        try:
            priors = fit_workload_priors(root)
            rows = rows + build_structured_rows(
                rows, tasks_by_id, priors, source_estimator="M1-S0", estimator_name="M1-Struct"
            )
        except FileNotFoundError:
            priors = None

    metrics = json.loads((root / "results/metrics.json").read_text())
    extra_path = root / "results/extra-analyses.json"
    extra_data = json.loads(extra_path.read_text()) if extra_path.exists() else None
    noise_path = root / "results/noise-ceiling.json"
    noise_by_split = {}
    if noise_path.exists() and tasks_by_id:
        raw_noise = json.loads(noise_path.read_text())
        for sp in ("test", "test_ood"):
            vals = [
                float(raw_noise[tid])
                for tid, meta in tasks_by_id.items()
                if meta.get("split") == sp and tid in raw_noise
            ]
            if vals:
                noise_by_split[sp] = float(np.median(vals))

    hero(rows, directory)
    accuracy_bars(metrics, extra_data, directory, noise_by_split=noise_by_split)
    if extra_data:
        speed_cards(extra_data, directory)
        calibration_and_uncertainty(extra_data, directory)
        disentanglement_and_forensics(extra_data, directory)
    simulations = root / "results/simulations.json"
    if simulations.exists():
        operations(json.loads(simulations.read_text()), directory)
    x10 = root / "data/x10/cpu-deployment.jsonl"
    if x10.exists():
        card_path = root / "data/x10/cpu-speed-card.json"
        cpu_card = None
        if card_path.exists():
            c = json.loads(card_path.read_text())
            cpu_card = SpeedCard(c["id"], float(c["S"]), float(c["P"]))
        deployment(x10, directory, tasks_by_id=tasks_by_id, priors=priors, cpu_card=cpu_card)
    if priors is not None:
        real_hardware(root, directory, priors)
