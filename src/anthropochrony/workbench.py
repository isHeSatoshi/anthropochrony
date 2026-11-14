"""Self-contained interactive research & systems workbench for Gradio / Hugging Face Spaces.

All exploratory and live CPU simulation callbacks strictly forbid executing pasted code:
prompts are inspected solely via static AST / token feature extraction and pre-computed
benchmark artifacts. Restricted problem statements (`LeetCodeDataset`, `CodeContests`,
`TACO`) are redacted automatically in rendered UI cards.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.ensemble import HistGradientBoostingRegressor

from anthropochrony.artifacts import read_jsonl
from anthropochrony.baselines.regressors import InputRegressor, input_features
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.eval.structured import (
    WorkloadPriors,
    apply_conformal_rows,
    build_structured_rows,
    fit_split_conformal,
    fit_workload_priors,
    structured_duration,
)
from anthropochrony.families import infer_prompt_family, static_ast_stats
from anthropochrony.parse import parse
from anthropochrony.release import RESTRICTED_SOURCES
from anthropochrony.sim import sla_tier_routing, timeout_tradeoffs
from anthropochrony.timemodel import SpeedCard, trajectory_time

SPEED_CARDS: tuple[SpeedCard, ...] = (
    SpeedCard("REF", 96.0, 26000.0),
    SpeedCard("seen-edge", 60.0, 3000.0),
    SpeedCard("seen-mid", 120.0, 12000.0),
    SpeedCard("seen-dc", 400.0, 40000.0),
    SpeedCard("heldout-8", 8.0, 400.0),
    SpeedCard("heldout-190", 190.0, 19000.0),
    SpeedCard("extrap-low", 1.5, 75.0),
    SpeedCard("extrap-high", 5000.0, 150000.0),
)
CARD_BY_ID: dict[str, SpeedCard] = {c.id: c for c in SPEED_CARDS}

PALETTE: dict[str, str] = {
    "E0": "#94A3B8",
    "E1": "#D55E00",
    "E2": "#B45309",
    "B1": "#475569",
    "B2": "#56B4E9",
    "B4": "#CC79A7",
    "M2-S0": "#E69F00",
    "M1-S0": "#0072B2",
    "M1-Struct": "#0F766E",
    "M1-Struct+CP": "#0F766E",
    "M1b": "#2563EB",
    "Always-Edge": "#DC2626",
    "Always-DC": "#334155",
}

ESTIMATOR_LABELS: dict[str, str] = {
    "E0": "E0 · Human-Frame Anchor",
    "E1": "E1 · Base Verbal (3.35B)",
    "E2": "E2 · Base Calc (Verbal)",
    "B1": "B1 · Structural Tabular HGB",
    "B2": "B2 · BGE + Structural HGB",
    "B4": "B4 · DeepSeek-V4-Pro (Frontier)",
    "M2-S0": "M2-S0 · Teacher-Guessed SFT",
    "M1-S0": "M1-S0 · Measured SFT (Verbal)",
    "M1-Struct": "M1-Struct · Workload + Exact Timing",
    "M1-Struct+CP": "M1-Struct+CP · 80% Conformal SLA",
    "M1b": "M1b · Conformal Verbal CQR",
}

COHORTS: tuple[str, ...] = (
    "Showcase (PROJ-8 + Curated Demo)",
    "Real CPU Deployment (X10 · llama.cpp @ 20 tok/s)",
    "Held-Out In-Distribution (test · n=255)",
    "Held-Out Unseen Family OOD (test_ood · CLASS · n=93)",
)


def _new_figure(figsize: tuple[float, float] = (8.4, 4.2)) -> Figure:
    fig = Figure(figsize=figsize, dpi=140)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#FFFFFF")
    return fig


def _format_seconds(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    if val >= 3600:
        return f"{val / 3600:.2f} h ({val:,.0f} s)"
    if val >= 60:
        return f"{val / 60:.1f} min ({val:.1f} s)"
    if val >= 10:
        return f"{val:.2f} s"
    return f"{val:.3f} s"


@dataclass
class LiveWorkloadPredictor:
    """Zero-GPU train-only structural HGB predictor (`B1` + pass-prob head) for live prompts."""

    regressor: InputRegressor
    pass_model: HistGradientBoostingRegressor
    priors: WorkloadPriors
    conformal_multiplier: float

    @classmethod
    def fit(
        cls,
        tasks_by_id: dict[str, dict],
        traces_path: Path,
        priors: WorkloadPriors,
        conformal_multiplier: float = 2.087,
    ) -> LiveWorkloadPredictor | None:
        if not traces_path.exists() or not tasks_by_id:
            return None
        train_tasks = [t for t in tasks_by_id.values() if t.get("split") == "train"]
        if len(train_tasks) < 20:
            return None
        traces = read_jsonl(traces_path)
        reg = InputRegressor(privileged=False).fit(train_tasks, traces)

        grouped: dict[str, list[dict]] = defaultdict(list)
        for tr in traces:
            if not tr.get("harness_error"):
                grouped[tr["task_id"]].append(tr)
        x_pass, y_pass = [], []
        for task in train_tasks:
            runs = grouped.get(task["task_id"], [])
            if not runs:
                continue
            x_pass.append(input_features(task, reg.families, False))
            y_pass.append(float(np.mean([1.0 if r.get("first_pass") else 0.0 for r in runs])))
        pass_m = HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, random_state=20261109
        ).fit(x_pass, np.asarray(y_pass))
        return cls(
            regressor=reg,
            pass_model=pass_m,
            priors=priors,
            conformal_multiplier=conformal_multiplier,
        )

    def predict(
        self,
        prompt_text: str,
        *,
        decode_tok_s: float,
        prefill_tok_s: float,
        latency_ms: float,
        sla_deadline_s: float,
    ) -> dict:
        family = infer_prompt_family(prompt_text)
        stats = static_ast_stats(prompt_text)
        approx_uncached = max(24, int(round(len(prompt_text) / 3.6)))
        pseudo_task = {
            "task_turn": prompt_text,
            "family": family,
            "prompt_tokens": approx_uncached + 510,
            "uncached_task_tokens": approx_uncached,
            "public_examples": {},
        }
        q = self.regressor.quantities(pseudo_task)
        feat_vec = [input_features(pseudo_task, self.regressor.families, False)]
        pass_prob = float(np.clip(self.pass_model.predict(feat_vec)[0], 0.02, 0.98))

        o_tokens = float(q["o_tokens"])
        attempts = float(q["attempts"])
        card = SpeedCard("custom", float(decode_tok_s), float(prefill_tok_s))
        latency_s = float(latency_ms) / 1000.0

        t_hat = float(
            structured_duration(
                o_tokens,
                attempts,
                float(approx_uncached),
                family,
                card,
                self.priors,
                latency_s=latency_s,
            )
            or 0.5
        )
        lo_s = t_hat / self.conformal_multiplier
        hi_s = t_hat * self.conformal_multiplier

        edge_card = CARD_BY_ID["seen-edge"]
        dc_card = CARD_BY_ID["seen-dc"]
        edge_hi = (
            float(
                structured_duration(
                    o_tokens,
                    attempts,
                    float(approx_uncached),
                    family,
                    edge_card,
                    self.priors,
                    latency_s=latency_s,
                )
                or t_hat
            )
            * self.conformal_multiplier
        )
        dc_hi = (
            float(
                structured_duration(
                    o_tokens,
                    attempts,
                    float(approx_uncached),
                    family,
                    dc_card,
                    self.priors,
                    latency_s=latency_s,
                )
                or t_hat
            )
            * self.conformal_multiplier
        )

        u_total = float(approx_uncached) + max(0.0, attempts - 1.0) * (
            self.priors.feedback_tokens(family)
        )
        exec_s = attempts * self.priors.exec_per_attempt(family)
        prefill_s = u_total / card.P
        decode_s = o_tokens / card.S
        overhead_s = attempts * latency_s

        if hi_s <= sla_deadline_s:
            sla_verdict = f"PASSES {sla_deadline_s:.1f} s SLA at 80% conformal bound ({hi_s:.2f} s)"
        else:
            sla_verdict = (
                f"EXCEEDS {sla_deadline_s:.1f} s SLA at 80% conformal bound ({hi_s:.2f} s)"
            )

        if edge_hi <= sla_deadline_s:
            route = (
                f"Offload to Edge GPU (60 tok/s · $0.25/h) — 80% upper bound "
                f"`{edge_hi:.2f} s <= {sla_deadline_s:.1f} s`"
            )
        else:
            route = (
                f"Escalate to Datacenter GPU (400 tok/s · $1.80/h) — Edge 80% bound "
                f"`{edge_hi:.2f} s > {sla_deadline_s:.1f} s` (DC bound `{dc_hi:.2f} s`)"
            )

        return {
            "family": family,
            "uncached_tokens": approx_uncached,
            "ast_nodes": int(stats["ast_nodes"]),
            "n_functions": int(stats["n_functions"]),
            "n_classes": int(stats["n_classes"]),
            "n_branches": int(stats["n_branches"]),
            "n_loops": int(stats["n_loops"]),
            "o_tokens": o_tokens,
            "attempts": attempts,
            "pass_prob": pass_prob,
            "u_tokens": u_total,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "exec_s": exec_s,
            "overhead_s": overhead_s,
            "point_s": t_hat,
            "lo_s": lo_s,
            "hi_s": hi_s,
            "sla_verdict": sla_verdict,
            "route": route,
        }


class WorkbenchEngine:
    """In-memory engine backing the 4-tab Anthropochrony interactive workbench."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.metrics = self._read_json("results/metrics.json")
        self.extra = self._read_json("results/extra-analyses.json")
        self.simulations = self._read_json("results/simulations.json")

        self.tasks_by_id: dict[str, dict] = {}
        for p in (self.root / "data/tasks/tasks.jsonl", self.root / "data/demo/tasks.jsonl"):
            if p.exists():
                for row in read_jsonl(p):
                    self.tasks_by_id.setdefault(row["task_id"], row)

        self.demo_tasks: list[dict] = (
            read_jsonl(self.root / "data/demo/tasks.jsonl")
            if (self.root / "data/demo/tasks.jsonl").exists()
            else []
        )

        # PROJ-8 CPU + GPU showcase (`results/proj8-showcase.json`)
        self.proj8_by_id: dict[str, dict] = {}
        proj8_raw = self._read_json("results/proj8-showcase.json")
        if proj8_raw:
            m1_ref = proj8_raw.get("m1-s0", {}).get("tasks", {})
            base_ref = proj8_raw.get("base", {}).get("tasks", {})
            for item in proj8_raw.get("cpu_deployment", {}).get("tasks", []):
                tid = item["task_id"]
                merged = dict(item)
                merged["family"] = self.tasks_by_id.get(tid, {}).get("family", "PROJ-8")
                merged["m1_ref_median_s"] = m1_ref.get(tid, {}).get("median_time_s")
                merged["m1_ref_tokens"] = m1_ref.get(tid, {}).get("median_output_tokens")
                merged["base_ref_median_s"] = base_ref.get(tid, {}).get("median_time_s")
                self.proj8_by_id[tid] = merged

        # Real Desktop CPU Deployment (`X10`, `n=100` tasks, `data/x10/cpu-deployment.jsonl`)
        self.x10_by_id: dict[str, dict] = {}
        x10_path = self.root / "data/x10/cpu-deployment.jsonl"
        if x10_path.exists():
            for item in read_jsonl(x10_path):
                tid = item["task_id"]
                enriched = dict(item)
                enriched["split"] = self.tasks_by_id.get(tid, {}).get("split", "test")
                enriched["base_parsed"] = parse(item.get("base_raw", ""))
                enriched["m1_parsed"] = parse(item.get("tuned_raw", ""))
                enriched["human_parsed"] = parse(item.get("human_raw", ""))
                self.x10_by_id[tid] = enriched

        try:
            self.priors = fit_workload_priors(self.root)
        except FileNotFoundError:
            self.priors = WorkloadPriors(
                feedback_by_family={"FUNC": 120.0, "LIB": 160.0, "ALGO_IO": 110.0},
                feedback_default=125.0,
                exec_per_attempt_by_family={"FUNC": 0.119, "LIB": 0.268, "ALGO_IO": 0.068},
                exec_per_attempt_default=0.094,
            )

        conformal_mult = float(
            self.extra.get("conformal_calibration", {})
            .get("calibration", {})
            .get("struct_multiplier", 2.087)
        )

        self.estimates_index: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        self.human_anchor_index: dict[str, float] = {}
        self.traces_by_run_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._load_traces_and_estimates(conformal_mult)

        self.live_predictor = LiveWorkloadPredictor.fit(
            self.tasks_by_id,
            self.root / "data/measure/base/traces.jsonl",
            self.priors,
            conformal_multiplier=conformal_mult,
        )

    def _read_json(self, rel: str) -> dict:
        path = self.root / rel
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _load_traces_and_estimates(self, conformal_mult: float) -> None:
        for run in ("base", "m1-s0", "m2-s0"):
            tp = self.root / f"data/measure/{run}/traces.jsonl"
            if tp.exists():
                for tr in read_jsonl(tp):
                    self.traces_by_run_task[(run, tr["task_id"])].append(tr)

        est_path = self.root / "data/eval/estimates.jsonl"
        if not est_path.exists():
            return
        base_rows = normalized_rows(read_jsonl(est_path))
        m1_struct = build_structured_rows(
            base_rows,
            self.tasks_by_id,
            self.priors,
            source_estimator="M1-S0",
            estimator_name="M1-Struct",
        )
        val_traces = defaultdict(list)
        for (run, tid), trs in self.traces_by_run_task.items():
            if run == "base" and self.tasks_by_id.get(tid, {}).get("split") == "val":
                val_traces[tid].extend(trs)
        if any(r.get("split") == "val" for r in m1_struct):
            cal = fit_split_conformal(
                base_rows,
                m1_struct,
                val_traces_by_task=val_traces or None,
                card=CARD_BY_ID["REF"],
                alpha=0.20,
            )
            m1b, struct_cp = apply_conformal_rows(base_rows, m1_struct, cal)
        else:
            m1b, struct_cp = [], []
            for r in m1_struct:
                cp_row = dict(r)
                cp_row["estimator"] = "M1-Struct+CP"
                if cp_row.get("point_s"):
                    cp_row["lo_s"] = cp_row["point_s"] / conformal_mult
                    cp_row["hi_s"] = cp_row["point_s"] * conformal_mult
                struct_cp.append(cp_row)

        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        human_grouped: dict[str, list[float]] = defaultdict(list)
        for row in base_rows + m1_struct + m1b + struct_cp:
            if row.get("frame") == "human" and row.get("estimator") == "E0":
                if row.get("point_s") and row["point_s"] > 0:
                    human_grouped[row["task_id"]].append(math.log10(row["point_s"]))
                continue
            if row.get("frame", "self") != "self":
                continue
            grouped[(row["task_id"], row["card_id"], row["estimator"])].append(row)

        for tid, vals in human_grouped.items():
            self.human_anchor_index[tid] = float(10 ** np.mean(vals))

        for (tid, cid, est), items in grouped.items():
            pts = [math.log10(r["point_s"]) for r in items if r.get("point_s") and r["point_s"] > 0]
            los = [math.log10(r["lo_s"]) for r in items if r.get("lo_s") and r["lo_s"] > 0]
            his = [math.log10(r["hi_s"]) for r in items if r.get("hi_s") and r["hi_s"] > 0]
            toks = [float(r["tokens"]) for r in items if r.get("tokens") and r["tokens"] > 0]
            atts = [float(r["attempts"]) for r in items if r.get("attempts") and r["attempts"] > 0]
            probs = [
                float(r["pass_prob"])
                for r in items
                if r.get("pass_prob") is not None and math.isfinite(float(r["pass_prob"]))
            ]
            actuals = [float(r["T_med_s"]) for r in items if r.get("T_med_s") and r["T_med_s"] > 0]
            self.estimates_index[(tid, cid)][est] = {
                "point_s": float(10 ** np.mean(pts)) if pts else None,
                "lo_s": float(10 ** np.mean(los)) if los else None,
                "hi_s": float(10 ** np.mean(his)) if his else None,
                "tokens": float(np.mean(toks)) if toks else None,
                "attempts": float(np.mean(atts)) if atts else None,
                "pass_prob": float(np.mean(probs)) if probs else None,
                "actual_s": float(np.median(actuals)) if actuals else None,
                "target_source": items[0].get("target_source", "base"),
            }

    # ------------------------------------------------------------------
    # Tab 1: Benchmark, Speed-Card & Real CPU Trajectory Explorer
    # ------------------------------------------------------------------
    def cohort_choices(self, cohort: str) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        if cohort.startswith("Showcase"):
            seen = set()
            for tid, item in sorted(self.proj8_by_id.items()):
                choices.append((f"[PROJ-8 CPU · {item['wall_s']:.0f}s] {tid}", tid))
                seen.add(tid)
            for row in self.demo_tasks:
                tid = row["task_id"]
                if tid not in seen:
                    choices.append((f"[{row['family']} · {row['split']}] {tid}", tid))
                    seen.add(tid)
            return choices

        if cohort.startswith("Real CPU"):
            for tid, item in sorted(self.x10_by_id.items()):
                tag = "PASS" if item.get("passed") else "FAIL"
                choices.append(
                    (
                        f"[{item['family']} · {item['split']} · {tag} · {item['wall_s']:.1f}s] {tid}",
                        tid,
                    )
                )
            return choices

        target_split = "test_ood" if "test_ood" in cohort else "test"
        permissive, restricted = [], []
        for tid, task in sorted(self.tasks_by_id.items()):
            if task.get("split") != target_split:
                continue
            if (tid, "REF") not in self.estimates_index:
                continue
            is_restricted = task.get("source") in RESTRICTED_SOURCES or bool(
                task.get("restricted_statement")
            )
            label = f"[{task['family']} · {task['source'].split('/')[-1]}] {tid}"
            if is_restricted:
                restricted.append((f"{label} (redacted statement)", tid))
            else:
                permissive.append((label, tid))
        return permissive + restricted

    def safe_prompt_markdown(self, task_id: str) -> str:
        task = self.tasks_by_id.get(task_id)
        if not task:
            return f"Task `{task_id}` metadata unavailable."
        source = task.get("source", "unknown")
        rev = str(task.get("source_revision", ""))[:10]
        lic = task.get("license", "unknown")
        header = (
            f"**Task:** `{task_id}` &nbsp;|&nbsp; **Family:** `{task.get('family')}` "
            f"&nbsp;|&nbsp; **Split:** `{task.get('split')}` &nbsp;|&nbsp; "
            f"**Source:** `{source}` (`{rev}` · `{lic}`)"
        )
        if (
            source in RESTRICTED_SOURCES
            or task.get("restricted_statement")
            or not task.get("task_turn")
        ):
            return (
                f"{header}\n\n"
                "> **Statement redacted for redistribution license compliance.** "
                f"Source dataset `{source}` prohibits redistributing raw problem prose; "
                "all measured trajectories, speed-card scaling curves, and model self-estimates "
                "for this task remain inspectable below."
            )
        snippet = task["task_turn"].strip()
        if len(snippet) > 1800:
            snippet = snippet[:1800] + "\n... [truncated in preview]"
        return f"{header}\n\n~~~~markdown\n{snippet}\n~~~~"

    def explore_task(self, cohort: str, task_id: str, card_id: str) -> tuple[str, str, Figure]:
        if not task_id:
            opts = self.cohort_choices(cohort)
            task_id = opts[0][1] if opts else ""
        card = CARD_BY_ID.get(card_id, CARD_BY_ID["REF"])
        prompt_md = self.safe_prompt_markdown(task_id)
        est_map = self.estimates_index.get((task_id, card.id), {})

        cpu_banner = ""
        if task_id in self.proj8_by_id:
            p8 = self.proj8_by_id[task_id]
            cpu_banner = (
                f"### Real Desktop CPU Showcase (`llama.cpp` Q4_K_M @ `20 tok/s`, `P = 210 tok/s`, local `L = 0 ms`)\n"
                f"- **Measured CPU wall-clock:** `{p8['wall_s']:.2f} s` (`{p8['attempts']}` attempts; "
                f"GPU `REF` $K=8$ median: `{_format_seconds(p8.get('m1_ref_median_s'))}` across `{p8.get('m1_ref_tokens') or 0:.0f}` output tokens)\n"
                f"- **Untuned Base verbal (`E1`):** `{_format_seconds(p8['base_estimate_s'])}` "
                f"(**{p8['base_estimate_s'] / p8['wall_s']:.1f}×** human-time overestimate; human-frame `E0`: `{_format_seconds(p8['human_estimate_s'])}`)\n"
                f"- **Measured SFT verbal (`M1-S0`):** `{_format_seconds(p8['tuned_estimate_s'])}` "
                f"(**`{p8['tuned_estimate_s'] / p8['wall_s']:.2f}×` of wall-clock — within 2×!**)\n\n"
            )
        elif task_id in self.x10_by_id:
            x = self.x10_by_id[task_id]
            m1_p = x["m1_parsed"]
            b_p = x["base_parsed"]
            x_card = SpeedCard(
                "demo-llamacpp",
                float(x.get("estimate_card", {}).get("S", 20.0)),
                float(x.get("estimate_card", {}).get("P", 200.0)),
            )
            struct_x10 = structured_duration(
                m1_p.tokens,
                m1_p.attempts,
                float(self.tasks_by_id.get(task_id, {}).get("uncached_task_tokens", 160.0)),
                x["family"],
                x_card,
                self.priors,
                latency_s=0.0,
            )
            cpu_banner = (
                f"### Real Desktop CPU Deployment (`X10` · `llama.cpp` Q4_K_M @ `{x_card.S:g} tok/s`, `L = 0 ms`)\n"
                f"- **Measured single-run CPU wall-clock:** `{x['wall_s']:.2f} s` "
                f"(`{'PASSED' if x.get('passed') else 'FAILED'}`, `{x['attempts']}` attempt(s))\n"
                f"- **Base verbal (`E1`):** `{_format_seconds(b_p.point_s)}` &nbsp;|&nbsp; "
                f"**`M1-S0` verbal:** `{_format_seconds(m1_p.point_s)}` &nbsp;|&nbsp; "
                f"**`M1-Struct` (`L = 0`):** **`{_format_seconds(struct_x10)}`**\n"
                f'- **Model\'s live `M1` output:** *"{x.get("tuned_raw", "").strip()}"*\n\n'
            )

        base_trs = self.traces_by_run_task.get(("base", task_id), [])
        m1_trs = self.traces_by_run_task.get(("m1-s0", task_id), [])
        base_times = [trajectory_time(t, card) for t in base_trs]
        m1_times = [trajectory_time(t, card) for t in m1_trs]

        ref_actual = float(np.median(base_times)) if base_times else None
        if ref_actual is None and est_map:
            for v in est_map.values():
                if v.get("actual_s"):
                    ref_actual = v["actual_s"]
                    break
        own_actual = float(np.median(m1_times)) if m1_times else ref_actual

        lines = [cpu_banner]
        if est_map:
            lines.append(
                f"### Speed-Card Comparison on `{card.id}` (`S = {card.S:g} tok/s`, `P = {card.P:g} tok/s`)\n"
            )
            if base_times:
                pass_cnt = sum(1 for t in base_trs if t.get("passed"))
                lines.append(
                    f"- **Base $K={len(base_times)}$ trajectories:** median `{_format_seconds(ref_actual)}` "
                    f"(range `{min(base_times):.2f} s` – `{max(base_times):.2f} s`, pass `{pass_cnt}/{len(base_trs)}`)"
                )
            if m1_times:
                m1_pass = sum(1 for t in m1_trs if t.get("passed"))
                lines.append(
                    f"- **`M1-S0` own $K={len(m1_times)}$ trajectories:** median `{_format_seconds(own_actual)}` "
                    f"(range `{min(m1_times):.2f} s` – `{max(m1_times):.2f} s`, pass `{m1_pass}/{len(m1_trs)}`)\n"
                )
            lines.append(
                "| Estimator | Target source | Target (`T_med`) | Predicted `T_hat` | 80% / Verbal Interval | Ratio | `|log10 Err|` | Within 2× | Workload (`tok`, `att`, `p1`) |\n"
                "|---|:---:|---:|---:|---:|---:|---:|:---:|---|"
            )
            order = (
                "E1",
                "E2",
                "B4",
                "B1",
                "B2",
                "M2-S0",
                "M1-S0",
                "M1-Struct",
                "M1-Struct+CP",
            )
            if task_id in self.human_anchor_index and card.id == "REF":
                h_s = self.human_anchor_index[task_id]
                act = ref_actual or 1.0
                err = abs(math.log10(h_s / act))
                lines.append(
                    f"| {ESTIMATOR_LABELS['E0']} | `base` | {_format_seconds(act)} | {_format_seconds(h_s)} | — | "
                    f"`{h_s / act:.1f}×` | `{err:.3f}` | {'Yes' if err <= math.log10(2) else 'No'} | human reference |"
                )
            for est in order:
                rec = est_map.get(est)
                if not rec or not rec.get("point_s"):
                    continue
                tsrc = rec.get("target_source") or "base"
                act = (
                    own_actual
                    if tsrc == "own" and own_actual
                    else (ref_actual or rec.get("actual_s"))
                )
                pt = rec["point_s"]
                interval_str = (
                    f"[{rec['lo_s']:.2f}s, {rec['hi_s']:.2f}s]"
                    if rec.get("lo_s") and rec.get("hi_s")
                    else "—"
                )
                if act and act > 0:
                    ratio_s = f"`{pt / act:.2f}×`"
                    err_val = abs(math.log10(pt / act))
                    err_s = f"`{err_val:.3f}`"
                    w2 = "**Yes**" if err_val <= math.log10(2.0) else "No"
                else:
                    ratio_s, err_s, w2 = "—", "—", "—"
                wk_parts = []
                if rec.get("tokens"):
                    wk_parts.append(f"{rec['tokens']:.0f} tok")
                if rec.get("attempts"):
                    wk_parts.append(f"{rec['attempts']:.1f} att")
                if rec.get("pass_prob") is not None:
                    wk_parts.append(f"p={rec['pass_prob']:.2f}")
                wk_str = ", ".join(wk_parts) if wk_parts else "point-only"
                lines.append(
                    f"| **{ESTIMATOR_LABELS.get(est, est)}** | `{tsrc}` | {_format_seconds(act)} | "
                    f"**{_format_seconds(pt)}** | {interval_str} | {ratio_s} | {err_s} | {w2} | {wk_str} |"
                )
        elif not cpu_banner:
            lines.append("No pre-computed evaluation rows found for this task/card pair.")

        fig = self._plot_task_comparison(
            task_id, card, est_map, base_times, m1_times, ref_actual, own_actual
        )
        return prompt_md, "\n".join(lines), fig

    def _plot_task_comparison(
        self,
        task_id: str,
        card: SpeedCard,
        est_map: dict[str, dict],
        base_times: list[float],
        m1_times: list[float],
        ref_actual: float | None,
        own_actual: float | None,
    ) -> Figure:
        fig = _new_figure((8.6, 4.3))
        ax = fig.add_subplot(111)

        if not est_map and task_id in self.proj8_by_id:
            p8 = self.proj8_by_id[task_id]
            labels = [
                "Measured CPU Wall-Clock",
                "M1-S0 Tuned Self-Estimate",
                "Base E1 Self-Estimate",
                "Human-Frame E0 Anchor",
            ]
            vals = [
                float(p8["wall_s"]),
                float(p8["tuned_estimate_s"]),
                float(p8["base_estimate_s"]),
                float(p8["human_estimate_s"]),
            ]
            colors = ["#0F172A", PALETTE["M1-S0"], PALETTE["E1"], PALETTE["E0"]]
            y = np.arange(len(labels))
            ax.barh(y, vals, color=colors, height=0.52, alpha=0.9)
            ax.set_xscale("log")
            ax.axvspan(
                vals[0] / 2.0,
                vals[0] * 2.0,
                color="#10B981",
                alpha=0.14,
                label="Within 2× of wall-clock",
            )
            ax.axvline(vals[0], color="#0F172A", linestyle="--", linewidth=1.4)
            for yi, v in zip(y, vals, strict=True):
                ax.text(
                    v * 1.14,
                    yi,
                    _format_seconds(v),
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                )
            ax.set_yticks(y, labels, fontsize=9)
            ax.set_xlabel("Duration (seconds, log scale)", fontsize=9.5)
            ax.set_title(
                f"{task_id} · Real Desktop CPU Deployment (llama.cpp Q4_K_M @ 20 tok/s)",
                fontsize=10.2,
                fontweight="bold",
            )
            ax.set_xlim(min(vals) * 0.4, max(vals) * 5.2)
            ax.grid(axis="x", which="both", alpha=0.22)
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            return fig

        rows_to_plot: list[tuple[str, float, float | None, float | None, str]] = []
        for est in ("E1", "E2", "B4", "B2", "M2-S0", "M1-S0", "M1-Struct", "M1-Struct+CP"):
            rec = est_map.get(est)
            if rec and rec.get("point_s"):
                rows_to_plot.append(
                    (
                        ESTIMATOR_LABELS.get(est, est),
                        float(rec["point_s"]),
                        rec.get("lo_s"),
                        rec.get("hi_s"),
                        PALETTE.get(est, "#334155"),
                    )
                )

        if not rows_to_plot:
            ax.text(
                0.5,
                0.5,
                "Select a benchmark task to visualize estimates.",
                ha="center",
                va="center",
            )
            ax.axis("off")
            return fig

        target = own_actual or ref_actual
        if target and target > 0:
            ax.axvspan(
                target / 2.0,
                target * 2.0,
                color="#10B981",
                alpha=0.13,
                label="Within 2× of T_med",
            )
            ax.axvline(
                target,
                color="#0F172A",
                linestyle="--",
                linewidth=1.6,
                label=f"Measured T_med ({target:.2f}s)",
            )

        trajs = m1_times or base_times
        if trajs:
            ax.scatter(
                trajs,
                np.full(len(trajs), -0.85),
                color="#0F172A",
                alpha=0.65,
                s=34,
                zorder=4,
                label=f"Measured K={len(trajs)} trajectories",
            )

        for idx, (_label, pt, lo, hi, color) in enumerate(rows_to_plot):
            if lo and hi and hi > lo:
                ax.plot(
                    [lo, hi],
                    [idx, idx],
                    color=color,
                    linewidth=3.2,
                    alpha=0.45,
                    solid_capstyle="round",
                )
                ax.scatter([lo, hi], [idx, idx], color=color, marker="|", s=75, linewidths=2.0)
            ax.scatter([pt], [idx], color=color, s=68, zorder=5)
            ax.text(pt * 1.12, idx, _format_seconds(pt), va="center", fontsize=8, color="#1E293B")

        y_ticks = ([-0.85] if trajs else []) + list(range(len(rows_to_plot)))
        y_labels = (["Measured K=8 runs"] if trajs else []) + [r[0] for r in rows_to_plot]
        ax.set_yticks(y_ticks, y_labels, fontsize=8.5)
        ax.set_xscale("log")
        all_x = [r[1] for r in rows_to_plot] + [r[3] for r in rows_to_plot if r[3]] + list(trajs)
        if target:
            all_x.append(target)
        ax.set_xlim(max(1e-3, min(all_x) * 0.45), max(all_x) * 3.8)
        ax.set_xlabel(f"Completion Time on `{card.id}` (seconds, log scale)", fontsize=9.5)
        ax.set_title(
            f"{task_id} · `{card.id}` ({card.S:g} tok/s decode, {card.P:g} tok/s prefill)",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(axis="x", which="both", alpha=0.22)
        ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Tab 2: Zero-GPU Live Prompt Workload & Hardware Simulator
    # ------------------------------------------------------------------
    def starter_prompts(self) -> list[tuple[str, str]]:
        return [(f"[{row['family']}] {row['task_id']}", row["task_id"]) for row in self.demo_tasks]

    def starter_prompt_text(self, task_id: str) -> str:
        task = self.tasks_by_id.get(task_id)
        if task and task.get("source") not in RESTRICTED_SOURCES and task.get("task_turn"):
            return task["task_turn"]
        if self.demo_tasks:
            return self.demo_tasks[0]["task_turn"]
        return (
            "Task:\n```python\ndef merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
            '    """Merge overlapping closed intervals and return them sorted."""\n```\n'
        )

    def simulate_prompt(
        self,
        prompt_text: str,
        decode_tok_s: float,
        prefill_tok_s: float,
        latency_ms: float,
        sla_deadline_s: float,
    ) -> tuple[str, Figure]:
        text = (prompt_text or "").strip()
        if not text:
            fig = _new_figure((8.0, 3.6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Paste or select a Python coding prompt.", ha="center", va="center")
            ax.axis("off")
            return "Please provide a non-empty Python task specification.", fig

        if self.live_predictor is None:
            fig = _new_figure((8.0, 3.6))
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Train trace cache unavailable in minimal checkout.",
                ha="center",
                va="center",
            )
            ax.axis("off")
            return "Live structural predictor requires `data/measure/base/traces.jsonl`.", fig

        pred = self.live_predictor.predict(
            text,
            decode_tok_s=float(decode_tok_s),
            prefill_tok_s=float(prefill_tok_s),
            latency_ms=float(latency_ms),
            sla_deadline_s=float(sla_deadline_s),
        )

        md = (
            "### Instant CPU Workload Decomposition with an Exact Timing Model *(Static AST only — zero code execution)*\n"
            f"- **Inferred Workload Family:** `{pred['family']}` &nbsp;|&nbsp; "
            f"**Prompt Complexity:** `~{pred['uncached_tokens']} tok`, `{pred['ast_nodes']} AST nodes`, "
            f"`{pred['n_functions']} def`, `{pred['n_classes']} class`, `{pred['n_branches']} branch`, `{pred['n_loops']} loop`\n"
            f"- **Predicted Agent Workload:** `{pred['o_tokens']:.0f}` generated tokens across "
            f"`{pred['attempts']:.2f}` expected attempts &nbsp;|&nbsp; "
            f"**Calibrated 1st-Attempt Pass Prob:** `{pred['pass_prob'] * 100:.1f}%`\n\n"
            "| Physics Component | Formula Term | Predicted Contribution |\n"
            "|---|---|---:|\n"
            f"| **Decode Generation** | `{pred['o_tokens']:.0f} tok / {decode_tok_s:g} tok/s` | `{_format_seconds(pred['decode_s'])}` |\n"
            f"| **Uncached Prefill + Repair Feedback** | `{pred['u_tokens']:.0f} tok / {prefill_tok_s:g} tok/s` | `{_format_seconds(pred['prefill_s'])}` |\n"
            f"| **Sandboxed Test Execution (`{pred['family']}` prior)** | `{pred['attempts']:.2f} × {self.priors.exec_per_attempt(pred['family']):.3f} s` | `{_format_seconds(pred['exec_s'])}` |\n"
            f"| **Per-Call Serving Latency (`L`)** | `{pred['attempts']:.2f} × {latency_ms:g} ms` | `{_format_seconds(pred['overhead_s'])}` |\n"
            f"| **Workload + Exact-Timing Point Estimate (`T_Struct`)** | Exact serving timing model | **`{_format_seconds(pred['point_s'])}`** |\n"
            f"| **80% Split-Conformal SLA Interval** | `val`-calibrated `× / ÷ {self.live_predictor.conformal_multiplier:.3f}` | **`[{_format_seconds(pred['lo_s'])}, {_format_seconds(pred['hi_s'])}]`** |\n\n"
            f"- **SLA Tripwire Recommendation (`k = 1.0×` upper bound):** Abort runaway trajectory at **`{pred['hi_s']:.2f} s`** — {pred['sla_verdict']}\n"
            f"- **Multi-Tier Router Decision:** {pred['route']}"
        )

        fig = _new_figure((8.4, 3.8))
        ax = fig.add_subplot(111)
        s_grid = np.geomspace(1.5, 5000.0, 60)
        pts, los, his = [], [], []
        ratio_ps = max(20.0, float(prefill_tok_s) / max(1.0, float(decode_tok_s)))
        mult = self.live_predictor.conformal_multiplier
        for s_val in s_grid:
            c = SpeedCard("grid", float(s_val), float(s_val * ratio_ps))
            t_val = float(
                structured_duration(
                    pred["o_tokens"],
                    pred["attempts"],
                    float(pred["uncached_tokens"]),
                    pred["family"],
                    c,
                    self.priors,
                    latency_s=float(latency_ms) / 1000.0,
                )
                or 0.1
            )
            pts.append(t_val)
            los.append(t_val / mult)
            his.append(t_val * mult)

        ax.fill_between(
            s_grid, los, his, color=PALETTE["M1-Struct"], alpha=0.18, label="80% Conformal SLA Band"
        )
        ax.plot(
            s_grid, pts, color=PALETTE["M1-Struct"], linewidth=2.2, label="Disentangled T_Struct(S)"
        )
        ax.axhline(
            sla_deadline_s,
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label=f"SLA Deadline ({sla_deadline_s:g}s)",
        )
        ax.scatter(
            [decode_tok_s],
            [pred["point_s"]],
            color="#0F172A",
            s=75,
            zorder=5,
            label=f"Selected Hardware ({decode_tok_s:g} tok/s -> {pred['point_s']:.2f}s)",
        )
        ax.errorbar(
            [decode_tok_s],
            [pred["point_s"]],
            yerr=[[pred["point_s"] - pred["lo_s"]], [pred["hi_s"] - pred["point_s"]]],
            fmt="none",
            ecolor="#0F172A",
            elinewidth=2.0,
            capsize=4,
            zorder=5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Decode Throughput S (tokens / second, log scale)", fontsize=9.5)
        ax.set_ylabel("Predicted Completion Time (s, log scale)", fontsize=9.5)
        ax.set_title(
            f"Hardware Scaling Curve · {pred['family']} ({pred['o_tokens']:.0f} tok, {pred['attempts']:.2f} attempts)",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        fig.tight_layout()
        return md, fig

    # ------------------------------------------------------------------
    # Tab 3: Fleet SLA Tripwire, Tier-Routing & Closed-API ROI Calculator
    # ------------------------------------------------------------------
    def compute_fleet_roi(
        self,
        monthly_tasks_k: float,
        sla_deadline_s: float,
        tripwire_k: float,
        edge_gpu_usd_hr: float,
        dc_gpu_usd_hr: float,
        closed_api_usd_per_1k: float,
    ) -> tuple[str, Figure]:
        monthly_tasks = float(monthly_tasks_k) * 1000.0
        edge_card = CARD_BY_ID["seen-edge"]
        dc_card = CARD_BY_ID["seen-dc"]
        ref_card = CARD_BY_ID["REF"]

        test_ids = sorted(
            tid
            for tid, task in self.tasks_by_id.items()
            if task.get("split") == "test" and task.get("family") != "CLASS"
        )

        tripwire_methods = ("E1", "E2", "B4", "M1-S0", "M1b", "M1-Struct+CP")
        tripwire_rows = []
        for est in tripwire_methods:
            run_key = "m1-s0" if est.startswith("M1") else "base"
            act, up, ok = [], [], []
            for tid in test_ids:
                rec = self.estimates_index.get((tid, "REF"), {}).get(est)
                if not rec or not rec.get("point_s"):
                    continue
                upper_s = max(float(rec.get("hi_s") or rec["point_s"]), float(rec["point_s"]))
                for tr in self.traces_by_run_task.get((run_key, tid), ()):
                    act.append(trajectory_time(tr, ref_card))
                    up.append(upper_s)
                    ok.append(bool(tr.get("passed")))
            if not act:
                continue
            stats = timeout_tradeoffs(act, up, ok, multipliers=(float(tripwire_k),))[0]
            saved_s_per_run = float(stats["failed_time_saved_s"]) / len(act)
            monthly_gpu_hrs_saved = (saved_s_per_run * monthly_tasks) / 3600.0
            monthly_saved_usd = monthly_gpu_hrs_saved * float(dc_gpu_usd_hr)
            tripwire_rows.append(
                {
                    "est": est,
                    "false_stop_pct": float(stats["false_stop_rate"]) * 100.0,
                    "saved_share_pct": float(stats["failed_time_saved_share"]) * 100.0,
                    "monthly_saved_usd": monthly_saved_usd,
                }
            )

        routing_rows = []
        m1_e_all, m1_d_all = [], []
        for tid in test_ids:
            for tr in self.traces_by_run_task.get(("m1-s0", tid), ()):
                m1_e_all.append(trajectory_time(tr, edge_card))
                m1_d_all.append(trajectory_time(tr, dc_card))

        if m1_e_all:
            for name, hi_arr in (
                ("Always-Edge", np.zeros(len(m1_e_all))),
                ("Always-DC", np.full(len(m1_e_all), 1e9)),
            ):
                st = sla_tier_routing(
                    m1_e_all,
                    m1_d_all,
                    hi_arr,
                    sla_s=float(sla_deadline_s),
                    edge_usd_per_hour=float(edge_gpu_usd_hr),
                    dc_usd_per_hour=float(dc_gpu_usd_hr),
                )
                compute_mo = float(st["passing_cost_usd_per_1k_runs"]) * float(monthly_tasks_k)
                routing_rows.append(
                    {
                        "est": name,
                        "viol_pct": float(st["sla_violation_rate"]) * 100.0,
                        "edge_pct": float(st["edge_routed_share"]) * 100.0,
                        "compute_mo": compute_mo,
                        "est_fee_mo": 0.0,
                        "total_mo": compute_mo,
                    }
                )

        self_hosted_est_per_1k = (0.28 / 3600.0) * float(dc_gpu_usd_hr) * 1000.0

        for est in ("E1", "B4", "M1-S0", "M1b", "M1-Struct+CP"):
            run_key = "m1-s0" if est.startswith("M1") else "base"
            e_runs, d_runs, hi_runs = [], [], []
            for tid in test_ids:
                rec = self.estimates_index.get((tid, "seen-edge"), {}).get(est)
                if not rec or not rec.get("point_s"):
                    continue
                upper_s = max(float(rec.get("hi_s") or rec["point_s"]), float(rec["point_s"]))
                for tr in self.traces_by_run_task.get((run_key, tid), ()):
                    e_runs.append(trajectory_time(tr, edge_card))
                    d_runs.append(trajectory_time(tr, dc_card))
                    hi_runs.append(upper_s)
            if not e_runs:
                continue
            st = sla_tier_routing(
                e_runs,
                d_runs,
                hi_runs,
                sla_s=float(sla_deadline_s),
                edge_usd_per_hour=float(edge_gpu_usd_hr),
                dc_usd_per_hour=float(dc_gpu_usd_hr),
            )
            compute_mo = float(st["passing_cost_usd_per_1k_runs"]) * float(monthly_tasks_k)
            fee_per_1k = float(closed_api_usd_per_1k) if est == "B4" else self_hosted_est_per_1k
            fee_mo = fee_per_1k * float(monthly_tasks_k)
            routing_rows.append(
                {
                    "est": est,
                    "viol_pct": float(st["sla_violation_rate"]) * 100.0,
                    "edge_pct": float(st["edge_routed_share"]) * 100.0,
                    "compute_mo": compute_mo,
                    "est_fee_mo": fee_mo,
                    "total_mo": compute_mo + fee_mo,
                }
            )

        dc_baseline = next((r["total_mo"] for r in routing_rows if r["est"] == "Always-DC"), 0.0)
        b4_total = next((r["total_mo"] for r in routing_rows if r["est"] == "B4"), 0.0)
        ours = next((r for r in routing_rows if r["est"] == "M1-Struct+CP"), None)

        md_lines = [
            f"### Empirical Fleet Simulation on `2,040` Held-Out `test` Trajectories (`{monthly_tasks:,.0f}` tasks / month)",
        ]
        if ours:
            vs_dc = dc_baseline - ours["total_mo"]
            vs_b4 = b4_total - ours["total_mo"]
            md_lines.append(
                f"- **Headline TCO (`M1-Struct+CP` self-hosted 3.35B):** **`${ours['total_mo']:,.2f} / mo`** "
                f"(`{ours['viol_pct']:.1f}%` SLA violations, `{ours['edge_pct']:.1f}%` safely offloaded to Edge) — "
                f"saves **`${vs_dc:,.2f} / mo` (`${vs_dc * 12:,.0f} / yr`)** vs `Always-DC` and "
                f"**`${vs_b4:,.2f} / mo` (`${vs_b4 * 12:,.0f} / yr`)** vs closed frontier API (`B4`)!\n"
            )

        md_lines.append(
            f"#### A. SLA Timeout Tripwire Trade-Off (`abort if t > {tripwire_k:.2f} × upper_bound`)\n"
            "| Policy | Passing Jobs Falsely Aborted | Wasted Failing GPU Time Saved | Monthly Failing Compute Saved |\n"
            "|---|---:|---:|---:|"
        )
        for r in tripwire_rows:
            md_lines.append(
                f"| **{ESTIMATOR_LABELS.get(r['est'], r['est'])}** | `{r['false_stop_pct']:.2f}%` | "
                f"`{r['saved_share_pct']:.1f}%` | `${r['monthly_saved_usd']:,.2f} / mo` |"
            )

        md_lines.append(
            f"\n#### B. Multi-Tier Edge (`${edge_gpu_usd_hr:.2f}/h`) vs Datacenter (`${dc_gpu_usd_hr:.2f}/h`) Routing + Estimator API ROI (`SLA = {sla_deadline_s:.1f} s`)\n"
            "| Routing + Estimator Stack | SLA Violation Rate | Edge Offload Share | Agent GPU Compute | Estimator Inference Cost | Total Fleet TCO |\n"
            "|---|---:|---:|---:|---:|---:|"
        )
        for r in routing_rows:
            md_lines.append(
                f"| **{ESTIMATOR_LABELS.get(r['est'], r['est'])}** | `{r['viol_pct']:.1f}%` | "
                f"`{r['edge_pct']:.1f}%` | `${r['compute_mo']:,.2f}/mo` | "
                f"`${r['est_fee_mo']:,.2f}/mo` | **`${r['total_mo']:,.2f}/mo`** |"
            )

        fig = _new_figure((9.2, 4.0))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for r in tripwire_rows:
            c = PALETTE.get(r["est"], "#334155")
            ax1.scatter(r["false_stop_pct"], r["saved_share_pct"], color=c, s=78, zorder=4)
            ax1.annotate(
                r["est"],
                (r["false_stop_pct"], r["saved_share_pct"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold" if r["est"] == "M1-Struct+CP" else "normal",
            )
        ax1.set_xlabel("Passing Runs Falsely Aborted (%)", fontsize=9)
        ax1.set_ylabel("Failing GPU Compute Saved (%)", fontsize=9)
        ax1.set_title(f"Timeout Tripwire @ k = {tripwire_k:.2f}×", fontsize=9.8, fontweight="bold")
        ax1.grid(True, alpha=0.24)

        for r in routing_rows:
            c = PALETTE.get(r["est"], "#334155")
            ax2.scatter(r["viol_pct"], r["total_mo"], color=c, s=78, zorder=4)
            ax2.annotate(
                r["est"],
                (r["viol_pct"], r["total_mo"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold" if r["est"] == "M1-Struct+CP" else "normal",
            )
        ax2.set_xlabel(f"SLA Violations > {sla_deadline_s:g}s (%)", fontsize=9)
        ax2.set_ylabel("Total Monthly Fleet + API Cost (USD)", fontsize=9)
        ax2.set_title("Multi-Tier Routing + API TCO Pareto", fontsize=9.8, fontweight="bold")
        ax2.grid(True, alpha=0.24)

        fig.tight_layout()
        return "\n".join(md_lines), fig
