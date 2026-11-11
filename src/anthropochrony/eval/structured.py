"""Post-hoc workload decomposition with an exact timing model, split-conformal calibration, and audits.

All procedures in this module are strictly post-hoc inference or diagnostic analyses:
- Zero model weights or pre-registered confirmatory verdicts (`H1`--`H6`) are altered.
- Workload execution/feedback priors use `train`-only statistics.
- Conformal quantiles and hybrid ensemble weights are fit strictly on `val`.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.eval.errors import log_error
from anthropochrony.eval.metrics import (
    arithmetic_consistency,
    interval_metrics,
    pairwise_accuracy,
    speed_elasticity,
    success_metrics,
    summary,
    task_records,
)
from anthropochrony.eval.reporting import normalized_rows
from anthropochrony.eval.stats import json_safe, stratified_bootstrap
from anthropochrony.eval.targets import own_trajectory_targets
from anthropochrony.parse import parse
from anthropochrony.timemodel import PER_CALL_LATENCY_S, SpeedCard, completion_time, trajectory_time


def _mdae(records):
    return float(np.median([r["abs_error"] for r in records]))


def _metric_line(rows, resamples=2000):
    records = task_records(rows)
    stats = summary(records)
    return {
        "target_source": (rows[0].get("target_source") or "base") if rows else "base",
        "n_tasks": stats["n_tasks"],
        "mdae_log10": stats["mdae_log10"],
        "ci95": stratified_bootstrap(records, _mdae, resamples=resamples) if resamples else None,
        "within2x": stats["within2x"],
        "within10x": stats["within10x"],
        "gm_ratio": stats["gm_ratio"],
        "signed_median_bias": stats["signed_median_bias"],
        "spearman": stats["spearman"],
        "pearson_log": stats["pearson_log"],
        "aur": stats["aur"],
        "parse_rate": stats["parse_rate"],
    }


@dataclass(frozen=True)
class WorkloadPriors:
    """Strictly train-split-derived feedback token and per-attempt sandbox wall-time medians."""

    feedback_by_family: dict[str, float]
    feedback_default: float
    exec_per_attempt_by_family: dict[str, float]
    exec_per_attempt_default: float

    def feedback_tokens(self, family: str) -> float:
        return float(self.feedback_by_family.get(family, self.feedback_default))

    def exec_per_attempt(self, family: str) -> float:
        return float(self.exec_per_attempt_by_family.get(family, self.exec_per_attempt_default))

    def as_dict(self) -> dict:
        return asdict(self)


def fit_workload_priors(root: Path) -> WorkloadPriors:
    """Compute or load strictly train-only priors for workload decomposition + exact timing."""
    root = Path(root)
    fb_path = root / "data/measure/feedback-medians.json"
    cached_path = root / "results/workload-priors.json"
    traces_path = root / "data/measure/base/traces.jsonl"
    tasks_path = root / "data/tasks/tasks.jsonl"

    if fb_path.exists() and traces_path.exists() and tasks_path.exists():
        feedback = {k: float(v) for k, v in json.loads(fb_path.read_text(encoding="utf-8")).items()}
        train_fb = [v for k, v in feedback.items() if k != "CLASS"]
        fb_default = float(np.median(train_fb or list(feedback.values())))
        tasks = {t["task_id"]: t for t in read_jsonl(tasks_path)}
        by_family: dict[str, list[float]] = defaultdict(list)
        for trace in read_jsonl(traces_path):
            meta = tasks.get(trace["task_id"])
            if not meta or meta.get("split") != "train":
                continue
            wall = [float(a["exec"]["wall_s"]) for a in trace.get("attempts", ())]
            if wall:
                by_family[meta["family"]].append(float(np.median(wall)))
        exec_fam = {fam: float(np.median(vals)) for fam, vals in sorted(by_family.items())}
        exec_default = float(np.median(list(exec_fam.values()))) if exec_fam else 0.1
        return WorkloadPriors(
            feedback_by_family=feedback,
            feedback_default=fb_default,
            exec_per_attempt_by_family=exec_fam,
            exec_per_attempt_default=exec_default,
        )
    if cached_path.exists():
        raw = json.loads(cached_path.read_text(encoding="utf-8"))
        return WorkloadPriors(
            feedback_by_family={k: float(v) for k, v in raw["feedback_by_family"].items()},
            feedback_default=float(raw["feedback_default"]),
            exec_per_attempt_by_family={
                k: float(v) for k, v in raw["exec_per_attempt_by_family"].items()
            },
            exec_per_attempt_default=float(raw["exec_per_attempt_default"]),
        )
    raise FileNotFoundError("Neither raw train traces nor cached workload-priors.json found")


def structured_duration(
    tokens: float | None,
    attempts: float | None,
    uncached_task_tokens: float,
    family: str,
    card: SpeedCard,
    priors: WorkloadPriors,
    *,
    latency_s: float | None = None,
) -> float | None:
    """Combine predicted hardware-invariant workload `(tokens, attempts)` with exact physics."""
    if tokens is None or not math.isfinite(tokens) or tokens <= 0:
        return None
    calls = max(1.0, float(attempts if attempts and math.isfinite(attempts) else 1.0))
    u_total = float(uncached_task_tokens) + max(0.0, calls - 1.0) * priors.feedback_tokens(family)
    exec_s = calls * priors.exec_per_attempt(family)
    return completion_time(
        u_total,
        float(tokens),
        exec_s,
        card,
        latency_s=latency_s,
        calls=calls,
    )


def build_structured_rows(
    rows: list[dict],
    tasks_by_id: dict[str, dict],
    priors: WorkloadPriors,
    *,
    source_estimator: str = "M1-S0",
    estimator_name: str = "M1-Struct",
    latency_s: float | None = None,
) -> list[dict]:
    """Create workload-decomposition + exact-timing `*-Struct` rows from an estimator breakdown."""
    out = []
    for row in rows:
        if row.get("estimator") != source_estimator or row.get("frame", "self") != "self":
            continue
        task = tasks_by_id.get(row["task_id"])
        if not task:
            continue
        card = SpeedCard(row["card_id"], float(row["S"]), float(row["P"]))
        pred = structured_duration(
            row.get("tokens"),
            row.get("attempts"),
            float(task.get("uncached_task_tokens", 120.0)),
            row["family"],
            card,
            priors,
            latency_s=latency_s,
        )
        point_s = pred if pred is not None else row.get("point_s")
        updated = dict(row)
        updated["estimator"] = estimator_name
        updated["point_s"] = point_s
        updated["verbal_point_s"] = row.get("point_s")
        updated["has_human_units"] = False
        if "parsed" in updated and isinstance(updated["parsed"], dict):
            updated["parsed"] = dict(updated["parsed"], point_s=point_s, has_human_units=False)
        out.append(updated)
    return out


@dataclass(frozen=True)
class ConformalCalibration:
    """Validation-calibrated finite-sample conformal multipliers in log10 duration space."""

    alpha: float
    n_val_tasks: int
    finite_sample_level: float
    q_cqr_log10: float
    q_struct_log10: float
    q_cqr_median_log10: float
    q_struct_median_log10: float

    @property
    def cqr_factor(self) -> float:
        return float(10.0**self.q_cqr_log10)

    @property
    def struct_factor(self) -> float:
        return float(10.0**self.q_struct_log10)

    def as_dict(self) -> dict:
        return asdict(self) | {
            "cqr_factor": self.cqr_factor,
            "struct_factor": self.struct_factor,
        }


def _cluster_quantile(scores_by_task: list[list[float]], level: float) -> float:
    flat = sorted({float(s) for group in scores_by_task for s in group})
    if not flat:
        raise ValueError("Empty validation nonconformity scores")
    # Pad uneven run counts with per-task empirical CDF means so every validation task has equal weight
    for candidate in flat:
        task_cov = [float(np.mean([s <= candidate for s in group])) for group in scores_by_task]
        if float(np.mean(task_cov)) >= level:
            return candidate
    return flat[-1]


def fit_split_conformal(
    val_m1_rows: list[dict],
    val_struct_rows: list[dict],
    *,
    val_traces_by_task: dict[str, list[dict]] | None = None,
    card: SpeedCard | None = None,
    alpha: float = 0.20,
) -> ConformalCalibration:
    """Fit CQR (`M1b`) and residual split-conformal (`M1-Struct+CP`) strictly on `val`."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    m1_by_task: dict[str, list[dict]] = defaultdict(list)
    for row in val_m1_rows:
        if (
            row.get("estimator") == "M1-S0"
            and row.get("split") == "val"
            and row.get("card_id") == "REF"
            and row.get("frame", "self") == "self"
        ):
            m1_by_task[row["task_id"]].append(row)
    st_by_task: dict[str, list[dict]] = defaultdict(list)
    for row in val_struct_rows:
        if (
            row.get("estimator") == "M1-Struct"
            and row.get("split") == "val"
            and row.get("card_id") == "REF"
            and row.get("frame", "self") == "self"
        ):
            st_by_task[row["task_id"]].append(row)

    task_ids = sorted(set(m1_by_task) & set(st_by_task))
    if not task_ids:
        raise ValueError("No paired validation tasks found for conformal calibration")
    n_val = len(task_ids)
    level = min(1.0, math.ceil((n_val + 1) * (1.0 - alpha)) / n_val)
    ref_card = card or SpeedCard("REF", 96.0, 26000.0)

    cqr_traj_scores: list[list[float]] = []
    res_traj_scores: list[list[float]] = []
    cqr_med_scores: list[list[float]] = []
    res_med_scores: list[list[float]] = []

    for tid in task_ids:
        m_rows = m1_by_task[tid]
        s_rows = st_by_task[tid]
        lo = float(10 ** np.mean([math.log10(r["lo_s"]) for r in m_rows if r.get("lo_s")]))
        hi = float(10 ** np.mean([math.log10(r["hi_s"]) for r in m_rows if r.get("hi_s")]))
        st_pred = float(
            10 ** np.mean([math.log10(r["point_s"]) for r in s_rows if r.get("point_s")])
        )
        y_med = float(m_rows[0]["actual_s"])
        cqr_med = max(math.log10(lo) - math.log10(y_med), math.log10(y_med) - math.log10(hi))
        res_med = abs(math.log10(y_med) - math.log10(st_pred))
        cqr_med_scores.append([cqr_med])
        res_med_scores.append([res_med])

        if val_traces_by_task and tid in val_traces_by_task:
            runs = [trajectory_time(tr, ref_card) for tr in val_traces_by_task[tid]]
        else:
            runs = [y_med]
        cqr_traj_scores.append(
            [max(math.log10(lo) - math.log10(y), math.log10(y) - math.log10(hi)) for y in runs]
        )
        res_traj_scores.append([abs(math.log10(y) - math.log10(st_pred)) for y in runs])

    return ConformalCalibration(
        alpha=alpha,
        n_val_tasks=n_val,
        finite_sample_level=level,
        q_cqr_log10=_cluster_quantile(cqr_traj_scores, level),
        q_struct_log10=_cluster_quantile(res_traj_scores, level),
        q_cqr_median_log10=_cluster_quantile(cqr_med_scores, level),
        q_struct_median_log10=_cluster_quantile(res_med_scores, level),
    )


def apply_conformal_rows(
    m1_rows: list[dict],
    struct_rows: list[dict],
    cal: ConformalCalibration,
) -> tuple[list[dict], list[dict]]:
    """Produce `M1b` (CQR on `M1-S0`) and `M1-Struct+CP` rows with calibrated `[lo_s, hi_s]`."""
    m1b_rows = []
    c_factor = cal.cqr_factor
    for row in m1_rows:
        if row.get("estimator") != "M1-S0" or row.get("frame", "self") != "self":
            continue
        lo = row.get("lo_s") or row.get("point_s")
        hi = row.get("hi_s") or row.get("point_s")
        updated = dict(row, estimator="M1b")
        if lo and hi and lo > 0 and hi > 0:
            new_lo, new_hi = float(lo) / c_factor, float(hi) * c_factor
            updated["lo_s"] = new_lo
            updated["hi_s"] = new_hi
            if isinstance(updated.get("parsed"), dict):
                updated["parsed"] = dict(updated["parsed"], lo_s=new_lo, hi_s=new_hi)
        m1b_rows.append(updated)

    struct_cp_rows = []
    s_factor = cal.struct_factor
    for row in struct_rows:
        pt = row.get("point_s")
        updated = dict(row, estimator="M1-Struct+CP")
        if pt and pt > 0:
            new_lo, new_hi = float(pt) / s_factor, float(pt) * s_factor
            updated["lo_s"] = new_lo
            updated["hi_s"] = new_hi
            if isinstance(updated.get("parsed"), dict):
                updated["parsed"] = dict(updated["parsed"], lo_s=new_lo, hi_s=new_hi)
        struct_cp_rows.append(updated)
    return m1b_rows, struct_cp_rows


def hybrid_ensemble_analysis(
    struct_rows: list[dict],
    b2_rows: list[dict],
    *,
    resamples: int = 2000,
) -> dict:
    """Tune geometric blend weight `w` on `val` (`REF`) between `M1-Struct` and tabular `B2`."""
    st_by_split: dict[str, dict[str, dict]] = {}
    b2_by_split: dict[str, dict[str, dict]] = {}
    for split in ("val", "test", "test_ood"):
        st_sub = [r for r in struct_rows if r["split"] == split and r["card_id"] == "REF"]
        b2_sub = [r for r in b2_rows if r["split"] == split and r["card_id"] == "REF"]
        if st_sub and b2_sub:
            st_by_split[split] = {r["task_id"]: r for r in task_records(st_sub)}
            b2_by_split[split] = {r["task_id"]: r for r in task_records(b2_sub)}

    def blend_records(split: str, w_m1: float) -> list[dict]:
        st_map = st_by_split[split]
        b2_map = b2_by_split[split]
        recs = []
        for tid in sorted(set(st_map) & set(b2_map)):
            sr, br = st_map[tid], b2_map[tid]
            if sr["predicted"] is None or br["predicted"] is None:
                continue
            pred = float(
                10
                ** (w_m1 * math.log10(sr["predicted"]) + (1.0 - w_m1) * math.log10(br["predicted"]))
            )
            # Evaluate against M1-Struct's target (own on test/test_ood, base on val)
            err = log_error(pred, sr["actual"])
            recs.append(
                {
                    "task_id": tid,
                    "family": sr["family"],
                    "abs_error": abs(err),
                    "signed_error": err,
                    "parsed_abs_error": abs(err),
                    "predicted": pred,
                    "actual": sr["actual"],
                    "parse_rate": 1.0,
                    "aur": 0.0,
                }
            )
        return recs

    grid = [round(i * 0.05, 2) for i in range(21)]
    val_curve = {}
    best_w, best_val = 0.5, float("inf")
    for w in grid:
        val_mdae = summary(blend_records("val", w))["mdae_log10"]
        val_curve[f"{w:.2f}"] = val_mdae
        if val_mdae < best_val:
            best_val = val_mdae
            best_w = w

    splits_out = {}
    for split in ("val", "test", "test_ood"):
        if split not in st_by_split:
            continue
        recs_sel = blend_records(split, best_w)
        recs_eq = blend_records(split, 0.5)
        recs_b2 = blend_records(split, 0.0)
        recs_m1 = blend_records(split, 1.0)
        s_sel = summary(recs_sel)
        s_eq = summary(recs_eq)
        s_b2 = summary(recs_b2)
        s_m1 = summary(recs_m1)
        splits_out[split] = {
            "target_source": "base" if split == "val" else "own",
            "val_selection_beat_b2": bool(s_sel["mdae_log10"] < s_b2["mdae_log10"]),
            "val_selected_blend": {
                "w_m1_struct": best_w,
                "mdae_log10": s_sel["mdae_log10"],
                "ci95": stratified_bootstrap(recs_sel, _mdae, resamples=resamples)
                if resamples
                else None,
                "within2x": s_sel["within2x"],
                "spearman": s_sel["spearman"],
            },
            "equal_geometric_blend": {
                "w_m1_struct": 0.5,
                "mdae_log10": s_eq["mdae_log10"],
                "ci95": stratified_bootstrap(recs_eq, _mdae, resamples=resamples)
                if resamples
                else None,
                "within2x": s_eq["within2x"],
                "spearman": s_eq["spearman"],
            },
            "standalone_b2_same_target": {
                "w_m1_struct": 0.0,
                "mdae_log10": s_b2["mdae_log10"],
                "within2x": s_b2["within2x"],
                "spearman": s_b2["spearman"],
            },
            "standalone_m1_struct_same_target": {
                "w_m1_struct": 1.0,
                "mdae_log10": s_m1["mdae_log10"],
                "within2x": s_m1["within2x"],
                "spearman": s_m1["spearman"],
            },
        }
    return {
        "val_selected_w_m1_struct": best_w,
        "val_grid_mdae_log10": val_curve,
        "splits": splits_out,
    }


def audit_m3_curation(root: Path, rows: list[dict], resamples: int = 2000) -> dict:
    """Forensic audit of historical `M3` `raw_extra` leakage, volume confound, and own-targets."""
    root = Path(root)
    tasks = list(read_jsonl(root / "data/tasks/tasks.jsonl"))
    tasks_by_id = {t["task_id"]: t for t in tasks}
    raw_extra_tasks = [t for t in tasks if t["split"] == "raw_extra"]
    raw_by_family = Counter(t["family"] for t in raw_extra_tasks)
    leaked_class_ids = sorted(t["task_id"] for t in raw_extra_tasks if t["family"] == "CLASS")

    sft_counts = {}
    for variant in ("m1", "m2", "m3"):
        p = root / f"data/sft/{variant}/train/data.jsonl"
        if p.exists():
            sft_rows = list(read_jsonl(p))
            sft_counts[variant] = {
                "total": len(sft_rows),
                "by_family": dict(sorted(Counter(r["family"] for r in sft_rows).items())),
                "by_frame": dict(sorted(Counter(r["frame"] for r in sft_rows).items())),
            }

    # Matched all-own-targets scoring using data/labels/m3-s0.parquet
    own_table = {}
    m3_parquet = root / "data/labels/m3-s0.parquet"
    if m3_parquet.exists():
        import pandas as pd

        m3_labels = pd.read_parquet(m3_parquet).to_dict(orient="records")
        m3_eval_rows = [
            r for r in rows if r["estimator"] == "M3-S0" and r["split"] in {"test", "test_ood"}
        ]
        m3_own = normalized_rows(
            own_trajectory_targets(m3_eval_rows, m3_labels, expected_source="m3-s0")
        )
        for split in ("test", "test_ood"):
            own_table[f"M3-S0|{split}|own"] = _metric_line(
                [
                    r
                    for r in m3_own
                    if r["split"] == split and r["card_id"] == "REF" and r["frame"] == "self"
                ],
                resamples=resamples,
            )
            for est in ("M1-S0", "M2-S0"):
                own_table[f"{est}|{split}|own"] = _metric_line(
                    [
                        r
                        for r in rows
                        if r["estimator"] == est
                        and r["split"] == split
                        and r["card_id"] == "REF"
                        and r["frame"] == "self"
                    ],
                    resamples=resamples,
                )

    m1_fam = sft_counts.get("m1", {}).get("by_family", {})
    m3_fam = sft_counts.get("m3", {}).get("by_family", {})
    volume_gain_pct = {
        fam: round(100.0 * (m3_fam.get(fam, 0) - m1_fam.get(fam, 0)) / m1_fam[fam], 1)
        for fam in sorted(m1_fam)
        if m1_fam[fam] > 0
    }

    return {
        "raw_extra_total_tasks": len(raw_extra_tasks),
        "raw_extra_by_family": dict(sorted(raw_by_family.items())),
        "leaked_class_task_ids": leaked_class_ids,
        "leaked_class_sft_rows_in_m3": m3_fam.get("CLASS", 0),
        "non_class_extra_sft_rows_in_m3": (
            sft_counts.get("m3", {}).get("total", 0)
            - sft_counts.get("m1", {}).get("total", 0)
            - m3_fam.get("CLASS", 0)
        ),
        "sft_train_counts": sft_counts,
        "family_volume_gain_pct_m3_vs_m1": volume_gain_pct,
        "matched_all_own_targets_ref": own_table,
        "pipeline_remediation_verified": all(
            tasks_by_id[tid]["family"] == "CLASS" for tid in leaked_class_ids
        ),
    }


def x1_interval_calibration(
    root: Path,
    rows: list[dict],
    m1b_rows: list[dict],
    struct_cp_rows: list[dict],
) -> dict:
    """Evaluate 80% intervals on both task medians and `X1` fresh K=8 trajectories."""
    root = Path(root)
    ref_card = SpeedCard("REF", 96.0, 26000.0)
    traces_cache: dict[str, dict[str, list[float]]] = {}
    for run in ("base", "m1-s0", "m2-s0", "m3-s0"):
        p = root / f"data/measure/{run}/traces.jsonl"
        if p.exists():
            by_t: dict[str, list[float]] = defaultdict(list)
            for tr in read_jsonl(p):
                by_t[tr["task_id"]].append(trajectory_time(tr, ref_card))
            traces_cache[run] = by_t

    pooled = rows + m1b_rows + struct_cp_rows
    specs = [
        ("E2", "base"),
        ("M2-S0", "m2-s0"),
        ("M1-S0", "m1-s0"),
        ("M1b", "m1-s0"),
        ("M1-Struct+CP", "m1-s0"),
    ]
    out: dict[str, dict] = {"fresh_trajectories": {}, "task_medians": {}}
    for split in ("test", "test_ood"):
        out["fresh_trajectories"][split] = {}
        out["task_medians"][split] = {}
        for est, trace_run in specs:
            sub = [
                r
                for r in pooled
                if r["estimator"] == est
                and r["split"] == split
                and r["card_id"] == "REF"
                and r.get("frame", "self") == "self"
                and r.get("lo_s")
                and r.get("hi_s")
                and r["lo_s"] > 0
                and r["hi_s"] >= r["lo_s"]
            ]
            if not sub:
                continue
            by_t = defaultdict(list)
            for r in sub:
                by_t[r["task_id"]].append(r)
            med_y, med_lo, med_hi = [], [], []
            run_y, run_lo, run_hi = [], [], []
            for tid, rlist in sorted(by_t.items()):
                lo = float(10 ** np.mean([math.log10(r["lo_s"]) for r in rlist]))
                hi = float(10 ** np.mean([math.log10(r["hi_s"]) for r in rlist]))
                med_y.append(float(rlist[0]["actual_s"]))
                med_lo.append(lo)
                med_hi.append(hi)
                for y in traces_cache.get(trace_run, {}).get(tid, ()):
                    run_y.append(y)
                    run_lo.append(lo)
                    run_hi.append(hi)
            out["task_medians"][split][est] = interval_metrics(med_y, med_lo, med_hi) | {
                "n_tasks": len(med_y)
            }
            if run_y:
                out["fresh_trajectories"][split][est] = interval_metrics(run_y, run_lo, run_hi) | {
                    "n_runs": len(run_y),
                    "n_tasks": len(med_y),
                }
    return out


def x8_first_pass_calibration(root: Path, rows: list[dict]) -> dict:
    """Pre-registered `X8` first-attempt pass probability calibration (`Brier`, `ECE`, `AUROC`)."""
    root = Path(root)
    empirical: dict[str, dict[str, float]] = {}
    for run in ("base", "m1-s0", "m2-s0", "m3-s0"):
        p = root / f"data/measure/{run}/traces.jsonl"
        if p.exists():
            by_t: dict[str, list[float]] = defaultdict(list)
            for tr in read_jsonl(p):
                by_t[tr["task_id"]].append(float(tr["first_pass"]))
            empirical[run] = {tid: float(np.mean(vals)) for tid, vals in by_t.items()}

    teacher_q = {}
    tq_path = root / "data/teacher/quantities.json"
    if tq_path.exists():
        teacher_q = json.loads(tq_path.read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in read_jsonl(root / "data/tasks/tasks.jsonl")}

    def reliability_bins(probs: list[float], outcomes: list[float], n_bins: int = 5) -> list[dict]:
        p_arr = np.asarray(probs, dtype=float)
        y_arr = np.asarray(outcomes, dtype=float)
        idx = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
        bins = []
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                bins.append(
                    {
                        "bin": b,
                        "count": int(mask.sum()),
                        "mean_predicted": float(p_arr[mask].mean()),
                        "mean_empirical": float(y_arr[mask].mean()),
                    }
                )
        return bins

    specs = [
        ("E2", "base"),
        ("M1-S0", "m1-s0"),
        ("M2-S0", "m2-s0"),
        ("M3-S0", "m3-s0"),
    ]
    out: dict[str, dict] = {}
    for split in ("test", "test_ood"):
        out[split] = {}
        for est, run in specs:
            sub = [
                r
                for r in rows
                if r["estimator"] == est
                and r["split"] == split
                and r["card_id"] == "REF"
                and r.get("frame", "self") == "self"
                and r.get("pfp") is not None
            ]
            by_t = defaultdict(list)
            for r in sub:
                by_t[r["task_id"]].append(float(r["pfp"]))
            if not by_t or run not in empirical:
                continue
            tids = sorted(t for t in by_t if t in empirical[run])
            probs = [float(np.mean(by_t[t])) for t in tids]
            raw_rates = [empirical[run][t] for t in tids]
            binary = [1 if y >= 0.5 else 0 for y in raw_rates]
            # Continuous empirical pass@1 Brier & ECE + binary majority-pass AUROC
            sm_bin = success_metrics(probs, binary, bins=10)
            p_arr, y_arr = np.asarray(probs), np.asarray(raw_rates)
            assign = np.minimum((p_arr * 10).astype(int), 9)
            ece_emp = float(
                sum(
                    (assign == b).mean()
                    * abs(p_arr[assign == b].mean() - y_arr[assign == b].mean())
                    for b in range(10)
                    if (assign == b).any()
                )
            )
            out[split][est] = {
                "n_tasks": len(tids),
                "brier_empirical_rate": float(np.mean((p_arr - y_arr) ** 2)),
                "ece_empirical_rate": ece_emp,
                "brier_majority_pass": sm_bin["brier"],
                "ece_majority_pass": sm_bin["ece"],
                "auroc_majority_pass": sm_bin["auroc"],
                "mean_predicted_pfp": float(p_arr.mean()),
                "mean_empirical_pfp": float(y_arr.mean()),
                "reliability_5bin": reliability_bins(probs, raw_rates, n_bins=5),
            }
        # Direct DeepSeek-V4-Pro teacher (`B4`) on base empirical pass@1
        if teacher_q and "base" in empirical:
            tids = sorted(
                tid
                for tid, meta in tasks.items()
                if meta["split"] == split and tid in teacher_q and tid in empirical["base"]
            )
            if tids:
                probs = [float(np.clip(teacher_q[t]["p_first_pass"], 0.0, 1.0)) for t in tids]
                raw_rates = [empirical["base"][t] for t in tids]
                binary = [1 if y >= 0.5 else 0 for y in raw_rates]
                sm_bin = success_metrics(probs, binary, bins=10)
                p_arr, y_arr = np.asarray(probs), np.asarray(raw_rates)
                assign = np.minimum((p_arr * 10).astype(int), 9)
                ece_emp = float(
                    sum(
                        (assign == b).mean()
                        * abs(p_arr[assign == b].mean() - y_arr[assign == b].mean())
                        for b in range(10)
                        if (assign == b).any()
                    )
                )
                out[split]["B4"] = {
                    "n_tasks": len(tids),
                    "brier_empirical_rate": float(np.mean((p_arr - y_arr) ** 2)),
                    "ece_empirical_rate": ece_emp,
                    "brier_majority_pass": sm_bin["brier"],
                    "ece_majority_pass": sm_bin["ece"],
                    "auroc_majority_pass": sm_bin["auroc"],
                    "mean_predicted_pfp": float(p_arr.mean()),
                    "mean_empirical_pfp": float(y_arr.mean()),
                    "reliability_5bin": reliability_bins(probs, raw_rates, n_bins=5),
                }
    return out


def token_and_arithmetic_audit(
    root: Path,
    rows: list[dict],
    tasks_by_id: dict[str, dict],
    priors: WorkloadPriors,
) -> dict:
    """Pre-registered `T8` token-level MdAE and internal verbal arithmetic consistency."""
    import pandas as pd

    root = Path(root)
    tok_lookup: dict[tuple[str, str], float] = {}
    for run in ("base", "m1-s0", "m2-s0", "m3-s0"):
        pq = root / f"data/labels/{run}.parquet"
        if pq.exists():
            df = pd.read_parquet(pq, columns=["task_id", "card_id", "TOK_med"])
            for rec in df[df["card_id"] == "REF"].to_dict(orient="records"):
                tok_lookup[(run, rec["task_id"])] = float(rec["TOK_med"])

    run_for_est = {"E2": "base", "M1-S0": "m1-s0", "M2-S0": "m2-s0", "M3-S0": "m3-s0"}
    out: dict[str, dict] = {}
    for split in ("test", "test_ood"):
        out[split] = {}
        for est, label_run in run_for_est.items():
            sub = [
                r
                for r in rows
                if r["estimator"] == est
                and r["split"] == split
                and r["card_id"] == "REF"
                and r.get("frame", "self") == "self"
            ]
            if not sub:
                continue
            by_t = defaultdict(list)
            for r in sub:
                by_t[r["task_id"]].append(r)
            tok_errs = []
            arith_bare = []
            arith_lat = []
            for tid, rlist in sorted(by_t.items()):
                t_meta = tasks_by_id.get(tid, {})
                u1 = float(t_meta.get("uncached_task_tokens", 120.0))
                fam = rlist[0]["family"]
                tok_med = tok_lookup.get((label_run, tid)) or tok_lookup.get(("base", tid))
                per_row_tok = []
                per_row_bare = []
                per_row_lat = []
                for r in rlist:
                    tok = r.get("tokens")
                    pt = r.get("point_s")
                    if tok and tok > 0:
                        if tok_med and tok_med > 0:
                            per_row_tok.append(abs(math.log10(tok / tok_med)))
                        if pt and pt > 0:
                            card = SpeedCard(r["card_id"], float(r["S"]), float(r["P"]))
                            calls = max(1.0, float(r.get("attempts") or 1.0))
                            u_tot = u1 + max(0.0, calls - 1.0) * priors.feedback_tokens(fam)
                            ex = calls * priors.exec_per_attempt(fam)
                            per_row_bare.append(arithmetic_consistency(pt, tok, u_tot, ex, card))
                            implied_lat = completion_time(
                                u_tot,
                                tok,
                                ex,
                                card,
                                latency_s=PER_CALL_LATENCY_S,
                                calls=calls,
                            )
                            per_row_lat.append(abs(log_error(pt, implied_lat)))
                if per_row_tok:
                    tok_errs.append(float(np.mean(per_row_tok)))
                if per_row_bare:
                    arith_bare.append(float(np.mean(per_row_bare)))
                if per_row_lat:
                    arith_lat.append(float(np.mean(per_row_lat)))
            out[split][est] = {
                "n_tasks_with_tokens": len(tok_errs),
                "token_mdae_log10": float(np.median(tok_errs)) if tok_errs else None,
                "token_within2x": float(np.mean(np.asarray(tok_errs) <= math.log10(2)))
                if tok_errs
                else None,
                "arithmetic_inconsistency_bare_mdae": float(np.median(arith_bare))
                if arith_bare
                else None,
                "arithmetic_inconsistency_with_latency_mdae": float(np.median(arith_lat))
                if arith_lat
                else None,
            }
    return out


def x10_cpu_deployment_struct(root: Path, priors: WorkloadPriors) -> dict | None:
    """Evaluate `M1-Struct` (`latency_s = 0.0`, local llama.cpp) on real desktop CPU (`X10`)."""
    root = Path(root)
    cpu_path = root / "data/x10/cpu-deployment.jsonl"
    card_path = root / "data/x10/cpu-speed-card.json"
    tasks_path = root / "data/tasks/tasks.jsonl"
    if not (cpu_path.exists() and card_path.exists() and tasks_path.exists()):
        return None

    card_info = json.loads(card_path.read_text(encoding="utf-8"))
    cpu_card = SpeedCard(card_info["id"], float(card_info["S"]), float(card_info["P"]))
    tasks = {t["task_id"]: t for t in read_jsonl(tasks_path)}
    cpu_rows = list(read_jsonl(cpu_path))

    base_errs, m1_errs, st_errs = [], [], []
    base_preds, m1_preds, st_preds, actuals = [], [], [], []
    for r in cpu_rows:
        actual = float(r["wall_s"])
        b_est = parse(r["base_raw"])
        m_est = parse(r["tuned_raw"])
        b_pt = b_est.point_s
        m_pt = m_est.point_s
        meta = tasks.get(r["task_id"], {})
        # Local llama.cpp runs on localhost without remote HF RPC latency (L=0.0)
        st_pt = structured_duration(
            m_est.tokens,
            m_est.attempts,
            float(meta.get("uncached_task_tokens", 120.0)),
            r["family"],
            cpu_card,
            priors,
            latency_s=0.0,
        )
        st_pt = st_pt if st_pt is not None else m_pt
        base_errs.append(abs(log_error(b_pt, actual)))
        m1_errs.append(abs(log_error(m_pt, actual)))
        st_errs.append(abs(log_error(st_pt, actual)))
        actuals.append(actual)
        base_preds.append(b_pt or 1e9)
        m1_preds.append(m_pt or 1e9)
        st_preds.append(st_pt or 1e9)

    from scipy.stats import spearmanr

    def _pack(errs, preds):
        arr = np.asarray(errs, dtype=float)
        return {
            "n_tasks": len(arr),
            "mdae_log10": float(np.median(arr)),
            "within2x": float(np.mean(arr <= math.log10(2))),
            "within10x": float(np.mean(arr <= 1.0)),
            "spearman": float(spearmanr(preds, actuals).statistic),
        }

    return {
        "speed_card": card_info,
        "base_verbal": _pack(base_errs, base_preds),
        "m1_verbal": _pack(m1_errs, m1_preds),
        "m1_struct_local": _pack(st_errs, st_preds),
    }


def unit_and_wording_diagnostics(rows: list[dict]) -> dict:
    """Pre-registered calendar-unit sensitivity (§6) and 8-template wording robustness (`X11`)."""
    # Calendar-vs-working-day sensitivity on E0, E1, M1-S0 where raw_text is non-redacted
    calendar_out = {}
    for est in ("E0", "E1", "M1-S0"):
        sub = [
            r
            for r in rows
            if r["estimator"] == est
            and r["split"] == "test"
            and r["card_id"] == "REF"
            and r.get("frame", "self") == "self"
        ]
        if not sub:
            continue
        cal_rows = []
        changed = 0
        inspected = 0
        for r in sub:
            raw_text = r.get("raw_text")
            if raw_text:
                inspected += 1
                cal_est = parse(raw_text, calendar=True)
                if cal_est.point_s != r.get("point_s"):
                    changed += 1
                cal_rows.append(dict(r, point_s=cal_est.point_s))
            else:
                cal_rows.append(r)
        calendar_out[est] = {
            "working_day_mdae_log10": summary(task_records(sub))["mdae_log10"],
            "calendar_day_mdae_log10": summary(task_records(cal_rows))["mdae_log10"],
            "non_redacted_rows_inspected": inspected,
            "rows_changed_by_calendar_factor": changed,
        }

    # X11 wording sensitivity across templates s01..s08 and speed-card phrasings sc01..sc04
    wording_out: dict[str, dict] = {}
    for est in ("E1", "M1-S0", "M2-S0"):
        sub = [
            r
            for r in rows
            if r["estimator"] == est
            and r["split"] == "test"
            and r["card_id"] == "REF"
            and r.get("frame", "self") == "self"
        ]
        by_tpl: dict[str, list[dict]] = defaultdict(list)
        by_sc: dict[str, list[dict]] = defaultdict(list)
        for r in sub:
            if r.get("template_id"):
                by_tpl[r["template_id"]].append(r)
            if r.get("phrasing_id"):
                by_sc[r["phrasing_id"]].append(r)
        tpl_mdaes = {k: summary(task_records(v))["mdae_log10"] for k, v in sorted(by_tpl.items())}
        sc_mdaes = {k: summary(task_records(v))["mdae_log10"] for k, v in sorted(by_sc.items())}
        wording_out[est] = {
            "by_template_mdae_log10": tpl_mdaes,
            "template_iqr": float(
                np.percentile(list(tpl_mdaes.values()), 75)
                - np.percentile(list(tpl_mdaes.values()), 25)
            )
            if tpl_mdaes
            else None,
            "by_phrasing_mdae_log10": sc_mdaes,
            "phrasing_range": float(max(sc_mdaes.values()) - min(sc_mdaes.values()))
            if sc_mdaes
            else None,
        }
    return {"calendar_unit_sensitivity": calendar_out, "wording_sensitivity_x11": wording_out}


def compute_posthoc_bundle(root: Path, resamples: int = 2000) -> dict:
    """Compute workload-decomposition + exact-timing, conformal, hybrid, and diagnostic artifacts."""
    root = Path(root)
    rows = normalized_rows(read_jsonl(root / "data/eval/estimates.jsonl"))
    tasks = {t["task_id"]: t for t in read_jsonl(root / "data/tasks/tasks.jsonl")}
    priors = fit_workload_priors(root)
    write_json(root / "results/workload-priors.json", json_safe(priors.as_dict()))

    m1_struct = build_structured_rows(
        rows, tasks, priors, source_estimator="M1-S0", estimator_name="M1-Struct"
    )
    m2_struct = build_structured_rows(
        rows, tasks, priors, source_estimator="M2-S0", estimator_name="M2-Struct"
    )
    m3_struct = build_structured_rows(
        rows, tasks, priors, source_estimator="M3-S0", estimator_name="M3-Struct"
    )
    e2_struct = build_structured_rows(
        rows, tasks, priors, source_estimator="E2", estimator_name="E2-Struct"
    )

    val_traces = defaultdict(list)
    base_traces_path = root / "data/measure/base/traces.jsonl"
    if base_traces_path.exists():
        for tr in read_jsonl(base_traces_path):
            if tasks.get(tr["task_id"], {}).get("split") == "val":
                val_traces[tr["task_id"]].append(tr)

    cal = fit_split_conformal(
        rows,
        m1_struct,
        val_traces_by_task=val_traces or None,
        card=SpeedCard("REF", 96.0, 26000.0),
        alpha=0.20,
    )
    m1b_rows, m1_struct_cp_rows = apply_conformal_rows(rows, m1_struct, cal)

    # Post-hoc & full baseline summary table on REF
    pooled = rows + m1_struct + m2_struct + m3_struct + e2_struct
    all_estimators = (
        "E0",
        "E1",
        "E4",
        "E2",
        "E2-Struct",
        "B0",
        "B1",
        "B4",
        "B2",
        "B2+",
        "N",
        "M2-S0",
        "M2-Struct",
        "M3-S0",
        "M3-Struct",
        "M1-S0",
        "M1-Struct",
    )
    ref_summary = {}
    for split in ("val", "test", "test_ood"):
        ref_summary[split] = {}
        for est in all_estimators:
            sub = [
                r
                for r in pooled
                if r["estimator"] == est
                and r["split"] == split
                and r["card_id"] == "REF"
                and r.get("frame", "self") == "self"
            ]
            if sub:
                ref_summary[split][est] = _metric_line(sub, resamples=resamples)

    # Speed-card conditioning + extrapolation across all 8 cards on test and test_ood
    speed_models = ("E1", "E2", "B2", "M2-S0", "M1-S0", "M1-Struct")
    speed_cards_out: dict[str, dict] = {}
    elasticity_out: dict[str, dict] = {}
    for split in ("test", "test_ood"):
        cards = sorted(
            {r["card_id"] for r in pooled if r["split"] == split and r["estimator"] == "M1-S0"}
        )
        speed_cards_out[split] = {}
        for cid in cards:
            speed_cards_out[split][cid] = {}
            for est in speed_models:
                sub = [
                    r
                    for r in pooled
                    if r["estimator"] == est
                    and r["split"] == split
                    and r["card_id"] == cid
                    and r.get("frame", "self") == "self"
                ]
                if sub:
                    s = summary(task_records(sub))
                    speed_cards_out[split][cid][est] = {
                        "mdae_log10": s["mdae_log10"],
                        "within2x": s["within2x"],
                        "gm_ratio": s["gm_ratio"],
                    }
        # Per-task speed elasticity median across tasks
        elasticity_out[split] = {}
        for est in speed_models:
            by_task_card: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for cid in cards:
                sub = [
                    r
                    for r in pooled
                    if r["estimator"] == est
                    and r["split"] == split
                    and r["card_id"] == cid
                    and r.get("frame", "self") == "self"
                ]
                for rec in task_records(sub):
                    if rec["predicted"] is not None:
                        by_task_card[rec["task_id"]].append((rec["predicted"], rec["actual"]))
            slopes = [
                speed_elasticity([p for p, _ in pairs], [a for _, a in pairs])
                for pairs in by_task_card.values()
                if len(pairs) >= 4
            ]
            valid_slopes = [s for s in slopes if s is not None]
            elasticity_out[split][est] = float(np.median(valid_slopes)) if valid_slopes else None

    # Overall and counter-intuitive pairwise ordering accuracy (against E1 human-frame intuition)
    human_e1 = {
        rec["task_id"]: rec["predicted"]
        for rec in task_records(
            [
                r
                for r in rows
                if r["estimator"] == "E1"
                and r["split"] == "test"
                and r["card_id"] == "REF"
                and r.get("frame") == "human"
            ]
        )
    }
    pairwise_out = {}
    for est in ("E1", "E2", "B2", "B4", "M2-S0", "M1-S0", "M1-Struct"):
        sub = [
            r
            for r in pooled
            if r["estimator"] == est
            and r["split"] == "test"
            and r["card_id"] == "REF"
            and r.get("frame", "self") == "self"
        ]
        if sub:
            recs = task_records(sub)
            pairwise_out[est] = {
                "overall": pairwise_accuracy(recs),
                "counter_intuitive_vs_human_frame": pairwise_accuracy(
                    recs, human_predictions=human_e1
                ),
            }

    b2_rows = [r for r in rows if r["estimator"] == "B2" and r.get("frame", "self") == "self"]
    return {
        "workload_priors": priors.as_dict(),
        "conformal_calibration": cal.as_dict(),
        "posthoc_ref_summary": ref_summary,
        "speed_card_disentanglement": {
            "by_split_and_card": speed_cards_out,
            "median_task_speed_elasticity": elasticity_out,
        },
        "hybrid_ensemble": hybrid_ensemble_analysis(m1_struct, b2_rows, resamples=resamples),
        "x1_interval_calibration": x1_interval_calibration(root, rows, m1b_rows, m1_struct_cp_rows),
        "x8_first_pass_calibration": x8_first_pass_calibration(root, rows),
        "token_and_arithmetic_audit": token_and_arithmetic_audit(root, rows, tasks, priors),
        "m3_curation_audit": audit_m3_curation(root, rows, resamples=resamples),
        "pairwise_ordering_test": pairwise_out,
        "x10_cpu_deployment_struct": x10_cpu_deployment_struct(root, priors),
    } | unit_and_wording_diagnostics(rows)
