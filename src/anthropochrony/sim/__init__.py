"""Exploratory scheduling and timeout analyses, never training rewards."""

import numpy as np


def scheduling(actual, estimates, queues=1000, queue_size=50, seed=20261109):
    actual, estimates = np.asarray(actual), np.asarray(estimates)
    if len(actual) < queue_size or len(actual) != len(estimates):
        raise ValueError("Need at least queue_size paired tasks")
    rng = np.random.default_rng(seed)
    ratios, fifo = [], []
    for _ in range(queues):
        ids = rng.choice(len(actual), queue_size, replace=False)
        times = actual[ids]
        oracle = np.cumsum(np.sort(times)).mean()
        ratios.append(np.cumsum(times[np.argsort(estimates[ids], kind="stable")]).mean() / oracle)
        fifo.append(np.cumsum(times).mean() / oracle)
    return {
        "sjf_relative_to_oracle": float(np.mean(ratios)),
        "fifo_relative_to_oracle": float(np.mean(fifo)),
        "n_queues": queues,
    }


def timeout_tradeoffs(actual, upper, passed, multipliers=(0.75, 1, 1.25, 1.5, 2, 3)):
    actual, upper, passed = np.asarray(actual), np.asarray(upper), np.asarray(passed, dtype=bool)
    rows = []
    for multiplier in multipliers:
        cap = multiplier * upper
        saved = np.maximum(actual - cap, 0)
        rows.append(
            {
                "multiplier": multiplier,
                "false_stop_rate": float(np.mean(actual[passed] > cap[passed]))
                if passed.any()
                else None,
                "failed_time_saved_s": float(saved[~passed].sum()),
                "failed_time_saved_share": float(saved[~passed].sum() / actual[~passed].sum())
                if (~passed).any()
                else None,
            }
        )
    return rows


def sla_tier_routing(
    edge_actual_s,
    dc_actual_s,
    edge_upper_s,
    *,
    sla_s=15.0,
    edge_usd_per_hour=0.25,
    dc_usd_per_hour=1.80,
):
    """Route tasks to cheap Edge GPU if predicted upper bound <= SLA, else fast DC GPU."""
    edge_t = np.asarray(edge_actual_s, dtype=float)
    dc_t = np.asarray(dc_actual_s, dtype=float)
    pred_edge_hi = np.asarray(edge_upper_s, dtype=float)
    if not (len(edge_t) == len(dc_t) == len(pred_edge_hi)) or len(edge_t) == 0:
        raise ValueError("Paired non-empty edge/dc durations and predictions required")

    use_edge = pred_edge_hi <= sla_s
    realized_s = np.where(use_edge, edge_t, dc_t)
    cost_per_run = np.where(
        use_edge,
        edge_t / 3600.0 * edge_usd_per_hour,
        dc_t / 3600.0 * dc_usd_per_hour,
    )
    dc_only_cost = float(np.mean(dc_t / 3600.0 * dc_usd_per_hour) * 1000.0)
    routed_cost = float(np.mean(cost_per_run) * 1000.0)
    return {
        "sla_s": float(sla_s),
        "edge_routed_share": float(np.mean(use_edge)),
        "sla_violation_rate": float(np.mean(realized_s > sla_s)),
        "passing_cost_usd_per_1k_runs": routed_cost,
        "cost_savings_vs_always_dc_pct": float(100.0 * (1.0 - routed_cost / dc_only_cost))
        if dc_only_cost > 0
        else 0.0,
        "mean_latency_s": float(np.mean(realized_s)),
    }
