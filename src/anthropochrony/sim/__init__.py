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


def timeout_tradeoffs(actual, upper, passed, multipliers=(1, 1.5, 2, 3)):
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
