"""Scoring for the MetaTime head-to-head: the benchmark's metrics, intervals and paired tests.

Every function works on plain per-task numbers, so the committed records under
`results/metatime/` are enough to rebuild `results/metatime.json` without a GPU and
without the benchmark's task text.
"""

import math

import numpy as np

DRAWS = 5000
SEED = 20261114


def _pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):  # average ranks over ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def ordering_accuracy(estimate, actual):
    """Share of task pairs ordered as their measured times; a tied estimate counts half."""
    total = score = 0.0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            if actual[i] == actual[j]:
                continue
            total += 1
            product = (estimate[i] - estimate[j]) * (actual[i] - actual[j])
            score += 1.0 if product > 0 else 0.5 if product == 0 else 0.0
    return score / total if total else None


def metrics(estimate, actual):
    """The benchmark's metrics for one estimator; an unparsed estimate counts as an infinite miss."""
    estimate = np.array([np.nan if value is None else value for value in estimate], dtype=float)
    actual = np.asarray(actual, dtype=float)
    parsed = np.isfinite(estimate) & (estimate > 0)
    log_error = np.full(len(actual), np.inf)
    log_error[parsed] = np.abs(np.log(estimate[parsed] / actual[parsed]))
    ratio = np.full(len(actual), np.inf)
    ratio[parsed] = estimate[parsed] / actual[parsed]
    e, a = estimate[parsed], actual[parsed]
    return {
        "n": int(len(actual)),
        "parsed": int(parsed.sum()),
        "median_ratio": float(np.median(ratio)),
        "median_abs_log_error": float(np.median(log_error)),
        "within_2x": float(np.mean(log_error <= math.log(2))),
        "pearson_log": _pearson(np.log(e), np.log(a)) if len(e) else None,
        "pearson_raw": _pearson(e, a) if len(e) else None,
        "spearman": _pearson(_ranks(e), _ranks(a)) if len(e) else None,
        "ordering_accuracy": ordering_accuracy(e, a) if len(e) else None,
        "mae_s": float(np.mean(np.abs(e - a))) if len(e) else None,
    }


def _draws(n, draws=DRAWS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def intervals(
    estimate, actual, keys=("median_ratio", "median_abs_log_error", "pearson_log", "within_2x")
):
    """Percentile 95% intervals from resampling tasks."""
    estimate = [None if value is None else float(value) for value in estimate]
    samples = {key: [] for key in keys}
    for index in _draws(len(actual)):
        row = metrics([estimate[i] for i in index], [actual[i] for i in index])
        for key in keys:
            if row[key] is not None and math.isfinite(row[key]):
                samples[key].append(row[key])
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in samples.items()
        if values
    }


def paired_log_error(estimate_a, actual_a, estimate_b, actual_b):
    """Difference in median absolute log error (a minus b) over the same resampled tasks.

    Each system is scored against its own measured times; the pairing is by task.
    """

    def errors(estimate, actual):
        return np.array(
            [
                abs(math.log(e / a)) if e is not None and e > 0 else math.inf
                for e, a in zip(estimate, actual, strict=True)
            ]
        )

    a, b = errors(estimate_a, actual_a), errors(estimate_b, actual_b)
    index = _draws(len(a))
    difference = np.median(a[index], axis=1) - np.median(b[index], axis=1)
    return {
        "difference": float(np.median(a) - np.median(b)),
        "ci95": [float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))],
        "share_of_draws_a_lower": float(np.mean(difference < 0)),
        "tasks_a_closer": int(np.sum(a < b)),
        "tasks": int(len(a)),
    }


def retest_ceiling(runs_by_task):
    """How well one run predicts the median of the task's other runs (run-to-run noise)."""
    errors = []
    for runs in runs_by_task:
        for i, value in enumerate(runs):
            others = [other for j, other in enumerate(runs) if j != i]
            if others:
                errors.append(abs(math.log(value / float(np.median(others)))))
    return float(np.median(errors)) if errors else None
