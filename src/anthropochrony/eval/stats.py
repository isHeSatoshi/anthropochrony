import math
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def stratified_bootstrap(records, statistic, resamples=10000, seed=20261109):
    if not records:
        raise ValueError("Cannot bootstrap an empty dataset")
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["family"]].append(i)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indices = np.concatenate(
            [rng.choice(group, len(group), replace=True) for _, group in sorted(groups.items())]
        )
        value = statistic([records[i] for i in indices])
        if value is not None and not math.isnan(value):
            values.append(value)
    if not values:
        return [None, None]
    return [float(x) for x in np.quantile(values, [0.025, 0.975], method="nearest")]


def paired_wilcoxon(first, second):
    if len(first) != len(second) or not len(first):
        raise ValueError("Paired observations required")
    # Both failed parses are a zero difference. One failure ranks beyond all finite differences.
    finite = [
        abs(a - b)
        for a, b in zip(first, second, strict=True)
        if math.isfinite(a) and math.isfinite(b)
    ]
    bound = max(finite, default=0) + 1
    delta = [
        0
        if math.isinf(a) and math.isinf(b)
        else bound
        if math.isinf(a)
        else -bound
        if math.isinf(b)
        else a - b
        for a, b in zip(first, second, strict=True)
    ]
    if not any(delta):
        return 1.0
    return float(wilcoxon(delta, alternative="less", zero_method="wilcox").pvalue)


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0
    for i, key in enumerate(ordered):
        running = max(running, (len(ordered) - i) * pvalues[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value
