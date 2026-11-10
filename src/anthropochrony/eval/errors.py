"""Scalar error definition shared by GPU scoring and CPU statistical analysis."""

import math


def log_error(predicted, actual):
    if actual <= 0 or not math.isfinite(actual):
        raise ValueError("Ground-truth durations must be positive and finite")
    if predicted is None or predicted <= 0 or not math.isfinite(predicted):
        return math.inf
    return math.log10(predicted / actual)
