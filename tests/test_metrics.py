import math

import numpy as np
import pytest

from anthropochrony.eval.metrics import interval_metrics, summary, task_records
from anthropochrony.eval.stats import holm, paired_wilcoxon, stratified_bootstrap


def test_failures_count_against_median_and_shares():
    rows = [
        {"task_id": str(i), "family": "FUNC", "point_s": point, "actual_s": 10}
        for i, point in enumerate([None, None, 10])
    ]
    metrics = summary(task_records(rows))
    assert math.isinf(metrics["mdae_log10"])
    assert metrics["within2x"] == pytest.approx(1 / 3)
    assert metrics["parse_rate"] == pytest.approx(1 / 3)
    assert metrics["gm_ratio"] == 1


def test_task_unit_averages_templates_before_median():
    rows = [
        {"task_id": "a", "family": "FUNC", "point_s": 10, "actual_s": 10},
        {"task_id": "a", "family": "FUNC", "point_s": 1000, "actual_s": 10},
        {"task_id": "b", "family": "FUNC", "point_s": 10, "actual_s": 10},
    ]
    assert summary(task_records(rows))["mdae_log10"] == 0.5


def test_stratified_bootstrap_preserves_family_counts():
    records = [{"family": "A"}] + [{"family": "B"}] * 9
    seen = []

    def statistic(sample):
        seen.append(sum(r["family"] == "A" for r in sample))
        return seen[-1]

    assert stratified_bootstrap(records, statistic, resamples=50) == [1, 1]
    assert set(seen) == {1}


def test_holm_and_failed_pair_ties():
    assert holm({"H2": 0.01, "H3": 0.04, "H5": 0.03}) == {"H2": 0.03, "H5": 0.06, "H3": 0.06}
    assert paired_wilcoxon([np.inf] * 5, [np.inf] * 5) == 1


def test_winkler_independent_example():
    assert interval_metrics([1, 3, 8], [2, 2, 2], [4, 4, 4])["winkler"] == pytest.approx(56 / 3)
