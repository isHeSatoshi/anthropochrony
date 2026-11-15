import math

import pytest

from anthropochrony.eval import metatime


def test_metrics_match_hand_values():
    actual = [10.0, 20.0, 40.0, 80.0]
    estimate = [20.0, 20.0, 20.0, 160.0]
    row = metatime.metrics(estimate, actual)
    assert row["median_ratio"] == 1.5
    assert math.isclose(row["median_abs_log_error"], math.log(2))
    assert row["within_2x"] == 1.0
    assert row["mae_s"] == (10 + 0 + 20 + 80) / 4
    assert row["parsed"] == 4


def test_unparsed_estimate_is_an_infinite_miss():
    row = metatime.metrics([10.0, None, None], [10.0, 10.0, 10.0])
    assert row["parsed"] == 1
    assert math.isinf(row["median_abs_log_error"])
    assert math.isclose(row["within_2x"], 1 / 3)
    assert row["pearson_log"] is None


def test_perfect_and_reversed_orderings():
    actual = [1.0, 2.0, 3.0, 4.0]
    assert metatime.ordering_accuracy([10, 20, 30, 40], actual) == 1.0
    assert metatime.ordering_accuracy([40, 30, 20, 10], actual) == 0.0
    assert metatime.ordering_accuracy([5, 5, 5, 5], actual) == 0.5
    assert math.isclose(metatime.metrics([10, 20, 30, 40], actual)["spearman"], 1.0)


def test_paired_comparison_prefers_the_closer_system():
    actual = [float(value) for value in range(5, 45)]
    close = [value * 1.2 for value in actual]
    far = [value * 3.0 for value in actual]
    result = metatime.paired_log_error(close, actual, far, actual)
    assert result["difference"] < 0
    assert result["ci95"][1] < 0
    assert result["share_of_draws_a_lower"] == 1.0
    assert result["tasks_a_closer"] == len(actual)


def test_intervals_bracket_the_point_estimate():
    actual = [float(value) for value in range(5, 45)]
    estimate = [value * (1.5 if index % 2 else 0.8) for index, value in enumerate(actual)]
    point = metatime.metrics(estimate, actual)
    interval = metatime.intervals(estimate, actual)
    low, high = interval["median_abs_log_error"]
    assert low <= point["median_abs_log_error"] <= high


def test_retest_ceiling_is_zero_for_identical_runs():
    assert metatime.retest_ceiling([[2.0, 2.0, 2.0], [5.0, 5.0]]) == 0.0
    assert metatime.retest_ceiling([[1.0, 2.0]]) > 0


def test_leave_one_out_constant_and_rescaling_use_only_the_other_tasks():
    actual = [1.0, 2.0, 4.0, 8.0, 100.0]
    assert metatime.leave_one_out_constant(actual) == [6.0, 6.0, 5.0, 3.0, 3.0]
    # A system that is exactly three times too high everywhere is exact once rescaled.
    rescaled = metatime.leave_one_out_rescaled([3 * value for value in actual], actual)
    assert rescaled == pytest.approx(actual)
    # An unparsed answer stays unparsed and does not shift the others.
    partial = metatime.leave_one_out_rescaled([3.0, None, 12.0, 24.0, 300.0], actual)
    assert partial[1] is None
    assert [partial[i] for i in (0, 2, 3, 4)] == pytest.approx([1.0, 4.0, 8.0, 100.0])


def test_conformal_range_uses_the_finite_sample_rank():
    # Nine calibration scores ln(1.1) .. ln(1.9): rank ceil(10 x 0.8) = 8 picks ln(1.8).
    actual = [10.0] * 9
    estimate = [10.0 * (1 + k / 10) for k in range(1, 10)]
    estimate[0] = 10.0 / 1.1  # the score is symmetric in the direction of the miss
    row = metatime.conformal_range(estimate, actual)
    assert row["calibration_n"] == 9 and row["rank"] == 8
    assert row["factor"] == pytest.approx(1.8)
    assert metatime.conformal_range([10.0] * 3, [10.0] * 3) is None  # rank 4 of 3
    assert metatime.conformal_range([None] * 3 + estimate, [10.0] * 3 + actual) is None
    assert metatime.share_inside([10.0, 10.0, None, 10.0], [18.0, 19.0, 10.0, 5.6], 1.8) == 0.5
