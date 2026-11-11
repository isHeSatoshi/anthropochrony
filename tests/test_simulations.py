import pytest

from anthropochrony.sim import scheduling, sla_tier_routing, timeout_tradeoffs


def test_sjf_scheduling_oracle_beats_fifo():
    actual = [float(i + 1) for i in range(60)]
    oracle_est = list(actual)
    reverse_est = list(reversed(actual))
    res_oracle = scheduling(actual, oracle_est, queues=50, queue_size=50)
    res_bad = scheduling(actual, reverse_est, queues=50, queue_size=50)
    assert res_oracle["sjf_relative_to_oracle"] == pytest.approx(1.0)
    assert res_oracle["fifo_relative_to_oracle"] > 1.2
    assert res_bad["sjf_relative_to_oracle"] > res_oracle["fifo_relative_to_oracle"]


def test_timeout_tradeoffs_monotonic_in_multiplier():
    actual = [5.0, 8.0, 30.0, 60.0]
    upper = [10.0, 10.0, 10.0, 10.0]
    passed = [True, True, False, False]
    rows = timeout_tradeoffs(actual, upper, passed, multipliers=(0.75, 1.0, 2.0))
    assert [r["multiplier"] for r in rows] == [0.75, 1.0, 2.0]
    assert rows[0]["false_stop_rate"] >= rows[1]["false_stop_rate"] >= rows[2]["false_stop_rate"]
    assert rows[1]["false_stop_rate"] == 0.0
    assert rows[1]["failed_time_saved_s"] == pytest.approx(20.0 + 50.0)


def test_sla_tier_routing_trades_cost_and_sla_compliance():
    edge_s = [4.0, 6.0, 22.0, 30.0]
    dc_s = [1.0, 1.5, 4.0, 6.0]
    calibrated_hi = [6.0, 9.0, 25.0, 35.0]
    human_inflated_hi = [600.0, 900.0, 1800.0, 3600.0]

    res_cal = sla_tier_routing(edge_s, dc_s, calibrated_hi, sla_s=15.0)
    res_e1 = sla_tier_routing(edge_s, dc_s, human_inflated_hi, sla_s=15.0)

    assert res_cal["edge_routed_share"] == pytest.approx(0.5)
    assert res_cal["sla_violation_rate"] == 0.0
    assert res_e1["edge_routed_share"] == 0.0
    assert res_cal["passing_cost_usd_per_1k_runs"] < res_e1["passing_cost_usd_per_1k_runs"]
    assert res_cal["cost_savings_vs_always_dc_pct"] == pytest.approx(200.0 / 22.5)
