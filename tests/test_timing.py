import pytest

from anthropochrony.timemodel import (
    SpeedCard,
    attempt_budget,
    completion_time,
    output_token_count,
    uncached_tokens,
)


def test_formula_independent_hand_calculation():
    assert completion_time(100, 200, 3, SpeedCard("test", 20, 100), latency_s=0) == 14
    assert completion_time(5000, 350, 4, SpeedCard("test", 35, 5000), latency_s=0) == 15
    # The adopted fallback adds one latency per model call by default.
    card = SpeedCard("test", 20, 100)
    assert completion_time(100, 200, 3, card) == pytest.approx(14.967)
    assert completion_time(100, 200, 3, card, calls=2.5) == pytest.approx(14 + 2.5 * 0.967)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_speed(value):
    with pytest.raises(ValueError):
        SpeedCard("bad", value, 1)


def test_budget_boundary():
    assert attempt_budget(7936, "FUNC") == 256
    assert attempt_budget(7937, "FUNC") == 0
    assert attempt_budget(0, "FUNC") == 2048
    assert attempt_budget(0, "PROJ-8") == 4096
    assert attempt_budget(0, "ALGO_IO") == 4096
    assert attempt_budget(5000, "ALGO_IO") == 3192


def test_stop_accounting_once():
    assert output_token_count([9, 6], 6) == 2
    assert output_token_count([9], 6) == 2
    assert output_token_count([9, 261001], 261001) == 2
    assert output_token_count([9], None) == 1


def test_uncached_retry():
    assert uncached_tokens(800, 510) == 290
    assert uncached_tokens(1200, 510, 800, 300) == 100
    with pytest.raises(ValueError):
        uncached_tokens(10, 510)
