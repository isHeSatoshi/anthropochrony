import math

import pytest

from anthropochrony.parse import parse


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("6-9 hours", math.sqrt(6 * 9) * 3600),
        ("between two and four hours", math.sqrt(8) * 3600),
        ("a couple of hours", 7200),
        ("a few minutes", 180),
        ("half an hour", 1800),
        ("two days", 57600),
        ("1 week", 144000),
        ("1 month", 576000),
        ("40 s–2.5 min", math.sqrt(40 * 150)),
        ("Design: 10 min\nImplement: 20 min\nTest: 5 min", 2100),
        ("Design: 10 min\nCode: 20 min\nTotal: 30 min", 1800),
        ("2 hours and 30 minutes", 9000),
    ],
)
def test_duration(text, expected):
    assert parse(text).point_s == pytest.approx(expected)


def test_canonical():
    result = parse(
        "About 30 s (likely 20–55 s). Breakdown: ~900 output tokens at "
        "35 tok/s ≈ 26 s, plus ~4.3 s reading the task and running tests; "
        "1 attempt expected. First-attempt pass chance: about 70%."
    )
    assert (
        result.point_s,
        result.lo_s,
        result.hi_s,
        result.tokens,
        result.attempts,
        result.pfp,
    ) == (
        30,
        20,
        55,
        900,
        1,
        0.7,
    )
    assert not result.has_human_units


def test_invalid_and_judge():
    for text in ("I cannot tell", "0 seconds", "-2 seconds"):
        assert parse(text).point_s is None
    assert parse("1 day", calendar=True).point_s == 86400
    result = parse(
        "unclear",
        judge=lambda *_: {"has_estimate": True, "point_seconds": 60, "units_used": ["minute"]},
    )
    assert result.parse_method == "judge"
    assert result.point_s == 60


def test_code_is_not_an_estimate():
    code = "```python\ndef f(s):\n    return 2 s\n```\nTime Complexity: 5 s"
    assert parse("```python\nx = 3 s\n").point_s is None
    assert parse(code).point_s == 5


def test_non_plan_answer_uses_first_duration_paragraph_only():
    text = "Time estimate: 5-10 minutes\n\n**Explanation:** the loop runs in 2 s."
    assert parse(text).point_s == pytest.approx(math.sqrt(50) * 60)
    plan = "Design: 10 min\n\nCode: 20 min"
    assert parse(plan, plan=True).point_s == 1800
