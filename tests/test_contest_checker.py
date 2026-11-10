import importlib.util

import pytest

from anthropochrony.prompts import SPEC

spec = importlib.util.spec_from_file_location(
    "sandbox_worker", SPEC.parent.parent / "src/anthropochrony/sandbox/worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        # Independent upstream OutputsMatchTest examples, same pinned commit.
        ("abc def", "abc def", True),
        ("abc deg", "abc def", False),
        ("abc def", "abc def 123", False),
        ("abc DEF", "abc def", True),
        ("abc 123", "abc 123.000001", True),
        ("abc 123", "abc 123.1", False),
        # Boundary and malformed-token checks, not a tuned acceptance tolerance.
        ("YES\n", "Yes ", True),
        ("0.000009", "0", True),
        ("0.000011", "0", False),
        ("1_0", "10", False),
        ("nan", "1", False),
        ("wrong", "", False),
    ],
)
def test_upstream_contest_output_comparison(actual, expected, matches):
    assert worker.outputs_match(actual, expected) == matches
