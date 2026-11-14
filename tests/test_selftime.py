import importlib.util
import json
import shlex
import sys
from collections import deque
from pathlib import Path

import pytest

from anthropochrony import selftime
from anthropochrony.cli import main
from anthropochrony.eval.structured import structured_duration
from anthropochrony.prompts import estimate_messages, task_turn
from anthropochrony.selftime import (
    EstimateUnavailable,
    SelfTimeEstimator,
    approx_task_tokens,
    default_priors,
    load_defaults,
)
from anthropochrony.timemodel import SpeedCard

ROOT = Path(__file__).resolve().parents[1]
TASK = "Write a function `add(a, b)` that returns the sum of two integers."
TUNED = (
    "About 23 s (likely 18–29 s). Breakdown: ~330 output tokens at 20 tok/s ≈ 17 s, plus ~6.2 s "
    "reading the task and running tests; all 3 attempts expected. "
    "First-attempt pass chance: under 10%."
)
HUMAN = "A developer would need about 2 hours for this."
CARD = SpeedCard("fake", 20, 200)


class FakeBackend:
    def __init__(self, *responses, card=CARD):
        self.responses = deque(responses)
        self.card = card
        self.calls = []
        self.calibrations = 0

    def calibrate(self):
        self.calibrations += 1
        return self.card

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.popleft()


class GenerateOnly:
    def __init__(self, *responses):
        self.responses = deque(responses)

    def generate(self, messages, **kwargs):
        return self.responses.popleft()


class FakeLlama:
    """Stands in for LlamaBackend; the test sets the queued answers."""

    responses = deque()
    created = []

    def __init__(self, base_url, tuned_url, model_path, bench, threads=8, gpu_layers=0):
        FakeLlama.created.append((base_url, tuned_url, model_path, bench, threads, gpu_layers))

    def calibrate(self):
        return SpeedCard("demo-llamacpp", 20, 200)

    def generate(self, messages, **kwargs):
        return FakeLlama.responses.popleft()


@pytest.fixture
def fake_llama(monkeypatch):
    FakeLlama.responses = deque()
    FakeLlama.created = []
    monkeypatch.setattr(selftime, "LlamaBackend", FakeLlama)
    return FakeLlama


def test_m1_struct_matches_structured_duration_with_zero_latency():
    backend = FakeBackend(TUNED)
    estimator = SelfTimeEstimator(backend)
    estimate = estimator.estimate(TASK, family="FUNC")
    expected = structured_duration(
        330, 3, approx_task_tokens(TASK), "FUNC", CARD, default_priors(), latency_s=0.0
    )
    factor = load_defaults()["struct_conformal_factor"]
    assert estimate.method == "m1-struct"
    assert estimate.seconds == pytest.approx(expected)
    assert estimate.low == pytest.approx(expected / factor)
    assert estimate.high == pytest.approx(expected * factor)
    assert (estimate.tokens, estimate.attempts, estimate.verbal_seconds) == (330, 3, 23)
    assert estimate.card == CARD.as_dict() and estimate.family == "FUNC"
    assert estimate.raw_text == TUNED
    assert backend.calls == [
        estimate_messages({"task_turn": task_turn(TASK)}, CARD, estimator="agent")
    ]


def test_speed_card_is_measured_once_or_taken_as_given():
    backend = FakeBackend(TUNED, TUNED)
    estimator = SelfTimeEstimator(backend)
    estimator.estimate(TASK)
    estimator.estimate(TASK)
    assert backend.calibrations == 1
    explicit = SelfTimeEstimator(GenerateOnly(TUNED), SpeedCard("given", 50, 900))
    assert explicit.estimate(TASK).card == {"id": "given", "S": 50.0, "P": 900.0}
    with pytest.raises(ValueError, match="SpeedCard"):
        SelfTimeEstimator(GenerateOnly())


def test_task_tokens_override_and_latency_are_used():
    estimator = SelfTimeEstimator(FakeBackend(TUNED), latency_s=0.5)
    estimate = estimator.estimate(TASK, family="LIB", task_tokens=1000)
    expected = structured_duration(330, 3, 1000, "LIB", CARD, default_priors(), latency_s=0.5)
    assert estimate.seconds == pytest.approx(expected)


def test_verbal_fallback_when_tokens_are_missing():
    estimator = SelfTimeEstimator(FakeBackend("About 40 s (likely 30–60 s)."))
    estimate = estimator.estimate(TASK)
    factor = load_defaults()["verbal_conformal_factor"]
    assert estimate.method == "verbal-fallback"
    assert estimate.tokens is None
    assert estimate.seconds == estimate.verbal_seconds == 40
    assert estimate.low == pytest.approx(30 / factor)
    assert estimate.high == pytest.approx(60 * factor)
    assert "verbal estimate" in estimate.summary()


def test_unparseable_answer_raises():
    with pytest.raises(EstimateUnavailable):
        SelfTimeEstimator(FakeBackend("I cannot say.")).estimate(TASK)
    with pytest.raises(ValueError, match="empty"):
        SelfTimeEstimator(FakeBackend()).estimate("   ")


def test_family_is_inferred_or_validated():
    estimator = SelfTimeEstimator(FakeBackend(TUNED, TUNED, TUNED))
    assert estimator.estimate("Read n from stdin and print n squared.").family == "ALGO_IO"
    assert estimator.estimate("```python\nimport numpy as np\n```").family == "LIB"
    assert estimator.estimate(TASK).family == "FUNC"
    with pytest.raises(ValueError, match="Unknown family"):
        estimator.estimate(TASK, family="PROJ")


def test_tripwire_scales_the_upper_bound():
    estimator = SelfTimeEstimator(FakeBackend(TUNED))
    estimate = estimator.estimate(TASK)
    assert estimator.tripwire_seconds(estimate) == pytest.approx(1.5 * estimate.high)
    assert estimator.tripwire_seconds(estimate, k=1.0) == estimate.high
    for k in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            estimator.tripwire_seconds(estimate, k=k)


def test_human_estimate_uses_the_human_frame():
    backend = FakeBackend(HUMAN)
    human = SelfTimeEstimator(backend).human_estimate(TASK)
    assert human.seconds == 7200 and human.raw_text == HUMAN
    assert human.summary() == "~2.0 h for a human developer"
    task = {"task_turn": task_turn(TASK)}
    assert backend.calls == [estimate_messages(task, CARD, template_id="h09", estimator="agent")]


def test_human_answer_in_own_time_format_gives_no_number():
    human = SelfTimeEstimator(FakeBackend(TUNED)).human_estimate(TASK)
    assert human.seconds is None and human.own_time_format and human.raw_text == TUNED
    assert "answered with its own time" in human.summary()


def test_summary_reads_plainly():
    estimate = SelfTimeEstimator(FakeBackend(TUNED)).estimate(TASK)
    text = estimate.summary()
    assert text.startswith("~") and "on this machine (80% range" in text
    assert "~330 output tokens over 3 attempts" in text
    assert json.loads(json.dumps(estimate.as_dict()))["method"] == "m1-struct"


def test_packaged_defaults_match_results():
    defaults = load_defaults()
    priors = json.loads((ROOT / "results/workload-priors.json").read_text())
    calibration = json.loads((ROOT / "results/extra-analyses.json").read_text())[
        "conformal_calibration"
    ]
    assert defaults["workload_priors"] == priors
    assert defaults["struct_conformal_factor"] == calibration["struct_factor"]
    assert defaults["verbal_conformal_factor"] == calibration["cqr_factor"]


def test_from_llama_server_requires_local_server_and_speed():
    for url in ("http://example.com:8082", "https://127.0.0.1:8082", "http://127.0.0.1"):
        with pytest.raises(ValueError, match="local llama.cpp"):
            SelfTimeEstimator.from_llama_server(url, speed=(20, 200))
    with pytest.raises(ValueError, match="speed"):
        SelfTimeEstimator.from_llama_server("http://127.0.0.1:8082")
    estimator = SelfTimeEstimator.from_llama_server("http://127.0.0.1:8082", speed=(20, 200))
    assert estimator.speed_card() == SpeedCard("manual", 20, 200)
    assert estimator.latency_s == 0.0


def test_cli_estimate_json(fake_llama, capsys, tmp_path):
    fake_llama.responses.extend([TUNED, HUMAN])
    task = tmp_path / "task.md"
    task.write_text(TASK)
    main(["estimate", "--task", str(task), "--speed", "20,200", "--human", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["self"]["method"] == "m1-struct"
    assert payload["self"]["card"] == {"id": "manual", "S": 20.0, "P": 200.0}
    assert payload["human"]["seconds"] == 7200
    url = "http://127.0.0.1:8082"
    assert fake_llama.created == [(url, url, None, None, 8, 0)]


def test_cli_estimate_text_measures_speed(fake_llama, capsys):
    fake_llama.responses.append(TUNED)
    main(
        [
            "estimate",
            "--text",
            TASK,
            "--llama-bench",
            "/opt/llama-bench",
            "--gguf",
            "m.gguf",
            "--gpu-layers",
            "99",
            "--family",
            "LIB",
        ]
    )
    output = capsys.readouterr().out
    assert output.startswith("~") and "on this machine" in output
    assert fake_llama.created[0][2:] == ("m.gguf", "/opt/llama-bench", 8, 99)


def test_cli_estimate_errors(fake_llama, capsys):
    with pytest.raises(SystemExit, match="--speed S,P"):
        main(["estimate", "--text", TASK])
    for bad in ("20", "20,x", "0,200"):
        with pytest.raises(SystemExit):
            main(["estimate", "--text", TASK, "--speed", bad])
        assert "argument --speed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["estimate", "--speed", "20,200"])


def test_cli_speedcard(fake_llama, capsys):
    main(["speedcard", "--llama-bench", "/opt/llama-bench", "--gguf", "m.gguf"])
    assert json.loads(capsys.readouterr().out) == {"id": "demo-llamacpp", "S": 20.0, "P": 200.0}
    with pytest.raises(SystemExit):
        main(["speedcard", "--gguf", "m.gguf"])


def load_example():
    path = ROOT / "examples/agent_with_tripwire.py"
    spec = importlib.util.spec_from_file_location("agent_with_tripwire", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def python_command(code):
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_example_tripwire_stops_slow_runs():
    example = load_example()
    fast = example.run_with_tripwire(python_command("import sys; sys.stdin.read()"), TASK, 30)
    assert not fast.tripped and fast.returncode == 0
    slow = example.run_with_tripwire(python_command("import time; time.sleep(30)"), TASK, 0.3)
    assert slow.tripped and slow.returncode is None and slow.elapsed_s < 10


def test_example_end_to_end(fake_llama, capsys, tmp_path):
    fake_llama.responses.append(TUNED)
    task = tmp_path / "task.md"
    task.write_text(TASK)
    command = python_command("import sys; assert 'add(a, b)' in sys.stdin.read()")
    code = load_example().main(["--task", str(task), "--speed", "20,200", "--cmd", command])
    output = capsys.readouterr().out
    assert code == 0
    assert "Tripwire: stop after" in output and "Finished with exit code 0" in output
