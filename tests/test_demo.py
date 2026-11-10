import copy
import runpy

import pytest

from anthropochrony.demo import BackendUnavailable, ContextExhausted, Demo, LlamaBackend
from anthropochrony.prompts import SPEC, task_turn
from anthropochrony.sandbox import Sandbox
from anthropochrony.timemodel import SpeedCard


class FixtureBackend:
    def __init__(self):
        self.calls = []
        self.coding_calls = 0

    def calibrate(self):
        return SpeedCard("fixture", 10, 100)

    def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        if kwargs.get("coding"):
            self.coding_calls += 1
            return "```python\ndef f(x): return x + " + str(self.coding_calls - 1) + "\n```"
        return "Fixture estimate, not research data"


def fixture_task():
    return {
        "task_id": "synthetic-only",
        "family": "FUNC",
        "gt": {"passes": True},
        "task_turn": task_turn("Implement f(x), adding one."),
        "harness": "evalplus_check",
        "test_code": "assert f(3) == 4",
        "time_limits": {"per_task_s": 3},
    }


def test_demo_without_backend_does_not_invent_results():
    with pytest.raises(BackendUnavailable):
        Demo([fixture_task()]).estimate("synthetic-only")


@pytest.mark.security
def test_demo_separate_estimates_real_sandbox_feedback_and_allowlist():
    backend = FixtureBackend()
    demo = Demo([fixture_task()], backend, Sandbox())
    output = demo.estimate("synthetic-only")
    assert len(backend.calls) == 3 and output["card"]["S"] == 10
    assert all(len(messages) == 2 for messages, _ in backend.calls)
    with pytest.raises(ValueError, match="allowlisted"):
        demo.run("synthetic-only", pasted="untrusted pasted input")
    with pytest.raises(ValueError, match="allowlisted"):
        demo.run("unknown")
    result = demo.run("synthetic-only", card=SpeedCard("session", 20, 200))
    assert result["passed"] and len(result["attempts"]) == 2
    assert backend.calls[-1][1]["coding"] and backend.calls[-1][1]["max_tokens"] == 2048
    assert len(backend.calls[-1][0]) == 4
    assert "Fixture estimate" not in str(backend.calls[-1][0])


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:80@outside.example",
        "http://outside.example:8080",
        "https://localhost:8080",
        "http://localhost:8080/private",
        "http://localhost:8080?x=1",
    ],
)
def test_demo_rejects_nonlocal_backend_or_url_credentials(url):
    with pytest.raises(ValueError):
        LlamaBackend(url, "http://localhost:8081", "model.gguf", "llama-bench")


def test_demo_context_cap_and_sampling(monkeypatch):
    backend = LlamaBackend("http://localhost:8080", "http://localhost:8081", "m", "bench")
    calls = []

    def request(url, path, data):
        calls.append((path, data))
        if path == "/apply-template":
            return {"prompt": "fixture"}
        if path == "/tokenize":
            return {"tokens": [1] * 7800}
        return {"choices": [{"message": {"content": "fixture"}}]}

    monkeypatch.setattr(backend, "request", request)
    assert backend.generate([], max_tokens=2048, coding=True) == "fixture"
    assert calls[-1][1]["max_tokens"] == 392
    assert calls[-1][1]["temperature"] == 0.1 and calls[-1][1]["top_p"] == 0.95
    monkeypatch.setattr(
        backend,
        "request",
        lambda url, path, data: (
            {"prompt": "fixture"} if path == "/apply-template" else {"tokens": [1] * 8000}
        ),
    )
    with pytest.raises(ContextExhausted):
        backend.generate([])


def test_gradio_builds_without_backend(tmp_path):
    pytest.importorskip("gradio")
    functions = runpy.run_path(str(SPEC.parent.parent / "space/app.py"))
    interface = functions["app"](root=tmp_path)
    config = interface.get_config_file()
    assert "Anthropochrony" in config["title"]
    assert len(config["dependencies"]) == 2
    assert not config.get("analytics_enabled")
