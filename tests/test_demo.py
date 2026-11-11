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


@pytest.mark.release_data
def test_workbench_engine_redaction_and_zero_execution_safety():
    from anthropochrony.workbench import COHORTS, WorkbenchEngine

    root = SPEC.parent.parent
    if not (root / "data/eval/estimates.jsonl").exists():
        pytest.skip(
            "needs the released research data; run `uv run python scripts/fetch_release.py`"
        )
    engine = WorkbenchEngine(root)
    # 1) Showcase & OOD CLASS exploration
    p_md, table_md, fig = engine.explore_task(COHORTS[0], "PROJ8-todo-cli", "REF")
    assert "PROJ8-todo-cli" in p_md
    assert "Real Desktop CPU Showcase" in table_md
    assert fig is not None

    ood_opts = engine.cohort_choices(COHORTS[3])
    assert ood_opts
    _, ood_table, _ = engine.explore_task(COHORTS[3], ood_opts[0][1], "extrap-low")
    assert "M1-Struct" in ood_table
    # 2) Restricted source statement redaction check
    restricted_ids = [
        tid
        for tid, t in engine.tasks_by_id.items()
        if t.get("source") in {"newfacade/LeetCodeDataset", "deepmind/code_contests", "BAAI/TACO"}
    ]
    if restricted_ids:
        redacted_md = engine.safe_prompt_markdown(restricted_ids[0])
        assert "Statement redacted for redistribution license compliance" in redacted_md

    # 3) Tab 2 Live simulator parses hostile code statically without ever executing it
    hostile_prompt = (
        "Task:\n```python\nimport os\nos.kill(1, 9)\n"
        "class Trap:\n    def boom(self):\n        raise RuntimeError('never run')\n```\n"
    )
    fast_md, _ = engine.simulate_prompt(hostile_prompt, 400.0, 40000.0, 25.0, 15.0)
    slow_md, _ = engine.simulate_prompt(hostile_prompt, 8.0, 400.0, 25.0, 15.0)
    assert "CLASS" in fast_md
    assert "80% Split-Conformal SLA Interval" in fast_md
    assert fast_md != slow_md

    # 4) Tab 3 empirical fleet ROI calculator
    roi_md, roi_fig = engine.compute_fleet_roi(250.0, 15.0, 1.0, 0.25, 1.80, 2.50)
    assert "M1-Struct+CP" in roi_md
    assert "DeepSeek-V4-Pro" in roi_md
    assert roi_fig is not None


def test_gradio_full_workbench_builds_on_root():
    pytest.importorskip("gradio")
    root = SPEC.parent.parent
    functions = runpy.run_path(str(root / "space/app.py"))
    interface = functions["app"](root=root)
    config = interface.get_config_file()
    assert "Anthropochrony" in config["title"]
    assert not config.get("analytics_enabled")
    assert len(config["dependencies"]) >= 8
