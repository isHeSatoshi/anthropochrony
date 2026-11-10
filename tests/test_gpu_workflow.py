import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anthropochrony.gen.performance import select_batch_config, summarize_gpu
from anthropochrony.measure import validation
from anthropochrony.timemodel import SpeedCard, completion_time, trajectory_time


def test_telemetry_excludes_engine_loading_and_post_generation_idle():
    samples = [
        {
            "monotonic_s": t,
            "gpu_utilization_pct": value,
            "memory_utilization_pct": 80,
            "memory_used_mb": 44000,
            "memory_total_mb": 46068,
            "power_w": 250,
        }
        for t, value in ((0, 5), (1, 90), (2, 100), (3, 0))
    ]
    result = summarize_gpu(samples, 0.5, 2.5)
    assert result["samples"] == 2
    assert result["mean_gpu_utilization_pct"] == 95
    assert result["peak_memory_used_mb"] == 44000
    assert summarize_gpu(samples, 4, 5)["mean_gpu_utilization_pct"] is None


def test_batch_selection_uses_measured_token_rate_and_no_preemptions():
    def candidate(rate, preemptions, samples=10):
        return {"tokens_per_second": rate, "preemptions": preemptions, "gpu": {"samples": samples}}

    best = candidate(5000, 0)
    assert (
        select_batch_config(
            [
                candidate(8000, 2),
                candidate(9000, None),
                candidate(7000, 0, samples=0),
                best,
                candidate(4000, 0),
            ]
        )
        is best
    )
    with pytest.raises(ValueError, match="No measured"):
        select_batch_config([candidate(9000, None)])


def test_preemption_counter_does_not_read_the_vllm_histogram_as_a_counter():
    from anthropochrony.gen.performance import preemptions

    # Shapes/names from vLLM v0.33.0 v1/metrics/{reader,loggers}.py.
    histogram = SimpleNamespace(
        name="vllm:request_num_preemptions", count=800, sum=3, buckets={"+Inf": 800}
    )
    counters = [
        SimpleNamespace(name="vllm:num_preemptions", value=value, labels={"engine": str(i)})
        for i, value in enumerate((1, 2))
    ]
    llm = SimpleNamespace(get_metrics=lambda: counters + [histogram])
    assert preemptions(llm) == 3
    counters[0].value = counters[1].value = 0
    assert preemptions(llm) == 0
    llm.get_metrics = lambda: [histogram]
    assert preemptions(llm) is None


def test_latency_fallback_stays_in_the_single_time_formula():
    card = SpeedCard("fixture", 1, 1)
    records = []
    for tokens in (1, 2, 4, 8):
        trace = {"attempts": [{"u_tokens": 0, "o_tokens": tokens, "exec": {"wall_s": 0}}]}
        records.append(
            {"actual_wall_s": tokens + 1, "trace": trace, "calls": [{"rpc_wall_s": tokens + 1}]}
        )
    assert validation.fit_latency(records, card) == 1
    raw = validation.validation_metrics(records, card)
    assert raw["median_absolute_relative_error"] == pytest.approx((1 / 3 + 1 / 5) / 2)
    corrected = validation.validation_metrics(records, card, latency_s=1)
    assert corrected["median_absolute_relative_error"] == 0
    assert corrected["log_time_r2"] == 1
    assert completion_time(2, 3, 4, card, latency_s=0.5, calls=3) == 10.5
    assert (
        trajectory_time({"attempts": records[0]["trace"]["attempts"] * 2}, card, latency_s=1) == 4
    )
    with pytest.raises(ValueError):
        completion_time(1, 1, 1, card, latency_s=-1)


def test_observer_times_real_request_to_final_test_not_reconstructed_sums(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(validation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(validation, "initial_states", lambda *a: [{"messages": [], "done": False}])
    monkeypatch.setattr(validation, "prepare_round", lambda *a: [{"request_id": "fixture"}])

    def apply(states, *args, **kwargs):
        clock[0] += 3
        states[0].update(
            done=True, attempts=[{"u_tokens": 0, "o_tokens": 2, "exec": {"wall_s": 3}}]
        )
        return states, [{"text": "```python\ndef f(): return 1\n```"}]

    def request(message):
        clock[0] += 2
        return {"outputs": [], "performance": {"elapsed_s": 1.5}}

    monkeypatch.setattr(validation, "apply_round", apply)
    records, replays = validation.observe_sequential(
        [{"task_id": "fixture", "family": "FUNC"}],
        None,
        {},
        SpeedCard("fixture", 1, 1),
        SimpleNamespace(request=request),
        None,
        tmp_path,
    )
    assert records[0]["actual_wall_s"] == 5
    assert records[0]["calls"][0]["rpc_wall_s"] == 2
    assert replays[0]["inline_wall_s"] == 3


def test_authenticated_job_keeps_hf_token_local_and_reserves_hard_timeout(tmp_path, monkeypatch):
    from anthropochrony.budget import Budget
    from anthropochrony.gen import session

    captured = {}
    job = SimpleNamespace(id="fixture-job", owner=SimpleNamespace(name="fixture"))
    monkeypatch.setattr(session, "get_token", lambda: "FAKE_TEST_NOT_A_TOKEN")
    monkeypatch.setattr(session, "sync_bucket", lambda *a, **k: None)

    def bootstrap(*args, **kwargs):
        captured["bootstrap"] = kwargs
        return ["pinned-server"], {}, {"SBX_TOKEN": "FAKE_DERIVED_CAPABILITY"}, kwargs["volumes"]

    def run_job(**kwargs):
        captured["job"] = kwargs
        return job

    monkeypatch.setattr(session.protocol, "_bootstrap_job_spec", bootstrap)
    monkeypatch.setattr(session, "HfApi", lambda: SimpleNamespace(run_job=run_job))
    monkeypatch.setattr(session, "wait_for_allocation", lambda api, job, **kwargs: job)
    server = SimpleNamespace(
        wait_ready=lambda t: None,
        close=lambda: None,
        _client=SimpleNamespace(event_hooks={}),
    )
    monkeypatch.setattr(session.protocol._SandboxServer, "from_job", lambda **kwargs: server)
    monkeypatch.setattr(session, "Sandbox", lambda **kwargs: SimpleNamespace(**kwargs))
    result, connection = session.launch_session(tmp_path, "fixture", "pinned-image", tmp_path)
    assert result is job
    assert connection.owns_sandbox is False
    assert captured["bootstrap"]["forward_hf_token"] is False
    assert "HF_TOKEN" not in captured["job"]["secrets"]
    assert captured["job"]["attempts"] == 1
    assert captured["job"]["timeout"] == 7200
    assert "expose_public" not in captured["job"]
    assert captured["job"]["volumes"][1].path == "code/fixture"
    assert captured["job"]["volumes"][1].read_only is True
    assert Budget(tmp_path).total() == 3.62
    assert server._client.event_hooks["response"] == [session.check_authorization]


@pytest.mark.parametrize("status", [401, 403])
def test_session_stops_on_first_authorization_failure(status):
    from anthropochrony.gen.session import check_authorization

    with pytest.raises(PermissionError, match="renew access"):
        check_authorization(SimpleNamespace(status_code=status))


def test_hardware_queue_has_a_separate_bounded_startup_window(monkeypatch):
    from anthropochrony.gen import session

    clock = [0]
    monkeypatch.setattr(
        session,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    job = SimpleNamespace(id="fixture", owner=SimpleNamespace(name="fixture"))
    running = SimpleNamespace(status=SimpleNamespace(stage="RUNNING"))
    statuses = iter([SimpleNamespace(status=SimpleNamespace(stage="SCHEDULING"))] * 50 + [running])
    api = SimpleNamespace(inspect_job=lambda **kwargs: next(statuses))
    assert session.wait_for_allocation(api, job, timeout_s=1200) is running
    assert clock[0] == 750
    api = SimpleNamespace(
        inspect_job=lambda **kwargs: SimpleNamespace(status=SimpleNamespace(stage="SCHEDULING"))
    )
    with pytest.raises(TimeoutError):
        session.wait_for_allocation(api, job, timeout_s=30)
    assert clock[0] == 780


def test_job_reservation_includes_port_fee_and_partial_billable_minutes():
    from anthropochrony.budget import job_cost

    assert job_cost("l40sx1", 7200, exposed_ports=True) == 3.62
    assert job_cost("l40sx1", 61) == 0.06


@pytest.mark.parametrize("change", ["source", "staged", "unexpected"])
def test_immutable_payload_can_be_reused_but_never_silently_overwritten(tmp_path, change):
    from anthropochrony.jobs import payload

    root, target = tmp_path / "source", tmp_path / "staged"
    (root / "src").mkdir(parents=True)
    original = root / "src/module.py"
    original.write_text("fixture = 1\n")
    payload(root, target)
    before = (target / "SOURCE_IDENTITY.json").stat().st_mtime_ns
    assert payload(root, target) == target
    assert (target / "SOURCE_IDENTITY.json").stat().st_mtime_ns == before
    if change == "source":
        original.write_text("fixture = 2\n")
    elif change == "staged":
        (target / "src/module.py").write_text("fixture = 3\n")
    else:
        (target / "unexpected-user-file").write_text("Preserve this user file.")
    with pytest.raises(FileExistsError, match="changed"):
        payload(root, target)
    if change == "unexpected":
        assert (target / "unexpected-user-file").read_text() == "Preserve this user file."


@pytest.mark.parametrize("error", ["Unconfirmed transport failure", "HTTP 401", "HTTP 403"])
def test_failed_submission_does_not_claim_zero_actual_billing(tmp_path, monkeypatch, error):
    from anthropochrony import jobs
    from anthropochrony.budget import Budget

    monkeypatch.setattr(
        jobs.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr=error)
    )
    exception = PermissionError if "HTTP" in error else RuntimeError
    with pytest.raises(exception):
        jobs.launch(
            tmp_path,
            "fixture-failed-launch",
            flavor="l40sx1",
            timeout_s=1200,
            image="fixture-image",
            command=["fixture-command"],
        )
    assert Budget(tmp_path).total() == 0.6
    ledger = json.loads((tmp_path / "artifacts/budget.json").read_text())["entries"][0]
    assert "actual_usd" not in ledger
    assert ledger["state"] == "submission_failed_reservation_retained"
    assert "full reservation is retained" in (tmp_path / "RUNLOG.md").read_text()


def test_estimate_context_budget_preserves_five_shot_prompts_without_batch_failure():
    from anthropochrony.gen.engine import prepare_estimate_requests

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: [2] + [7] * (int(messages[0]["content"]) - 1)
    )
    prompts = [
        {"est_id": str(length), "messages": [{"content": str(length)}], "max_tokens": 512}
        for length in (50, 99, 100, 110)
    ]
    requests, exhausted = prepare_estimate_requests(tokenizer, prompts, context=100)
    assert [row["max_tokens"] for row in requests] == [50, 1]
    assert {row["seed"] for row in requests} == {0}
    assert set(exhausted) == {"100", "110"}
    assert all(row["text"] == "" and row["o_tokens"] == 0 for row in exhausted.values())
    assert all(row["finish_reason"] == "context_exhausted" for row in exhausted.values())
    assert all(len(row["prompt_token_ids"]) == int(row["request_id"]) for row in requests)


def test_production_measurement_keeps_one_batch_and_reports_counter_delta(monkeypatch):
    from anthropochrony.gen import performance

    counters = iter((4, 6))
    monkeypatch.setattr(performance, "preemptions", lambda llm: next(counters))
    monitor = SimpleNamespace(
        samples=[{"fixture": True}],
        errors=[],
        summary=lambda start, end: {"samples": 1, "mean_gpu_utilization_pct": 99},
    )

    class Monitor:
        def __enter__(self):
            return monitor

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(performance, "GPUMonitor", Monitor)
    requests = [{"prompt_token_ids": [1, 2]}] * 800
    calls = []

    def generate(batch, **kwargs):
        calls.append((batch, kwargs))
        return [{"request_id": "fixture"}]

    engine = SimpleNamespace(
        llm=None, generate=generate, last_performance={"tokens_per_second": 5000}
    )
    rows, report, samples = performance.measured_generate(engine, requests, greedy=True)
    assert len(calls) == 1
    assert calls[0] == (requests, {"greedy": True})
    assert rows == [{"request_id": "fixture"}]
    assert report["preemptions"] == 2
    assert report["requests"] == 800
    assert report["prompt_tokens"] == 1600
    assert report["gpu"]["mean_gpu_utilization_pct"] == 99
    assert samples == monitor.samples


def test_vendored_auditkit_runs_actual_callable_without_site_packages(tmp_path):
    from anthropochrony.jobs import payload

    root = Path(__file__).resolve().parents[1]
    target = payload(root, tmp_path / "code", include_auditkit=True)
    provenance = json.loads((target / "vendor/VENDOR_PROVENANCE.json").read_text())
    assert provenance["version"] == "1.1.6"
    assert (target / "vendor/auditkit-1.1.6.dist-info/licenses/LICENSE.md").is_file()
    program = (
        "import sys; "
        f"sys.path[:0] = {[str(target / 'src'), str(target / 'vendor')]!r}; "
        "import auditkit; "
        "from anthropochrony.eval.audit import evaluate; "
        "p=[{'est_id':'fixture','messages':[{'role':'user','content':'Estimate.'}],'T_med_s':2}]; "
        f"r,f=evaluate(p,lambda inputs:['2 s.']*len(inputs),{str(tmp_path / 'evaluation')!r},"
        "model_sha='fixture',code_sha='fixture'); "
        "assert auditkit.__version__=='1.1.6'; assert len(r.predictions)==1; "
        "assert r.predictions[0].raw_output=='2 s.'"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _adoption_fixture(root, latency=0.967):
    directory = root / "data/calibration/obs"
    directory.mkdir(parents=True)
    records = []
    for i in range(100):
        tokens = 1 + i % 10
        trace = {"attempts": [{"u_tokens": 0, "o_tokens": tokens, "exec": {"wall_s": 0}}]}
        records.append(
            {
                "task_id": str(i),
                "family": ("FUNC", "LIB", "ALGO_FN", "ALGO_IO")[i % 4],
                "actual_wall_s": tokens + latency,
                "trace": trace,
                "calls": [{"rpc_wall_s": tokens + latency}],
            }
        )
    from anthropochrony.artifacts import write_json, write_jsonl

    write_jsonl(directory / "observed-loops.jsonl", records)
    write_json(directory / "reference.json", {"reference": {"id": "REF", "S": 1, "P": 1}})
    write_json(directory / "pool-8.json", {"median_pooled_to_inline_ratio": 1.01})
    write_json(
        root / "results/gates/G2.json",
        {"gate": "G2", "passed": False, "checks": {"median": False}, "evidence": {}},
    )


def test_latency_fallback_adoption_preserves_raw_failure_and_applies_same_thresholds(tmp_path):
    from anthropochrony.gates import require_gate

    _adoption_fixture(tmp_path)
    evidence = validation.adopt_latency_fallback(tmp_path, "data/calibration/obs")
    require_gate(tmp_path, "G2")
    assert evidence["raw_metrics"]["median_absolute_relative_error"] > 0.1
    assert evidence["metrics"]["median_absolute_relative_error"] < 1e-3
    assert evidence["cross_validation"]["worst_heldout_log_time_r2"] > 0.99
    raw = json.loads((tmp_path / "results/gates/G2-raw-attempt.json").read_text())
    assert raw["passed"] is False
    with pytest.raises(ValueError, match="already"):
        validation.adopt_latency_fallback(tmp_path, "data/calibration/obs")


def test_latency_fallback_refuses_a_latency_that_is_not_the_central_constant(tmp_path):
    _adoption_fixture(tmp_path, latency=2.0)
    with pytest.raises(ValueError, match="does not match"):
        validation.adopt_latency_fallback(tmp_path, "data/calibration/obs")
