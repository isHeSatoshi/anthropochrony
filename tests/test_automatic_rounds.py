import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from transformers import AutoTokenizer

from anthropochrony.artifacts import file_hash, read_jsonl, write_json, write_jsonl
from anthropochrony.gates import GateError, record_gate
from anthropochrony.measure import automatic, remote
from anthropochrony.measure.rounds import Measurement
from anthropochrony.prompts import SPEC, task_turn
from anthropochrony.sandbox import Sandbox
from anthropochrony.timemodel import SpeedCard

ROOT = SPEC.parent.parent


def fixture_config():
    return yaml.safe_load((ROOT / "configs/measure.yaml").read_text()) | {"k": 2}


@pytest.mark.security
def test_automatic_real_rounds_namespace_ids_and_resume_partial_checkpoint(tmp_path, monkeypatch):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "pins/tiny-aya-global", local_files_only=True)
    task = {
        "task_id": "synthetic-automatic-only",
        "family": "FUNC",
        "split": "val",
        "task_turn": task_turn("Implement f(x): return x + 1"),
        "harness": "evalplus_check",
        "test_code": "assert f(3) == 4",
        "time_limits": {"per_task_s": 5},
    }
    experiments = {
        name: Measurement(tmp_path / name, [task], fixture_config(), SpeedCard(name, 35, 5000))
        for name in ("ref", "slow")
    }
    calls, executing, crashed = [], [], [False]

    def generator_factory(directory):
        def generate_file(requests):
            # A remote output cache is reused, not a second paid submission.
            output = directory / "generation.jsonl"
            if not output.exists():
                rows = read_jsonl(requests)
                calls.append([row["request_id"] for row in rows])
                write_jsonl(
                    output,
                    [
                        {
                            "request_id": row["request_id"],
                            "text": "```python\ndef f(x): return x + 1\n```",
                            "prompt_tokens_total": len(row["prompt_token_ids"]),
                            "o_tokens": 22,
                            "finish_reason": "stop",
                            "stop_reason": 261001,
                        }
                        for row in rows
                    ],
                )
            # generate_file returns only after the simulated VM is terminal.
            executing.append("gpu-terminal")
            return read_jsonl(output)

        return SimpleNamespace(generate_file=generate_file)

    actual_apply = automatic.apply_round

    def crash_second_condition(*args):
        executing.append("cpu")
        assert "gpu-terminal" in executing
        if len(executing) == 3 and not crashed[0]:
            crashed[0] = True
            raise RuntimeError("fixture interruption after first checkpoint")
        return actual_apply(*args)

    monkeypatch.setattr(automatic, "apply_round", crash_second_condition)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        automatic.run_experiments(
            experiments, tokenizer, Sandbox(), tmp_path / "batches", generator_factory
        )
    assert all(s["done"] for s in experiments["ref"].states())
    assert not any(s["done"] for s in experiments["slow"].states())
    # Simulate a download/split interruption as well. Existing full batch must
    # satisfy the remaining condition without requiring the finished condition.
    generation = experiments["slow"].directory / "generation-r1.jsonl"
    generation.unlink()
    Path(f"{generation}.manifest.json").unlink()
    result = automatic.run_experiments(
        experiments, tokenizer, Sandbox(), tmp_path / "batches", generator_factory
    )
    assert len(calls) == 1 and len(set(calls[0])) == 4
    assert all(any(key.startswith(f"{name}:") for key in calls[0]) for name in experiments)
    assert all(s["passed"] and s["n_attempts"] == 1 for rows in result.values() for s in rows)
    automatic.run_experiments(
        experiments, tokenizer, Sandbox(), tmp_path / "batches", generator_factory
    )
    assert len(calls) == 1


def test_automatic_cached_individual_output_rejects_changed_content(tmp_path, monkeypatch):
    experiment = SimpleNamespace(
        directory=tmp_path / "ref",
        tasks=[],
        config={"workers": 1},
        states=lambda: [],
        save=lambda states: None,
    )
    experiment.directory.mkdir()
    requests = [{"request_id": "fixture"}]
    request = experiment.directory / "requests-r1.jsonl"
    output = experiment.directory / "generation-r1.jsonl"
    write_jsonl(request, requests, config=experiment.config)
    write_jsonl(output, [{"request_id": "fixture"}], inputs=[request], config=experiment.config)
    output.write_text('{"request_id":"tampered"}\n')
    monkeypatch.setattr(automatic, "prepare_round", lambda *a: requests)
    with pytest.raises(ValueError, match="content hashes"):
        automatic.run_experiments(
            {"ref": experiment},
            None,
            None,
            tmp_path / "batches",
            lambda *a: pytest.fail("Cached output must not launch a new job"),
        )


def remote_fixture(tmp_path):
    config = fixture_config()
    pins = {
        "subject": {"repo": config["model_id"], "revision": config["model_sha"]},
        "generation_image": "fixture-image",
    }
    for name in remote.GENERATION_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pins) if name == "configs/pins.yaml" else "fixture source\n")
    for gate in ("G0-security", "G1", "G2"):
        record_gate(tmp_path, gate, {"fixture": True})
    instance = remote.HFRemoteRound(tmp_path, tmp_path / "batch", config)
    requests = instance.directory / "requests.jsonl"
    write_jsonl(requests, [{"request_id": "fixture"}])
    return instance, requests


def write_remote_outputs(instance, directory, request_sha):
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "generation.jsonl"
    record = write_jsonl(output, [{"request_id": "fixture"}], config=instance.config)
    record["input_hashes"] = {"/code/requests.jsonl": request_sha}
    record["source_identity"] = {"files": instance._source_hashes()}
    Path(f"{output}.manifest.json").write_text(json.dumps(record))
    write_json(directory / "generation.gpu.json", [])
    write_json(directory / "generation.performance.json", {"fixture": True})


@pytest.mark.parametrize("change", ["none", "requests", "source", "telemetry", "missing-telemetry"])
def test_remote_complete_cache_is_hash_checked_before_any_api(tmp_path, monkeypatch, change):
    instance, requests = remote_fixture(tmp_path)
    write_remote_outputs(instance, instance.directory, file_hash(requests))
    monkeypatch.setattr(remote, "HfApi", lambda: pytest.fail("Complete cache cannot access HF"))
    if change == "requests":
        requests.write_text('{"request_id":"changed"}\n')
    elif change == "source":
        (tmp_path / remote.GENERATION_SOURCES[0]).write_text("changed source")
    elif change == "telemetry":
        (instance.directory / "generation.gpu.json").write_text("[1]")
    elif change == "missing-telemetry":
        (instance.directory / "generation.gpu.json").unlink()
    if change == "none":
        assert instance.generate_file(requests) == [{"request_id": "fixture"}]
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            instance.generate_file(requests)


def test_remote_download_failure_never_commits_partial_cache(tmp_path):
    instance, requests = remote_fixture(tmp_path)

    def download(bucket_id, paths, **kwargs):
        directory = Path(paths[0][1]).parent
        write_remote_outputs(instance, directory, file_hash(requests))
        (directory / "generation.gpu.json").write_text("tampered")

    with pytest.raises(ValueError, match="content hash"):
        instance._download(
            SimpleNamespace(download_bucket_files=download), "fixture", file_hash(requests)
        )
    assert not (instance.directory / "generation.jsonl").exists()
    assert not list(instance.directory.glob(".download-*"))


def test_remote_denies_fine_tuned_identity_and_missing_gate_before_submission(
    tmp_path, monkeypatch
):
    instance, requests = remote_fixture(tmp_path)
    (tmp_path / "results/gates/G2.json").unlink()
    monkeypatch.setattr(remote, "launch", lambda *a, **kw: pytest.fail("G2 must block paid launch"))
    with pytest.raises(GateError, match="G2"):
        instance.generate_file(requests)
    changed = instance.config | {"model_sha": "fine-tuned"}
    with pytest.raises(ValueError, match="pinned base"):
        remote.HFRemoteRound(tmp_path, tmp_path / "other", changed)


def test_quality_gate_cannot_pass_g3_without_independent_invariance(tmp_path):
    from anthropochrony.workflow import measurement_gate

    tasks = [{"task_id": str(i), "family": "FUNC"} for i in range(100)]
    states = [
        {
            "task_id": task["task_id"],
            "done": True,
            "first_pass": True,
            "attempts": [{"finish_reason": "stop", "exec": {"status": "pass"}}],
        }
        for task in tasks
        for _ in range(8)
    ]
    measurement_gate(tmp_path, states, tasks, require_invariance=False)
    assert not (tmp_path / "results/gates/G3.json").exists()
    assert json.loads((tmp_path / "results/gates/G3-base-pilot-quality.json").read_text())["passed"]
    with pytest.raises(GateError, match="speed_card_invariance"):
        measurement_gate(tmp_path, states, tasks)


def test_production_outputs_are_saved_before_post_generation_statistics_fail(monkeypatch):
    from anthropochrony.gen import performance

    class Monitor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    calls, saved = [0], []

    def preemptions(llm):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("fixture statistics failure")
        return 0

    monkeypatch.setattr(performance, "GPUMonitor", Monitor)
    monkeypatch.setattr(performance, "preemptions", preemptions)
    engine = SimpleNamespace(llm=None, generate=lambda *a, **kw: [{"text": "preserved"}])
    with pytest.raises(RuntimeError, match="statistics failure"):
        performance.measured_generate(engine, [], output_sink=saved.extend)
    assert saved == [{"text": "preserved"}]


@pytest.mark.parametrize("outcome", ["COMPLETED", "ERROR", "401", "403"])
def test_remote_job_resumes_without_double_payment_and_stops_on_denial(
    tmp_path, monkeypatch, outcome
):
    from anthropochrony.budget import Budget

    instance, requests = remote_fixture(tmp_path)
    events, launches = [], []

    def launch(root, name, **kwargs):
        launches.append(kwargs)
        Budget(root).reserve(name, 0.6)
        Budget(root).update(name, state="launched", job_id="fixture-job")
        write_json(root / f"artifacts/jobs/{name}.json", {"job_id": "fixture-job"})

    class API:
        def inspect_job(self, **kwargs):
            events.append("inspect")
            if outcome in {"401", "403"}:
                raise PermissionError(f"{outcome}; renew access")
            return SimpleNamespace(status=SimpleNamespace(stage=outcome))

        def cancel_job(self, **kwargs):
            events.append("cancel")

        def download_bucket_files(self, bucket, paths, **kwargs):
            events.append("download")
            assert events[-2] == "inspect" and outcome == "COMPLETED"
            assert kwargs["raise_on_missing_files"] is True and len(paths) == 6
            write_remote_outputs(instance, Path(paths[0][1]).parent, file_hash(requests))

    monkeypatch.setattr(remote, "launch", launch)
    monkeypatch.setattr(remote, "payload", lambda *args, **kwargs: tmp_path / "staged")
    monkeypatch.setattr(remote, "HfApi", API)
    if outcome == "COMPLETED":
        assert instance.generate_file(requests) == [{"request_id": "fixture"}]
        assert instance.generate_file(requests) == [{"request_id": "fixture"}]
        assert events == ["inspect", "download"]
    else:
        exception = PermissionError if outcome in {"401", "403"} else RuntimeError
        with pytest.raises(exception):
            instance.generate_file(requests)
        if outcome == "ERROR":
            # Reinspection is allowed, but never another reservation/submission.
            with pytest.raises(RuntimeError):
                instance.generate_file(requests)
            assert events == ["inspect", "inspect"]
        else:
            assert events == ["inspect", "cancel"]
    assert len(launches) == 1
    assert launches[0]["flavor"] == "l40sx1"
    assert launches[0]["secret"] is False
    assert Budget(tmp_path).total() == 0.6


def test_automatic_cli_cannot_preview_through_missing_g2(tmp_path):
    from anthropochrony.cli import main, parser

    args = parser().parse_args(["measure", "--auto", "--dry-run"])
    assert args.auto and args.phase == "pilot"
    record_gate(tmp_path, "G0-security", {"fixture": True})
    record_gate(tmp_path, "G1", {"fixture": True})
    with pytest.raises(SystemExit, match="G2"):
        main(["--root", str(tmp_path), "measure", "--auto", "--dry-run"])


@pytest.mark.security
def test_retry_after_overlong_sampled_output_counts_only_reusable_prefix(tmp_path):
    """Degenerate length-capped text can re-tokenize shorter than it was sampled."""
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "pins/tiny-aya-global", local_files_only=True)
    task = {
        "task_id": "synthetic-retokenize",
        "family": "FUNC",
        "split": "val",
        "task_turn": task_turn("Implement f(x): return x + 1"),
        "harness": "evalplus_check",
        "test_code": "assert f(3) == 4",
        "time_limits": {"per_task_s": 5},
    }
    experiment = Measurement(tmp_path / "m", [task], fixture_config(), SpeedCard("c", 35, 5000))

    def generator_factory(directory):
        def generate_file(requests):
            return [
                {
                    "request_id": row["request_id"],
                    "text": "```python\ndef f(x): return x\n```",
                    "prompt_tokens_total": len(row["prompt_token_ids"]),
                    # Far more sampled tokens than the text can re-tokenize to.
                    "o_tokens": 400,
                    "finish_reason": "length",
                    "stop_reason": None,
                }
                for row in read_jsonl(requests)
            ]

        return SimpleNamespace(generate_file=generate_file)

    states = automatic.run_experiments(
        {"m": experiment}, tokenizer, Sandbox(), tmp_path / "b", generator_factory
    )["m"]
    attempts = states[0]["attempts"]
    assert len(attempts) == 3
    for previous, current in zip(attempts, attempts[1:], strict=False):
        assert current["u_tokens"] > 0
        assert current["u_tokens"] > current["prompt_tokens_total"] - (
            previous["prompt_tokens_total"] + previous["o_tokens"]
        )
