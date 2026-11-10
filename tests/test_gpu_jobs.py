import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anthropochrony import gpu_jobs
from anthropochrony.artifacts import manifest, write_json
from anthropochrony.budget import Budget
from anthropochrony.gates import GateError


def make_root(tmp_path, gates=("G0-security", "G1", "G2")):
    for name in (*gpu_jobs.COMMON_SOURCES, "jobs/fixture.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    (tmp_path / "configs/pins.yaml").write_text('{"generation_image": "fixture-image"}')
    for gate in gates:
        write_json(tmp_path / f"results/gates/{gate}.json", {"gate": gate, "passed": True})
    (tmp_path / "data").mkdir()
    (tmp_path / "RUNLOG.md").write_text("")
    inputs = tmp_path / "input.txt"
    inputs.write_text("payload")
    return inputs


def build(tmp_path, inputs, **extra):
    return gpu_jobs.GPUJob(
        tmp_path,
        tmp_path / "data/job",
        kind="fixture",
        script="jobs/fixture.py",
        files=[(inputs, "input.txt")],
        arguments=["--input", "/code/input.txt"],
        outputs=["result.json", "nested/run.json"],
        timeout_s=600,
        context={"label": "x"},
        **extra,
    )


def write_outputs(directory):
    for name in ("result.json", "nested/run.json"):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"name": name})


def test_gpu_job_downloads_nested_hashed_outputs_once(tmp_path, monkeypatch):
    inputs = make_root(tmp_path)
    events, launches = [], []

    def launch(root, name, **kwargs):
        launches.append(kwargs)
        Budget(root).reserve(name, 0.3)
        Budget(root).update(name, state="launched", job_id="job-1")
        write_json(root / f"artifacts/jobs/{name}.json", {"job_id": "job-1"})

    class API:
        def inspect_job(self, **kwargs):
            events.append("inspect")
            return SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"))

        def download_bucket_files(self, bucket, paths, **kwargs):
            events.append("download")
            assert len(paths) == 4 and kwargs["raise_on_missing_files"] is True
            for remote, local in paths:
                assert remote.startswith("automatic/fixture-")
                assert Path(local).parent.exists()
            write_outputs(Path(paths[0][1]).parent)

    monkeypatch.setattr(gpu_jobs, "launch", launch)
    monkeypatch.setattr(gpu_jobs, "payload", lambda *a, **k: tmp_path / "staged")
    monkeypatch.setattr(gpu_jobs, "HfApi", API)
    job = build(tmp_path, inputs)
    assert job.run() == job.directory
    assert (job.directory / "nested/run.json").exists()
    build(tmp_path, inputs).run()
    assert events == ["inspect", "download"]
    assert len(launches) == 1 and launches[0]["flavor"] == "l40sx1"
    assert launches[0]["command"][-2:] == ["--output-dir", f"/out/automatic/{job.name}"]
    assert Budget(tmp_path).total() == 0.3


def test_gpu_job_cache_rejects_tampering_and_other_contexts(tmp_path, monkeypatch):
    inputs = make_root(tmp_path)
    job = build(tmp_path, inputs)
    job.directory.mkdir(parents=True)
    write_outputs(job.directory)
    write_json(job.directory / "job-context.json", job.context)
    monkeypatch.setattr(gpu_jobs, "HfApi", lambda: pytest.fail("A valid cache needs no HF call"))
    assert build(tmp_path, inputs).run() == job.directory
    (job.directory / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="content hash"):
        build(tmp_path, inputs).run()
    write_outputs(job.directory)
    inputs.write_text("changed")
    with pytest.raises(ValueError, match="different job context"):
        build(tmp_path, inputs).run()


def test_gpu_job_requires_gates_before_any_payment(tmp_path, monkeypatch):
    inputs = make_root(tmp_path, gates=("G0-security", "G1"))
    monkeypatch.setattr(gpu_jobs, "launch", lambda *a, **k: pytest.fail("No launch without G2"))
    with pytest.raises(GateError, match="G2"):
        build(tmp_path, inputs).run()


def test_gpu_job_downloads_fail_closed_on_bad_hash(tmp_path, monkeypatch):
    inputs = make_root(tmp_path)

    def launch(root, name, **kwargs):
        Budget(root).reserve(name, 0.3)
        write_json(root / f"artifacts/jobs/{name}.json", {"job_id": "job-2"})

    class API:
        def inspect_job(self, **kwargs):
            return SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"))

        def download_bucket_files(self, bucket, paths, **kwargs):
            write_outputs(Path(paths[0][1]).parent)
            (Path(paths[0][1]).parent / "result.json").write_text("{}")
            manifest(Path(paths[0][1]).parent / "nested/run.json")

    monkeypatch.setattr(gpu_jobs, "launch", launch)
    monkeypatch.setattr(gpu_jobs, "payload", lambda *a, **k: tmp_path / "staged")
    monkeypatch.setattr(gpu_jobs, "HfApi", API)
    job = build(tmp_path, inputs)
    with pytest.raises(ValueError, match="failed its hash"):
        job.run()
    assert not (job.directory / "result.json").exists()
    assert json.loads((tmp_path / "artifacts/budget.json").read_text())["entries"][0][
        "terminal_verified"
    ]


def test_queue_timeout_settles_unstarted_job_at_zero_and_retries_with_new_name(
    tmp_path, monkeypatch
):
    from anthropochrony.measure.remote import QueueTimeout

    inputs = make_root(tmp_path)
    launched, canceled = [], []

    def launch(root, name, **kwargs):
        launched.append(name)
        Budget(root).reserve(name, 0.3)
        Budget(root).update(name, state="launched", job_id=f"job-{len(launched)}")
        write_json(root / f"artifacts/jobs/{name}.json", {"job_id": f"job-{len(launched)}"})

    class API:
        def cancel_job(self, **kwargs):
            canceled.append(kwargs["job_id"])

        def inspect_job(self, **kwargs):
            return SimpleNamespace(status=SimpleNamespace(stage="CANCELED"), started_at=None)

        def download_bucket_files(self, bucket, paths, **kwargs):
            write_outputs(Path(paths[0][1]).parent)

    monkeypatch.setattr(gpu_jobs, "launch", launch)
    monkeypatch.setattr(gpu_jobs, "payload", lambda *a, **k: tmp_path / "staged")
    monkeypatch.setattr(gpu_jobs, "HfApi", API)

    def queue_forever(*args, **kwargs):
        raise QueueTimeout("never allocated")

    monkeypatch.setattr(gpu_jobs, "wait_for_job", queue_forever)
    with pytest.raises(QueueTimeout):
        build(tmp_path, inputs).run()
    assert canceled == ["job-1"] and Budget(tmp_path).total() == 0
    monkeypatch.setattr(gpu_jobs, "wait_for_job", lambda *a, **k: "COMPLETED")
    job = build(tmp_path, inputs)
    job.run()
    assert launched[1] == f"{launched[0]}-retry1" and job.name == launched[1]
    assert Budget(tmp_path).total() == 0.3
