import pytest

from anthropochrony import gpu_jobs
from anthropochrony.artifacts import write_json, write_jsonl
from anthropochrony.gates import GateError
from anthropochrony.measure.finetuned import MergedRound, adapter_identity


def identity(sha="a" * 64):
    return {"model_id": "user/lora", "model_sha": "commit", "adapter_weights_sha256": sha}


def setup(tmp_path, monkeypatch, merged_sha):
    for run in ("m1-s0", "m2-s0"):
        write_json(tmp_path / f"adapters/{run}/identity.json", identity())
    requests = tmp_path / "requests.jsonl"
    write_jsonl(
        requests,
        [{"request_id": "m1-s0:a", "prompt_token_ids": [1]}]
        + [{"request_id": "m2-s0:b", "prompt_token_ids": [2]}],
    )
    seen = {}

    def run(self):
        seen["job"] = self
        self.directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.directory / "generation.jsonl", [{"request_id": "m1-s0:a"}])
        write_json(
            self.directory / "merge-evidence.json",
            {r: {"adapter_weights_sha256": merged_sha} for r in ("m1-s0", "m2-s0")},
        )
        return self.directory

    monkeypatch.setattr(gpu_jobs.GPUJob, "run", run)
    # Source hashes are read from the root, so give the fixture root the required files.
    for name in (*gpu_jobs.COMMON_SOURCES, "jobs/ft_round.py", "jobs/model_prep.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return requests, seen


def test_merged_round_requires_matching_adapter_hash_and_serves_each_run(tmp_path, monkeypatch):
    requests, seen = setup(tmp_path, monkeypatch, "a" * 64)
    round_ = MergedRound(tmp_path, tmp_path / "data/batch", {"engine": {}}, ["m1-s0", "m2-s0"])
    assert round_.generate_file(requests) == [{"request_id": "m1-s0:a"}]
    job = seen["job"]
    assert job.script == "jobs/ft_round.py" and "G6" in job.required_gates
    assert job.timeout_s == 900 * 2 + 600
    assert set(job.context["extra"]["identities"]) == {"m1-s0", "m2-s0"}


def test_merged_round_rejects_wrong_adapter_and_unknown_runs(tmp_path, monkeypatch):
    requests, _ = setup(tmp_path, monkeypatch, "b" * 64)
    with pytest.raises(ValueError, match="unexpected adapter"):
        MergedRound(tmp_path, tmp_path / "d", {}, ["m1-s0", "m2-s0"]).generate_file(requests)
    with pytest.raises(GateError, match="no adapter identity"):
        MergedRound(tmp_path, tmp_path / "d2", {}, ["m1-s0"]).generate_file(requests)
    with pytest.raises(GateError, match="not been downloaded"):
        adapter_identity(tmp_path, "m1-s2")
