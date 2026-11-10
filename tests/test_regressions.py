import copy
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from anthropochrony import artifacts
from anthropochrony.eval.hypotheses import confirmatory
from anthropochrony.eval.prompts import build_evaluation_prompts
from anthropochrony.jobs import payload
from anthropochrony.measure.rounds import apply_round, initial_states
from anthropochrony.prompts import SPEC, task_turn
from anthropochrony.timemodel import SpeedCard

ROOT = SPEC.parent.parent


def test_manifest_without_git_binary(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(artifacts.subprocess, "run", no_git)
    artifacts.write_json(tmp_path / "result.json", {"generation_count": 10})
    sidecar = json.loads((tmp_path / "result.json.manifest.json").read_text())
    assert sidecar["git_sha"] == "unavailable"
    assert sidecar["artifact_sha256"] == artifacts.file_hash(tmp_path / "result.json")


def test_payload_preserves_identity_without_git(tmp_path, monkeypatch):
    code = payload(ROOT, tmp_path / "payload")
    identity = json.loads((code / "SOURCE_IDENTITY.json").read_text())
    assert identity["source_sha"]
    assert "uv.lock" in identity["files"]
    assert not any("token" in p.lower() and p.startswith(".") for p in identity["files"])

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(artifacts.subprocess, "run", no_git)
    assert artifacts.git_sha(code) == identity["git_sha"]


@pytest.mark.parametrize("split", ["train", "val", "test", "test_ood"])
def test_eval_prompt_ids_distinct_and_heldout_templates(split):
    task = {
        "task_id": "fixture",
        "family": "FUNC",
        "split": split,
        "task_turn": task_turn("Implement f(x)."),
    }
    labels = [{"task_id": "fixture", "card_id": "REF", "S": 35, "P": 5000, "T_med_s": 2}]
    rows = build_evaluation_prompts([task], labels, split=split)
    assert len({r["est_id"] for r in rows}) == len(rows)
    if split == "val":
        e1 = [r for r in rows if r["estimator"] == "E1" and r["frame"] == "self"]
        assert len(e1) == 2
        assert len({r["template_id"] for r in e1}) == 2
    if split == "train":
        assert len(rows) == 1 and rows[0]["frame"] == "human"
    if split in {"test", "test_ood"}:
        assert all(int(r["template_id"][1:]) >= 21 for r in rows if r["frame"] == "self")


def test_round_replay_is_idempotent_and_changed_outputs_rejected():
    task = {"task_id": "fixture", "family": "FUNC", "task_turn": "fixture"}
    config = {
        "k": 1,
        "model_sha": "fixture",
        "model_id": "fixture",
        "temperature": 0.1,
        "top_p": 0.95,
    }
    states = initial_states([task], SpeedCard("REF", 35, 5000), config)
    output = {
        "request_id": states[0]["run_id"],
        "text": "answer",
        "o_tokens": 1,
        "prompt_tokens_total": 3,
    }
    states[0]["attempts"] = [{"text_sha": artifacts.digest("answer"), "o_tokens": 1}]
    states[0]["done"] = True
    requests = [{"request_id": output["request_id"], "prompt_token_ids": [1, 2, 3]}]
    saved = copy.deepcopy(states)
    result, texts = apply_round(states, [task], requests, [output], None, 1)
    assert result == saved and texts == []
    with pytest.raises(ValueError, match="different generation"):
        apply_round(states, [task], requests, [output | {"text": "changed"}], None, 1)
    with pytest.raises(ValueError, match="do not match"):
        apply_round(states, [task], requests * 2, [output], None, 1)


def test_calibration_uses_v033_metrics(monkeypatch):
    spec = importlib.util.spec_from_file_location("calibration", ROOT / "jobs/calibrate_l4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=dict))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(stdout="fixture clocks"))

    class Engine:
        calls = 0

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda *a, **kw: list(range(6000)))

        def generate(self, prompts, params):
            self.calls += 1
            count = params.max_tokens
            return [
                SimpleNamespace(
                    metrics=SimpleNamespace(
                        first_token_ts=2, scheduled_ts=1, last_token_ts=2 + (count - 1) / 50
                    ),
                    outputs=[SimpleNamespace(token_ids=[1] * count)],
                )
            ]

    engine = Engine()
    result = module.calibrate(engine)
    assert result["reference"] == {"id": "REF", "S": 50, "P": 770}
    assert len(result["records"]) == 35 and engine.calls == 36
    assert not result["validation_gate_passed"]


def test_unmeasured_hypotheses_do_not_pass():
    assert all(r["status"] == "not_measured" for r in confirmatory([]).values())
