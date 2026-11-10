import pytest
import yaml
from transformers import AutoTokenizer

from anthropochrony.artifacts import read_jsonl
from anthropochrony.measure.rounds import Measurement
from anthropochrony.prompts import SPEC, task_turn
from anthropochrony.sandbox import Sandbox
from anthropochrony.timemodel import SpeedCard

ROOT = SPEC.parent.parent


class FixtureEngine:
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, 0

    def generate(self, requests):
        self.calls += 1
        text = "```python\ndef f(x): return x + 1\n```"
        return [
            {
                "request_id": row["request_id"],
                "text": text,
                "prompt_tokens_total": len(row["prompt_token_ids"]),
                "o_tokens": 22,
                "finish_reason": "stop",
                "stop_reason": 261001,
            }
            for row in requests
        ]


@pytest.mark.security
def test_real_tokenizer_sandbox_round_and_resume(tmp_path):
    path = ROOT / "pins/tiny-aya-global"
    if not path.exists():
        pytest.skip("Run preflight to download the pinned tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    engine = FixtureEngine(tokenizer)
    task = {
        "task_id": "synthetic-only",
        "family": "FUNC",
        "split": "train",
        "task_turn": task_turn("Implement f(x): return x + 1"),
        "harness": "evalplus_check",
        "test_code": "assert f(3) == 4",
        "time_limits": {"per_task_s": 5},
    }
    config = yaml.safe_load((ROOT / "configs/measure.yaml").read_text()) | {"k": 2}
    measurement = Measurement(tmp_path, [task], config, SpeedCard("fixture", 35, 5000))
    states = measurement.run(engine, Sandbox())
    assert engine.calls == 1
    assert all(s["passed"] and s["n_attempts"] == 1 for s in states)
    assert len(read_jsonl(tmp_path / "traces.jsonl")) == 2
    measurement.run(engine, Sandbox())
    assert engine.calls == 1
    changed = Measurement(
        tmp_path, [task | {"task_turn": "changed"}], config, SpeedCard("fixture", 35, 5000)
    )
    with pytest.raises(ValueError, match="Refusing to resume"):
        changed.states()
