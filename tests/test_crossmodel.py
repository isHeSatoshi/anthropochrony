from pathlib import Path

import pytest

from anthropochrony.crossmodel import EOT
from anthropochrony.crossmodel.rounds import system_token_count
from anthropochrony.gen.engine import render

PIN = Path(__file__).resolve().parents[1] / "pins/aya-expanse-8b"


@pytest.fixture(scope="module")
def tokenizer():
    if not (PIN / "tokenizer.json").exists():
        pytest.skip("Aya Expanse tokenizer pin is not downloaded")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(PIN, local_files_only=True)


def test_assistant_turn_ends_with_single_end_of_turn(tokenizer):
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    prompt = render(tokenizer, messages[:2])
    completion = tokenizer.encode("yo", add_special_tokens=False) + [EOT]
    assert render(tokenizer, messages, generation_prompt=False) == prompt + completion


def test_system_prefix_ends_at_first_end_of_turn(tokenizer):
    count = system_token_count(tokenizer, "S")
    ids = render(tokenizer, [{"role": "system", "content": "S"}, {"role": "user", "content": ""}])
    assert ids[count - 1] == EOT
    assert EOT not in ids[: count - 1]
