from anthropochrony.prompts import compile_prompts, compile_spec, system_prompt
from anthropochrony.timemodel import SpeedCard


def test_prompt_sync():
    assert compile_prompts(check=True) == 68
    data = compile_spec()
    assert data["templates"]["q21"]["split"] == "heldout"
    assert data["templates"]["q01"]["split"] == "train"


def test_speed_card_and_no_card():
    card = SpeedCard("check", 35, 5000)
    assert "5,000" in system_prompt(card)
    assert "2,100 output tokens per minute" in system_prompt(card, "sc06")
    assert "serving speed" not in system_prompt(card, estimator="E0")
