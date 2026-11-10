import math
import random

from anthropochrony.artifacts import digest
from anthropochrony.timemodel import SpeedCard


def sample_cards(task_id, config, n=6):
    rng = random.Random(int(digest({"task": task_id, "seed": 20261109})[:16], 16))
    profiles = list(config["profiles"].values())
    cards = []
    while len(cards) < n:
        profile = rng.choice(profiles)
        speed = math.exp(rng.uniform(*map(math.log, profile["S"])))
        ratio = math.exp(rng.uniform(*map(math.log, profile["ratio"])))
        card = SpeedCard(f"train-{task_id}-{len(cards)}", speed, speed * ratio)
        low, high = config["heldout_decode_band"]
        if low <= card.S <= high:
            continue
        if any((c.S, c.P) == (card.S, card.P) for c in cards):
            continue
        cards.append(card)
    return cards


def covered(card, profiles):
    return any(
        p["S"][0] <= card.S <= p["S"][1] and p["ratio"][0] <= card.P / card.S <= p["ratio"][1]
        for p in profiles.values()
    )
