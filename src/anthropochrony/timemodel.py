"""The sole implementation of the pre-registered time formula."""

import math
from dataclasses import dataclass


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def round_speed(value):
    return float(f"{positive(value, 'speed'):.2g}")


@dataclass(frozen=True)
class SpeedCard:
    id: str
    S: float
    P: float

    def __post_init__(self):
        object.__setattr__(self, "S", round_speed(self.S))
        object.__setattr__(self, "P", round_speed(self.P))

    def as_dict(self):
        return {"id": self.id, "S": self.S, "P": self.P}


# Adopted S2 fallback (PREREGISTRATION.md section 7, 2026-11-09): the raw
# formula missed G2 on 100 real HF-generation/local-sandbox loops (median error
# 36.1%). One RPC round trip per model call costs a constant ~0.967 s that no
# token or execution term captures. Every label, baseline and teacher target
# must use this same value, so callers rely on the default.
PER_CALL_LATENCY_S = 0.967


def completion_time(u_tokens, o_tokens, exec_s, card, *, latency_s=None, calls=1):
    """Works for trajectories and for baselines predicting aggregate quantities.

    `calls` is the number of model calls (attempts) the aggregate covers.
    """
    if latency_s is None:
        latency_s = PER_CALL_LATENCY_S
    for value in (u_tokens, o_tokens, exec_s, latency_s, calls):
        if value < 0 or not math.isfinite(value):
            raise ValueError("Token counts and execution time must be finite and non-negative")
    return u_tokens / card.P + o_tokens / card.S + exec_s + latency_s * calls


def trajectory_time(trace, card, *, latency_s=None):
    return sum(
        completion_time(
            a["u_tokens"], a["o_tokens"], a["exec"]["wall_s"], card, latency_s=latency_s
        )
        for a in trace["attempts"]
    )


def uncached_tokens(prompt_total, system_tokens, previous_prompt=None, previous_output=None):
    if previous_prompt is None:
        value = prompt_total - system_tokens
    else:
        if previous_output is None:
            raise ValueError("Previous output count is required for a retry")
        value = prompt_total - previous_prompt - previous_output
    if value < 0:
        raise ValueError("Negative uncached tokens: chat rendering or stop accounting changed")
    return value


def attempt_budget(prompt_tokens, family, context=8192):
    if prompt_tokens < 0:
        raise ValueError("Prompt length cannot be negative")
    remaining = context - prompt_tokens
    if remaining < 256:
        return 0
    long_output = family.startswith("PROJ") or family == "ALGO_IO"
    return min(4096 if long_output else 2048, remaining)


def output_token_count(token_ids, stop_reason, stop_ids=(6, 261001)):
    """vLLM may return the terminal token in token_ids or only in stop_reason."""
    ids = list(token_ids)
    if stop_reason in stop_ids and (not ids or ids[-1] != stop_reason):
        return len(ids) + 1
    return len(ids)
