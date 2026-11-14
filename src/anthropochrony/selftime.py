"""Ask the released Anthropochrony model how long a coding task will take on this machine.

The tuned model predicts its own workload (output tokens and attempts). M1-Struct turns that
workload into seconds with the exact timing formula and train-only priors, and the interval
uses the split-conformal factor fit on the validation split.
"""

import json
import math
from dataclasses import asdict, dataclass
from functools import cache
from importlib import resources

from anthropochrony.demo import LlamaBackend
from anthropochrony.eval.structured import WorkloadPriors, structured_duration
from anthropochrony.families import FAMILIES, infer_prompt_family
from anthropochrony.parse import parse
from anthropochrony.prompts import estimate_messages, task_turn
from anthropochrony.timemodel import SpeedCard

HUMAN_TEMPLATE = "h09"
# Same pasted-task approximation as the workbench and Space: about 3.6 characters per token.
CHARS_PER_TOKEN = 3.6
MIN_TASK_TOKENS = 24.0


class EstimateUnavailable(ValueError):
    """The model answer contained neither a token count nor a duration."""


@cache
def load_defaults() -> dict:
    path = resources.files("anthropochrony").joinpath("selftime_defaults.json")
    return json.loads(path.read_text(encoding="utf-8"))


def default_priors() -> WorkloadPriors:
    return WorkloadPriors(**load_defaults()["workload_priors"])


def approx_task_tokens(task_text: str) -> float:
    return max(MIN_TASK_TOKENS, len(task_text) / CHARS_PER_TOKEN)


def format_seconds(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


@dataclass(frozen=True)
class SelfTimeEstimate:
    seconds: float
    low: float
    high: float
    tokens: float | None
    attempts: float | None
    verbal_seconds: float | None
    family: str
    card: dict
    method: str
    raw_text: str

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        text = (
            f"~{format_seconds(self.seconds)} on this machine "
            f"(80% range {format_seconds(self.low)} to {format_seconds(self.high)}"
        )
        if self.tokens:
            calls = int(max(1.0, self.attempts or 1.0))
            plural = "s" if calls != 1 else ""
            text += f"; ~{self.tokens:,.0f} output tokens over {calls} attempt{plural}"
        if self.method == "verbal-fallback":
            text += "; verbal estimate, no token count was given"
        return text + ")"


@dataclass(frozen=True)
class HumanTimeEstimate:
    seconds: float | None
    raw_text: str
    own_time_format: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.own_time_format:
            return "No human-developer estimate: the model answered with its own time instead"
        if self.seconds is None:
            return "No human-developer estimate could be parsed"
        return f"~{format_seconds(self.seconds)} for a human developer"


class SelfTimeEstimator:
    """Estimate Tiny Aya's own wall-clock time for a Python coding task.

    `backend` needs `generate(messages)`; it also needs `calibrate()` unless `card` is given.
    `latency_s` is the per-call overhead: 0 for a local llama-server.
    """

    def __init__(
        self,
        backend,
        card: SpeedCard | None = None,
        *,
        latency_s: float = 0.0,
        priors: WorkloadPriors | None = None,
        conformal_factor: float | None = None,
        verbal_conformal_factor: float | None = None,
    ):
        if card is None and not callable(getattr(backend, "calibrate", None)):
            raise ValueError("Give a SpeedCard or a backend with calibrate()")
        defaults = load_defaults()
        self.backend = backend
        self.latency_s = float(latency_s)
        self.priors = priors if priors is not None else default_priors()
        if conformal_factor is None:
            conformal_factor = defaults["struct_conformal_factor"]
        if verbal_conformal_factor is None:
            verbal_conformal_factor = defaults["verbal_conformal_factor"]
        self.conformal_factor = float(conformal_factor)
        self.verbal_conformal_factor = float(verbal_conformal_factor)
        if self.latency_s < 0 or not math.isfinite(self.latency_s):
            raise ValueError("latency_s must be finite and non-negative")
        if self.conformal_factor < 1 or self.verbal_conformal_factor < 1:
            raise ValueError("Conformal factors must be at least 1")
        self._card = card

    @classmethod
    def from_llama_server(
        cls,
        url: str,
        *,
        speed: tuple[float, float] | None = None,
        llama_bench: str | None = None,
        gguf: str | None = None,
        threads: int = 8,
        gpu_layers: int = 0,
        **options,
    ) -> "SelfTimeEstimator":
        """Use a local llama-server that serves the tuned GGUF.

        Pass `speed=(S, P)` (decode and prefill tokens/s) or `llama_bench` and `gguf` so the
        speed is measured once with llama-bench on first use.
        """
        if speed is None and not (llama_bench and gguf):
            raise ValueError("Give speed=(S, P), or llama_bench and gguf to measure it")
        backend = LlamaBackend(url, url, gguf, llama_bench, threads=threads, gpu_layers=gpu_layers)
        card = SpeedCard("manual", *speed) if speed is not None else None
        return cls(backend, card, **options)

    def speed_card(self) -> SpeedCard:
        if self._card is None:
            self._card = self.backend.calibrate()
        return self._card

    def estimate(
        self, task_text: str, family: str | None = None, *, task_tokens: float | None = None
    ) -> SelfTimeEstimate:
        """`task_tokens` overrides the character-based count of the task's prompt tokens."""
        if not task_text.strip():
            raise ValueError("The task text is empty")
        family = family or infer_prompt_family(task_text)
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}; expected one of {', '.join(FAMILIES)}")
        card = self.speed_card()
        messages = estimate_messages({"task_turn": task_turn(task_text)}, card, estimator="agent")
        raw = self.backend.generate(messages)
        parsed = parse(raw)
        uncached = approx_task_tokens(task_text) if task_tokens is None else float(task_tokens)
        point = structured_duration(
            parsed.tokens,
            parsed.attempts,
            uncached,
            family,
            card,
            self.priors,
            latency_s=self.latency_s,
        )
        if point is not None:
            method = "m1-struct"
            low, high = point / self.conformal_factor, point * self.conformal_factor
        elif parsed.point_s is not None:
            # Without a token count, fall back to M1b: the verbal interval widened by its own
            # validation-fit conformal factor.
            method = "verbal-fallback"
            point = parsed.point_s
            low = (parsed.lo_s or point) / self.verbal_conformal_factor
            high = (parsed.hi_s or point) * self.verbal_conformal_factor
        else:
            raise EstimateUnavailable(f"No token count or duration in the model answer: {raw!r}")
        return SelfTimeEstimate(
            seconds=float(point),
            low=float(low),
            high=float(high),
            tokens=parsed.tokens,
            attempts=parsed.attempts,
            verbal_seconds=parsed.point_s,
            family=family,
            card=card.as_dict(),
            method=method,
            raw_text=raw,
        )

    def human_estimate(self, task_text: str) -> HumanTimeEstimate:
        """The same model's estimate of how long a human developer would take."""
        if not task_text.strip():
            raise ValueError("The task text is empty")
        messages = estimate_messages(
            {"task_turn": task_turn(task_text)},
            self.speed_card(),
            template_id=HUMAN_TEMPLATE,
            estimator="agent",
        )
        raw = self.backend.generate(messages)
        parsed = parse(raw)
        # The quantized model sometimes answers the human question with its own token
        # breakdown; that number is its own time, not a human's, so it is not reported.
        if parsed.tokens is not None:
            return HumanTimeEstimate(None, raw, own_time_format=True)
        return HumanTimeEstimate(parsed.point_s, raw)

    def tripwire_seconds(self, estimate: SelfTimeEstimate, k: float = 1.5) -> float:
        """Timeout for a run: `k` times the 80% upper bound.

        The default 1.5 comes from the real quantized runs (`x10_deployment_checks`): at 1.0 the
        bound stopped 15-18% of passing runs there, at 1.5 none.
        """
        if not math.isfinite(k) or k <= 0:
            raise ValueError("k must be positive and finite")
        return estimate.high * k


def measure_speed_card(
    llama_bench: str, gguf: str, *, threads: int = 8, gpu_layers: int = 0
) -> SpeedCard:
    """Measure decode (S) and prefill (P) tokens/s with llama-bench."""
    # calibrate() only runs llama-bench locally; it never contacts this URL.
    url = "http://127.0.0.1:8082"
    return LlamaBackend(url, url, gguf, llama_bench, threads, gpu_layers).calibrate()
