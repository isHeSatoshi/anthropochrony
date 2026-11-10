"""Conservative duration extraction, with an explicit cached-judge fallback hook."""

import math
import re
from dataclasses import asdict, dataclass, field

WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "a couple of": 2,
    "a couple": 2,
    "a few": 3,
    "half an": 0.5,
    "half a": 0.5,
    "an": 1,
    "a": 1,
}
NUMBER = r"(?:\d[\d,]*(?:\.\d+)?|" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")"
UNIT = r"(?:seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h|days?|weeks?|months?)"
DURATION = re.compile(
    rf"(?<![-\d.])\b(?:(?:between\s+)?(?P<lo>{NUMBER})\s*(?P<lu>{UNIT})?\s*"
    rf"(?:[–—-]|to|and)\s*)?(?P<hi>{NUMBER})\s*(?P<hu>{UNIT})\b",
    re.I,
)
FACTORS = {"s": 1, "min": 60, "h": 3600, "day": 8 * 3600, "week": 40 * 3600, "month": 160 * 3600}


def unit(value):
    value = value.lower()
    if value.startswith(("s",)):
        return "s"
    if value.startswith("min"):
        return "min"
    if value.startswith("h"):
        return "h"
    return value.rstrip("s")


def number(value):
    value = value.lower()
    return WORDS[value] if value in WORDS else float(value.replace(",", ""))


@dataclass
class Estimate:
    point_s: float | None = None
    lo_s: float | None = None
    hi_s: float | None = None
    tokens: float | None = None
    attempts: float | None = None
    pfp: float | None = None
    units_used: list[str] = field(default_factory=list)
    has_human_units: bool = False
    parse_method: str = "none"

    def as_dict(self):
        return asdict(self)


CODE_BLOCK = re.compile(r"```.*?(?:```|\Z)", re.S)


def answer_segment(text, plan=False):
    """Prose outside code fences; a non-plan answer is its first paragraph with a duration."""
    prose = CODE_BLOCK.sub("\n\n", text).strip()
    if plan:
        return prose
    for paragraph in re.split(r"\n\s*\n", prose):
        if DURATION.search(paragraph):
            return paragraph
    return prose


def parse(text, *, calendar=False, judge=None, question="", plan=False):
    factors = FACTORS | ({"day": 86400, "week": 604800, "month": 2628000} if calendar else {})
    text = answer_segment(text, plan)
    matches = list(DURATION.finditer(text))
    units = sorted({unit(m[g]) for m in matches for g in ("lu", "hu") if m[g]})
    result = Estimate(
        units_used=units, has_human_units=bool(set(units) & {"h", "day", "week", "month"})
    )
    canonical = re.match(r"^About ([\d.,]+)\s*(s|min|h)\b", text)
    if canonical:
        result.point_s = number(canonical[1]) * factors[canonical[2]]
        interval = re.search(
            r"likely ([\d.,]+)\s*(s|min|h)?\s*[–-]\s*([\d.,]+)\s*(s|min|h)\b", text
        )
        if interval:
            result.lo_s = number(interval[1]) * factors[interval[2] or interval[4]]
            result.hi_s = number(interval[3]) * factors[interval[4]]
        tokens = re.search(r"~([\d,]+) output tokens", text)
        attempts = re.search(r"(1|about 2|all 3) attempts? expected", text)
        pfp = re.search(r"pass chance: (?:under|about|over) (\d+)%", text)
        result.tokens = number(tokens[1]) if tokens else None
        result.attempts = float(attempts[1].split()[-1]) if attempts else None
        result.pfp = float(pfp[1]) / 100 if pfp else None
        result.parse_method = "regex"
    elif matches:
        # A labeled total overrides component durations. Otherwise sum the listed steps.
        totals = [
            m
            for m in matches
            if re.search(
                r"(?:total|overall|in all|altogether)[^.\n:]*:?\s*$",
                text[max(0, m.start() - 70) : m.start()],
                re.I,
            )
        ]
        selected = totals[-1:] or matches
        lows, highs = [], []
        for match in selected:
            hi = number(match["hi"]) * factors[unit(match["hu"])]
            lo = (
                number(match["lo"]) * factors[unit(match["lu"] or match["hu"])]
                if match["lo"]
                else hi
            )
            if (
                match["lu"]
                and re.search(r"\band\b", match[0])
                and not match[0].lower().startswith("between")
            ):
                lo = hi = lo + hi
            if lo > hi or lo <= 0:
                return _judge(result, judge, text, question)
            lows.append(lo)
            highs.append(hi)
        result.lo_s, result.hi_s = sum(lows), sum(highs)
        result.point_s = math.sqrt(result.lo_s * result.hi_s)
        result.parse_method = "regex"
    if result.point_s is not None and (result.point_s <= 0 or not math.isfinite(result.point_s)):
        result.point_s = result.lo_s = result.hi_s = None
        result.parse_method = "none"
    return result if result.point_s is not None else _judge(result, judge, text, question)


def _judge(result, judge, text, question):
    if judge is None:
        return result
    answer = judge(question, text)
    point = answer.get("point_seconds")
    if (
        answer.get("has_estimate")
        and isinstance(point, (float, int))
        and math.isfinite(point)
        and point > 0
    ):
        result.point_s = point
        result.lo_s = answer.get("lo_seconds")
        result.hi_s = answer.get("hi_seconds")
        result.units_used = sorted({unit(u) for u in answer.get("units_used", [])})
        result.has_human_units = bool(set(result.units_used) & {"h", "day", "week", "month"})
        result.parse_method = "judge"
    return result
