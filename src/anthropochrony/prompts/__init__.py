"""Compile prompt text from the binding Markdown specification, never duplicate it."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs" / "PROMPTS.md"
COMPILED = Path(__file__).with_name("templates.yaml")


def compile_spec(path=SPEC):
    text = Path(path).read_text()
    blocks = re.findall(r"^```text\n(.*?)^```$", text, re.M | re.S)
    if len(blocks) != 8:
        raise ValueError(f"Prompt specification changed: expected 8 text blocks, got {len(blocks)}")
    names = ["system", "task", "feedback", "target", "example", "teacher", "synthetic", "judge"]
    result = dict(zip(names, (b.rstrip("\n") for b in blocks), strict=True))
    templates = {}
    for line in text.splitlines():
        match = re.match(r"^\| ((?:sc|q|h|g|p|w)\d\d) \| (.*) \|$", line)
        if not match:
            continue
        template_id, tail = match.groups()
        parts = tail.split(" | ")
        split = parts[0] if len(parts) == 2 else "heldout"
        templates[template_id] = {"split": split, "text": parts[-1].strip("`")}
    if len(templates) != 68:
        raise ValueError(f"Expected 68 prompt templates, got {len(templates)}")
    result["templates"] = templates
    result["e1_instruction"] = re.search(
        r"E1 adds one more line to the system prompt: `([^`]+)`", text
    ).group(1)
    result["no_code_feedback"] = re.search(r"1\. No code block: `(.*)`", text).group(1)
    result["timeout_feedback"] = re.search(r"2\. Timeout: `(.*)`", text).group(1)
    return result


def compile_prompts(check=False):
    rendered = yaml.safe_dump(compile_spec(), sort_keys=False, allow_unicode=True, width=110)
    if check:
        if not COMPILED.exists() or COMPILED.read_text() != rendered:
            raise ValueError("Compiled prompts are stale. Run anthropochrony prompts")
    else:
        COMPILED.write_text(rendered)
    return len(compile_spec()["templates"])


_CACHE = {}


def load():
    if not COMPILED.exists():
        raise RuntimeError("Compile prompts first: anthropochrony prompts")
    stat = COMPILED.stat()
    key = (str(COMPILED), stat.st_mtime_ns, stat.st_size)
    if key not in _CACHE:
        # Pure-Python YAML parsing costs ~25 ms; evaluation builds tens of thousands of prompts.
        _CACHE.clear()
        _CACHE[key] = yaml.safe_load(COMPILED.read_text())
    return _CACHE[key]


def fmt_speed(value):
    rounded = float(f"{value:.2g}")
    return f"{rounded:,.10f}".rstrip("0").rstrip(".") if rounded % 1 else f"{rounded:,.0f}"


def system_prompt(card, phrasing="sc01", estimator="E1"):
    data = load()
    speed = ""
    if card is not None and estimator not in {"E0", "M5"}:
        speed = data["templates"][phrasing]["text"].format(
            S=fmt_speed(card.S), P=fmt_speed(card.P), S_per_min=fmt_speed(card.S * 60)
        )
    text = data["system"].replace("{speed_card}", speed).rstrip()
    if estimator == "E1":
        text += "\n" + data["e1_instruction"]
    return text


def task_turn(spec):
    return load()["task"].replace("{spec}", spec)


def estimate_messages(task, card, template_id="q21", phrasing="sc07", estimator="E1"):
    data = load()
    question = data["templates"][template_id]["text"].replace("{S}", fmt_speed(card.S))
    return [
        {"role": "system", "content": system_prompt(card, phrasing, estimator)},
        {"role": "user", "content": task["task_turn"] + "\n\n---\n" + question},
    ]


def feedback(attempt, execution):
    data = load()
    if execution["status"] == "no_code":
        summary = data["no_code_feedback"]
    elif execution["status"] == "timeout":
        summary = data["timeout_feedback"].replace("{timeout_s}", str(execution["wall_s"]))
    else:
        summary = "\n".join(execution.get("first_failure", "").splitlines()[-25:])[-1200:]
    return data["feedback"].replace("{j}", str(attempt)).replace("{failure_summary}", summary)
