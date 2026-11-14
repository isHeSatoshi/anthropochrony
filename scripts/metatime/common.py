"""Shared paths and loaders for the MetaTime head-to-head scripts."""

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTIFACT_URL = "https://anonymous.4open.science/api/repo/Time-Estimation-B286/zip"

# The benchmark's own prospective question (experiments/framework.py in the artifact).
PRE_ESTIMATE = (
    "I'm going to ask you to complete the following task:\n\n{description}\n\n"
    "Before you do it, estimate how long it will take YOU (the AI) to complete this task.\n"
    "Give your estimate in seconds or minutes. Be specific with a number.\n\n"
    'Just provide the time estimate, nothing else. Example format: "45 seconds" or "2 minutes"\n'
)


def fetch_suite(directory):
    """Download the public MetaTime artifact once; we do not redistribute its tasks."""
    directory = Path(directory)
    if not (directory / "tasks" / "task_suite.py").exists():
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "artifact.zip"
        urllib.request.urlretrieve(ARTIFACT_URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(directory)
    return directory


def load_tasks(directory):
    directory = fetch_suite(directory)
    sys.path.insert(0, str(directory))
    from tasks.task_suite import ALL_TASKS

    return [
        {
            "id": task.id,
            "category": task.category,
            "complexity": task.complexity,
            "description": task.description,
            "prompt": task.prompt,
        }
        for task in ALL_TASKS
    ]


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- The self-time interface used for training and for our rows of the comparison. ---

SYSTEM = (
    "{speed_card}\n"
    "When asked how long a task will take you, answer for your own run on this deployment: "
    "the time to read the prompt and write your complete response."
)
PAPER_ESTIMATE = (
    "Task: {description}\n\nHow long will it take you to complete this? Give a specific "
    'estimate in seconds.\n\nReply with just the number and unit, e.g. "30 seconds" or '
    '"2 minutes".'
)
OWN_QUESTIONS = [
    "{description}\n\n---\nBefore you start: how long will this take you, from reading the "
    "prompt to the last token of your response?",
    "You are about to be given this task:\n\n{description}\n\nHow many seconds will your own "
    "run take on this deployment?",
]
# (question, sampling weight); the benchmark's own wording dominates.
QUESTIONS = [
    (PRE_ESTIMATE, 0.4),
    (PAPER_ESTIMATE, 0.2),
    (OWN_QUESTIONS[0], 0.2),
    (OWN_QUESTIONS[1], 0.2),
]
TRAIN_PHRASINGS = ["sc01", "sc02", "sc03", "sc04", "sc05", "sc06"]
HELDOUT_PHRASING = "sc07"
FULL_PROMPT_CHARS = 3000
ROOT = Path(__file__).resolve().parents[2]


def sig(value, digits):
    return float(f"{value:.{digits}g}")


def plain(value):
    return f"{value:.10f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.0f}"


def speed_line(S, P, phrasing):
    import yaml

    templates = yaml.safe_load((ROOT / "src/anthropochrony/prompts/templates.yaml").read_text())
    return templates["templates"][phrasing]["text"].format(
        S=plain(S), P=plain(P), S_per_min=plain(S * 60)
    )


def system_prompt(S, P, phrasing=HELDOUT_PHRASING):
    return SYSTEM.format(speed_card=speed_line(S, P, phrasing))


def sample_card(rng):
    """One stated speed from the project's deployment profiles, shown to two significant digits."""
    import math

    import yaml

    config = yaml.safe_load((ROOT / "configs/speeds.yaml").read_text())
    low, high = config["heldout_decode_band"]
    while True:
        profile = rng.choice(sorted(config["profiles"].values(), key=lambda p: p["S"][0]))
        speed = sig(math.exp(rng.uniform(*map(math.log, profile["S"]))), 2)
        prefill = sig(speed * math.exp(rng.uniform(*map(math.log, profile["ratio"]))), 2)
        if not low <= speed <= high:
            return speed, prefill


def task_view(row, view):
    if view == "description":
        return row["description"]
    text = row["prompt"]
    return text if len(text) <= FULL_PROMPT_CHARS else text[:FULL_PROMPT_CHARS] + " …"


def answer_text(n_in, n_out, S, P):
    """Workload first, then the time that follows from it at the stated speeds."""
    prompt_tokens = int(sig(n_in, 2))
    response_tokens = max(1, int(sig(n_out, 2)))
    seconds = sig(prompt_tokens / P + response_tokens / S, 3)
    return (
        f"Prompt: about {prompt_tokens} tokens. Response: about {response_tokens} tokens.\n"
        f"Time: {plain(seconds)} seconds"
    )


def parse_answer(text):
    """Read our answer format; None for any part that is missing."""
    import re

    def find(pattern):
        match = re.search(pattern, text or "")
        return float(match.group(1)) if match else None

    return {
        "prompt_tokens": find(r"Prompt: about (\d+(?:\.\d+)?) tokens"),
        "response_tokens": find(r"Response: about (\d+(?:\.\d+)?) tokens"),
        "seconds": find(r"Time: (\d+(?:\.\d+)?) seconds"),
    }
