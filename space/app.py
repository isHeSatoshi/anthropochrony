"""Anthropochrony demo: a small model that knows how long its own answers take.

Two llama.cpp servers run on this machine's CPU: the untrained Tiny Aya Global and the same
model with the G1-kit adapter (built with CuratorKIT and AlignTune) merged in, both 4-bit
GGUF. The app measures the G1-kit model's speed, asks both models how long a task will take,
then runs the task on the G1-kit model and times it, as in the benchmark head-to-head. The second tab shows the
benchmark results from the repository's committed records.

GGUF quantization and the CPU change the numbers; the benchmark results are for bf16 on a
GPU. This page is an illustration, not evidence.
"""

import csv
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
LLAMA = Path(os.environ.get("LLAMA_DIR", "/opt/llama"))
GGUF_REPO = "isHeSatoshi/anthropochrony-tiny-aya-global-g1-kit-gguf"
MODELS = {"base": ("base-q4_k_m.gguf", 8091), "g1": ("g1-kit-q4_k_m.gguf", 8092)}
THREADS = int(os.environ.get("THREADS", os.cpu_count() or 2))
MAX_RUN_TOKENS = int(os.environ.get("MAX_RUN_TOKENS", 768))
REPO = "https://github.com/isHeSatoshi/anthropochrony"

# The benchmark's prospective question and the stated-speed system prompt used in training
# (the held-out phrasing), copied from scripts/metatime/common.py and configs/metatime.yaml.
QUESTION = (
    "I'm going to ask you to complete the following task:\n\n{description}\n\n"
    "Before you do it, estimate how long it will take YOU (the AI) to complete this task.\n"
    "Give your estimate in seconds or minutes. Be specific with a number.\n\n"
    'Just provide the time estimate, nothing else. Example format: "45 seconds" or "2 minutes"\n'
)
SYSTEM = (
    "{speed}\n"
    "When asked how long a task will take you, answer for your own run on this deployment: "
    "the time to read the prompt and write your complete response."
)
SPEED_LINE = "Benchmark for this deployment: {S} tokens/sec generation, {P} tokens/sec prompt."
EXAMPLES = [
    "Write a haiku about autumn.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Write a 600-word short story about a lighthouse keeper.",
    "List five tips for a first job interview.",
]


def model_path(name):
    local = os.environ.get(f"GGUF_{name.upper()}")
    if local:
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(GGUF_REPO, MODELS[name][0])


def start_servers():
    env = dict(os.environ, LD_LIBRARY_PATH=str(LLAMA))
    for name, (_, port) in MODELS.items():
        subprocess.Popen(
            [
                str(LLAMA / "llama-server"),
                "-m",
                model_path(name),
                "--port",
                str(port),
                "-t",
                str(THREADS),
                "-c",
                "4096",
                "--jinja",
                "-ngl",
                "0",
                # The GGUF does not mark <|END_RESPONSE|> as end of generation.
                "--override-kv",
                "tokenizer.ggml.eot_token_id=int:261001",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    for _, port in MODELS.values():
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)


def chat(name, messages, max_tokens, temperature=0.0, top_p=1.0):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": 0,
        # Every request reads its whole prompt, as the benchmark's timed runs did.
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{MODELS[name][1]}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    reply = json.load(urllib.request.urlopen(request, timeout=900))
    wall = time.perf_counter() - started
    text = reply["choices"][0]["message"]["content"] or ""
    text = re.sub(r"<\|[A-Z_]+\|>", "", text).strip()
    return text, reply, wall


def sig2(value):
    return float(f"{value:.2g}")


def plain(value):
    return f"{value:.10f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.0f}"


def measure_speed():
    """Decode and prompt speed of the G1-kit model on this machine (median of 3)."""
    decode, prefill = [], []
    filler = "The quick brown fox jumps over the lazy dog. " * 60
    for _ in range(3):
        _, reply, _ = chat(
            "g1",
            [{"role": "user", "content": "Count from 1 to 200, one number per line."}],
            128,
        )
        decode.append(reply["timings"]["predicted_per_second"])
        _, reply, _ = chat(
            "g1",
            [{"role": "user", "content": filler + f"\nReply with the single word OK. ({_})"}],
            4,
        )
        prefill.append(reply["timings"]["prompt_per_second"])
    S, P = sig2(sorted(decode)[1]), sig2(sorted(prefill)[1])
    return (
        S,
        P,
        f"Measured on this Space: **{plain(S)} tokens/s** writing, **{plain(P)} tokens/s** reading.",
    )


def read_seconds(text):
    """A simplified reading of the answer's time (the benchmark has its own parser)."""
    match = re.search(r"Time:\s*([\d.]+)\s*seconds", text) or re.search(
        r"([\d.]+)\s*(seconds?|secs?|s\b|minutes?|mins?|hours?)", text, re.I
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) if match.lastindex and match.lastindex > 1 else "seconds").lower()
    return value * (60 if unit.startswith("m") else 3600 if unit.startswith("h") else 1)


def estimate(task, S, P):
    if not task.strip():
        raise gr.Error("Type a task first.")
    if not S or not P:
        S, P, _ = measure_speed()
    system = SYSTEM.format(speed=SPEED_LINE.format(S=plain(S), P=plain(P)))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": QUESTION.format(description=task.strip())},
    ]
    answers = {name: chat(name, messages, 64)[0] for name in ("base", "g1")}
    seconds = {name: read_seconds(text) for name, text in answers.items()}
    return (
        S,
        P,
        answers["base"],
        answers["g1"],
        {"untrained": seconds["base"], "g1": seconds["g1"]},
        "Now run the task to see how long it really takes.",
    )


def run_task(task, estimates):
    if not task.strip():
        raise gr.Error("Type a task first.")
    # As in the benchmark's timed runs: G1-kit does the task, with the model card's sampling.
    text, reply, wall = chat(
        "g1", [{"role": "user", "content": task.strip()}], MAX_RUN_TOKENS, 0.1, 0.95
    )
    tokens = reply["timings"]["predicted_n"]
    capped = reply["choices"][0].get("finish_reason") == "length"
    lines = [f"**Measured: {wall:.1f} s** ({tokens} tokens written)."]
    if capped:
        lines.append(
            f"The answer hit this demo's {MAX_RUN_TOKENS}-token cap, so the real time is longer."
        )
    for label, key in (("Untrained Tiny Aya", "untrained"), ("Tiny Aya + G1-kit", "g1")):
        guess = (estimates or {}).get(key)
        if guess:
            lines.append(
                f"- {label} said {guess:.1f} s: **{guess / wall:.2f}x** the measured time."
            )
        else:
            lines.append(f"- {label}: no time could be read from its answer.")
    return text, "\n".join(lines)


def load_tables():
    summary = json.loads((HERE / "data/summary.json").read_text())
    rows = [
        [
            r["system"],
            f"{r['typical_factor_off']:.2f}x",
            f"{r['median_abs_log_error']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]",
            f"{100 * r['within_2x']:.0f}%",
            f"{r['median_estimate_over_actual']:.2f}x",
        ]
        for r in summary
    ]
    with (HERE / "data/tasks.csv").open() as handle:
        tasks = list(csv.DictReader(handle))
    return rows, tasks


def task_rows(tasks, category):
    out = []
    for t in tasks:
        if category != "all" and t["category"] != category:
            continue
        out.append(
            [
                t["task"],
                t["category"],
                f"{float(t['gpt55_estimate_s']):g} / {float(t['gpt55_actual_s']):g}",
                f"{float(t['opus48_estimate_s']):g} / {float(t['opus48_actual_s']):g}",
                f"{float(t['g1_estimate_s']):g} / {float(t['tiny_aya_actual_s']):g}",
            ]
        )
    return out


def build():
    summary_rows, tasks = load_tables()
    categories = ["all"] + sorted({t["category"] for t in tasks})
    with gr.Blocks(title="Anthropochrony: a model that knows its own time") as demo:
        gr.Markdown(
            "# Anthropochrony\n"
            "Ask an AI how long a task will take it and it answers on a human clock. "
            "We trained a small open model, Cohere Labs' Tiny Aya Global (3.35B), to know its own. "
            f"[Code and write-up]({REPO})"
        )
        with gr.Tab("Try it on this Space"):
            gr.Markdown(
                "1. Measure this machine's speed. 2. Ask the untrained model and the model with "
                "our G1-kit adapter how long a task will take. 3. Run the task on the G1-kit model "
                "and time it (its answers are as long as the untrained model's).\n\n"
                "*Runs on a shared CPU with 4-bit models, so it is slow and noisy, and the numbers "
                "are an illustration. The benchmark numbers (next tab) are bf16 on a GPU.*"
            )
            S = gr.Number(label="Writing speed, tokens/s", precision=None)
            P = gr.Number(label="Reading speed, tokens/s", precision=None)
            speed_note = gr.Markdown()
            gr.Button("1. Measure this Space's speed").click(
                measure_speed, outputs=[S, P, speed_note]
            )
            task = gr.Textbox(label="Task", lines=3, value=EXAMPLES[1])
            gr.Examples(EXAMPLES, inputs=task)
            ask = gr.Button("2. Ask both models how long it will take", variant="primary")
            with gr.Row():
                base_answer = gr.Textbox(label="Untrained Tiny Aya says")
                g1_answer = gr.Textbox(label="Tiny Aya + G1-kit says")
            guesses = gr.State({})
            note = gr.Markdown()
            ask.click(
                estimate,
                inputs=[task, S, P],
                outputs=[S, P, base_answer, g1_answer, guesses, note],
            )
            go = gr.Button("3. Run the task and time it")
            verdict = gr.Markdown()
            response = gr.Textbox(label="The G1-kit model's response", lines=10)
            go.click(run_task, inputs=[task, guesses], outputs=[response, verdict])
        with gr.Tab("Benchmark results"):
            gr.Markdown(
                "**MetaTime single-turn suite, 68 tasks.** Each model estimates its own run time "
                "before starting and is scored against its own measured time, with the "
                "benchmark's question, parser and metrics. GPT-5.5 and Claude Opus 4.8 are the "
                "benchmark paper's published numbers. Error = median absolute log error "
                "(lower is better), shown with a 95% bootstrap interval."
            )
            gr.Dataframe(
                summary_rows,
                headers=[
                    "System",
                    "Typically off by",
                    "Error [95% CI]",
                    "Within 2x",
                    "Estimate / actual",
                ],
                interactive=False,
            )
            figure = HERE / "data/F9-metatime-head-to-head.png"
            if figure.exists():
                gr.Image(str(figure), show_label=False)
            gr.Markdown(
                "Per task: estimate / measured seconds, each model on its own clock. Task ids "
                "only; the benchmark's task text is not redistributed."
            )
            category = gr.Dropdown(categories, value="all", label="Category")
            table = gr.Dataframe(
                task_rows(tasks, "all"),
                headers=["Task", "Category", "GPT-5.5", "Claude Opus 4.8", "Tiny Aya + G1-kit"],
                interactive=False,
            )
            category.change(lambda c: task_rows(tasks, c), inputs=category, outputs=table)
        with gr.Tab("About"):
            gr.Markdown(
                "The G1-kit adapter was trained with AlignTune on how long the untrained model's "
                "own answers were to the 1,342 public prompts that passed CuratorKIT (Dolly-15k, "
                "CodeAlpaca-20k, no_robots, GSM8K and 100 coding tasks), each shown with a stated "
                "machine speed. It answers with the "
                "work first and the time second, for example `Prompt: about 380 tokens. "
                "Response: about 300 tokens. Time: 9.2 seconds`.\n\n"
                "- Models: https://huggingface.co/isHeSatoshi\n"
                f"- Code, rules declared before the first run, and the write-up: {REPO}\n"
                "- Base model: CohereLabs/tiny-aya-global (CC-BY-NC-4.0). This demo and the "
                "adapter are non-commercial."
            )
    return demo


if __name__ == "__main__":
    start_servers()
    build().queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
