"""Local/private Gradio demo. Publication is a separate, user-approved action."""

import os
from pathlib import Path

import gradio as gr

from anthropochrony.artifacts import read_jsonl
from anthropochrony.demo import BackendUnavailable, Demo, LlamaBackend
from anthropochrony.timemodel import SpeedCard

ROOT = Path(__file__).resolve().parents[1]


def app(root=ROOT, backend=None):
    root = Path(root)
    path = root / "data/demo/tasks.jsonl"
    tasks = read_jsonl(path) if path.exists() else []
    controller = Demo(tasks, backend=backend)
    with gr.Blocks(title="Anthropochrony", analytics_enabled=False) as interface:
        session_card = gr.State(None)
        gr.Markdown(
            "# Anthropochrony\n"
            "### Measuring and Correcting Human-Time Bias in LLM Self-Estimates\n"
            "Compare an agent's own clock with its estimate for a human developer. "
            "Speed cards are measured, not assumed. Pasted tasks are never executed."
        )
        if backend is None:
            gr.Markdown(
                "**Research pipeline in progress.** No trained backend is attached. "
                "This interface does not invent before/after results."
            )
        task = gr.Dropdown(choices=list(controller.tasks), label="Verified task")
        pasted = gr.Textbox(label="Or paste a task (estimates only)", lines=5)
        estimate = gr.Button("Measure speed and estimate", variant="primary")
        with gr.Row():
            base = gr.Textbox(label="Base model: own time")
            tuned = gr.Textbox(label="Anthropochrony: own time")
            human = gr.Textbox(label="Human developer's time")
        speed = gr.JSON(label="Measured serving speed")
        run = gr.Button("Run verified task in sandbox")
        result = gr.JSON(label="Measured run and test results")

        def estimate_action(task_id, custom):
            try:
                output = controller.estimate(task_id, custom)
            except (BackendUnavailable, KeyError) as error:
                raise gr.Error(str(error)) from error
            return (
                output["base"],
                output["anthropochrony"],
                output["human"],
                output["card"],
                output["card"],
            )

        def run_action(task_id, custom, measured_card):
            try:
                if measured_card is None:
                    raise BackendUnavailable(
                        "Measure speed and request estimates in this session first"
                    )
                return controller.run(task_id, pasted=custom, card=SpeedCard(**measured_card))
            except (BackendUnavailable, ValueError, KeyError) as error:
                raise gr.Error(str(error)) from error

        estimate.click(estimate_action, [task, pasted], [base, tuned, human, speed, session_card])
        run.click(run_action, [task, pasted, session_card], result)
    return interface


if __name__ == "__main__":
    backend = None
    if os.environ.get("ANTHROPOCHRONY_GGUF"):
        backend = LlamaBackend(
            os.environ["ANTHROPOCHRONY_BASE_URL"],
            os.environ["ANTHROPOCHRONY_TUNED_URL"],
            os.environ["ANTHROPOCHRONY_GGUF"],
            os.environ["ANTHROPOCHRONY_LLAMA_BENCH"],
        )
    app(backend=backend).launch(server_name="127.0.0.1", share=False)
