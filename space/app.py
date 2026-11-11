"""Local/private Gradio workbench. Publication is a separate, user-approved action.

Exposes:
1. Benchmark, 8-Speed-Card & Real Desktop CPU (`X10` / `PROJ-8`) Trajectory Explorer
2. Zero-GPU Instant CPU Prompt Workload & Hardware Scaling Simulator (never executes pasted code)
3. Empirical Fleet SLA Tripwire, Multi-Tier GPU Routing & Closed-API ROI Calculator
4. Live `llama.cpp` GGUF + Bubblewrap Sandbox Runner (`Demo` + `LlamaBackend`, curated tasks only)

When `root` does not contain `results/metrics.json` (e.g., isolated unit-test directories),
renders the minimal two-action `Demo` sandbox interface (`len(config["dependencies"]) == 2`).
"""

import os
from pathlib import Path

import gradio as gr

from anthropochrony.artifacts import read_jsonl
from anthropochrony.demo import BackendUnavailable, Demo, LlamaBackend
from anthropochrony.eval.structured import structured_duration
from anthropochrony.parse import parse
from anthropochrony.timemodel import SpeedCard

ROOT = Path(__file__).resolve().parents[1]


def _build_minimal_sandbox(controller: Demo, backend) -> gr.Blocks:
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
                "**No live `llama.cpp` backend is attached.** "
                "This interface does not invent live GGUF outputs without a local server."
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


def app(root=ROOT, backend=None):
    root = Path(root)
    path = root / "data/demo/tasks.jsonl"
    tasks = read_jsonl(path) if path.exists() else []
    controller = Demo(tasks, backend=backend)

    if not (root / "results" / "metrics.json").exists():
        return _build_minimal_sandbox(controller, backend)

    from anthropochrony.workbench import COHORTS, SPEED_CARDS, WorkbenchEngine

    engine = WorkbenchEngine(root)

    init_cohort = COHORTS[0]
    init_choices = engine.cohort_choices(init_cohort)
    init_task = init_choices[0][1] if init_choices else ""
    init_prompt_md, init_table_md, init_fig = engine.explore_task(init_cohort, init_task, "REF")

    starter_opts = engine.starter_prompts()
    init_starter = starter_opts[0][1] if starter_opts else ""
    init_starter_txt = engine.starter_prompt_text(init_starter)
    init_sim_md, init_sim_fig = engine.simulate_prompt(init_starter_txt, 96.0, 26000.0, 967.0, 15.0)

    init_roi_md, init_roi_fig = engine.compute_fleet_roi(250.0, 15.0, 1.0, 0.25, 1.80, 2.50)

    with gr.Blocks(
        title="Anthropochrony — Measuring and Correcting Human-Time Bias in LLM Self-Estimates",
        analytics_enabled=False,
    ) as interface:
        session_card = gr.State(None)
        gr.Markdown(
            "# Anthropochrony: Measuring and Correcting Human-Time Bias in LLM Self-Estimates\n"
            "**Backbone:** `CohereLabs/tiny-aya-global` (`3.35B`, QLoRA via CuratorKIT $\\rightarrow$ AlignTune $\\rightarrow$ AuditKIT) "
            "&nbsp;|&nbsp; **Core Finding:** 5-shot prompting (`E2`) nearly ties verbal SFT in-distribution (`0.240` vs `0.242` MdAE) "
            "but collapses on unseen class-level tasks (`CLASS` OOD `0.503`, `30.1%` within 2×). Measured-trajectory SFT (`M1-S0`, `0.300` OOD) + "
            "**Workload Decomposition with an Exact Timing Model (`M1-Struct`, zero retraining)** eliminates the `0.272` $\\text{log}_{10}$ "
            "verbal-arithmetic bottleneck, reaching **`0.269` OOD (`58.1%` within 2×, `4.6×` higher $\\rho$)**, **`0.213` on `test` (`71.8%` within 2×)**, "
            "**3.17× lower slow-hardware error** (`0.155` at `1.5 tok/s`), **63.0% within 2×** on live CPU (`llama.cpp` Q4_K_M), and **10.5× fewer SLA false aborts** (`1.33%`)."
        )

        with gr.Tabs():
            # ==========================================================
            # TAB 1: Benchmark, Speed-Card & Real CPU Explorer
            # ==========================================================
            with gr.Tab("1 · Benchmark & CPU Trajectory Explorer"):
                gr.Markdown(
                    "Inspect pre-computed model self-estimates, **workload decomposition + exact timing (`M1-Struct`)**, **80% conformal SLA intervals**, "
                    "explicit **`Target source` (`own` vs `base`)**, and real $K=8$ measured agent trajectories across held-out `test`, unseen-family `test_ood` (`CLASS`), "
                    "and real `llama.cpp` desktop CPU runs (`X10` & `PROJ-8`). Restricted third-party problem statements are redacted automatically."
                )
                with gr.Row():
                    cohort_dd = gr.Dropdown(
                        choices=list(COHORTS),
                        value=init_cohort,
                        label="Evaluation Cohort",
                        scale=2,
                    )
                    task_dd = gr.Dropdown(
                        choices=init_choices,
                        value=init_task,
                        label="Benchmark Task",
                        scale=3,
                    )
                    card_dd = gr.Dropdown(
                        choices=[
                            (
                                f"{c.id} (S={c.S:g} tok/s, P={c.P:g} tok/s)",
                                c.id,
                            )
                            for c in SPEED_CARDS
                        ],
                        value="REF",
                        label="Hardware Speed Card",
                        scale=2,
                    )
                explore_plot = gr.Plot(
                    value=init_fig, label="Log-Scale Estimator & Trajectory Comparison"
                )
                explore_table = gr.Markdown(value=init_table_md)
                with gr.Accordion("Task Prompt & Provenance (License-Aware)", open=False):
                    explore_prompt = gr.Markdown(value=init_prompt_md)

                def on_cohort_change(cohort_name, card_id):
                    opts = engine.cohort_choices(cohort_name)
                    first_tid = opts[0][1] if opts else ""
                    p_md, t_md, fig = engine.explore_task(cohort_name, first_tid, card_id)
                    return gr.update(choices=opts, value=first_tid), p_md, t_md, fig

                cohort_dd.change(
                    on_cohort_change,
                    [cohort_dd, card_dd],
                    [task_dd, explore_prompt, explore_table, explore_plot],
                )
                task_dd.change(
                    engine.explore_task,
                    [cohort_dd, task_dd, card_dd],
                    [explore_prompt, explore_table, explore_plot],
                )
                card_dd.change(
                    engine.explore_task,
                    [cohort_dd, task_dd, card_dd],
                    [explore_prompt, explore_table, explore_plot],
                )

            # ==========================================================
            # TAB 2: Zero-GPU Live Prompt Workload & Hardware Simulator
            # ==========================================================
            with gr.Tab("2 · Instant CPU Prompt & Hardware Simulator"):
                gr.Markdown(
                    "**Zero-GPU Interactive Mode:** Paste *any* Python coding specification (or load a starter task) and dial your "
                    "serving throughput `(S, P, L)`. Static AST/token features decompose expected `(o_tokens, attempts, exec_s)` via "
                    "train-only structural models and plug into exact serving physics + `val`-calibrated 80% conformal bounds. "
                    "**Pasted code is parsed statically and never executed.**"
                )
                starter_dd = gr.Dropdown(
                    choices=starter_opts,
                    value=init_starter,
                    label="Load Curated Starter Specification",
                )
                live_prompt = gr.Textbox(
                    lines=9,
                    value=init_starter_txt,
                    label="Python Task Specification (Static AST Analysis Only — Never Executed)",
                )
                with gr.Row():
                    sim_s = gr.Slider(
                        1.5, 5000.0, value=96.0, step=0.5, label="Decode Speed S (tok/s)"
                    )
                    sim_p = gr.Slider(
                        50.0, 50000.0, value=26000.0, step=50.0, label="Prefill Speed P (tok/s)"
                    )
                    sim_lat = gr.Slider(
                        0.0, 1500.0, value=967.0, step=1.0, label="Per-Call Serving Latency L (ms)"
                    )
                    sim_sla = gr.Slider(
                        2.0, 120.0, value=15.0, step=1.0, label="SLA Deadline D (seconds)"
                    )
                sim_btn = gr.Button(
                    "Decompose Workload & Simulate Hardware Scaling", variant="primary"
                )
                sim_plot = gr.Plot(
                    value=init_sim_fig, label="Hardware Scaling Curve & 80% Conformal Band"
                )
                sim_md = gr.Markdown(value=init_sim_md)

                def on_starter_select(tid, s, p, lat, sla):
                    txt = engine.starter_prompt_text(tid)
                    md, fig = engine.simulate_prompt(txt, s, p, lat, sla)
                    return txt, md, fig

                starter_dd.change(
                    on_starter_select,
                    [starter_dd, sim_s, sim_p, sim_lat, sim_sla],
                    [live_prompt, sim_md, sim_plot],
                )
                sim_btn.click(
                    engine.simulate_prompt,
                    [live_prompt, sim_s, sim_p, sim_lat, sim_sla],
                    [sim_md, sim_plot],
                )

            # ==========================================================
            # TAB 3: Fleet SLA Tripwire, Tier-Routing & API ROI
            # ==========================================================
            with gr.Tab("3 · SLA Tripwire, Routing & Closed-API ROI"):
                gr.Markdown(
                    "Simulate production coding-agent fleet economics directly on the **2,040 fresh held-out `test` trajectories**. "
                    "Compare self-hosted `M1-Struct+CP` (`3.35B`) against raw verbal bounds and closed frontier estimator APIs (`B4`)."
                )
                with gr.Row():
                    roi_vol = gr.Slider(
                        10.0,
                        5000.0,
                        value=250.0,
                        step=10.0,
                        label="Monthly Volume (Thousands of Tasks)",
                    )
                    roi_sla = gr.Slider(
                        5.0, 60.0, value=15.0, step=1.0, label="SLA Latency Target (seconds)"
                    )
                    roi_k = gr.Slider(
                        0.75, 2.5, value=1.0, step=0.25, label="Timeout Tripwire Multiplier k"
                    )
                with gr.Row():
                    roi_edge = gr.Slider(
                        0.05, 1.50, value=0.25, step=0.05, label="Edge GPU Price ($ / hour)"
                    )
                    roi_dc = gr.Slider(
                        0.50, 6.00, value=1.80, step=0.10, label="Datacenter GPU Price ($ / hour)"
                    )
                    roi_api = gr.Slider(
                        0.20,
                        10.00,
                        value=2.50,
                        step=0.10,
                        label="Closed Frontier API ($ / 1k calls)",
                    )
                roi_btn = gr.Button("Recompute Empirical Fleet Economics", variant="primary")
                roi_plot = gr.Plot(
                    value=init_roi_fig, label="Empirical Tripwire & Multi-Tier TCO Pareto"
                )
                roi_md = gr.Markdown(value=init_roi_md)

                roi_btn.click(
                    engine.compute_fleet_roi,
                    [roi_vol, roi_sla, roi_k, roi_edge, roi_dc, roi_api],
                    [roi_md, roi_plot],
                )

            # ==========================================================
            # TAB 4: Live llama.cpp + Bubblewrap Sandbox Runner
            # ==========================================================
            with gr.Tab("4 · Live llama.cpp + Sandbox Runner"):
                gr.Markdown(
                    "Attach localhost `llama.cpp` servers (`ANTHROPOCHRONY_BASE_URL`, `ANTHROPOCHRONY_TUNED_URL`, "
                    "`ANTHROPOCHRONY_GGUF`, `ANTHROPOCHRONY_LLAMA_BENCH`) to calibrate your machine's real `(S, P)` speed card, "
                    "query live neural self-estimates vs human-frame estimates, and run allowlisted tasks inside bubblewrap (`bwrap`). "
                    "Pasted tasks are never executed."
                )
                if backend is None:
                    gr.Markdown(
                        "**No live `llama.cpp` server attached in this session.** "
                        "Tabs 1–3 run immediately on CPU without a live server; attach local GGUF servers to enable live generation here."
                    )
                task = gr.Dropdown(
                    choices=list(controller.tasks),
                    value=next(iter(controller.tasks), None),
                    label="Verified allowlisted task",
                )
                pasted = gr.Textbox(
                    label="Or paste a task (estimates only — never executed)", lines=5
                )
                estimate = gr.Button("Measure speed and estimate", variant="primary")
                with gr.Row():
                    base = gr.Textbox(label="Base model: own time")
                    tuned = gr.Textbox(label="Anthropochrony verbal + M1-Struct (L=0)")
                    human = gr.Textbox(label="Human developer's time")
                speed = gr.JSON(label="Measured serving speed")
                run = gr.Button("Run verified task in sandbox")
                result = gr.JSON(label="Measured run and test results")

                def estimate_action(task_id, custom):
                    try:
                        output = controller.estimate(task_id, custom)
                    except (BackendUnavailable, KeyError) as error:
                        raise gr.Error(str(error)) from error
                    tuned_text = output["anthropochrony"]
                    parsed = parse(tuned_text)
                    card = SpeedCard(**output["card"])
                    u_tok = (
                        max(24.0, len(custom) / 3.6)
                        if (custom or "").strip()
                        else float(
                            engine.tasks_by_id.get(task_id, {}).get("uncached_task_tokens", 160.0)
                        )
                    )
                    fam = (
                        "FUNC"
                        if (custom or "").strip()
                        else engine.tasks_by_id.get(task_id, {}).get("family", "FUNC")
                    )
                    struct_s = structured_duration(
                        parsed.tokens,
                        parsed.attempts,
                        u_tok,
                        fam,
                        card,
                        engine.priors,
                        latency_s=0.0,
                    )
                    if struct_s is not None:
                        tuned_text = (
                            f"{tuned_text}\n\n"
                            f"[M1-Struct L=0: {struct_s:.2f}s | 80% Conformal SLA: "
                            f"{struct_s / 2.087:.2f}s .. {struct_s * 2.087:.2f}s]"
                        )
                    return (
                        output["base"],
                        tuned_text,
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
                        return controller.run(
                            task_id, pasted=custom, card=SpeedCard(**measured_card)
                        )
                    except (BackendUnavailable, ValueError, KeyError) as error:
                        raise gr.Error(str(error)) from error

                estimate.click(
                    estimate_action, [task, pasted], [base, tuned, human, speed, session_card]
                )
                run.click(run_action, [task, pasted, session_card], result)
    return interface


if __name__ == "__main__":
    live_backend = None
    if os.environ.get("ANTHROPOCHRONY_GGUF"):
        live_backend = LlamaBackend(
            os.environ["ANTHROPOCHRONY_BASE_URL"],
            os.environ["ANTHROPOCHRONY_TUNED_URL"],
            os.environ["ANTHROPOCHRONY_GGUF"],
            os.environ["ANTHROPOCHRONY_LLAMA_BENCH"],
        )
    app(backend=live_backend).launch(server_name="127.0.0.1", share=False)
