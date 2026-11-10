"""Write model and dataset cards into artifacts/release (private staging).

Numbers are read from results/metrics.json so cards cannot drift from the evaluation.
Usage: uv run python scripts/build_cards.py [--models-collection URL --data-collection URL]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "isHeSatoshi"
BASE = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
GITHUB = f"https://github.com/{USER}/anthropochrony"
AUP = "https://docs.cohere.com/docs/c4ai-acceptable-use-policy"

NOTICE = (
    "This work is a derivative of Cohere Labs Tiny Aya Global and is released under "
    "**CC-BY-NC-4.0** (non-commercial). Use is also subject to the "
    f"[Cohere Labs Acceptable Use Policy]({AUP}). Attribution: Tiny Aya Global, Cohere Labs."
)

DESCRIPTIONS = {
    "m1-s0": (
        "**Main model (M1, seed 0).** Trained on the model's own measured, curated completion "
        "times, with a spread of speed cards and phrasings, plus human-frame preservation examples."
    ),
    "m2-s0": (
        "**Control (M2, seed 0).** Same tasks, formats and size as M1, but the labels are a "
        "teacher model's guessed times (DeepSeek-V4-Pro). Used to test measured vs guessed labels."
    ),
    "m3-s0": (
        "**Control (M3, seed 0).** Measured labels like M1 but with the CuratorKIT quality gates "
        "skipped (uncurated). Used to test whether curation helps."
    ),
    "lr-0.0001": "One-epoch learning-rate sweep run (LR 1e-4) on the M1 data. Not for use.",
    "lr-0.0002": "One-epoch learning-rate sweep run (LR 2e-4) on the M1 data. Not for use.",
}

LIMITS = """## Limitations

- One training seed. No seed-to-seed variance was estimated.
- In distribution, base Tiny Aya with five measured examples in the prompt is as accurate as M1 (median error 0.24 log10 in both), and a text-feature regressor is more accurate (0.14).
- Stated 80% ranges cover about 19% of fresh runs (poorly calibrated).
- Weak ranking on a task family never seen in training (Spearman 0.08), and poor extrapolation to a much slower stated speed (0.49).
- Estimates are about *this* model on *this* task format (Python tasks, up to 3 attempts, 8 sampled runs, stated speeds). They are not predictions for other models or for real hardware unless the stated speed is the real speed.
- Merged bf16 weights flip a few near-tied digits compared with the PEFT adapter (see the preregistration deviation on G6)."""


def metric(rows, estimator, split, frame="self"):
    for row in rows:
        if (
            row["estimator"] == estimator
            and row["split"] == split
            and row["frame"] == frame
            and row["card_id"] == "REF"
            and row["target_source"] in ("own", "base")
        ):
            if estimator.startswith("M") and row["target_source"] != "own":
                continue
            return row
    return None


def adapter_card(name, metrics, collection):
    run = name
    header = f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
tags: [lora, tiny-aya, self-estimation, time-estimation, coding-agents]
---
"""
    results = ""
    key = run.replace("-s0", "-S0").upper()
    row_test = metric(metrics, key, "test")
    row_ood = metric(metrics, key, "test_ood")
    if row_test and row_ood:
        results = f"""## Results (reference speed card, one seed)

| Split | n | Median abs. log10 error | Within 2× | Spearman |
|---|---|---|---|---|
| test (held-out tasks) | {row_test["n_tasks"]} | {row_test["mdae_log10"]:.3f} | {row_test["within2x"]:.1%} | {row_test["spearman"]:.2f} |
| test_ood (unseen task family) | {row_ood["n_tasks"]} | {row_ood["mdae_log10"]:.3f} | {row_ood["within2x"]:.1%} | {row_ood["spearman"]:.2f} |

Errors are against this model's own re-measured median completion time. Base model with the speed card: 1.375 (test) and 1.199 (test_ood). Full tables, CIs and hypotheses: [article]({GITHUB}/blob/main/paper/article.md).
"""
    capability = ""
    if run == "m1-s0":
        capability = (
            "\n**Capability check.** IFEval (first 200 prompts) prompt-level strict accuracy fell from "
            "0.750 (base) to 0.700 (one seed, binomial SE about 3 points); ARC-Challenge "
            "acc_norm 0.458 to 0.456; pass@1 on held-out coding tasks 0.209 to 0.214.\n"
        )
    return (
        header
        + f"""# Anthropochrony LoRA adapter: {run}

{DESCRIPTIONS[run]}

{NOTICE}

**What it does.** Given a system prompt that states the serving speed (decode and prefill tokens per second) and a coding task, the model answers "how long will it take you to finish this task?" with an estimate in seconds, an interval and a short token-and-speed breakdown. It is trained on the model's own measured run times.

**Base model.** [`{BASE}`](https://huggingface.co/{BASE}) at revision `{REVISION}`, bf16.

**Training.** LoRA r=16, alpha=32, dropout 0.05, all attention and MLP projections, 2 epochs (the sweep runs: 1), assistant-only loss, max sequence length 3,072, effective batch 32, one L40S. Learning rate 2e-4 (chosen on validation from {{1e-4, 2e-4}}). Pipeline: CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6. Training logs: `{USER}/anthropochrony-logs`.

**Use.** Merge the adapter into the bf16 base model before serving (vLLM crashes serving LoRA on this architecture):

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{BASE}", revision="{REVISION}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{USER}/anthropochrony-tiny-aya-global-lora-{run}").merge_and_unload()
```

Use the exact prompts in the repository (`docs/PROMPTS.md`).

{results}{capability}
**Intended use.** Research on model self-knowledge, scheduling and budgeting experiments for coding agents, non-commercial. **Out of scope:** safety-critical timing, commercial use, estimates for other models or hardware without matching speed cards.

{LIMITS}

**Code and write-up.** {GITHUB}
**Collection.** {collection}
"""
    )


def gguf_card(metrics, collection):
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
tags: [gguf, llama.cpp, tiny-aya, self-estimation]
---
# Anthropochrony GGUF (merged M1-s0 and base)

{NOTICE}

Files (llama.cpp tag b12110):

| File | What |
|---|---|
| `m1-s0-q4_k_m.gguf`, `m1-s0-q8_0.gguf` | Base `{BASE}` @ `{REVISION}` merged with the M1 seed-0 adapter |
| `base-q4_k_m.gguf` | Un-tuned base model, for before/after comparisons |

SHA-256 values are in `gguf-manifest.json`. On an 8-thread desktop CPU, M1 Q4_K_M ran at about 200 tokens/s prefill and 20 tokens/s decode.

**Evidence.** 100 held-out FUNC/LIB/ALGO_FN tasks run end to end in the agent loop with the CPU's measured speeds in the prompt: tuned median |log10 error| 0.279 and 52% within 2× of the measured wall-clock; base 0.935 and 3% (Q4_K_M).

**Caveat.** This GGUF was exported before the merged-parity gate (G6) was formally passed, from the same adapter that later passed. Quantization adds a further perturbation to the numbers shown in the article, which are for bf16 merged serving.

{LIMITS}

**Code and write-up.** {GITHUB}
**Collection.** {collection}
"""


DATASETS = {
    "anthropochrony-tasks": (
        "Task manifest",
        "1,465 Python tasks after CuratorKIT filtering (869 train, 114 validation, 255 test, 93 test_ood unseen family, 134 extra), with source, license, split, near-duplicate group, token counts and ground-truth timing of the reference solution. Also the drop log, the PROJ-8 showcase tasks (our own, Apache-2.0) and `rebuild_messages.py`.",
        "Rows from LeetCodeDataset, CodeContests and TACO carry **IDs and provenance only** (`restricted_statement: true`): no statement, tests, reference code or paraphrase. Rebuild them from the pinned public sources with `rebuild_messages.py` (see the repository). ClassEval rows are CC BY-NC 4.0.",
    ),
    "anthropochrony-traces": (
        "Measured runs",
        "Per-run summaries for the base model and for M1, M2 and M3 seed 0 (K=8 runs per task, up to 3 attempts): token counts, finish reasons, test status and wall time, plus generated text for unrestricted sources, the labels built from them (`labels-*.parquet`, one row per task and speed card), and the PROJ-8 showcase runs.",
        "Generated text for tasks from LeetCodeDataset, CodeContests and TACO is withheld. Run summaries are kept.",
    ),
    "anthropochrony-sft": (
        "Fine-tuning sets",
        "The M1 (measured, curated), M2 (teacher-guessed) and M3 (measured, uncurated) supervised sets: chat `messages`, label, label source, frame (self or human), speed card id, and the drop log with reasons. M1 has 6,077 training examples (5,214 self-frame, 863 human-frame).",
        "User turns for restricted sources contain the marker `{{TASK_TURN}}` instead of the statement; human-frame answers for those rows are cut to their first line.",
    ),
    "anthropochrony-eval": (
        "Estimates, results and figures",
        "Every estimate from every estimator on every split, frame and speed card (`estimates.jsonl`, 55k rows), the parsed values, per-task analysis rows, `results/` (metrics, hypotheses, simulations, ROI, extra analyses, gate evidence), the CPU deployment results (`x10/`), capability pairs and the figures. `notebooks/reproduce_results.ipynb` in the code repository rebuilds the headline table from the `results/` folder here.",
        "Raw model answers and statements are withheld for restricted sources, and few-shot prompts are withheld for all rows because they embed demonstration tasks.",
    ),
    "anthropochrony-logs": (
        "Logs and provenance",
        "`RUNLOG.md` (timestamped run log), the pre-registration with dated deviations, the plan and prompts, budget ledger, acceptance-gate evidence (G0 to G6), training logs and `trainer_state.json` for every adapter, HF Jobs summaries, lm-eval results, and the speed reference.",
        "No credentials. Job IDs and spend are included on purpose.",
    ),
}


def dataset_card(name, collection):
    title, contents, restrictions = DATASETS[name]
    return f"""---
license: cc-by-nc-4.0
language: [en]
tags: [coding-agents, time-estimation, self-estimation, tiny-aya]
pretty_name: {name}
---
# {name}: {title}

Part of **Anthropochrony: Measuring and Correcting Human-Time Bias in LLM Self-Estimates** (Track 1, problem #8). Code, article and reproduction: {GITHUB}

**Contents.** {contents}

**Restrictions.** {restrictions}

**Licenses.** Our annotations, labels and run data: CC-BY-NC-4.0 (they derive from Tiny Aya Global outputs and include CC BY-NC 4.0 ClassEval). Upstream task sources keep their own licenses: HumanEval+/MBPP+ Apache-2.0 (HumanEval MIT, MBPP CC-BY-4.0), BigCodeBench Apache-2.0, ClassEval CC BY-NC 4.0, CodeContests CC-BY-4.0, LeetCodeDataset Apache-2.0 metadata with LeetCode-owned statements, TACO Apache-2.0 metadata with third-party statements. Model-generated text is from [{BASE}](https://huggingface.co/{BASE}) (CC-BY-NC-4.0 plus the [AUP]({AUP})). Teacher guesses are from DeepSeek-V4-Pro.

**Intended use.** Research on model self-knowledge and time estimation for coding agents. Not for commercial use.

**Limitations.** One training seed; labels are medians of 8 sampled runs under stated speeds, not measurements on the stated hardware; Python tasks only; the parser audit covered 60 rows and was done by the research agent.

**Collection.** {collection}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-collection", default="")
    parser.add_argument("--data-collection", default="")
    args = parser.parse_args()
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    out = ROOT / "artifacts/release"
    for run in DESCRIPTIONS:
        target = out / "models" / f"anthropochrony-tiny-aya-global-lora-{run}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(adapter_card(run, metrics, args.models_collection))
    target = out / "models/anthropochrony-tiny-aya-global-gguf"
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(gguf_card(metrics, args.models_collection))
    for name in DATASETS:
        target = out / "datasets" / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(dataset_card(name, args.data_collection))


if __name__ == "__main__":
    main()
